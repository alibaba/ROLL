"""Exercise ROLL phase transitions with the installed HybridDeviceOptimizer."""
import copy
import inspect
import os
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.skipif(os.environ.get("RUN_CPU_OPTIMIZER_TESTS") != "1", reason="requires Megatron")
@pytest.mark.parametrize("requested, expected", [(None, True), (False, False), (True, True)])
def test_bf16_gradient_precision_preserves_explicit_choice(tmp_path, requested, expected):
    from mcore_adapter import TrainingArguments

    kwargs = {} if requested is None else {"accumulate_allreduce_grads_in_fp32": requested}
    args = TrainingArguments(output_dir=str(tmp_path), bf16=True, use_cpu=True, report_to=[], **kwargs)
    assert args.accumulate_allreduce_grads_in_fp32 is expected


@pytest.mark.skipif(os.environ.get("RUN_CPU_OPTIMIZER_TESTS") != "1", reason="requires Megatron")
def test_typed_arguments_accept_bounded_staging(tmp_path):
    from mcore_adapter import TrainingArguments

    args = TrainingArguments(
        output_dir=str(tmp_path), use_cpu=True, report_to=[],
        optimizer_cpu_offload=True, overlap_cpu_optimizer_d2h_h2d=True,
        bounded_cpu_grad_staging=True,
    )
    assert args.bounded_cpu_grad_staging is True


@pytest.mark.skipif(os.environ.get("RUN_CPU_OPTIMIZER_TESTS") != "1", reason="requires Megatron")
@pytest.mark.parametrize("missing", ["optimizer_cpu_offload", "overlap_cpu_optimizer_d2h_h2d"])
def test_typed_arguments_reject_incompatible_bounded_staging(tmp_path, missing):
    from mcore_adapter import TrainingArguments

    kwargs = dict(optimizer_cpu_offload=True, overlap_cpu_optimizer_d2h_h2d=True,
                  bounded_cpu_grad_staging=True)
    kwargs[missing] = False
    with pytest.raises(ValueError, match=missing):
        TrainingArguments(output_dir=str(tmp_path), use_cpu=True, report_to=[], **kwargs)


@pytest.mark.skipif(os.environ.get("RUN_CPU_OPTIMIZER_TESTS") != "1", reason="requires Megatron")
def test_bounded_staging_requires_dependency_patch_before_distributed_setup(monkeypatch):
    from megatron.core.optimizer import OptimizerConfig
    from megatron.core.optimizer.cpu_offloading import HybridDeviceOptimizer
    from roll.third_party.megatron.optimizer import get_megatron_optimizer

    signature = inspect.signature(HybridDeviceOptimizer.__init__)
    signature = signature.replace(parameters=[p for p in signature.parameters.values()
                                              if p.name != "bounded_cpu_grad_staging"])
    monkeypatch.setattr(HybridDeviceOptimizer.__init__, "__signature__", signature, raising=False)
    config = OptimizerConfig(optimizer_cpu_offload=True, overlap_cpu_optimizer_d2h_h2d=True)
    config.bounded_cpu_grad_staging = True
    with pytest.raises(RuntimeError, match="patch_megatron_cpu_grad_staging"):
        get_megatron_optimizer(config, [])


@pytest.mark.skipif(os.environ.get("RUN_CPU_OPTIMIZER_TESTS") != "1", reason="requires Megatron and CUDA")
def test_roll_gpu_optimizer_does_not_require_cpu_staging_patch(tmp_path, monkeypatch):
    from pathlib import Path
    import torch.distributed as dist
    from megatron.core import parallel_state, tensor_parallel
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.optimizer import OptimizerConfig
    from megatron.core.optimizer.cpu_offloading import HybridDeviceOptimizer
    from roll.third_party.megatron.optimizer import get_megatron_optimizer
    from roll.third_party.megatron.optimizer_config import build_optimizer_config

    signature = inspect.signature(HybridDeviceOptimizer.__init__)
    signature = signature.replace(parameters=[p for p in signature.parameters.values()
                                              if p.name != "bounded_cpu_grad_staging"])
    monkeypatch.setattr(HybridDeviceOptimizer.__init__, "__signature__", signature, raising=False)
    monkeypatch.syspath_prepend(str(Path(__file__).parents[3] / "mcore_adapter/tests"))
    from test_qwen4_exp_model import tiny_config, make_model

    dist.init_process_group("nccl", init_method=f"file://{tmp_path}/rdzv", rank=0, world_size=1)
    parallel_state.initialize_model_parallel(1, 1)
    tensor_parallel.model_parallel_cuda_manual_seed(734)
    try:
        model_config = tiny_config()
        wrapped = DistributedDataParallel(model_config, DistributedDataParallelConfig(
            grad_reduce_in_fp32=False, use_distributed_optimizer=True, overlap_grad_reduce=False,
        ), make_model(model_config))
        config = build_optimizer_config(OptimizerConfig, dict(
            optimizer="adam", lr=0.001, bf16=True, params_dtype=torch.bfloat16,
            use_distributed_optimizer=True,
        ), SimpleNamespace())
        optimizer = get_megatron_optimizer(config, [wrapped])
        leaves = getattr(optimizer, "chained_optimizers", [optimizer])
        assert all(not isinstance(leaf.optimizer, HybridDeviceOptimizer) for leaf in leaves)
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


@pytest.mark.skipif(os.environ.get("RUN_CPU_OPTIMIZER_TESTS") != "1", reason="requires Megatron and CUDA")
@pytest.mark.parametrize("fraction", [1.0, 0.5])
def test_hybrid_adam_phase_keeps_cpu_state_and_next_update(fraction):
    from megatron.core.optimizer.cpu_offloading import HybridDeviceOptimizer
    from roll.third_party.megatron.offload_states_patch import offload_adam_states, reload_adam_states

    params = [torch.nn.Parameter(torch.tensor([1.0, -2.0], device="cuda")) for _ in range(2)]
    baseline = [torch.nn.Parameter(p.detach().clone()) for p in params]

    def make_optimizer(parameters):
        return HybridDeviceOptimizer(
            parameters, offload_fraction=fraction, cpu_optimizer_cls=torch.optim.AdamW,
            gpu_optimizer_cls=torch.optim.AdamW, overlap_cpu_optimizer_d2h_h2d=False,
            param_update_in_fp32=True, lr=0.01,
        )

    optimizer, reference = make_optimizer(params), make_optimizer(baseline)
    for step in range(2):
        for parameter in params + baseline:
            parameter.grad = torch.tensor([0.25, -0.5], device="cuda") * (step + 1)
        optimizer.step()
        reference.step()
        torch.cuda.synchronize()
        if step == 0:
            cpu_moments = [state[key] for sub in optimizer.cpu_optimizers for state in sub.state.values()
                           for key in ("exp_avg", "exp_avg_sq")]
            pointers = [value.data_ptr() for value in cpu_moments]
            offload_adam_states(optimizer, torch.device("cpu"))
            reload_adam_states(optimizer, torch.device("cuda", torch.cuda.current_device()))
            assert all(value.device.type == "cpu" for value in cpu_moments)
            assert [value.data_ptr() for value in cpu_moments] == pointers
    for actual, expected in zip(params, baseline):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.skipif(os.environ.get("RUN_CPU_OPTIMIZER_TESTS") != "1", reason="requires Megatron and CUDA")
@pytest.mark.parametrize("bounded", [False, True])
def test_roll_distributed_cpu_optimizer_update_and_phase(tmp_path, monkeypatch, bounded):
    """Real ROLL optimizer construction and DDP hooks must survive CPU phases."""
    import sys
    from pathlib import Path
    import torch.distributed as dist
    from megatron.core import parallel_state, tensor_parallel
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.distributed.finalize_model_grads import finalize_model_grads
    from megatron.core.optimizer import OptimizerConfig
    from megatron.core.optimizer.cpu_offloading import HybridDeviceOptimizer
    from roll.third_party.megatron.optimizer import get_megatron_optimizer
    from roll.third_party.megatron.optimizer_config import build_optimizer_config
    from roll.third_party.megatron.offload_states_patch import bind_megatron_offload_states_func

    monkeypatch.syspath_prepend(str(Path(__file__).parents[3] / "mcore_adapter/tests"))
    from test_qwen4_exp_model import tiny_config, make_model
    torch.cuda.set_device(0)
    dist.init_process_group("nccl", init_method=f"file://{tmp_path}/rdzv", rank=0, world_size=1)
    parallel_state.initialize_model_parallel(1, 1)
    tensor_parallel.model_parallel_cuda_manual_seed(733)
    try:
        config = tiny_config()
        model = make_model(config)
        # Exercise dense and expert ownership with one CUDA device. At EP=1
        # Megatron normally puts both kinds of parameter in the dense group.
        for name, parameter in model.named_parameters():
            if ".mlp.experts." in name:
                parameter.allreduce = False
        wrapped = DistributedDataParallel(config, DistributedDataParallelConfig(
            grad_reduce_in_fp32=False, use_distributed_optimizer=True, overlap_grad_reduce=False,
        ), model)
        train_args = SimpleNamespace(
            optimizer_cpu_offload=True, optimizer_offload_fraction=1.0,
            use_precision_aware_optimizer=True,
            overlap_cpu_optimizer_d2h_h2d=True, bounded_cpu_grad_staging=bounded,
        )
        optimizer_config = build_optimizer_config(OptimizerConfig, dict(
            optimizer="adam", lr=0.001, bf16=True, params_dtype=torch.bfloat16,
            use_distributed_optimizer=True, clip_grad=0.2,
        ), train_args)
        optimizer = get_megatron_optimizer(optimizer_config, [wrapped])
        leaves = getattr(optimizer, "chained_optimizers", [optimizer])
        assert len(leaves) == 2, "tiny MoE model must exercise dense and expert Hybrid instances"
        assert all(isinstance(leaf.optimizer, HybridDeviceOptimizer) for leaf in leaves)
        assert all(leaf.optimizer.bounded_cpu_grad_staging is bounded for leaf in leaves)
        bind_megatron_offload_states_func(optimizer)
        ids = torch.arange(32, device="cuda").remainder(15).add(1).unsqueeze(0)
        before = model.embedding.word_embeddings.weight.detach().clone()
        for step in range(2):
            wrapped.zero_grad_buffer()
            optimizer.zero_grad()
            losses = wrapped(ids, torch.arange(32, device="cuda").unsqueeze(0), None,
                             labels=ids.roll(-1, -1))
            losses.mean().backward()
            finalize_model_grads([wrapped])
            if bounded and step == 1:
                saved_model = copy.deepcopy(model.state_dict())
                saved_grads = {name: p.main_grad.clone() for name, p in model.named_parameters()
                               if p.requires_grad}
                saved_optimizers = []
                saved_inner_states = [copy.deepcopy(leaf.optimizer.state_dict()) for leaf in leaves]
                for leaf in leaves:
                    state = copy.deepcopy(leaf.state_dict())
                    state["param_state"] = copy.deepcopy(leaf.get_parameter_state_dp_zero())
                    state["param_state_sharding_type"] = "dp_zero_gather_scatter"
                    saved_optimizers.append(state)
            updated, _, _ = optimizer.step()
            assert updated
            torch.cuda.synchronize()
            if bounded:
                for leaf in leaves:
                    hybrid = leaf.optimizer
                    largest = max(p.numel() * p.element_size() for p in hybrid.cpu_copys_map_gpu_param)
                    assert hybrid._cpu_grad_staging_buffer.numel() == largest
                    assert not hybrid.cpu_copy_map_grad
                    assert all(p.grad is None for p in hybrid.cpu_copys_map_gpu_param)
                    assert all(state[key].dtype == torch.float32 and state[key].device.type == "cpu"
                               for state in hybrid.state.values()
                               for key in ("master_param", "exp_avg", "exp_avg_sq"))
            if step == 0:
                optimizer.offload_states()
                optimizer.reload_states()
        assert not torch.equal(before, model.embedding.word_embeddings.weight)
        if bounded:
            expected_model = copy.deepcopy(model.state_dict())
            expected_state = [copy.deepcopy(leaf.optimizer.state_dict()) for leaf in leaves]
            model.load_state_dict(saved_model)
            # Reconstruct through ROLL to exercise cold distributed resume,
            # including Hybrid's dummy initialization before state loading.
            optimizer = get_megatron_optimizer(optimizer_config, [wrapped])
            leaves = getattr(optimizer, "chained_optimizers", [optimizer])
            assert all(not leaf.optimizer.state for leaf in leaves)
            for leaf, state, saved_inner in zip(leaves, saved_optimizers, saved_inner_states):
                leaf.load_state_dict(state)
                assert leaf.optimizer.bounded_cpu_grad_staging
                for group, saved_group in zip(leaf.optimizer.param_groups, saved_inner["param_groups"]):
                    for parameter, saved_id in zip(group["params"], saved_group["params"]):
                        expected_master = saved_inner["state"][saved_id]["master_param"]
                        actual_master = leaf.optimizer.param_to_fp32_param[parameter]
                        torch.testing.assert_close(actual_master, expected_master, atol=0, rtol=0,
                                                   msg="distributed checkpoint must restore the CPU optimizer's actual master")
            for name, p in model.named_parameters():
                if p.requires_grad:
                    p.main_grad.copy_(saved_grads[name])
            updated, _, _ = optimizer.step()
            assert updated
            torch.cuda.synchronize()
            torch.testing.assert_close(model.state_dict(), expected_model, atol=0, rtol=0)
            for leaf, expected in zip(leaves, expected_state):
                torch.testing.assert_close(leaf.optimizer.state_dict()["state"], expected["state"], atol=0, rtol=0)
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()
