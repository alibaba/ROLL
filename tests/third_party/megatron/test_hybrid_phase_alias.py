"""Actual DistributedOptimizer trajectories and storage ownership across RL phases.

Run directly with pytest (one rank), or torchrun --nproc_per_node=4 -m pytest
for dense DP4 and expert DP2 / EP2 coverage. Requires the installed Megatron.
"""
import gc
from functools import wraps
import os
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _leaves(optimizer):
    return getattr(optimizer, "chained_optimizers", [optimizer])


def _cpu_state(optimizer):
    return [(value, value.data_ptr()) for leaf in _leaves(optimizer)
            for state in leaf.optimizer.state.values()
            for key, value in state.items()
            if key in ("master_param", "exp_avg", "exp_avg_sq")]


def _assert_cpu_state_identity(optimizer, saved):
    current = _cpu_state(optimizer)
    assert len(current) == len(saved)
    assert all(actual is expected and actual.device.type == "cpu" and pointer == saved_pointer
               for (actual, pointer), (expected, saved_pointer) in zip(current, saved))


def _assert_current_shards(optimizer):
    for leaf in _leaves(optimizer):
        hybrid = leaf.optimizer
        current = {id(p): p for group in leaf.shard_float16_groups for p in group}
        originals = [p for group in hybrid.param_groups for p in group["params"]]
        assert {id(p) for p in originals} == set(current), "Hybrid owns stale original shards"
        for original in originals:
            master = hybrid.param_to_fp32_param[original]
            assert hybrid.gpu_params_map_cpu_copy[original] is master
            assert hybrid.cpu_copys_map_gpu_param[master] is original
            assert hybrid.param_to_inner_param[original] is master
            assert hybrid.inner_param_to_orig_param[master] is original
            assert hybrid.fp32_param_to_orig_param[master] is original
            assert original in hybrid.state
        assert set(map(id, hybrid.state)) == set(current)
        for group_index, group_range in enumerate(leaf.opt_group_ranges):
            for shard, model_param in zip(leaf.shard_float16_groups[group_index], group_range["params"]):
                buf, dtype, bucket = leaf.model_param_gbuf_map[model_param]
                interval = leaf.gbuf_ranges[buf][dtype][bucket]["param_map"][model_param]["param"]
                expected = model_param.detach().view(-1)[interval.start:interval.end]
                assert shard.data_ptr() == expected.data_ptr()
                assert shard.device == expected.device


@pytest.fixture(scope="module")
def hybrid_parallel(tmp_path_factory):
    import torch.distributed as dist
    from megatron.core import parallel_state

    world = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    if world > 1:
        dist.init_process_group("nccl")
    else:
        path = tmp_path_factory.mktemp("hybrid-phase") / "rdzv"
        dist.init_process_group("nccl", init_method=f"file://{path}", rank=0, world_size=1)
    ep = 2 if world >= 2 else 1
    parallel_state.initialize_model_parallel(1, 1, expert_model_parallel_size=ep,
                                             expert_tensor_parallel_size=1)
    try:
        yield world, ep
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


@pytest.mark.skipif(os.environ.get("RUN_CPU_OPTIMIZER_TESTS") != "1", reason="requires Megatron and CUDA")
def test_optimizer_factory_releases_replaced_cpu_master_owners(hybrid_parallel, monkeypatch):
    """Temporary pre-shard Hybrid masters must not survive into cold restore."""
    from megatron.core import tensor_parallel
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.optimizer import OptimizerConfig
    from megatron.core.optimizer.cpu_offloading import HybridDeviceOptimizer
    from roll.third_party.megatron.optimizer import get_megatron_optimizer
    from roll.third_party.megatron.optimizer_config import build_optimizer_config

    monkeypatch.syspath_prepend(str(Path(__file__).parents[3] / "mcore_adapter/tests"))
    from test_qwen4_exp_model import tiny_config, make_model

    tensor_parallel.model_parallel_cuda_manual_seed(733)
    config = tiny_config()
    config.expert_model_parallel_size = hybrid_parallel[1]
    model = make_model(config)
    for name, param in model.named_parameters():
        if ".mlp.experts." in name:
            param.allreduce = False
    wrapped = DistributedDataParallel(config, DistributedDataParallelConfig(
        grad_reduce_in_fp32=False, use_distributed_optimizer=True, overlap_grad_reduce=False,
    ), model)
    optimizer_config = build_optimizer_config(OptimizerConfig, dict(
        optimizer="adam", lr=0.001, bf16=True, params_dtype=torch.bfloat16,
        use_distributed_optimizer=True, clip_grad=0.2,
    ), SimpleNamespace(optimizer_cpu_offload=True, optimizer_offload_fraction=1.0,
                       use_precision_aware_optimizer=True,
                       overlap_cpu_optimizer_d2h_h2d=True, bounded_cpu_grad_staging=True))
    initialized = []
    original = HybridDeviceOptimizer.__init__

    @wraps(original)
    def record_construction(self, *args, **kwargs):
        original(self, *args, **kwargs)
        initialized.append((weakref.ref(self), [weakref.ref(p) for p in self.param_to_fp32_param.values()]))

    monkeypatch.setattr(HybridDeviceOptimizer, "__init__", record_construction)
    # The factory must release large retired owners regardless of the process's
    # automatic GC threshold or allocation history.
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        optimizer = get_megatron_optimizer(optimizer_config, [wrapped])
        active = {id(leaf.optimizer) for leaf in _leaves(optimizer)}
        stale = [reference() for reference, _ in initialized
                 if reference() is not None and id(reference()) not in active]
        stale_bytes = sum(p.numel() * p.element_size() for owner in stale
                          for p in owner.param_to_fp32_param.values())
        assert not stale, f"Replaced Hybrid owners retain {stale_bytes} CPU master bytes"
        assert all(p() is None for owner, refs in initialized if owner() is None for p in refs)
        assert all(leaf.optimizer.param_to_fp32_param for leaf in _leaves(optimizer))
    finally:
        if was_enabled:
            gc.enable()
        gc.collect()


@pytest.mark.skipif(os.environ.get("RUN_CPU_OPTIMIZER_TESTS") != "1", reason="requires Megatron and CUDA")
@pytest.mark.parametrize("bounded,from_master", [(False, False), (True, False), (True, True)],
                         ids=["ordinary", "bounded", "cpu_master"])
def test_distributed_hybrid_phase_matches_uninterrupted_trajectory(hybrid_parallel, monkeypatch, bounded, from_master):
    import torch.distributed as dist
    from megatron.core import tensor_parallel
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.distributed.finalize_model_grads import finalize_model_grads
    from megatron.core.optimizer import OptimizerConfig
    from roll.third_party.megatron.optimizer import get_megatron_optimizer
    from roll.third_party.megatron.optimizer_config import build_optimizer_config
    from roll.third_party.megatron.offload_states_patch import (
        bind_megatron_offload_states_func, MegatronOffloadStateType,
    )

    monkeypatch.syspath_prepend(str(Path(__file__).parents[3] / "mcore_adapter/tests"))
    from test_qwen4_exp_model import tiny_config, make_model

    world, ep = hybrid_parallel

    def build(master_offload=False):
        torch.manual_seed(733)
        tensor_parallel.model_parallel_cuda_manual_seed(733)
        config = tiny_config()
        config.expert_model_parallel_size = ep
        config.expert_tensor_parallel_size = 1
        model = make_model(config)
        for name, param in model.named_parameters():
            if ".mlp.experts." in name:
                param.allreduce = False
        wrapped = DistributedDataParallel(config, DistributedDataParallelConfig(
            grad_reduce_in_fp32=False, use_distributed_optimizer=True, overlap_grad_reduce=False,
        ), model)
        optimizer_config = build_optimizer_config(OptimizerConfig, dict(
            optimizer="adam", lr=0.001, bf16=True, params_dtype=torch.bfloat16,
            use_distributed_optimizer=True, clip_grad=0.2,
        ), SimpleNamespace(optimizer_cpu_offload=True, optimizer_offload_fraction=1.0,
                           use_precision_aware_optimizer=True,
                           overlap_cpu_optimizer_d2h_h2d=True, bounded_cpu_grad_staging=bounded,
                           offload_model_from_cpu_master=master_offload))
        optimizer = get_megatron_optimizer(optimizer_config, [wrapped])
        assert len(_leaves(optimizer)) == 2
        dense, expert = _leaves(optimizer)
        assert dist.get_world_size(dense.data_parallel_group) == world
        assert dist.get_world_size(expert.data_parallel_group) == world // ep
        bind_megatron_offload_states_func(optimizer)
        return model, wrapped, optimizer

    actual, actual_ddp, optimizer = build(from_master)
    baseline, baseline_ddp, reference = build()
    if from_master:
        # BaseWorker parks the model before its first optimizer step too.
        optimizer.offload_states()
        optimizer.reload_states()
        torch.testing.assert_close(actual.state_dict(), baseline.state_dict(), atol=0, rtol=0)
    ids = torch.arange(32, device="cuda").remainder(15).add(1).unsqueeze(0)
    positions = torch.arange(32, device="cuda").unsqueeze(0)

    def update(wrapped, opt, step):
        torch.manual_seed(1000 + step)
        tensor_parallel.model_parallel_cuda_manual_seed(1000 + step)
        wrapped.zero_grad_buffer()
        opt.zero_grad()
        losses = wrapped(ids, positions, None, labels=ids.roll(-1, -1))
        losses.mean().backward()
        finalize_model_grads([wrapped])
        assert all(torch.isfinite(p.main_grad).all() for p in wrapped.parameters() if p.requires_grad)
        assert opt.step()[0]
        torch.cuda.synchronize()
        opt.zero_grad()

    for step in range(3):
        update(actual_ddp, optimizer, step)
        update(baseline_ddp, reference, step)
        torch.testing.assert_close(actual.state_dict(), baseline.state_dict(), atol=0, rtol=0)
        for leaf, ref_leaf in zip(_leaves(optimizer), _leaves(reference)):
            torch.testing.assert_close(leaf.optimizer.state_dict()["state"],
                                       ref_leaf.optimizer.state_dict()["state"], atol=0, rtol=0)
        if step == 2:
            break
        cpu_state = _cpu_state(optimizer)
        old_views = [weakref.ref(p) for leaf in _leaves(optimizer)
                     for group in leaf.shard_float16_groups for p in group]
        old_bases = [weakref.ref(p._base) for leaf in _leaves(optimizer)
                     for group in leaf.shard_float16_groups for p in group if p._base is not None]
        param_bytes = sum(b.param_data.numel() * b.param_data.element_size()
                          for leaf in _leaves(optimizer) for b in leaf.buffers)
        gc.collect()
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated()
        optimizer.offload_states()
        gc.collect()
        torch.cuda.synchronize()
        _assert_current_shards(optimizer)
        for group in actual_ddp.bucket_groups + actual_ddp.expert_parallel_bucket_groups:
            assert all(value is None for value in group.cached_param_buffer_shard_list)
            assert all(value is None for value in group.cached_grad_buffer_shard_list)
        assert all(ref() is None for ref in old_views), "old optimizer shards retain CUDA storage"
        assert all(ref() is None or ref().device.type == "cpu" for ref in old_bases)
        released = allocated - torch.cuda.memory_allocated()
        assert released >= param_bytes, ("old CUDA parameter allocation retained", released, param_bytes)
        _assert_cpu_state_identity(optimizer, cpu_state)
        if from_master:
            buffers = [b for leaf in _leaves(optimizer) for b in leaf.buffers]
            assert all(b.param_data.untyped_storage().nbytes() <= b.param_data.element_size()
                       for b in buffers), "parked model still owns a complete host parameter image"
        optimizer.reload_states(include=[MegatronOffloadStateType.model_params])
        _assert_current_shards(optimizer)
        assert all(p.main_grad.device.type == "cpu" and p.main_grad.numel() == 1
                   for p in actual.parameters() if p.requires_grad)
        with torch.no_grad():
            torch.testing.assert_close(actual_ddp(ids, positions, None, labels=ids.roll(-1, -1)),
                                       baseline_ddp(ids, positions, None, labels=ids.roll(-1, -1)),
                                       atol=0, rtol=0)
        assert all(p.main_grad.device.type == "cpu" and p.main_grad.numel() == 1
                   for p in actual.parameters() if p.requires_grad)
        optimizer.reload_states()
        assert all(p.main_grad.device.type == "cuda" and p.main_grad.shape == p.shape
                   for p in actual.parameters() if p.requires_grad)
        _assert_current_shards(optimizer)
        _assert_cpu_state_identity(optimizer, cpu_state)
        print(f"PHASE_STORAGE rank={dist.get_rank()} bounded={bounded} from_master={from_master} cycle={step} "
              f"parameter_bytes={param_bytes} released_bytes={released}", flush=True)
