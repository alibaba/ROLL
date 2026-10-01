"""Exercise actual ROLL checkpoint IO and the next distributed CPU Adam update."""
import copy
import json
import os
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


@pytest.mark.skipif(os.environ.get("RUN_CPU_OPTIMIZER_TESTS") != "1", reason="requires Megatron and CUDA")
@pytest.mark.parametrize("lora,from_master", [(False, False), (True, False), (False, True)],
                         ids=["backbone", "lora", "cpu_master"])
@pytest.mark.parametrize("initial_update", [True, False], ids=["after_update", "pristine"])
def test_strategy_checkpoint_resumes_model_optimizer_scheduler_and_rng(tmp_path, monkeypatch, lora, initial_update, from_master):
    import torch.distributed as dist
    from megatron.core import dist_checkpointing, parallel_state, tensor_parallel
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.distributed.finalize_model_grads import finalize_model_grads
    from megatron.core.dist_checkpointing.strategies.fully_parallel import FullyParallelSaveStrategyWrapper
    from megatron.core.optimizer import OptimizerConfig
    from mcore_adapter.models.model_factory import VirtualModels
    from mcore_adapter.checkpoint_write import streaming_save_strategy
    from mcore_adapter.patcher import patch_torch_validate_global_plan
    from mcore_adapter.trainer.utils import build_sharded_state_dict_metadata, get_megatron_lr_scheduler
    from mcore_adapter import TrainingArguments
    from roll.distributed.strategy.megatron_strategy import MegatronTrainStrategy
    from roll.third_party.megatron.offload_states_patch import bind_megatron_offload_states_func
    from roll.third_party.megatron.optimizer import get_megatron_optimizer
    from roll.third_party.megatron.optimizer_config import build_optimizer_config
    from roll.utils.checkpoint_manager import CheckpointManager
    from safetensors.torch import save_file

    monkeypatch.syspath_prepend(str(Path(__file__).parents[3] / "mcore_adapter/tests"))
    from test_qwen4_exp_model import tiny_config, make_model

    torch.cuda.set_device(0)
    dist.init_process_group("nccl", init_method=f"file://{tmp_path}/rdzv", rank=0, world_size=1)
    parallel_state.initialize_model_parallel(1, 1)
    patch_torch_validate_global_plan()
    try:
        assets = tmp_path / "assets"
        assets.mkdir()
        prefix = "model.language_model.layers.1.ple.ple_embedding."
        values = {
            prefix + "ngram_embedding.shard_0.weight": torch.arange(512).reshape(16, 32).bfloat16() / 512,
            prefix + "layer_multipliers": torch.tensor([13, 17, 29]),
            prefix + "ngram_heads_vocab_sizes": torch.full((4,), 16),
            prefix + "ngram_heads_offsets": torch.zeros(4, dtype=torch.long),
        }
        save_file(values, assets / "model.safetensors")
        (assets / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {name: "model.safetensors" for name in values}}))

        def build(seed):
            # LoRA checkpoints depend on an identical frozen backbone. Change
            # adapter initialization separately to prove it is restored.
            torch.manual_seed(733 if lora else seed)
            tensor_parallel.model_parallel_cuda_manual_seed(733 if lora else seed)
            config = tiny_config()
            model = make_model(config)
            model.load_external_assets(str(assets))
            if lora:
                from mcore_adapter.adapters import apply_megatron_lora, set_linear_is_expert
                from roll.configs.model_args import ModelArguments
                from roll.models.model_providers import setup_lora_training

                apply_megatron_lora()
                set_linear_is_expert(model)
                torch.manual_seed(seed)
                tensor_parallel.model_parallel_cuda_manual_seed(seed)
                model_args = ModelArguments(
                    lora_rank=8, lora_alpha=16, lora_dropout=0.0, autocast_adapter_dtype=False,
                    lora_target=r".*\.(linear_qkv|linear_proj|in_proj|out_proj|linear_fc1|linear_fc2|"
                                r"key_proj|value_proj|index_qk_proj|input_mix_weight_down|"
                                r"input_mix_weight_up|block_inject_weight)$",
                )
                model = setup_lora_training(config, model, model_args, is_trainable=True, is_mca=True)
                trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
                assert trainable and all("lora_" in name for name in trainable)
                for component in (".experts.", ".ple.", ".indexer.", ".hc."):
                    assert any(component in name for name in trainable), component
            for name, param in model.named_parameters():
                if ".mlp.experts." in name:
                    param.allreduce = False
            wrapped = DistributedDataParallel(config, DistributedDataParallelConfig(
                grad_reduce_in_fp32=False, use_distributed_optimizer=True, overlap_grad_reduce=False), model)
            args = TrainingArguments(
                output_dir=str(tmp_path), report_to=[], bf16=True, learning_rate=0.001,
                max_steps=4, warmup_steps=0, lr_scheduler_type="linear",
                ckpt_format="legacy" if lora else "torch_dist",
                use_distributed_optimizer=True, optimizer_cpu_offload=True, optimizer_offload_fraction=1.0,
                use_precision_aware_optimizer=True, overlap_cpu_optimizer_d2h_h2d=True,
                bounded_cpu_grad_staging=True,
                offload_model_from_cpu_master=from_master,
            )
            optimizer_config = build_optimizer_config(OptimizerConfig, dict(
                optimizer="adam", lr=0.001, min_lr=0.0, bf16=True, params_dtype=torch.bfloat16,
                use_distributed_optimizer=True, clip_grad=0.2), args)
            optimizer = get_megatron_optimizer(optimizer_config, [wrapped])
            bind_megatron_offload_states_func(optimizer)
            strategy = MegatronTrainStrategy.__new__(MegatronTrainStrategy)
            strategy.model = VirtualModels.__new__(VirtualModels)
            strategy.model.config = config
            strategy.model.models = [wrapped]
            strategy.models_unwrapped = [model]
            strategy.models_wrapped = [wrapped]
            strategy.optimizer = optimizer
            strategy.megatron_train_args = args
            strategy.scheduler = get_megatron_lr_scheduler(args, 4, optimizer)
            strategy.tokenizer = strategy.processor = None
            strategy.worker_config = SimpleNamespace(
                checkpoint_config={"async_upload": False},
                strategy_args=SimpleNamespace(strategy_config={}),
            )
            strategy.checkpoint_manager = CheckpointManager({})
            strategy.ckpt_sharding_metadata = build_sharded_state_dict_metadata(args)
            strategy.save_strategy = FullyParallelSaveStrategyWrapper(
                streaming_save_strategy(),
                parallel_state.get_data_parallel_group(), do_cache_distribution=True)
            strategy._validate_access_integrity = True
            from roll.distributed.store.local.backend import CPUOffloadBackend
            strategy._offload_backend = CPUOffloadBackend()
            strategy.worker = SimpleNamespace(cluster_name="resume_probe")
            return strategy

        def update(strategy, offset, capture_grads=None, expected_grads=None):
            wrapped = strategy.models_wrapped[0]
            wrapped.zero_grad_buffer()
            strategy.optimizer.zero_grad()
            ids = (torch.arange(32, device="cuda") + offset).remainder(15).add(1).unsqueeze(0)
            loss = wrapped(ids, torch.arange(32, device="cuda").unsqueeze(0), None,
                           labels=ids.roll(-1, -1)).mean()
            loss.backward()
            finalize_model_grads([wrapped])
            if capture_grads is not None or expected_grads is not None:
                gradients = {name: parameter.main_grad.detach().clone()
                             for name, parameter in wrapped.named_parameters() if parameter.requires_grad}
                if capture_grads is not None:
                    capture_grads.update(gradients)
                if expected_grads is not None:
                    torch.testing.assert_close(gradients, expected_grads, atol=0, rtol=0)
            step_result = strategy.optimizer.step()
            assert step_result[0]
            strategy.scheduler.step(1)
            torch.cuda.synchronize()
            return loss.detach()

        def rng_sample():
            tracker = tensor_parallel.get_cuda_rng_tracker()
            named_streams = {}
            for name in sorted(tracker.get_states()):
                with tracker.fork(name):
                    named_streams[name] = torch.rand(3, device="cuda")
            assert named_streams
            return random.random(), np.random.rand(), torch.rand(3), torch.rand(3, device="cuda"), named_streams

        baseline = build(733)
        if initial_update:
            update(baseline, 0)
        saved_model = copy.deepcopy(baseline.models_unwrapped[0].state_dict())
        saved_scheduler = copy.deepcopy(baseline.scheduler.state_dict())
        checkpoint = str(tmp_path / "checkpoint")
        checkpoint_step = int(initial_update)
        baseline.save_checkpoint(checkpoint, checkpoint_step, f"checkpoint-{checkpoint_step}", is_last_step=True)
        saved_optimizer_state = [copy.deepcopy(leaf.optimizer.state_dict()["state"])
                                 for leaf in baseline.optimizer.chained_optimizers]
        if not initial_update:
            assert all(state['step'].item() == 0 for leaf in baseline.optimizer.chained_optimizers
                       for state in leaf.optimizer.state.values())
        rng_expected = rng_sample()
        reference_gradients = {}
        loss_expected = update(baseline, 3, capture_grads=reference_gradients)
        resumed = build(999)
        assert any(not torch.equal(value, resumed.models_unwrapped[0].state_dict()[name])
                   for name, value in saved_model.items() if isinstance(value, torch.Tensor))
        import roll.distributed.strategy.megatron_strategy as strategy_module
        read_checkpoint = strategy_module.load_state_dict_from_checkpoint
        model_reads = []

        def check_model_load_has_no_training_grad_buffers(directory, *args, **kwargs):
            if not lora and str(directory) == checkpoint:
                buffers = [buffer for leaf in resumed.optimizer.chained_optimizers for buffer in leaf.buffers]
                assert buffers
                assert all(buffer.grad_data.device.type == "cpu" and buffer.grad_data.numel() == 1
                           for buffer in buffers), "model restore retained CUDA training gradients"
                model_reads.append(str(directory))
            return read_checkpoint(directory, *args, **kwargs)

        monkeypatch.setattr(strategy_module, "load_state_dict_from_checkpoint",
                            check_model_load_has_no_training_grad_buffers)
        resumed.load_checkpoint(checkpoint)
        if from_master:
            from roll.utils.offload_states import OffloadStateType

            cpu_owners = [value for leaf in resumed.optimizer.chained_optimizers
                          for state in leaf.optimizer.state.values()
                          for key, value in state.items()
                          if key in ("master_param", "exp_avg", "exp_avg_sq")]
            owner_ids = [(id(value), value.data_ptr()) for value in cpu_owners]
            resumed.offload_states()
            resumed.load_states(include=[OffloadStateType.model_params])
            torch.testing.assert_close(resumed.models_unwrapped[0].state_dict(), saved_model, atol=0, rtol=0)
            assert all(buffer.grad_data.numel() == 1 and buffer.grad_data.device.type == "cpu"
                       for leaf in resumed.optimizer.chained_optimizers for buffer in leaf.buffers)
            resumed.load_states()
            current_owners = [value for leaf in resumed.optimizer.chained_optimizers
                              for state in leaf.optimizer.state.values()
                              for key, value in state.items()
                              if key in ("master_param", "exp_avg", "exp_avg_sq")]
            assert [(id(value), value.data_ptr()) for value in current_owners] == owner_ids
        if not lora:
            assert model_reads == [checkpoint]
            assert all(buffer.grad_data.device.type == "cuda" and buffer.grad_data.numel() == buffer.numel
                       for leaf in resumed.optimizer.chained_optimizers for buffer in leaf.buffers)
        assert resumed.model.models is resumed.models_wrapped
        torch.testing.assert_close(resumed.models_unwrapped[0].state_dict(), saved_model, atol=0, rtol=0)
        assert resumed.scheduler.state_dict() == saved_scheduler
        for leaf, expected in zip(resumed.optimizer.chained_optimizers, saved_optimizer_state):
            torch.testing.assert_close(leaf.optimizer.state_dict()["state"], expected, atol=0, rtol=0)
        torch.testing.assert_close(rng_sample(), rng_expected, atol=0, rtol=0)
        torch.testing.assert_close(update(resumed, 3, expected_grads=reference_gradients), loss_expected, atol=0, rtol=0)
        torch.testing.assert_close(resumed.models_unwrapped[0].state_dict(),
                                   baseline.models_unwrapped[0].state_dict(), atol=0, rtol=0)
        for actual, expected in zip(resumed.optimizer.chained_optimizers, baseline.optimizer.chained_optimizers):
            torch.testing.assert_close(actual.optimizer.state_dict()["state"],
                                       expected.optimizer.state_dict()["state"], atol=0, rtol=0)
            # DistributedOptimizer restores a group-level step entry which
            # fresh Hybrid groups lack. Its canonical checkpoint state exports
            # the current per-parameter Adam step for both trajectories.
            assert actual.state_dict() == expected.state_dict()
        assert (Path(checkpoint) / "mca_external_assets.json").is_file()

        # A checkpoint cannot be resumed against a different frozen table.
        # Reject the saved identity before rewinding any training state.
        sidecar = Path(checkpoint) / "mca_external_assets.json"
        original_record = sidecar.read_text()
        record = json.loads(original_record)
        record["manifests"]["1"]["index_sha256"] = "0" * 64
        sidecar.write_text(json.dumps(record))
        before_rejected_load = copy.deepcopy(resumed.models_unwrapped[0].state_dict())
        before_optimizer = [copy.deepcopy(leaf.optimizer.state_dict()["state"])
                            for leaf in resumed.optimizer.chained_optimizers]
        before_scheduler = copy.deepcopy(resumed.scheduler.state_dict())
        with pytest.raises(ValueError, match="manifest mismatch"):
            resumed.load_checkpoint(checkpoint)
        assert resumed.model.models is resumed.models_wrapped
        torch.testing.assert_close(resumed.models_unwrapped[0].state_dict(), before_rejected_load, atol=0, rtol=0)
        assert resumed.scheduler.state_dict() == before_scheduler
        for leaf, before in zip(resumed.optimizer.chained_optimizers, before_optimizer):
            torch.testing.assert_close(leaf.optimizer.state_dict()["state"], before, atol=0, rtol=0)

        sidecar.write_text(original_record)
        if lora:
            # An adapter checkpoint with missing or extra trainable tensors is
            # not an exact resume, even though PEFT loads with strict=False.
            from mcore_adapter.checkpointing import get_checkpoint_name

            adapter_file = Path(get_checkpoint_name(str(Path(checkpoint) / "default")))
            adapter_bytes = adapter_file.read_bytes()
            adapter_state = torch.load(adapter_file, map_location="cpu", weights_only=True)
            key = next(key for key, value in adapter_state["model"].items()
                       if "lora_" in key and isinstance(value, torch.Tensor))
            for corruption in ("missing", "unexpected", "shape"):
                damaged = copy.deepcopy(adapter_state)
                if corruption == "missing":
                    del damaged["model"][key]
                elif corruption == "unexpected":
                    damaged["model"]["unknown.lora_A.weight"] = damaged["model"][key].clone()
                else:
                    damaged["model"][key] = damaged["model"][key].flatten()[:1]
                torch.save(damaged, adapter_file)
                try:
                    with pytest.raises(ValueError, match="adapter checkpoint"):
                        resumed.load_checkpoint(checkpoint)
                finally:
                    adapter_file.write_bytes(adapter_bytes)
                assert resumed.model.models is resumed.models_wrapped
                torch.testing.assert_close(resumed.models_unwrapped[0].state_dict(),
                                           before_rejected_load, atol=0, rtol=0)
                assert resumed.scheduler.state_dict() == before_scheduler
                for leaf, before in zip(resumed.optimizer.chained_optimizers, before_optimizer):
                    torch.testing.assert_close(leaf.optimizer.state_dict()["state"], before, atol=0, rtol=0)

        # Missing per-rank RNG means a partial training checkpoint. Reject it
        # before replacing any of the current, already-updated training state.
        rng_path = Path(checkpoint) / "rng_state/rng_state_0.pth"
        rng_backup = rng_path.with_suffix(".backup")
        rng_path.rename(rng_backup)
        try:
            with pytest.raises(FileNotFoundError, match="rng_state_0"):
                resumed.load_checkpoint(checkpoint)
        finally:
            rng_backup.rename(rng_path)
        assert resumed.model.models is resumed.models_wrapped
        torch.testing.assert_close(resumed.models_unwrapped[0].state_dict(), before_rejected_load, atol=0, rtol=0)
        assert resumed.scheduler.state_dict() == before_scheduler
        for leaf, before in zip(resumed.optimizer.chained_optimizers, before_optimizer):
            torch.testing.assert_close(leaf.optimizer.state_dict()["state"], before, atol=0, rtol=0)

        relocated = tmp_path / "relocated-assets"
        assets.rename(relocated)
        with pytest.raises(FileNotFoundError, match="asset checkpoint is missing"):
            resumed.load_checkpoint(checkpoint)
        assert resumed.model.models is resumed.models_wrapped
        torch.testing.assert_close(resumed.models_unwrapped[0].state_dict(), before_rejected_load, atol=0, rtol=0)
        resumed.load_checkpoint(checkpoint, external_asset_path=str(relocated))
        torch.testing.assert_close(resumed.models_unwrapped[0].state_dict(), saved_model, atol=0, rtol=0)
        assert resumed.scheduler.state_dict() == saved_scheduler
        assert resumed.models_unwrapped[0]._qwen4_ngram_asset_record["source"]["path"] == str(relocated)
        if not lora:
            from roll.third_party.megatron.offload_states_patch import (
                MegatronOffloadStateType, checkpoint_grad_buffer_offload,
            )

            import weakref

            for failure_stage in ("read", "install", "chained_read"):
                failed_payloads = []
                restored_leaves = []

                def fail_model_read(*args, **kwargs):
                    assert all(buffer.grad_data.device.type == "cpu" and buffer.grad_data.numel() == 1
                               for leaf in resumed.optimizer.chained_optimizers for buffer in leaf.buffers)
                    payload = torch.empty(1024 * 1024, device="cuda")
                    failed_payloads.append(weakref.ref(payload))
                    raise ValueError("injected model checkpoint payload failure")

                def fail_chained_read(*args, **kwargs):
                    try:
                        fail_model_read(*args, **kwargs)
                    except ValueError as error:
                        raise RuntimeError("injected model checkpoint payload failure") from error

                with monkeypatch.context() as failure_patch:
                    for leaf in resumed.optimizer.chained_optimizers:
                        original_reload = leaf.reload_states

                        def checked_reload(*args, _original=original_reload, **kwargs):
                            assert failed_payloads and all(ref() is None for ref in failed_payloads), (
                                "Failed loader payload remains live while rebuilding CUDA gradients"
                            )
                            restored_leaves.append(_original.__self__)
                            return _original(*args, **kwargs)

                        failure_patch.setattr(leaf, "reload_states", checked_reload)
                    if failure_stage == "install":
                        failure_patch.setattr(resumed.model, "load_state_dict", fail_model_read)
                    else:
                        failure_patch.setattr(
                            strategy_module, "load_state_dict_from_checkpoint",
                            fail_chained_read if failure_stage == "chained_read" else fail_model_read,
                        )
                    with pytest.raises((ValueError, RuntimeError), match="injected model checkpoint payload failure"):
                        resumed.load_checkpoint(checkpoint, external_asset_path=str(relocated))
                assert len(restored_leaves) == len(resumed.optimizer.chained_optimizers)
                assert resumed.model.models is resumed.models_wrapped
                assert resumed.scheduler.state_dict() == saved_scheduler
                assert all(buffer.grad_data.device.type == "cuda" and buffer.grad_data.numel() == buffer.numel
                           for leaf in resumed.optimizer.chained_optimizers for buffer in leaf.buffers)

            # The temporary context must preserve gradients already parked by
            # its caller. Full optimizer checkpoint templating itself requires
            # materialized bucket geometry and runs before this context.
            resumed.optimizer.offload_states(include=[MegatronOffloadStateType.other_params])
            with checkpoint_grad_buffer_offload(resumed.optimizer):
                assert all(buffer.grad_data.device.type == "cpu" and buffer.grad_data.numel() == 1
                           for leaf in resumed.optimizer.chained_optimizers for buffer in leaf.buffers)
            assert all(buffer.grad_data.device.type == "cpu" and buffer.grad_data.numel() == 1
                       for leaf in resumed.optimizer.chained_optimizers for buffer in leaf.buffers)
            resumed.optimizer.reload_states(include=[MegatronOffloadStateType.other_params])
            for leaf, expected in zip(resumed.optimizer.chained_optimizers, saved_optimizer_state):
                torch.testing.assert_close(leaf.optimizer.state_dict()["state"], expected, atol=0, rtol=0)
                hybrid = leaf.optimizer
                for sub in hybrid.cpu_optimizers:
                    for group in sub.param_groups:
                        for inner in group["params"]:
                            original = hybrid.inner_param_to_orig_param[inner]
                            torch.testing.assert_close(inner, hybrid.state[original]["master_param"], atol=0, rtol=0)
                            for key in ("exp_avg", "exp_avg_sq", "step"):
                                torch.testing.assert_close(sub.state[inner][key], hybrid.state[original][key], atol=0, rtol=0)
            torch.testing.assert_close(update(resumed, 3, expected_grads=reference_gradients),
                                       loss_expected, atol=0, rtol=0)
            torch.testing.assert_close(resumed.models_unwrapped[0].state_dict(),
                                       baseline.models_unwrapped[0].state_dict(), atol=0, rtol=0)
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()
