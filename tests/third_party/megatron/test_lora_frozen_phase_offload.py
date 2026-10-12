"""Check the real LoRA strategy's frozen parameters across an ordinary phase.

Reuse the existing native checkpoint/Adam continuation scenario to avoid a
second model/optimizer fixture with different settings. The added assertion
names the bug: default offload must move frozen weights as well as adapters.
"""
import os
from concurrent.futures import Future
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.skipif(os.environ.get("RUN_CPU_OPTIMIZER_TESTS") != "1", reason="requires Megatron and CUDA")
@pytest.mark.parametrize("phase", ["default", "admission", "admission_failure"])
def test_lora_phase_restores_frozen_weights_and_next_adam_update(tmp_path, monkeypatch, phase):
    from roll.distributed.strategy.megatron_strategy import MegatronTrainStrategy
    from tests.third_party.megatron.test_strategy_checkpoint_resume import (
        test_strategy_checkpoint_resumes_model_optimizer_scheduler_and_rng,
    )

    original_save = MegatronTrainStrategy.save_checkpoint
    checked = []

    def save_after_phase(strategy, *args, **kwargs):
        if not checked:
            parameters = dict(strategy.models_unwrapped[0].named_parameters())
            frozen = {name: value for name, value in parameters.items() if not value.requires_grad}
            trainable = {name: value for name, value in parameters.items() if value.requires_grad}
            assert frozen and trainable
            expected = {name: value.detach().cpu().clone() for name, value in parameters.items()}
            if phase in {"admission", "admission_failure"}:
                import torch.distributed as dist
                import roll.third_party.megatron.model_update as updates
                from roll.distributed.executor.worker import Worker

                def admit(**kwargs):
                    assert all(value.device.type == "cpu" for value in frozen.values()), (
                        "inference admission can wake its base while the actor frozen base is on CUDA"
                    )
                    assert all(value.device.type == "cuda" for value in trainable.values())
                    reply = Future()
                    if phase == "admission_failure":
                        reply.set_exception(RuntimeError("native adapter admission failed"))
                    else:
                        reply.set_result([True])
                    return reply

                updater = SimpleNamespace(
                    _infer_parallel_cpu_group=dist.new_group(backend="gloo"),
                    _co_infer_worker=SimpleNamespace(add_lora=SimpleNamespace(remote=admit)),
                    worker_config=SimpleNamespace(model_args=SimpleNamespace(lora_target=["linear_qkv"])),
                    models_unwrapped=strategy.models_unwrapped,
                    _model_update_buffer_size=1, _weights_meta={}, _broadcast_workers=[],
                )
                updater.model_update = updates.MegatronWeightUpdater._colocated_model_update.__get__(updater)
                strategy.weight_updaters = {"probe": updater}
                strategy.offload_nccl = False
                worker = SimpleNamespace(strategy=strategy, cluster_name="phase_probe")
                # The real pipeline first offloads the actor, then Worker loads
                # the states needed for export and finally admits the adapter.
                strategy.offload_states()
                with monkeypatch.context() as transport:
                    # Adapter buckets have completed; exercise the actual GPU
                    # sender's final admission transition without a second engine.
                    transport.setattr(updates, "gather_all_hf_weights", lambda *a, **kw: iter(()))
                    transport.setattr(updates.ray, "get", lambda refs: [ref.result() for ref in refs])
                    if phase == "admission_failure":
                        transport.delenv("roll_EXEC_FUNC_NAME", raising=False)
                        with pytest.raises(RuntimeError, match="native adapter admission failed"):
                            Worker.start_model_update(worker, model_update_name="probe")
                        assert all(value.device.type == "cpu" for value in parameters.values()), (
                            "failed model update retained actor parameters on CUDA"
                        )
                        assert "roll_EXEC_FUNC_NAME" not in os.environ
                    else:
                        Worker.start_model_update(worker, model_update_name="probe")
            strategy.offload_states()
            try:
                assert all(value.device.type == "cpu" for value in frozen.values()), (
                    "default LoRA phase offload retained frozen parameters on CUDA"
                )
                assert all(value.device.type == "cpu" for value in trainable.values())
            finally:
                strategy.load_states()
            assert all(value.device.type == "cuda" for value in parameters.values())
            for name, value in parameters.items():
                torch.testing.assert_close(value.detach().cpu(), expected[name], rtol=0, atol=0)
            checked.append(True)
        return original_save(strategy, *args, **kwargs)

    monkeypatch.setattr(MegatronTrainStrategy, "save_checkpoint", save_after_phase)
    test_strategy_checkpoint_resumes_model_optimizer_scheduler_and_rng(
        tmp_path, monkeypatch, lora=True, initial_update=True, from_master=False
    )
    assert checked == [True]
