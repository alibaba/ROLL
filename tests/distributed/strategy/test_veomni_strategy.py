"""Behavior-level tests for the VeOmni strategy."""

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch

from roll.distributed.scheduler.protocol import DataProto

try:
    import roll.distributed.strategy.veomni_strategy as veomni_strategy
except ImportError as error:
    pytest.skip(f"VeOmni strategy is unavailable in this environment: {error}", allow_module_level=True)

VeOmniInferStrategy = veomni_strategy.VeOmniInferStrategy
VeOmniTrainStrategy = veomni_strategy.VeOmniTrainStrategy


def test_async_ulysses_is_passed_to_parallel_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Async Ulysses should be propagated through VeOmni's parallel-state API."""
    from veomni.distributed import parallel_state

    captured: dict[str, Any] = {}
    monkeypatch.setattr(veomni_strategy.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(veomni_strategy.dist, "get_world_size", lambda: 4)
    monkeypatch.setattr(
        veomni_strategy,
        "current_platform",
        SimpleNamespace(device_type="cpu", communication_backend="gloo"),
    )
    monkeypatch.setattr(parallel_state, "init_parallel_state", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(
        parallel_state,
        "get_parallel_state",
        lambda: SimpleNamespace(
            ulysses_size=2,
            ulysses_rank=0,
            ulysses_group=object(),
            sp_enabled=True,
            async_enabled=True,
            dp_mode="fsdp2",
            dp_rank=0,
        ),
    )

    state = veomni_strategy._initialize_veomni_parallel_state(
        {"ulysses_parallel_size": 2, "enable_async_ulysses": True}
    )

    assert captured["async_enabled"] is True
    assert state.async_ulysses_enabled is True


def test_compute_diffusion_step_stats_replays_each_transition() -> None:
    """Diffusion transition statistics should preserve the recorded trajectory horizon."""
    strategy = VeOmniInferStrategy.__new__(VeOmniInferStrategy)
    strategy._noise_level = 0.7
    strategy._sde_type = "sde"
    calls: list[dict[str, Any]] = []

    def sample_previous_step(**kwargs: Any) -> tuple[torch.Tensor, ...]:
        calls.append(kwargs)
        step = float(len(calls))
        batch_size = kwargs["sample"].shape[0]
        return (
            kwargs["prev_sample"],
            torch.full((batch_size,), step),
            kwargs["prev_sample"] + step,
            torch.full((batch_size,), step / 10),
        )

    strategy.diffusion_scheduler = SimpleNamespace(sample_previous_step=sample_previous_step)
    stats = strategy.compute_diffusion_step_stats(
        model_output=torch.zeros(2, 3, 4, 4),
        all_latents=torch.zeros(2, 4, 4, 4),
        all_timesteps=torch.tensor([[900.0, 600.0, 300.0], [900.0, 600.0, 300.0]]),
    )

    assert torch.equal(stats.log_probs[0], torch.tensor([1.0, 2.0, 3.0]))
    assert torch.equal(stats.prev_sample_mean[0, :, 0, 0], torch.tensor([1.0, 2.0, 3.0]))
    assert torch.allclose(stats.std_dev_t[0], torch.tensor([0.1, 0.2, 0.3]))
    assert len(calls) == 3


def test_state_offload_moves_prompt_encoder(monkeypatch: pytest.MonkeyPatch) -> None:
    """The condition bundle should follow model state load and offload transitions."""
    strategy = VeOmniInferStrategy.__new__(VeOmniInferStrategy)
    strategy.model = object()
    strategy.optimizer = None
    strategy.prompt_encoder = Mock()
    strategy.cpu_offload_enabled = True
    monkeypatch.setattr(veomni_strategy, "current_platform", SimpleNamespace(device_type="cpu"))

    strategy.load_states()
    strategy.offload_states()

    assert [call.args[0] for call in strategy.prompt_encoder.to.call_args_list] == [
        torch.device("cpu"),
        "cpu",
    ]


def test_native_infer_initializes_through_veomni(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inference should build and parallelize its model through VeOmni's native path."""
    strategy = VeOmniInferStrategy.__new__(VeOmniInferStrategy)
    strategy.worker_config = SimpleNamespace(
        model_args=SimpleNamespace(dtype="fp32"),
        strategy_args=SimpleNamespace(
            strategy_config={
                "mixed_precision": False,
                "reduce_dtype": "bfloat16",
            }
        ),
    )
    strategy.scaler = None
    calls: list[str] = []
    monkeypatch.setattr(veomni_strategy, "_apply_veomni_parallel_state", lambda strategy: calls.append("parallel"))

    def build_diffusion_model(
        self: Any,
        mixed_precision: Any,
        model_init_dtype: str,
    ) -> None:
        calls.append("model")
        assert mixed_precision.enable is False
        assert model_init_dtype == "float32"
        self.model = object()
        self.prompt_encoder = object()

    monkeypatch.setattr(VeOmniInferStrategy, "_build_diffusion_model", build_diffusion_model)

    strategy.initialize()

    assert calls == ["parallel", "model"]
    assert strategy.param_dtype is torch.float32
    assert strategy.reduce_dtype is torch.bfloat16


def test_native_train_builds_optimizer_and_scheduler(monkeypatch: pytest.MonkeyPatch) -> None:
    """Training should extend the native model path with VeOmni optimizer state."""
    strategy = VeOmniTrainStrategy.__new__(VeOmniTrainStrategy)
    strategy.worker_config = SimpleNamespace(
        strategy_args=SimpleNamespace(strategy_config={}),
        training_args=SimpleNamespace(
            max_steps=10,
            adam_beta1=0.0,
            adam_beta2=0.999,
            learning_rate=1e-4,
            weight_decay=0.0,
            lr_scheduler_type="constant",
            get_warmup_steps=lambda max_steps: 0,
        ),
    )
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    calls: list[str] = []

    def initialize_model(self: Any, model_provider: Any = None) -> None:
        calls.append("model")
        self.model = model

    monkeypatch.setattr(VeOmniInferStrategy, "initialize", initialize_model)
    import veomni.optim

    monkeypatch.setattr(
        veomni.optim,
        "build_optimizer",
        lambda model, **kwargs: calls.append("optimizer") or optimizer,
    )
    monkeypatch.setattr(
        veomni.optim,
        "build_lr_scheduler",
        lambda optimizer, **kwargs: calls.append("scheduler") or object(),
    )
    monkeypatch.setattr(veomni_strategy.dist, "barrier", lambda: calls.append("barrier"))

    strategy.initialize()

    assert calls == ["model", "optimizer", "scheduler", "barrier"]
    assert strategy.optimizer is optimizer


def test_diffusion_forward_step_replays_micro_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    """The inference shell should replay and collate diffusion micro-batches."""
    strategy = VeOmniInferStrategy.__new__(VeOmniInferStrategy)
    strategy.model = torch.nn.Identity()
    strategy.model_fwd_context = nullcontext()
    strategy.param_dtype = torch.bfloat16
    strategy._forward_diffusion_model = lambda data: data.batch["prediction"]
    monkeypatch.setattr(veomni_strategy, "current_platform", SimpleNamespace(device_type="cpu"))

    batch = DataProto.from_dict(
        tensors={"prediction": torch.tensor([[1.0], [2.0]])},
        meta_info={"micro_batch_size": 1},
    )
    output = strategy.forward_step(
        batch,
        lambda data, prediction: (
            prediction.new_zeros(()),
            {"prediction": prediction},
        ),
    )

    assert torch.equal(output["prediction"], torch.tensor([[1.0], [2.0]]))


def test_diffusion_train_step_runs_algorithm_loss_and_optimizer(monkeypatch: pytest.MonkeyPatch) -> None:
    """The training shell should own backward/update while the callback owns the algorithm loss."""
    model = torch.nn.Linear(2, 1)
    strategy = VeOmniTrainStrategy.__new__(VeOmniTrainStrategy)
    strategy.model = model
    strategy.optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    strategy.scheduler = Mock()
    strategy.scaler = None
    strategy.param_dtype = torch.bfloat16
    strategy.model_fwd_context = nullcontext()
    strategy.model_bwd_context = nullcontext()
    strategy.cpu_offload_enabled = True
    strategy._guidance_scale = 0.0
    strategy.worker = SimpleNamespace(
        rank_info=SimpleNamespace(dp_size=1),
        pipeline_config=SimpleNamespace(
            max_grad_norm=1.0,
            is_offload_optimizer_states_in_train_step=False,
        ),
    )
    strategy.worker_config = SimpleNamespace(
        name="actor",
        apply_loss_scale=False,
        training_args=SimpleNamespace(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
        ),
    )
    strategy.clip_grad_norm = lambda max_norm: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    monkeypatch.setattr(
        veomni_strategy,
        "current_platform",
        SimpleNamespace(device_type="cpu", empty_cache=lambda: None),
    )

    batch = DataProto.from_dict(
        tensors={
            "features": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            "targets": torch.tensor([[1.0], [-1.0]]),
            "prompt_embeds": torch.zeros(2, 1, 1),
        },
    )
    initial_weight = model.weight.detach().clone()

    def loss_func(data: DataProto, model_context: dict[str, Any], strategy: Any) -> tuple[torch.Tensor, dict]:
        assert "prompt" in model_context
        prediction = strategy.model(data.batch["features"])
        loss = torch.nn.functional.mse_loss(prediction.float(), data.batch["targets"])
        return loss, {"loss": loss.detach().reshape(1)}

    metrics = strategy.train_step(batch, loss_func)

    assert len(metrics["loss"]) == 2
    assert strategy.scheduler.step.call_count == 2
    assert not torch.equal(model.weight.detach(), initial_weight)
