"""Behavior-level tests for the DMD pipeline."""

import json
import random
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from dacite import from_dict
import numpy as np
from omegaconf import OmegaConf
import pytest
import torch
from torch.utils.data import DataLoader

import roll.pipeline.diffusion.dmd.dmd_pipeline as dmd_pipeline
from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.diffusion.dmd.dmd_algorithm import (
    DMDFlowMatchScheduler,
    DMD_KEY_DENOISED_TIMESTEP_FROM,
    DMD_KEY_DENOISED_TIMESTEP_TO,
    DMD_KEY_GENERATED,
    DMD_KEY_NOISY_LATENT,
    DMD_KEY_NOISY_LATENT_CA,
    DMD_KEY_PRED_FAKE_IMAGE,
    DMD_KEY_PRED_REAL_IMAGE,
    DMD_KEY_PRED_REAL_COND_CA,
    DMD_KEY_PRED_REAL_UNCOND_CA,
    DMD_KEY_PRED_X0,
    DMD_KEY_PROMPTS,
    DMD_KEY_TIMESTEP,
    DMD_KEY_TIMESTEP_CA,
    compute_ddmd_generator_loss,
    compute_dmd_fake_score_loss,
    compute_dmd_generator_loss,
    generate_self_forcing_sample,
    sample_denoising_indices,
    sample_dmd_timesteps,
)
from roll.pipeline.diffusion.dmd.dmd_config import DMDConfig
from roll.pipeline.diffusion.dmd.dmd_pipeline import DMDPipeline
from roll.pipeline.diffusion.dmd.dmd_worker import (
    BaseDMDWorker,
    DMDFakeScoreWorker,
    DMDGeneratorWorker,
    ModelEMA,
)
from roll.pipeline.diffusion.models.wan.wan_self_forcing import WanPromptEncoder, WanVAEConfig
from roll.utils.worker_state import WorkerState


def _make_checkpoint_worker(output_dir: Path) -> DMDGeneratorWorker:
    worker = object.__new__(DMDGeneratorWorker)
    worker.worker_name = "generator-0-G0"
    worker.cluster_name = "generator"
    worker.pipeline_config = SimpleNamespace(
        output_dir=str(output_dir),
        resume_from_checkpoint=False,
        is_offload_states=False,
        ema=SimpleNamespace(enabled=True, weight=0.5, start_step=0, update_interval=1),
    )
    worker.strategy = SimpleNamespace(scheduler=None, scaler=None)
    worker.models = torch.nn.ModuleDict({"model": torch.nn.Linear(3, 2)})
    worker.model = worker.models["model"]
    worker.optimizer = torch.optim.AdamW(worker.model.parameters(), lr=0.01)
    worker.strategy.scheduler = torch.optim.lr_scheduler.StepLR(worker.optimizer, step_size=1)
    worker.model_ema = ModelEMA(worker.model, decay=0.5)
    worker.classification_head = None
    worker.classification_optimizer = None
    worker.classification_scheduler = None
    worker.step = 11
    return worker


@pytest.mark.parametrize(
    ("config_path", "generator_strategy", "self_forcing", "ddmd"),
    [
        ("examples/wan2_1-1.3B_dmd/dmd_config.yaml", "veomni_train", False, False),
        ("examples/wan2_1-1.3B_dmd/dmd_config.yaml", "veomni_train", False, True),
        ("examples/wan2_1-1.3B_dmd/dmd_self_forcing_config.yaml", "fsdp2_train", True, False),
    ],
)
def test_dmd_example_config_loads(
    tmp_path: Path,
    config_path: str,
    generator_strategy: str,
    self_forcing: bool,
    ddmd: bool,
) -> None:
    """Both checked-in DMD variants should instantiate the role-centric config."""
    config_dict = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    config_dict["max_steps"] = 1
    config_dict["output_dir"] = str(tmp_path / "output")
    config_dict["logging_dir"] = str(tmp_path / "logs")
    config_dict["checkpoint_config"]["output_dir"] = str(tmp_path / "checkpoint")
    config_dict["tracker_kwargs"]["log_dir"] = str(tmp_path / "tracker")
    if ddmd:
        config_dict["ddmd"]["enabled"] = True

    config = from_dict(data_class=DMDConfig, data=config_dict)
    config.set_max_steps(config.max_steps)

    assert [
        config.generator.strategy_args.strategy_name,
        config.fake_score.strategy_args.strategy_name,
        config.real_score.strategy_args.strategy_name,
    ] == [generator_strategy, "veomni_train", "veomni_infer"]
    assert config.diffusion_model_variant == "wan2_1"
    assert config.ddmd.enabled is ddmd
    assert config.self_forcing.enabled is self_forcing
    assert config.generator.model_args.model_name_or_path == config.pretrain
    assert config.generator.strategy_args.strategy_config["init_tokenizer_processor"] is not self_forcing
    assert config.fake_score.strategy_args.strategy_config["init_tokenizer_processor"] is False
    assert config.real_score.strategy_args.strategy_config["init_tokenizer_processor"] is False
    if self_forcing:
        generator_model_config = config.generator.model_args.model_config_kwargs
        assert generator_model_config["transformer_name_or_path"] == (
            "wan_models/Wan2.1-T2V-1.3B-Self-Forcing-ODEInit-Diffusers/transformer"
        )
        assert "initialization_checkpoint" not in generator_model_config


def test_dmd_flow_math_and_generator_gradient() -> None:
    """The flow convention and detached DMD target should produce the expected gradient."""
    scheduler = DMDFlowMatchScheduler(num_inference_steps=8, sigma_min=0.0)
    clean = torch.zeros(2, 1, 1, 1)
    noise = torch.ones_like(clean)
    timestep = scheduler.timesteps[:2]
    noisy = scheduler.add_noise(clean, noise, timestep)
    flow = noise - clean

    assert torch.allclose(scheduler.x0_from_flow_pred(flow_pred=flow, xt=noisy, timestep=timestep), clean)
    assert compute_dmd_fake_score_loss(flow, clean, noise).item() == 0.0

    generated_latents = torch.tensor([2.0, 4.0]).reshape(1, 2, 1, 1, 1).requires_grad_()
    loss, gradient_mean_abs = compute_dmd_generator_loss(
        generated_latents,
        torch.ones_like(generated_latents),
        torch.zeros_like(generated_latents),
    )
    loss.backward()

    assert loss.dtype == torch.float32
    assert torch.isfinite(gradient_mean_abs)
    assert torch.allclose(
        generated_latents.grad,
        torch.full_like(generated_latents, 1.0 / 6.0),
    )


def test_dmd_fake_score_and_classification_head_update_separately() -> None:
    """Fake-score diffusion and GAN classification must not share one backward/clip step."""
    worker = object.__new__(DMDFakeScoreWorker)
    worker.model = torch.nn.Linear(2, 2, bias=False)
    worker.classification_head = torch.nn.Linear(2, 1, bias=False)
    worker.optimizer = torch.optim.SGD(worker.model.parameters(), lr=0.1)
    worker.classification_optimizer = torch.optim.SGD(worker.classification_head.parameters(), lr=0.1)
    model_scheduler = torch.optim.lr_scheduler.StepLR(worker.optimizer, step_size=1)
    worker.classification_scheduler = torch.optim.lr_scheduler.StepLR(
        worker.classification_optimizer,
        step_size=1,
    )
    worker.strategy = SimpleNamespace(
        scaler=None,
        scheduler=model_scheduler,
        model_bwd_context=nullcontext(),
        clip_grad_norm=lambda max_norm: torch.nn.utils.clip_grad_norm_(
            worker.model.parameters(),
            max_norm,
        ),
    )
    worker.pipeline_config = SimpleNamespace(
        max_grad_norm=1.0,
        is_offload_states=False,
        is_offload_optimizer_states_in_train_step=False,
    )
    worker.worker_config = SimpleNamespace(name="fake_score")

    inputs = torch.tensor([[1.0, -2.0]])
    metrics: dict[str, float] = {}
    head_before = worker.classification_head.weight.detach().clone()
    worker._backward_and_step(worker.model(inputs).square().mean(), metrics)

    backbone_after_score = worker.model.weight.detach().clone()
    assert torch.equal(worker.classification_head.weight, head_before)

    frozen_features = worker.model(inputs).detach()
    worker._backward_classification_head(
        worker.classification_head(frozen_features).square().mean(),
        metrics,
    )

    assert torch.equal(worker.model.weight, backbone_after_score)
    assert not torch.equal(worker.classification_head.weight, head_before)
    assert "fake_score/grad_norm" in metrics
    assert "fake_score/classification_grad_norm" in metrics


def test_ddmd_generator_gradient_decomposes_dm_and_ca() -> None:
    """DDMD should add independently normalized DM and CFG-augmentation gradients."""
    generated_latents = torch.tensor([2.0, 4.0]).reshape(1, 2, 1, 1, 1).requires_grad_()
    loss, gradient_dm_mean_abs, gradient_ca_mean_abs = compute_ddmd_generator_loss(
        generated_latents=generated_latents,
        fake_pred_x0=torch.ones_like(generated_latents),
        real_pred_x0=torch.zeros_like(generated_latents),
        real_cond_pred_x0=torch.zeros_like(generated_latents),
        real_uncond_pred_x0=torch.ones_like(generated_latents),
        guidance_scale=4.0,
        gradient_scale=1.0,
    )
    loss.backward()

    assert torch.allclose(gradient_dm_mean_abs, torch.tensor(1.0 / 3.0))
    assert torch.allclose(gradient_ca_mean_abs, torch.tensor(1.0))
    assert torch.allclose(
        generated_latents.grad,
        torch.full_like(generated_latents, 2.0 / 3.0),
    )


def test_self_forcing_partial_block_gradient_window_backpropagates() -> None:
    """A block overlapping the gradient window must retain its autograd graph."""

    class ARModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def create_ar_state(self, **_: object) -> dict[str, object]:
            return {}

        def forward_ar(self, *, noisy_image_or_video: torch.Tensor, **_: object) -> torch.Tensor:
            return noisy_image_or_video * self.weight

    model = ARModel()
    noise_generator = torch.Generator().manual_seed(1)
    samples, gradient_mask, _, _ = generate_self_forcing_sample(
        model=model,
        scheduler=DMDFlowMatchScheduler(num_inference_steps=4, sigma_min=0.0),
        config=SimpleNamespace(
            num_frame_per_block=2,
            independent_first_frame=False,
            same_step_across_blocks=True,
            last_step_only=True,
            context_noise=0,
            num_training_frames=1,
        ),
        denoising_step_list=torch.tensor([1000]),
        noise=torch.randn(
            [1, 4, 1, 1, 1],
            generator=noise_generator,
        ),
        prompt={},
        noise_generator=noise_generator,
        exit_step_generator=torch.Generator().manual_seed(2),
    )

    samples[gradient_mask].sum().backward()

    assert samples[gradient_mask].requires_grad
    assert model.weight.grad is not None


def test_dmd_prompt_encoding_uses_provider_owned_device_context() -> None:
    """Self-Forcing DMD should release its unregistered text encoder after encoding."""
    text_encoder = Mock()
    worker = object.__new__(BaseDMDWorker)
    worker.pipeline_config = SimpleNamespace(is_offload_states=True)
    worker.prompt_encoder = WanPromptEncoder(
        tokenizer=object(),
        text_encoder=text_encoder,
        vae_config=WanVAEConfig(z_dim=16, scale_factor_spatial=8, scale_factor_temporal=4),
    )
    worker.diffusion_adapter = SimpleNamespace(
        encode_prompt=lambda **_: {"prompt_embeds": torch.ones(1)}
    )
    worker.device = torch.device("cuda:0")
    worker.dtype = torch.bfloat16

    prompt = worker._encode_prompt(["test"])

    assert torch.equal(prompt["prompt_embeds"], torch.ones(1))
    assert [call.args[0] for call in text_encoder.to.call_args_list] == [
        torch.device("cuda:0"),
        "cpu",
    ]


def test_dmd_sampling_is_reproducible_and_windowed() -> None:
    """Algorithm RNG should be DP-local and score timesteps should stay inside each sample window."""
    worker = object.__new__(BaseDMDWorker)
    worker.pipeline_config = SimpleNamespace(seed=42)
    worker.rank_info = SimpleNamespace(dp_rank=1)
    worker.device = torch.device("cpu")
    data = DataProto(meta_info={"global_step": 7})

    first = torch.rand(8, generator=worker._make_algorithm_generator(data, stream=3))
    torch.manual_seed(999)
    torch.rand(32)
    repeated = torch.rand(8, generator=worker._make_algorithm_generator(data, stream=3))
    worker.rank_info.dp_rank = 2
    other_dp = torch.rand(8, generator=worker._make_algorithm_generator(data, stream=3))

    assert torch.equal(first, repeated)
    assert not torch.equal(first, other_dp)
    assert sample_denoising_indices(
        8,
        4,
        torch.device("cpu"),
        generator=torch.Generator().manual_seed(11),
    ) == sample_denoising_indices(
        8,
        4,
        torch.device("cpu"),
        generator=torch.Generator().manual_seed(11),
    )

    timestep = sample_dmd_timesteps(
        SimpleNamespace(
            num_train_timestep=1000,
            min_score_timestep=0,
            sample_shift=1.0,
            ts_schedule=True,
            ts_schedule_max=True,
        ),
        torch.device("cpu"),
        denoised_timestep_from=torch.tensor([200, 800]),
        denoised_timestep_to=torch.tensor([100, 700]),
        batch_size=2,
        num_frames=3,
        generator=torch.Generator().manual_seed(1),
    )
    assert torch.all((100 <= timestep[0]) & (timestep[0] < 200))
    assert torch.all((700 <= timestep[1]) & (timestep[1] < 800))


def test_dmd_pipeline_orchestrates_three_roles(monkeypatch: pytest.MonkeyPatch) -> None:
    """The pipeline should keep generator and score role ordering outside workers."""
    monkeypatch.setattr(
        DataProto,
        "materialize_concat",
        staticmethod(lambda data_refs, global_keys=None: DataProto.concat(data_refs)),
    )
    generator, fake_score, real_score = Mock(), Mock(), Mock()
    call_order = Mock()
    for name, method in (
        ("generator_begin", generator.begin_generator_step),
        ("fake_forward", fake_score.forward_score),
        ("real_forward", real_score.forward_score),
        ("generator_finish", generator.finish_generator_step),
        ("generator_sample", generator.generate_no_grad),
        ("fake_train", fake_score.train_score_step),
    ):
        call_order.attach_mock(method, name)

    generator.begin_generator_step.return_value = [
        DataProto.from_dict(
            tensors={
                DMD_KEY_NOISY_LATENT: torch.zeros(2, 1, 1),
                DMD_KEY_NOISY_LATENT_CA: torch.zeros(2, 1, 1),
                DMD_KEY_TIMESTEP: torch.zeros(2, 1, dtype=torch.long),
                DMD_KEY_TIMESTEP_CA: torch.zeros(2, 1, dtype=torch.long),
            },
            non_tensors={DMD_KEY_PROMPTS: ["p0", "p1"]},
            meta_info={"metrics": {}},
        )
    ]
    generator.finish_generator_step.return_value = [
        DataProto(meta_info={"metrics": {"generator/loss": 0.25}})
    ]
    generator.generate_no_grad.return_value = [
        DataProto.from_dict(
            tensors={
                DMD_KEY_GENERATED: torch.zeros(2, 1, 1),
                DMD_KEY_DENOISED_TIMESTEP_FROM: torch.full((2,), 750),
                DMD_KEY_DENOISED_TIMESTEP_TO: torch.full((2,), 500),
            },
            non_tensors={DMD_KEY_PROMPTS: ["p0", "p1"]},
            meta_info={"metrics": {}},
        )
    ]
    fake_score.forward_score.return_value = [
        DataProto.from_dict(tensors={DMD_KEY_PRED_X0: torch.ones(2, 1, 1)}, meta_info={"metrics": {}})
    ]
    real_score.forward_score.return_value = [
        DataProto.from_dict(
            tensors={
                DMD_KEY_PRED_X0: torch.full((2, 1, 1), 2.0),
                DMD_KEY_PRED_REAL_COND_CA: torch.full((2, 1, 1), 3.0),
                DMD_KEY_PRED_REAL_UNCOND_CA: torch.full((2, 1, 1), 4.0),
            },
            meta_info={"metrics": {}},
        )
    ]
    fake_score.train_score_step.return_value = [
        DataProto(meta_info={"metrics": {"fake_score/loss": 0.5}})
    ]

    pipeline = object.__new__(DMDPipeline)
    pipeline.generator = generator
    pipeline.fake_score = fake_score
    pipeline.real_score = real_score
    pipeline.pipeline_config = SimpleNamespace(
        ddmd=SimpleNamespace(enabled=True),
        gan=SimpleNamespace(enabled=False),
    )
    pipeline.score_roles_colocated = False
    pipeline.gpu_tensor_transfer_enabled = False
    metrics = Mock()
    metrics.metrics = {}
    metrics.add_reduced_metrics.side_effect = metrics.metrics.update
    metrics.add_metric.side_effect = lambda key, value: metrics.metrics.update({key: value})
    batch = {DMD_KEY_PROMPTS: np.array(["p0", "p1"], dtype=object), "batch_idx": torch.arange(2)}

    pipeline._run_generator_phase(batch, global_step=7, metrics_mgr=metrics)
    pipeline._run_fake_score_phase(batch, global_step=7, metrics_mgr=metrics)

    assert [mock_call[0] for mock_call in call_order.mock_calls] == [
        "generator_begin",
        "fake_forward",
        "real_forward",
        "generator_finish",
        "generator_sample",
        "fake_train",
    ]
    finish_batch = generator.finish_generator_step.call_args.args[0]
    assert set(finish_batch.batch.keys()) == {
        DMD_KEY_PRED_FAKE_IMAGE,
        DMD_KEY_PRED_REAL_IMAGE,
        DMD_KEY_PRED_REAL_COND_CA,
        DMD_KEY_PRED_REAL_UNCOND_CA,
    }
    assert DMD_KEY_GENERATED in fake_score.train_score_step.call_args.args[0].batch
    assert metrics.metrics["generator/loss"] == 0.25
    assert metrics.metrics["fake_score/loss"] == 0.5


def test_dmd_checkpoint_roundtrip_restores_training_and_rng_state(tmp_path: Path) -> None:
    """A role checkpoint should restore model, optimizer, scheduler, EMA, step, and RNG state."""
    torch.manual_seed(123)
    np.random.seed(123)
    random.seed(123)
    source = _make_checkpoint_worker(tmp_path)
    loss = source.model(torch.randn(4, 3)).sum()
    source.optimizer.zero_grad()
    loss.backward()
    source.optimizer.step()
    source.strategy.scheduler.step()
    source.model_ema.update(source.model)
    checkpoint_dir = tmp_path / "checkpoint-3"
    source._save_checkpoint_to_dir(str(checkpoint_dir))
    expected_torch = torch.rand(2)
    expected_numpy = np.random.rand(2)
    expected_random = random.random()

    restored = _make_checkpoint_worker(tmp_path)
    restored._load_checkpoint_from_dir(str(checkpoint_dir))

    for key, value in source.models.state_dict().items():
        assert torch.allclose(restored.models.state_dict()[key], value)
    assert restored.optimizer.state_dict()["state"]
    assert restored.strategy.scheduler.state_dict() == source.strategy.scheduler.state_dict()
    assert restored.model_ema.state_dict().keys() == source.model_ema.state_dict().keys()
    assert restored.step == source.step
    assert torch.allclose(torch.rand(2), expected_torch)
    assert np.allclose(np.random.rand(2), expected_numpy)
    assert random.random() == expected_random


@pytest.mark.parametrize("role_upload_fails", [False, True])
def test_dmd_manifest_commits_only_complete_checkpoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    role_upload_fails: bool,
) -> None:
    """The manifest should be written only after every role upload succeeds."""
    pipeline = object.__new__(DMDPipeline)
    pipeline.pipeline_config = SimpleNamespace(
        output_dir=str(tmp_path),
        max_steps=10,
        save_steps=5,
        checkpoint_steps=[],
        checkpoint_config={"async_upload": False, "keep_local_file": True},
        gan=SimpleNamespace(enabled=False),
    )
    pipeline.state = WorkerState(step=4, log_history=[{"system/step": 5}])
    pipeline.checkpoint_manager = Mock()
    pipeline.executor = ThreadPoolExecutor(max_workers=1)
    pipeline.resume_futures = [pipeline.executor.submit(lambda: None)]
    pipeline.checkpoint_role_clusters = {}
    for name in ("generator", "fake_score"):
        cluster = Mock()
        cluster.do_checkpoint.return_value = [f"save-{name}"]
        cluster.wait_for_checkpoint_upload.return_value = [f"wait-{name}"]
        pipeline.checkpoint_role_clusters[name] = cluster
    pipeline._cleanup_old_checkpoints = Mock()
    monkeypatch.setattr(
        DataProto,
        "materialize_concat",
        staticmethod(lambda data_refs: DataProto(meta_info={"metrics": {}})),
    )

    def get_uploads(refs: list[str]) -> list[str]:
        if role_upload_fails and refs and refs[0].startswith("wait-"):
            raise RuntimeError("role upload failed")
        return refs

    monkeypatch.setattr(dmd_pipeline.ray, "get", get_uploads)

    def save_rng_state(save_dir: str, tag: str) -> None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        torch.save({}, Path(save_dir) / f"rng_state_{tag}.pth")

    monkeypatch.setattr(WorkerState, "save_rng_state", staticmethod(save_rng_state))
    manifest_path = tmp_path / "manifest" / "checkpoint-5" / dmd_pipeline.DMD_CHECKPOINT_MANIFEST

    if role_upload_fails:
        with pytest.raises(RuntimeError, match="role upload failed"):
            pipeline.do_checkpoint(global_step=5)
        assert not manifest_path.exists()
        pipeline._cleanup_old_checkpoints.assert_not_called()
    else:
        pipeline.do_checkpoint(global_step=5)
        manifest = json.loads(manifest_path.read_text())
        assert manifest["roles"] == ["fake_score", "generator"]
        assert pipeline.checkpoint_manager.upload.call_args.kwargs["local_state_path"] == str(
            tmp_path / "manifest" / "checkpoint-5"
        )
        pipeline._cleanup_old_checkpoints.assert_called_once_with()
    pipeline.executor.shutdown()


def test_dmd_data_cursor_resumes_shuffled_batches(tmp_path: Path) -> None:
    """Checkpointed epoch and batch offset should reproduce subsequent shuffled batches."""

    def make_pipeline(state: WorkerState) -> DMDPipeline:
        pipeline = object.__new__(DMDPipeline)
        pipeline.state = state
        pipeline._data_generator = torch.Generator()
        pipeline.dataloader = DataLoader(
            dataset=list(range(12)),
            batch_size=3,
            shuffle=True,
            drop_last=True,
            generator=pipeline._data_generator,
        )
        pipeline._dataloader_iter = None
        return pipeline

    state = WorkerState(
        kv={
            dmd_pipeline.DMD_DATA_STATE_KEY: {
                "seed": 17,
                "epoch": 0,
                "batch_offset": 0,
                "shuffle": True,
            }
        }
    )
    source = make_pipeline(state)
    for _ in range(3):
        source._next_batch()
    state.save_to_json(save_dir=str(tmp_path), tag="pipeline")
    restored = make_pipeline(WorkerState.load_from_json(load_dir=str(tmp_path), tag="pipeline"))

    expected = [source._next_batch() for _ in range(6)]
    actual = [restored._next_batch() for _ in range(6)]

    assert all(torch.equal(left, right) for left, right in zip(expected, actual, strict=True))
