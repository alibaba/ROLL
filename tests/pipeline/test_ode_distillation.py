"""Focused tests for the Self-Forcing ODE distillation path."""

import copy
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from dacite import from_dict
from diffusers.models.modeling_outputs import Transformer2DModelOutput
import numpy as np
from omegaconf import OmegaConf
import pytest
import torch
from torch.utils._pytree import tree_flatten

from roll.configs.model_args import ModelArguments
from roll.configs.worker_config import StrategyArguments
from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.diffusion.models.wan.wan_adapter import WanDiffusionAdapter
from roll.pipeline.diffusion.models.wan.wan_self_forcing import (
    WanDiffusionWrapper,
    WanPromptEncoder,
    WanVAEConfig,
    _create_block_causal_mask,
)
from roll.pipeline.diffusion.ode_distillation import ode_distillation_pipeline, ode_distillation_worker
from roll.pipeline.diffusion.ode_distillation.ode_distillation_config import ODEDistillationConfig
from roll.pipeline.diffusion.ode_distillation.ode_distillation_pipeline import (
    ODE_DISTILLATION_CHECKPOINT_MANIFEST,
    ODE_DISTILLATION_DATA_STATE_KEY,
    ODEDistillationPipeline,
)
from roll.pipeline.diffusion.ode_distillation.ode_distillation_worker import (
    BaseODEDistillationWorker,
    ODE_INPUT,
    ODE_PAIR_IDS,
    ODE_PROMPTS,
    ODE_STUDENT_ROLE,
    ODE_TARGET,
    ODE_TIMESTEP,
    ODEDistillationStudentWorker,
    ODETrajectoryTeacherWorker,
)
from roll.pipeline.diffusion.models.scheduling_flow_match_sde_discrete import FlowMatchSDEDiscreteScheduler
from roll.pipeline.diffusion.tensor_transfer import TensorTransferGroup
from roll.utils.checkpoint_manager import CheckpointManager
from roll.utils.import_utils import safe_import_class
from roll.utils.worker_state import WorkerState


def _build_cache_teacher_worker() -> ODETrajectoryTeacherWorker:
    worker = ODETrajectoryTeacherWorker.__new__(ODETrajectoryTeacherWorker)
    worker.pipeline_config = SimpleNamespace(
        denoising_step_list=[1000, 500],
        diffusion_model_variant="wan2_1",
        guidance_scale=6.0,
        latent_shape=[2, 1, 2, 2],
        negative_prompt="negative prompt",
        num_train_timesteps=1000,
        seed=42,
        teacher_num_steps=4,
        timestep_shift=5.0,
    )
    worker.worker_config = SimpleNamespace(
        model_args=ModelArguments(
            model_name_or_path="teacher-model",
            dtype="bf16",
        ),
        strategy_args=StrategyArguments(
            strategy_name="veomni_infer",
            strategy_config={"mixed_precision": True, "param_dtype": "bfloat16"},
        ),
    )
    worker.diffusion_adapter = WanDiffusionAdapter(transformer=SimpleNamespace())
    worker.scheduler = FlowMatchSDEDiscreteScheduler(num_train_timesteps=1000, shift=5.0)
    worker.dtype = torch.bfloat16
    worker.device = torch.device("cpu")
    worker.rank = 0
    worker._cache_config_fingerprint = worker._build_cache_config_fingerprint()
    return worker


def test_ode_distillation_example_wires_teacher_and_student(tmp_path: Path) -> None:
    """The example should use a native teacher and provider-built causal student."""
    config_dict = OmegaConf.to_container(
        OmegaConf.load("examples/wan2_1-1.3B_ode_distillation/ode_distillation_config.yaml"),
        resolve=True,
    )
    config_dict["output_dir"] = str(tmp_path / "output")
    config_dict["logging_dir"] = str(tmp_path / "logs")
    config_dict["checkpoint_config"]["output_dir"] = str(tmp_path / "checkpoint")
    config_dict["tracker_kwargs"]["log_dir"] = str(tmp_path / "tracker")

    config = from_dict(data_class=ODEDistillationConfig, data=config_dict)

    assert config.denoising_step_list == [1000, 750, 500, 250]
    assert config.diffusion_model_variant == "wan2_1"
    assert config.teacher.strategy_args.strategy_name == "veomni_infer"
    assert safe_import_class(config.teacher.worker_cls) is ODETrajectoryTeacherWorker
    assert config.student.strategy_args.strategy_name == "fsdp2_train"
    assert safe_import_class(config.student.worker_cls) is ODEDistillationStudentWorker
    assert config.student.training_args.gradient_accumulation_steps == 8
    assert config.student.model_args.model_config_kwargs["num_frame_per_block"] == 3


def test_ode_distillation_timesteps_match_self_forcing_rollout() -> None:
    """ODE regression inputs must be the exact nodes consumed by downstream Self-Forcing."""
    ode_config = OmegaConf.load("examples/wan2_1-1.3B_ode_distillation/ode_distillation_config.yaml")
    dmd_config = OmegaConf.load("examples/wan2_1-1.3B_dmd/dmd_self_forcing_config.yaml")
    assert list(ode_config.denoising_step_list) == list(dmd_config.denoising.step_list)
    assert ode_config.timestep_shift == dmd_config.timestep_shift

    scheduler = FlowMatchSDEDiscreteScheduler(
        num_train_timesteps=ode_config.num_train_timesteps,
        shift=ode_config.timestep_shift,
    )
    scheduler.set_timesteps(sigmas=np.linspace(1.0, 0.0, ode_config.teacher_num_steps + 1)[:-1])
    node_positions = torch.tensor(
        [
            (ode_config.num_train_timesteps - step)
            * ode_config.teacher_num_steps
            // ode_config.num_train_timesteps
            for step in [*ode_config.denoising_step_list, 0]
        ]
    )
    ode_timesteps = torch.cat((scheduler.timesteps.cpu(), torch.zeros(1)))[node_positions]

    rollout_scheduler = FlowMatchSDEDiscreteScheduler(
        num_train_timesteps=ode_config.num_train_timesteps,
        shift=dmd_config.timestep_shift,
    )
    rollout_steps = torch.tensor([*dmd_config.denoising.step_list, 0])
    rollout_timesteps = torch.cat((rollout_scheduler.timesteps.cpu(), torch.zeros(1)))[
        ode_config.num_train_timesteps - rollout_steps
    ]
    assert torch.equal(ode_timesteps, rollout_timesteps)


def test_wan_ode_input_keeps_block_timesteps_and_causal_mask() -> None:
    """ODE regression needs per-frame timesteps and bidirectional attention within each block."""
    class RecordingTransformer(torch.nn.Module):
        _roll_forward_features = True
        uses_per_frame_timesteps = True

        def __init__(self) -> None:
            super().__init__()
            self.timestep = None

        def forward(
            self,
            *,
            hidden_states: torch.Tensor,
            timestep: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
        ) -> torch.Tensor:
            del encoder_hidden_states
            self.timestep = timestep
            return hidden_states

    transformer = RecordingTransformer()
    adapter = WanDiffusionAdapter(transformer=transformer)
    timestep = torch.tensor([[1000.0, 1000.0, 500.0, 500.0]])
    adapter.forward_step(
        latents=torch.zeros(1, 4, 2, 1, 1),
        timestep=timestep,
        prompt={"prompt_embeds": torch.zeros(1, 1, 2)},
    )
    assert torch.equal(transformer.timestep, timestep)

    block_mask = _create_block_causal_mask(
        sequence_length=12,
        frame_sequence_length=2,
        num_frame_per_block=2,
        local_attn_size=-1,
        device=torch.device("cpu"),
    )
    zero = torch.tensor(0)
    assert block_mask.mask_mod(zero, zero, torch.tensor(0), torch.tensor(3))
    assert not block_mask.mask_mod(zero, zero, torch.tensor(0), torch.tensor(4))
    assert block_mask.mask_mod(zero, zero, torch.tensor(4), torch.tensor(7))
    assert not block_mask.mask_mod(zero, zero, torch.tensor(4), torch.tensor(8))


def test_wan_wrapper_exposes_only_fsdp_transformer_state(tmp_path: Path) -> None:
    """The wrapper should expose a traversable output and save only transformer state."""
    class TransformerStub(torch.nn.Module):
        config = SimpleNamespace(patch_size=(1, 1, 1))

        def save_config(self, save_directory: str) -> None:
            Path(save_directory, "config.json").write_text("{}", encoding="utf-8")

        def forward(self, hidden_states: torch.Tensor, **_: object) -> tuple[torch.Tensor]:
            return (hidden_states,)

    wrapper = WanDiffusionWrapper.__new__(WanDiffusionWrapper)
    torch.nn.Module.__init__(wrapper)
    wrapper.model = TransformerStub()
    wrapper.prompt_encoder = WanPromptEncoder(
        tokenizer=object(),
        text_encoder=torch.nn.Linear(2, 2, bias=False),
        vae_config=WanVAEConfig(z_dim=16, scale_factor_spatial=8, scale_factor_temporal=4),
    )
    wrapper.input_dtype = None
    wrapper.local_attn_size = -1
    wrapper.sink_size = 0
    wrapper.num_frame_per_block = 1
    wrapper._block_causal_masks = {}

    output = wrapper(
        hidden_states=torch.zeros(1, 2, 1, 1, 1),
        timestep=torch.ones(1, 1),
        encoder_hidden_states=torch.zeros(1, 1, 2),
        kv_cache=[{}],
    )

    assert isinstance(output, Transformer2DModelOutput)
    assert output.sample.shape == (1, 2, 1, 1, 1)
    assert tree_flatten(output)[0] == [output.sample]
    assert list(wrapper.named_children()) == [("model", wrapper.model)]
    wrapper.save_pretrained(
        str(tmp_path),
        state_dict={
            "model.weight": torch.arange(4, dtype=torch.float32),
            "prompt_encoder.weight": torch.ones(1),
        },
        safe_serialization=True,
    )

    from safetensors.torch import load_file

    saved_state = load_file(tmp_path / "diffusion_pytorch_model.safetensors")
    assert list(saved_state) == ["weight"]
    assert torch.equal(saved_state["weight"], torch.arange(4, dtype=torch.float32))


@pytest.mark.parametrize("raise_during_encode", [False, True])
def test_ode_prompt_encoder_restores_cpu_after_each_encoding(
    raise_during_encode: bool,
) -> None:
    """Provider-owned text encoders must follow the prompt call, including failures."""
    class AdapterStub:
        def encode_prompt(
            self,
            *,
            prompt_encoder: object,
            prompt_inputs: dict[str, object],
        ) -> dict[str, torch.Tensor]:
            del prompt_encoder, prompt_inputs
            if raise_during_encode:
                raise RuntimeError("prompt encoding failed")
            return {"prompt_embeds": torch.ones(1)}

    text_encoder = Mock()
    worker = BaseODEDistillationWorker.__new__(BaseODEDistillationWorker)
    worker.pipeline_config = SimpleNamespace(is_offload_states=True)
    worker.prompt_encoder = WanPromptEncoder(
        tokenizer=object(),
        text_encoder=text_encoder,
        vae_config=WanVAEConfig(z_dim=16, scale_factor_spatial=8, scale_factor_temporal=4),
    )
    worker.diffusion_adapter = AdapterStub()
    worker.device = torch.device("cuda:0")
    worker.dtype = torch.bfloat16

    if raise_during_encode:
        with pytest.raises(RuntimeError, match="prompt encoding failed"):
            worker._encode_prompt(["test"])
    else:
        assert torch.equal(worker._encode_prompt(["test"])["prompt_embeds"], torch.ones(1))

    assert [call.args[0] for call in text_encoder.to.call_args_list] == [
        torch.device("cuda:0"),
        "cpu",
    ]


def test_ode_distillation_student_resume_restores_backend_state_before_offload(tmp_path: Path) -> None:
    """Student resume must restore model, optimizer, scheduler, and RNG before offloading."""
    calls: list[tuple[str, object]] = []
    worker = ODEDistillationStudentWorker.__new__(ODEDistillationStudentWorker)
    worker.worker_config = SimpleNamespace(
        model_args=SimpleNamespace(),
        strategy_args=SimpleNamespace(strategy_config={}),
    )
    worker.worker_name = "student-0"
    strategy = Mock()
    strategy.optimizer = object()
    strategy.load_checkpoint.side_effect = lambda load_dir: calls.append(("load_checkpoint", load_dir))
    strategy.offload_states.side_effect = lambda: calls.append(("offload_states", None))
    prompt_encoder = SimpleNamespace(tokenizer=object())
    config = SimpleNamespace(
        diffusion_model_variant="wan2_1",
        resume_from_checkpoint=str(tmp_path / "checkpoint-10"),
        is_offload_states=True,
    )

    def initialize_runtime(pipeline_config: object, model_provider: object) -> None:
        calls.append(("initialize_runtime", model_provider))
        worker.pipeline_config = pipeline_config
        worker.strategy = strategy
        worker.prompt_encoder = prompt_encoder

    worker._initialize_runtime = initialize_runtime
    ODEDistillationStudentWorker.initialize.__wrapped__(worker, config)

    assert calls[0][0] == "initialize_runtime"
    assert calls[1:] == [
        ("load_checkpoint", os.path.join(config.resume_from_checkpoint, ODE_STUDENT_ROLE)),
        ("offload_states", None),
    ]
    assert worker.optimizer is strategy.optimizer
    assert strategy.tokenizer is prompt_encoder.tokenizer


def test_ode_distillation_dataloader_resumes_at_the_exact_next_batch() -> None:
    """Checkpointed epoch and batch offset must preserve the shuffled prompt stream."""
    dataset = [{"text": f"prompt-{index}", ODE_PAIR_IDS: index} for index in range(12)]
    config = SimpleNamespace(
        seed=42,
        teacher=SimpleNamespace(
            training_args=SimpleNamespace(
                per_device_train_batch_size=2,
                dataloader_num_workers=0,
                data_seed=17,
            )
        ),
        student=SimpleNamespace(
            training_args=SimpleNamespace(
                gradient_accumulation_steps=2,
                per_device_train_batch_size=2,
            )
        ),
    )

    def build_pipeline(state: WorkerState, resume_from_checkpoint: bool | str) -> ODEDistillationPipeline:
        pipeline = ODEDistillationPipeline.__new__(ODEDistillationPipeline)
        pipeline.pipeline_config = config
        pipeline.teacher = SimpleNamespace(dp_size=1)
        pipeline.student = SimpleNamespace(dp_size=1)
        pipeline.resume_from_checkpoint = resume_from_checkpoint
        pipeline.state = state
        pipeline._initialize_dataloader(dataset)
        return pipeline

    uninterrupted = build_pipeline(WorkerState(), False)
    for _ in range(7):
        uninterrupted._next_batch()
    saved_state = copy.deepcopy(uninterrupted.state)
    expected_batch = uninterrupted._next_batch()

    resumed = build_pipeline(saved_state, "/tmp/checkpoint-10")
    resumed_batch = resumed._next_batch()

    assert resumed.state.kv[ODE_DISTILLATION_DATA_STATE_KEY]["epoch"] == 1
    assert torch.equal(resumed_batch[ODE_PAIR_IDS], expected_batch[ODE_PAIR_IDS])
    assert resumed_batch["prompts"].tolist() == expected_batch["prompts"].tolist()

    resumed.teacher.dp_size = 2
    with pytest.raises(ValueError, match="teacher=4, student=2"):
        resumed._initialize_dataloader(dataset)


def test_ode_pair_cache_validates_metadata_and_shape(tmp_path: Path) -> None:
    """A cache hit must match its prompt, teacher config, format, and trajectory shape."""
    worker = _build_cache_teacher_worker()
    cache_path = tmp_path / "00007.pt"
    trajectory = torch.arange(24, dtype=torch.float16).reshape(3, 2, 1, 2, 2)

    worker._save_cached_trajectory(
        cache_path=str(cache_path),
        trajectory=trajectory,
        pair_id=7,
        prompt="matching prompt",
    )

    loaded_trajectory = worker._load_cached_trajectory(
        cache_path=str(cache_path),
        pair_id=7,
        prompt="matching prompt",
    )
    assert loaded_trajectory.dtype == torch.bfloat16
    assert torch.equal(loaded_trajectory.float(), trajectory.float())

    with pytest.raises(ValueError, match="'prompt'"):
        worker._load_cached_trajectory(
            cache_path=str(cache_path),
            pair_id=7,
            prompt="different prompt",
        )

    worker.pipeline_config.guidance_scale = 7.0
    worker._cache_config_fingerprint = worker._build_cache_config_fingerprint()
    with pytest.raises(ValueError, match="'config_fingerprint'"):
        worker._load_cached_trajectory(
            cache_path=str(cache_path),
            pair_id=7,
            prompt="matching prompt",
        )

    torch.save(trajectory, cache_path)
    with pytest.raises(ValueError, match="Unsupported ODE pair cache format"):
        worker._load_cached_trajectory(
            cache_path=str(cache_path),
            pair_id=7,
            prompt="matching prompt",
        )

    worker._save_cached_trajectory(
        cache_path=str(cache_path),
        trajectory=trajectory[:-1],
        pair_id=7,
        prompt="matching prompt",
    )
    with pytest.raises(ValueError, match="has shape"):
        worker._load_cached_trajectory(
            cache_path=str(cache_path),
            pair_id=7,
            prompt="matching prompt",
        )


def test_ode_teacher_batches_trajectories_and_reuses_per_sample_cache(tmp_path: Path) -> None:
    """A teacher local batch should share model forwards while retaining independent pair caches."""

    class AdapterStub:
        def __init__(self) -> None:
            self.forward_batch_sizes: list[int] = []

        def forward_step(self, *, latents: torch.Tensor, **_: object) -> SimpleNamespace:
            self.forward_batch_sizes.append(latents.shape[0])
            return SimpleNamespace(flow_pred=torch.zeros_like(latents))

    worker = _build_cache_teacher_worker()
    worker.pipeline_config.ode_pair_cache_dir = str(tmp_path)
    worker.pipeline_config.is_offload_states = False
    worker.pipeline_config.num_frame_per_block = 1
    worker.pipeline_config.num_ode_pairs = 16
    worker.strategy = SimpleNamespace()
    worker.cluster_name = "teacher"
    worker.world_size = 1
    worker.rank_info = SimpleNamespace(dp_rank=0)
    worker.dtype = torch.float32
    adapter = AdapterStub()
    worker.diffusion_adapter = adapter
    encoded_prompt_batches: list[list[str]] = []

    def encode_prompt(prompts: list[str]) -> dict[str, torch.Tensor]:
        encoded_prompt_batches.append(prompts)
        return {"prompt_embeds": torch.zeros(len(prompts), 1, 1)}

    worker._encode_prompt = encode_prompt

    def build_batch() -> DataProto:
        data = DataProto.from_dict(
            tensors={ODE_PAIR_IDS: torch.tensor([7, 8])},
            non_tensors={ODE_PROMPTS: ["prompt-7", "prompt-8"]},
        )
        data.meta_info = {
            "global_step": 0,
            "accumulation_step": 0,
            "gradient_accumulation_steps": 1,
        }
        return data

    first = ODETrajectoryTeacherWorker.make_regression_batch.__wrapped__(worker, build_batch())

    assert adapter.forward_batch_sizes == [2] * worker.pipeline_config.teacher_num_steps
    assert encoded_prompt_batches == [
        ["prompt-7", "prompt-8"],
        ["negative prompt", "negative prompt"],
    ]
    assert {key: tuple(value.shape) for key, value in first.batch.items()} == {
        ODE_INPUT: (2, 2, 1, 2, 2),
        ODE_TARGET: (2, 2, 1, 2, 2),
        ODE_TIMESTEP: (2, 2),
    }
    assert first.non_tensor_batch[ODE_PROMPTS].tolist() == ["prompt-7", "prompt-8"]

    (tmp_path / "00008.pt").unlink()
    adapter.forward_batch_sizes.clear()
    encoded_prompt_batches.clear()
    ODETrajectoryTeacherWorker.make_regression_batch.__wrapped__(worker, build_batch())

    assert adapter.forward_batch_sizes == [1] * worker.pipeline_config.teacher_num_steps
    assert encoded_prompt_batches == [["prompt-8"], ["negative prompt"]]

    worker._gpu_tensor_transfer_enabled = True
    worker._tensor_transfer_slots = {}
    direct = ODETrajectoryTeacherWorker.make_regression_batch.__wrapped__(worker, build_batch())
    assert set(direct.batch.keys()) == {ODE_TIMESTEP}
    assert torch.equal(worker._tensor_transfer_slots[ODE_INPUT], first.batch[ODE_INPUT])
    assert torch.equal(worker._tensor_transfer_slots[ODE_TARGET], first.batch[ODE_TARGET])
    assert torch.equal(direct.batch[ODE_TIMESTEP], first.batch[ODE_TIMESTEP])


def test_ode_student_consumes_direct_transfer_slots() -> None:
    """The direct path should train from local GPU slots while DataProto carries only control data."""

    class ScaleModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = torch.nn.Parameter(torch.ones(()))

    model = ScaleModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    worker = ODEDistillationStudentWorker.__new__(ODEDistillationStudentWorker)
    worker.pipeline_config = SimpleNamespace(is_offload_states=False, max_grad_norm=10.0, num_train_timesteps=1000)
    worker.device = torch.device("cpu")
    worker.dtype = torch.float32
    worker.model = model
    worker.optimizer = optimizer
    worker.diffusion_adapter = SimpleNamespace(
        forward_step=lambda latents, **_: SimpleNamespace(flow_pred=latents * model.scale)
    )
    worker._encode_prompt = lambda prompts: {"prompt_embeds": torch.zeros(len(prompts), 1, 1)}
    worker._gpu_tensor_transfer_enabled = True
    worker._tensor_transfer_slots = {
        ODE_INPUT: torch.ones(1, 2, 1, 1, 1),
        ODE_TARGET: torch.zeros(1, 2, 1, 1, 1),
    }
    worker.strategy = SimpleNamespace(
        scaler=None,
        scheduler=None,
        clip_grad_norm=lambda max_norm: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm),
    )
    batch = DataProto.from_dict(
        tensors={ODE_TIMESTEP: torch.full((1, 2), 500.0)},
        non_tensors={ODE_PROMPTS: ["test prompt"]},
        meta_info={"accumulation_step": 0, "gradient_accumulation_steps": 1},
    )

    output = ODEDistillationStudentWorker.train_step.__wrapped__(worker, batch)

    assert worker._tensor_transfer_slots == {}
    assert output.meta_info["metrics"]["student/loss"] == pytest.approx(0.25)
    assert model.scale.item() > 1.0


def test_tensor_transfer_requires_identical_role_topology() -> None:
    """Direct transfer maps local samples by their DP/TP/PP/CP coordinates."""

    def cluster(name: str, cp_rank: int = 0) -> SimpleNamespace:
        return SimpleNamespace(
            cluster_name=name,
            world_size=1,
            worker_rank_info=[
                SimpleNamespace(dp_rank=0, tp_rank=0, pp_rank=0, cp_rank=cp_rank),
            ],
            rank2devices={0: [{"node_rank": 0, "gpu_rank": 0}]},
        )

    assert TensorTransferGroup.compatibility_error(cluster("teacher"), cluster("student")) is None
    assert "different DP/TP/PP/CP layouts" in TensorTransferGroup.compatibility_error(
        cluster("teacher"),
        cluster("student", cp_rank=1),
    )


def _build_checkpoint_pipeline(
    tmp_path: Path,
    *,
    checkpoint_config: dict[str, object],
    max_steps: int,
    state_step: int,
    save_steps: int,
) -> ODEDistillationPipeline:
    pipeline = ODEDistillationPipeline.__new__(ODEDistillationPipeline)
    pipeline.pipeline_config = SimpleNamespace(
        output_dir=str(tmp_path),
        max_steps=max_steps,
        save_steps=save_steps,
        checkpoint_config=checkpoint_config,
    )
    pipeline.state = WorkerState(step=state_step, log_history=[{}])
    pipeline.checkpoint_clusters = []
    pipeline.checkpoint_manager = Mock()
    pipeline.checkpoint_manager.uploader = object()
    pipeline.executor = ThreadPoolExecutor(max_workers=1)
    pipeline.resume_futures = []
    pipeline._cleanup_old_checkpoints = Mock()
    return pipeline


def _patch_checkpoint_io(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        DataProto,
        "materialize_concat",
        staticmethod(lambda data_refs: DataProto(meta_info={"metrics": {}})),
    )
    monkeypatch.setattr(ode_distillation_pipeline.ray, "get", lambda refs: refs)

    def save_rng_state(save_dir: str, tag: str) -> None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        (Path(save_dir) / f"rng_state_{tag}.pth").write_bytes(b"rng")

    monkeypatch.setattr(WorkerState, "save_rng_state", staticmethod(save_rng_state))


def test_ode_distillation_final_checkpoint_is_resumable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The final step must save locally and auto-resume past an incomplete newer checkpoint."""
    checkpoint_step = 3
    checkpoint_dir = tmp_path / f"checkpoint-{checkpoint_step}"
    pipeline = _build_checkpoint_pipeline(
        tmp_path,
        checkpoint_config={},
        max_steps=checkpoint_step,
        state_step=checkpoint_step - 1,
        save_steps=1000,
    )
    pipeline.checkpoint_manager = CheckpointManager({})
    student = Mock()
    student.wait_for_checkpoint_upload.return_value = []

    def save_student(**_: object) -> list[str]:
        dcp_dir = checkpoint_dir / ODE_STUDENT_ROLE / "dcp"
        dcp_dir.mkdir(parents=True)
        (dcp_dir / ".metadata").write_bytes(b"metadata")
        (dcp_dir / "__0_0.distcp").write_bytes(b"shard")
        return ["student-checkpoint"]

    student.do_checkpoint.side_effect = save_student
    pipeline.checkpoint_clusters = [student]
    _patch_checkpoint_io(monkeypatch)

    pipeline.do_checkpoint(global_step=checkpoint_step, is_last_step=True)
    manifest = ODEDistillationPipeline._load_resume_manifest(str(checkpoint_dir))
    incomplete_dir = tmp_path / "checkpoint-4"
    (incomplete_dir / "pipeline").mkdir(parents=True)
    (incomplete_dir / "pipeline" / "worker_state_pipeline.json").write_text("{}", encoding="utf-8")
    (incomplete_dir / "pipeline" / "rng_state_pipeline.pth").write_bytes(b"rng")
    (incomplete_dir / ODE_STUDENT_ROLE / "dcp").mkdir(parents=True)
    (incomplete_dir / ODE_STUDENT_ROLE / "dcp" / ".metadata").write_bytes(b"metadata")
    (incomplete_dir / ODE_DISTILLATION_CHECKPOINT_MANIFEST).write_text(
        json.dumps(
            {
                "format_version": 1,
                "checkpoint_id": "checkpoint-4",
                "global_step": 4,
                "pipeline_step": 3,
                "roles": [ODE_STUDENT_ROLE],
            }
        ),
        encoding="utf-8",
    )
    resume_config = SimpleNamespace(
        resume_from_checkpoint=True,
        checkpoint_config={"type": "file_system", "output_dir": str(tmp_path)},
        output_dir=str(tmp_path),
    )

    assert manifest["global_step"] == checkpoint_step
    assert ODEDistillationPipeline._prepare_resume(resume_config) == manifest
    assert resume_config.resume_from_checkpoint == str(checkpoint_dir)
    pipeline._cleanup_old_checkpoints.assert_called_once_with()
    pipeline.executor.shutdown()


@pytest.mark.parametrize("failure_source", ["student", "pipeline"])
def test_ode_distillation_manifest_is_not_committed_before_uploads_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_source: str,
) -> None:
    """A failed student or pipeline upload must leave the checkpoint uncommitted."""
    pipeline = _build_checkpoint_pipeline(
        tmp_path,
        checkpoint_config={"async_upload": True, "keep_local_file": True},
        max_steps=10,
        state_step=4,
        save_steps=5,
    )
    student = Mock()
    student.do_checkpoint.return_value = ["save-student"]
    student.wait_for_checkpoint_upload.return_value = ["wait-student"]
    pipeline.checkpoint_clusters = [student]
    _patch_checkpoint_io(monkeypatch)
    if failure_source == "student":
        def fail_student_upload(_: list[str]) -> None:
            raise RuntimeError("student upload failed")

        monkeypatch.setattr(
            ode_distillation_pipeline.ray,
            "get",
            fail_student_upload,
        )
    else:
        pipeline.checkpoint_manager.upload.side_effect = RuntimeError("pipeline upload failed")

    try:
        with pytest.raises(RuntimeError, match=f"{failure_source} upload failed"):
            pipeline.do_checkpoint(global_step=5)
    finally:
        pipeline.executor.shutdown()

    manifest_path = tmp_path / "manifest" / "checkpoint-5" / ODE_DISTILLATION_CHECKPOINT_MANIFEST
    assert not manifest_path.exists()
    pipeline._cleanup_old_checkpoints.assert_not_called()


@pytest.mark.parametrize("uploader_configured", [False, True])
def test_ode_student_checkpoint_uses_shared_role_root_and_waitable_upload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    uploader_configured: bool,
) -> None:
    """Every student rank must save to one role root and expose the rank-zero upload future."""
    worker = ODEDistillationStudentWorker.__new__(ODEDistillationStudentWorker)
    worker.pipeline_config = SimpleNamespace(output_dir=str(tmp_path), is_offload_states=True)
    worker.worker_config = SimpleNamespace(checkpoint_config={"async_upload": True, "keep_local_file": True})
    worker.cluster_name = ODE_STUDENT_ROLE
    worker.strategy = Mock()
    worker.strategy.thread_executor = ThreadPoolExecutor(max_workers=1)
    worker.strategy.checkpoint_manager = SimpleNamespace(uploader=object())
    strategy_uploader = worker.strategy.checkpoint_manager.uploader
    worker.strategy.async_save_strategy = True
    async_save_modes: list[bool] = []
    worker.strategy.save_checkpoint.side_effect = lambda *args, **kwargs: (
        async_save_modes.append(worker.strategy.async_save_strategy) or {}
    )
    worker.checkpoint_manager = Mock()
    worker.checkpoint_manager.uploader = object() if uploader_configured else None
    worker._checkpoint_uploads = {}
    monkeypatch.setattr(ode_distillation_worker.dist, "is_available", lambda: True)
    monkeypatch.setattr(ode_distillation_worker.dist, "is_initialized", lambda: False)

    ODEDistillationStudentWorker.do_checkpoint.__wrapped__(
        worker,
        global_step=7,
        is_last_step=False,
    )
    ODEDistillationStudentWorker.wait_for_checkpoint_upload.__wrapped__(worker, "checkpoint-7")

    checkpoint_root = (
        tmp_path / ODE_STUDENT_ROLE / "checkpoint-7"
        if uploader_configured
        else tmp_path / "checkpoint-7"
    )
    worker.strategy.save_checkpoint.assert_called_once_with(
        str(checkpoint_root / ODE_STUDENT_ROLE),
        7,
        "checkpoint-7",
        is_last_step=False,
    )
    assert worker.strategy.async_save_strategy is True
    assert worker.strategy.checkpoint_manager.uploader is strategy_uploader
    assert async_save_modes == [False]
    worker.strategy.offload_states.assert_called_once_with()
    assert worker.checkpoint_manager.upload.call_count == int(uploader_configured)

    worker.strategy.offload_states.reset_mock()
    worker.strategy.save_checkpoint.side_effect = RuntimeError("serialization failed")
    with pytest.raises(RuntimeError, match="serialization failed"):
        ODEDistillationStudentWorker.do_checkpoint.__wrapped__(
            worker,
            global_step=8,
            is_last_step=False,
        )

    worker.strategy.offload_states.assert_called_once_with()
    assert worker.strategy.async_save_strategy is True
    worker.strategy.thread_executor.shutdown()
