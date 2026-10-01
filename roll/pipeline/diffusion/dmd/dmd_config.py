"""Configuration dataclasses for the DMD diffusion pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

from roll.configs.base_config import BaseConfig
from roll.configs.data_args import DataArguments
from roll.configs.worker_config import StrategyArguments, WorkerConfig


VEOMNI_TRAIN_STRATEGY = "veomni_train"
VEOMNI_INFER_STRATEGY = "veomni_infer"
FSDP2_TRAIN_STRATEGY = "fsdp2_train"


@dataclass
class DMDDenoisingConfig:
    """Denoising trajectory options consumed by algorithm pipelines."""

    step_list: list[int] = field(default_factory=lambda: [1000, 750, 500, 250])
    warp_step: bool = True


@dataclass
class DMDSampleConfig:
    """Pixel-space dimensions of samples generated during DMD training."""

    num_frames: int = 81
    height: int = 480
    width: int = 832


@dataclass
class DMDScoreTimestepConfig:
    """Timestep sampling and shift options for score/generator updates."""

    num_train_timestep: int = 1000
    min_score_timestep: int = 0
    sample_shift: float = 1.0
    ts_schedule: bool = False
    ts_schedule_max: bool = False


@dataclass
class DMDGuidanceConfig:
    """Classifier-free guidance options for real/fake score models."""

    scale: float = 5.0
    real_scale: float | None = None
    fake_scale: float | None = None
    negative_prompt: str = ""


@dataclass
class DMDDecoupledConfig:
    """Decoupled distribution-matching and CFG-augmentation options."""

    enabled: bool = False
    gradient_scale: float = 1.0


@dataclass
class DMDEMAConfig:
    """EMA options for generator weights."""

    enabled: bool = False
    weight: float = 0.99
    start_step: int = 200
    update_interval: int = 1


@dataclass
class DMDSelfForcingConfig:
    """Generator-side Self-Forcing training path options."""

    enabled: bool = False
    num_frame_per_block: int = 3
    independent_first_frame: bool = False
    same_step_across_blocks: bool = True
    last_step_only: bool = False
    context_noise: int = 0
    num_training_frames: int = 21


@dataclass
class DMDGANHeadConfig:
    """Classification head selected by the DMD2 GAN objective."""

    name: str = "residual_conv3d"
    hidden_dim: int = 512


@dataclass
class DMDGANConfig:
    """Optional DMD2 classification objective and real-latent data."""

    enabled: bool = False
    generator_loss_weight: float = 5e-3
    classification_loss_weight: float = 1e-2
    head: DMDGANHeadConfig = field(default_factory=DMDGANHeadConfig)
    noisy_input: bool = False
    max_timestep: int = 0
    real_data: DataArguments = field(default_factory=DataArguments)
    real_latent_key: str = "latent"
    real_prompt_key: str = "text"


@dataclass
class DMDConfig(BaseConfig):
    """ROLL config for DMD diffusion training."""

    full_determinism: bool = field(
        default=False,
        metadata={"help": "Use deterministic PyTorch and cuDNN kernels for DMD worker execution."},
    )
    pretrain: str | None = field(
        default=None,
        metadata={"help": "Path to the diffusion checkpoint directory."},
    )
    diffusion_model_variant: str = field(
        default="wan2_1",
        metadata={"help": "Diffusion model variant used to select the model adapter."},
    )
    sequence_length: int | None = 1
    timestep_shift: float = field(
        default=8.0,
        metadata={"help": "Flow-matching timestep shift used by DMD's scheduler."},
    )
    denoising: DMDDenoisingConfig = field(default_factory=DMDDenoisingConfig)
    sample: DMDSampleConfig = field(default_factory=DMDSampleConfig)
    score_timestep: DMDScoreTimestepConfig = field(default_factory=DMDScoreTimestepConfig)
    guidance: DMDGuidanceConfig = field(default_factory=DMDGuidanceConfig)
    ddmd: DMDDecoupledConfig = field(default_factory=DMDDecoupledConfig)
    self_forcing: DMDSelfForcingConfig = field(default_factory=DMDSelfForcingConfig)
    gan: DMDGANConfig = field(default_factory=DMDGANConfig)
    dfake_gen_update_ratio: int = 5
    share_prompt_embeddings: bool = field(
        default=True,
        metadata={
            "help": (
                "Reuse generator-encoded positive, negative, and GAN real prompt tensors in score workers, "
                "so score roles do not initialize local prompt encoders."
            ),
        },
    )
    gpu_tensor_transfer: bool = field(
        default=True,
        metadata={
            "help": (
                "Transfer large tensors directly between DMD role GPUs. Colocated workers use CUDA IPC and "
                "separated workers use NCCL. Disable to use CPU-backed DataProto transport."
            )
        },
    )
    ema: DMDEMAConfig = field(default_factory=DMDEMAConfig)
    max_grad_norm: float = field(default=10.0, metadata={"help": "Maximum gradient norm for trainable DMD roles."})
    checkpoint_steps: list[int] = field(
        default_factory=list,
        metadata={"help": "Additional DMD global steps that should trigger checkpointing."},
    )
    checkpoint_roles: list[str] = field(
        default_factory=lambda: ["generator", "fake_score"],
        metadata={
            "help": (
                "DMD roles to save. Saving both generator and fake_score also saves pipeline state for full resume; "
                "a subset is a model snapshot."
            )
        },
    )
    parallel_checkpoint_roles: bool = field(
        default=False,
        metadata={
            "help": (
                "Save different DMD checkpoint roles concurrently. All ranks within one role always save "
                "concurrently because DCP requires collective participation. Upload concurrency remains controlled "
                "by checkpoint_config.async_upload."
            )
        },
    )
    generator: WorkerConfig = field(
        default_factory=WorkerConfig,
        metadata={"help": "ROLL worker config for the DMD generator role."},
    )
    fake_score: WorkerConfig = field(
        default_factory=WorkerConfig,
        metadata={"help": "ROLL worker config for the trainable DMD fake-score role."},
    )
    real_score: WorkerConfig = field(
        default_factory=WorkerConfig,
        metadata={"help": "ROLL worker config for the frozen DMD real-score role."},
    )

    def __post_init__(self) -> None:
        if not self.denoising.step_list or all(step == 0 for step in self.denoising.step_list):
            raise ValueError("denoising.step_list must contain at least one non-zero timestep")
        if any(
            step < 0 or step > self.score_timestep.num_train_timestep
            for step in self.denoising.step_list
        ):
            raise ValueError("denoising.step_list must stay inside the diffusion training timestep range")
        if self.denoising.step_list[0] != self.score_timestep.num_train_timestep:
            raise ValueError("DMD noise-only generation requires denoising.step_list to start at num_train_timestep")
        if any(left <= right for left, right in zip(self.denoising.step_list, self.denoising.step_list[1:])):
            raise ValueError("denoising.step_list must be strictly decreasing")
        if self.ddmd.gradient_scale < 0:
            raise ValueError("ddmd.gradient_scale must be non-negative")
        if self.ddmd.enabled:
            real_guidance_scale = (
                self.guidance.real_scale if self.guidance.real_scale is not None else self.guidance.scale
            )
            if real_guidance_scale <= 1.0:
                raise ValueError("DDMD requires guidance.real_scale or guidance.scale to be greater than 1")
            if self.guidance.fake_scale not in (None, 0.0):
                raise ValueError("DDMD distribution matching requires guidance.fake_scale to be 0")
            if self.self_forcing.enabled:
                raise ValueError("DDMD currently supports only the native non-causal generator rollout")
        if self.self_forcing.enabled:
            if self.self_forcing.num_frame_per_block <= 0 or self.self_forcing.num_training_frames <= 0:
                raise ValueError("Self-Forcing block size and training window must be positive")
            if not 0 <= self.self_forcing.context_noise <= self.score_timestep.num_train_timestep:
                raise ValueError("self_forcing.context_noise must stay inside the diffusion training timestep range")
        if self.dfake_gen_update_ratio <= 0:
            raise ValueError("dfake_gen_update_ratio must be positive")
        if self.gan.generator_loss_weight < 0 or self.gan.classification_loss_weight < 0:
            raise ValueError("gan loss weights must be non-negative")
        if self.gan.head.name != "residual_conv3d":
            raise ValueError(f"Unsupported DMD GAN head: {self.gan.head.name}")
        if self.gan.head.hidden_dim <= 0:
            raise ValueError("gan.head.hidden_dim must be positive")
        if self.gan.enabled and self.gan.real_data.file_name is None:
            raise ValueError("gan.real_data.file_name is required when DMD2 GAN is enabled")
        if self.gan.enabled and self.gan.noisy_input and self.gan.max_timestep <= 0:
            raise ValueError("gan.max_timestep must be positive when gan.noisy_input is enabled")
        if self.gan.enabled and self.gan.max_timestep > self.score_timestep.num_train_timestep:
            raise ValueError("gan.max_timestep cannot exceed score_timestep.num_train_timestep")
        if self.ema.enabled and not 0.0 <= self.ema.weight < 1.0:
            raise ValueError("ema.weight must be in [0, 1)")
        if self.ema.enabled and (self.ema.start_step < 0 or self.ema.update_interval <= 0):
            raise ValueError("ema.start_step must be non-negative and ema.update_interval must be positive")
        super().__post_init__()

        role_defaults = [
            (
                self.generator,
                "generator",
                "roll.pipeline.diffusion.dmd.dmd_worker.DMDGeneratorWorker",
                VEOMNI_TRAIN_STRATEGY,
            ),
            (
                self.fake_score,
                "fake_score",
                "roll.pipeline.diffusion.dmd.dmd_worker.DMDFakeScoreWorker",
                VEOMNI_TRAIN_STRATEGY,
            ),
            (
                self.real_score,
                "real_score",
                "roll.pipeline.diffusion.dmd.dmd_worker.DMDRealScoreWorker",
                VEOMNI_INFER_STRATEGY,
            ),
        ]
        for worker_config, name, worker_cls, default_strategy_name in role_defaults:
            if worker_config.worker_cls is None:
                worker_config.worker_cls = worker_cls
            if worker_config.name is None:
                worker_config.name = name
            if worker_config.model_args.model_name_or_path is None:
                worker_config.model_args.model_name_or_path = self.pretrain
            worker_config.training_args.output_dir = self.output_dir
            worker_config.system_envs.update(
                {key: value for key, value in self.system_envs.items() if key not in worker_config.system_envs}
            )
            if worker_config.strategy_args is None:
                worker_config.strategy_args = StrategyArguments(strategy_name=default_strategy_name)
            provider_owns_prompt_encoder = (
                name == "generator"
                and self.self_forcing.enabled
                and worker_config.strategy_args.strategy_name == FSDP2_TRAIN_STRATEGY
            )
            init_tokenizer_processor = name == "generator" or not self.share_prompt_embeddings
            worker_config.strategy_args.strategy_config["init_tokenizer_processor"] = (
                init_tokenizer_processor and not provider_owns_prompt_encoder
            )

        for worker_config in (self.generator, self.fake_score):
            if worker_config.training_args.gradient_accumulation_steps != 1:
                raise ValueError(
                    f"DMD {worker_config.name}.training_args.gradient_accumulation_steps must be 1 because "
                    "DMD workers perform one optimizer step per local algorithm update."
                )
