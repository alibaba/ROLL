import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Union

from roll.configs.base_config import PPOConfig, RolloutMockConfig, RouterArguments
from roll.configs.worker_config import WorkerConfig, is_actor_infer_overlapping_with_any_cluster
from roll.pipeline.rlvr.rlvr_config import RewardConfig
from roll.utils.logging import get_logger


logger = get_logger()


@dataclass
class DiffusionConfig(PPOConfig):
    transformer_ckpt: Optional[str] = field(
        default=None,
        metadata={"help": "Optional standalone Diffusers transformer checkpoint."},
    )
    validation: WorkerConfig = field(default=None)
    val_before_train: bool = field(
        default=False,
        metadata={"help": "Run validation once before training starts."},
    )
    rewards: Optional[Dict[str, RewardConfig]] = field(
        default_factory=dict,
        metadata={"help": "Configuration for rollout-only reward workers."},
    )
    num_return_sequences_in_group: int = field(
        default=1,
        metadata={"help": "Number of images sampled per prompt during rollout."},
    )
    rollout_mock: Optional[RolloutMockConfig] = field(
        default=None,
        metadata={"help": "Optional rollout mock configuration for precision alignment testing."},
    )
    dump_step_output_idx: Any = field(
        default_factory=list,
        metadata={"help": "Step indices at which to dump output. Accepts a list or a string like 'list(range(0,100))'."},
    )
    dump_step_output_path: str = field(
        default="",
        metadata={"help": "Base path for dumping step output. Each run creates a subfolder named after the experiment."},
    )
    dump_step_output_type: List[str] = field(
        default_factory=list,
        metadata={"help": "Which phases to dump: 'train', 'val', or both. Empty list disables dump."},
    )
    diffusion_model_variant: str = field(
        default="qwen_image",
        metadata={"help": "Diffusion model variant for adapter selection (e.g. qwen_image, flux)."},
    )

    # ==================== DiffNFT hyperparameters (adv_estimator == "diffnft") ====================
    nft_beta: float = field(
        default=0.1,
        metadata={"help": "DiffNFT mixing coefficient between live and EMA adapters, in (0, 1]."},
    )
    adv_clip_max: float = field(
        default=5.0,
        metadata={"help": "DiffNFT loss scaling upper bound, > 0."},
    )
    ref_kl_coef: float = field(
        default=0.0,
        metadata={"help": "DiffNFT reference (base model) KL loss coefficient; 0 disables the ref forward."},
    )
    use_adaptive_weight: bool = field(
        default=True,
        metadata={"help": "DiffNFT adaptive per-sample loss weighting."},
    )
    training_timestep_fraction: Union[float, List[float]] = field(
        default=0.5,
        metadata={"help": "Fraction of the rollout timestep schedule used for training; "
                          "a float end value or a [start, end] range in [0, 1]."},
    )
    min_train_timestep: Optional[float] = field(
        default=None,
        metadata={"help": "Minimum training timestep in flow-matching units [0, 1]; None disables."},
    )
    max_train_timestep: Optional[float] = field(
        default=None,
        metadata={"help": "Maximum training timestep in flow-matching units [0, 1]; None disables."},
    )
    shuffle_train_timesteps: bool = field(
        default=True,
        metadata={"help": "Shuffle the resolved training timestep order each step."},
    )
    ema_range: List[int] = field(
        default_factory=lambda: [0, 0],
        metadata={"help": "Step boundaries for dynamic EMA decay schedule. "
                          "ema_range[i] <= step < ema_range[i+1] uses ema_decays[i]; "
                          "step >= ema_range[-1] uses ema_decays[-1]. "
                          "Example: [0, 75] means step 0-74 uses decays[0], step 75+ uses decays[1]."},
    )
    ema_decays: List[float] = field(
        default_factory=lambda: [0.99, 0.99],
        metadata={"help": "EMA decay values corresponding to ema_range boundaries. "
                          "shadow = decay * shadow + (1 - decay) * live. "
                          "Must have same length as ema_range."},
    )
    ema_interpolation: Literal["step", "linear"] = field(
        default="step",
        metadata={"help": "How to derive the decay between ema_range control points. "
                          "'step': piecewise-constant (jump at each boundary). "
                          "'linear': piecewise-linear ramp between adjacent control points, "
                          "held at ema_decays[-1] after the last one. "
                          "Example (delayed linear ramp, slope ~0.0075/step): "
                          "ema_range: [0, 75, 208], ema_decays: [0.0, 0.0, 0.999]."},
    )
    old_policy_update_interval: int = field(
        default=1,
        metadata={"help": "Update the EMA shadow adapter every N global steps. "
                          "Steps in between keep the shadow frozen, creating a "
                          "meaningful gap between live and old policy. "
                          "Set to 1 to update every step (original behavior)."},
    )

    def __post_init__(self):
        super().__post_init__()

        # Diffusion uses fixed sample and timestep counts per micro-batch, so locally normalized loss needs no scaling.
        self.actor_train.apply_loss_scale = False

        # Diffusion does not use reference model / KL loss
        if self.enable_reference or self.use_kl_loss or self.kl_loss_coef != 0.0:
            logger.warning(
                "Diffusion pipeline does not support reference model or KL loss; "
                "forcing enable_reference=False, use_kl_loss=False, kl_loss_coef=0.0"
            )
        self.enable_reference = False
        self.use_kl_loss = False
        self.kl_loss_coef = 0.0

        # Expand string syntax like "list(range(0,100))" into an actual list
        if isinstance(self.dump_step_output_idx, str):
            self.dump_step_output_idx = list(eval(self.dump_step_output_idx))

        self.algorithm = self.adv_estimator

        if self.algorithm == "diffnft":
            if not (0.0 < self.nft_beta <= 1.0):
                raise ValueError(f"nft_beta must be in (0, 1], got {self.nft_beta}")
            if self.adv_clip_max <= 0.0:
                raise ValueError(f"adv_clip_max must be > 0, got {self.adv_clip_max}")
            if len(self.ema_range) != len(self.ema_decays):
                raise ValueError(
                    f"ema_range and ema_decays must have the same length, "
                    f"got {len(self.ema_range)} vs {len(self.ema_decays)}"
                )
            if any(d < 0.0 or d >= 1.0 for d in self.ema_decays):
                raise ValueError(f"all ema_decays must be in [0, 1), got {self.ema_decays}")
            if list(self.ema_range) != sorted(self.ema_range):
                raise ValueError(f"ema_range must be sorted in ascending order, got {self.ema_range}")
            if self.ema_interpolation == "linear":
                if len(self.ema_range) < 2:
                    raise ValueError(
                        f"ema_interpolation='linear' requires at least 2 control points, got {len(self.ema_range)}"
                    )
                if len(set(self.ema_range)) != len(self.ema_range):
                    raise ValueError(
                        f"ema_interpolation='linear' requires strictly increasing ema_range, got {self.ema_range}"
                    )

            # DiffNFT needs a frozen EMA shadow adapter ("ema_lora") alongside the trainable
            # "default" adapter; declare it via model_args so model providers stay
            # algorithm-agnostic. Users may also set it explicitly in YAML.
            if self.actor_train.model_args.lora_target is None:
                raise ValueError("adv_estimator='diffnft' requires actor_train.model_args.lora_target")
            extra = self.actor_train.model_args.extra_frozen_lora_adapters or []
            if "ema_lora" not in extra:
                self.actor_train.model_args.extra_frozen_lora_adapters = extra + ["ema_lora"]

        self.actor_train.name = "actor_train"
        self.actor_infer.name = "actor_infer"
        self.reference.name = "reference"

        if self.user_defined_rollout_loop_cls == "roll.distributed.scheduler.user_defined_rollout_loop.UserDefinedRolloutLoop":
            self.user_defined_rollout_loop_cls = "roll.pipeline.diffusion.rollout_loop.DiffusionRolloutLoop"

        if self.actor_train.worker_cls is None:
            if self.algorithm == "flowgrpo":
                self.actor_train.worker_cls = "roll.pipeline.diffusion.actor_grpo_worker.ActorGRPOWorker"
            elif self.algorithm == "diffnft":
                self.actor_train.worker_cls = "roll.pipeline.diffusion.actor_nft_worker.ActorNFTWorker"
        if self.actor_infer.worker_cls is None:
            self.actor_infer.worker_cls = "roll.pipeline.base_worker.InferWorker"
        if self.reference.worker_cls is None:
            if self.algorithm == "flowgrpo":
                self.reference.worker_cls = "roll.pipeline.diffusion.actor_grpo_worker.ActorGRPOWorker"
            elif self.algorithm == "diffnft":
                self.reference.worker_cls = "roll.pipeline.diffusion.actor_nft_worker.ActorNFTWorker"
        if self.router_args is None:
            self.router_args = RouterArguments(router_name="PromptAffinityRouter", router_config=dict())
            self.router_args.max_running_requests = self.max_running_requests
        for reward_name, reward_config in self.rewards.items():
            reward_config.name = reward_config.name or reward_name

        # Build tag_2_domain mapping from rewards config (same pattern as RLVR)
        self.tag_2_domain = {
            tag: reward_name
            for reward_name, reward_config in self.rewards.items()
            for tag in reward_config.tag_included
        }

        self.reference.training_args.output_dir = self.output_dir

        if self.num_return_sequences_in_group > 1 and not self.is_num_return_sequences_expand:
            self.is_num_return_sequences_expand = True
            logger.warning(
                "Diffusion must set is_num_return_sequences_expand=True when num_return_sequences_in_group > 1"
            )

        if self.actor_infer.generating_args is not None:
            self.actor_infer.generating_args.num_return_sequences = self.num_return_sequences_in_group
            self.actor_infer.generating_args.max_new_tokens = self.sequence_length - self.prompt_length

    def set_max_steps(self, max_steps: int):
        backward_batch_size = (
            self.actor_train.training_args.per_device_train_batch_size
            * self.actor_train.training_args.gradient_accumulation_steps
        )
        num_return_sequences = self.num_return_sequences_in_group
        if self.actor_infer.generating_args is not None:
            num_return_sequences = self.actor_infer.generating_args.num_return_sequences
        self.actor_train.training_args.max_steps = max(
            1,
            max_steps
            * self.rollout_batch_size
            * num_return_sequences
            * self.ppo_epochs
            // max(1, backward_batch_size),
        )
        self.reference.training_args.max_steps = self.actor_train.training_args.max_steps
        self.max_steps = max_steps

    @property
    def is_actor_infer_colocated(self) -> bool:
        """Whether actor_infer overlaps with training/reference workers."""
        return is_actor_infer_overlapping_with_any_cluster(
            actor_infer=self.actor_infer,
            actor_train=self.actor_train,
            reference=self.reference,
        )

    def get_ema_decay(self, global_step: int) -> float:
        """Return the EMA decay for the given global step based on the dynamic schedule.

        'step' mode: piecewise-constant, ema_range[i] <= step < ema_range[i+1] uses ema_decays[i].
        'linear' mode: piecewise-linear interpolation between adjacent control points
        (ema_range[i], ema_decays[i]) -> (ema_range[i+1], ema_decays[i+1]); clamped to
        ema_decays[0] before the first point and ema_decays[-1] after the last.
        """
        if self.ema_interpolation == "linear":
            if global_step <= self.ema_range[0]:
                return float(self.ema_decays[0])
            if global_step >= self.ema_range[-1]:
                return float(self.ema_decays[-1])
            for i in range(len(self.ema_range) - 1):
                left, right = self.ema_range[i], self.ema_range[i + 1]
                if left <= global_step < right:
                    progress = (global_step - left) / (right - left)
                    return float(self.ema_decays[i] + (self.ema_decays[i + 1] - self.ema_decays[i]) * progress)
            return float(self.ema_decays[-1])
        for i in range(len(self.ema_range) - 1, -1, -1):
            if global_step >= self.ema_range[i]:
                return float(self.ema_decays[i])
        return float(self.ema_decays[0])

    def to_dict(self):
        return dataclasses.asdict(self)
