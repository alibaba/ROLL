import dataclasses
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Literal, Optional, Union, List

from roll.configs.worker_config import WorkerConfig, is_actor_infer_overlapping_with_any_cluster
from roll.platforms import current_platform
from roll.utils.config_utils import (calculate_megatron_dp_size,
                                     validate_megatron_batch_size)
from roll.utils.logging import get_logger

logger = get_logger()

@dataclass
class RolloutMockConfig:
    """Configuration for rollout dump/mock mechanism for precision alignment testing."""
    enable: bool = field(
        default=False,
        metadata={"help": "Enable rollout dump/mock mechanism for precision alignment testing"}
    )
    mode: Literal["dump", "mock"] = field(
        default="dump",
        metadata={"help": "dump: save rollout data, mock: load pre-recorded data"}
    )
    dump_dir: str = field(
        default="./rollout_mock_dumps",
        metadata={"help": "Storage directory for rollout dump/mock data"}
    )

@dataclass
class RouterArguments:
    router_name: Literal[
        "PromptAffinityRouter",
        "EnvAffinityRouter",
        "SglangRouter",
    ] = field(
        default=None,
        metadata={
            "help": "The name of the router."
        },
    )
    router_config: Dict = field(
        default_factory=dict,
        metadata={"help": "Configuration dictionary for the router."},
    )
    max_running_requests: int = field(
        default=128,
        metadata={"help": "The maximum number of running requests."}
    )
    enable_sample_level_affinity: bool = field(
        default=False,
        metadata={
            "help": (
                "Route separately submitted samples in a prompt group independently while keeping partial or "
                "multi-turn requests for the same sample on the same inference worker. Only affects "
                "affinity-aware routers."
            )
        },
    )

@dataclass
class ScheduleConfig:
    generate_opt_level: int = field(
        default=1,
        metadata={
            "help": "generate optimizing level: 0 use base batch generate interface, 1 use scheduler process requests"
        },
    )
    is_num_return_sequences_expand: bool = field(
        default=False,
        metadata={"help": "whether replicate `num_return_sequences` times in prompts or not."}
    )
    max_running_requests: int = field(
        default=128,
        metadata={"help": "The maximum number of running requests."}
    )
    is_use_additional_prompts: bool = field(
        default=False,
        metadata={"help": "Whether to use additional prompts or not."}
    )
    max_additional_running_prompts: int = field(
        default=16, metadata={"help": "The additional number of running prompts, beyond batch_size."}
    )
    user_defined_rollout_loop_cls: str = field(
        default="roll.distributed.scheduler.user_defined_rollout_loop.UserDefinedRolloutLoop",
        metadata={"help": "Path to class UserDefinedRolloutLoop."}
    )
    router_args: RouterArguments = field(
        default=None,
        metadata={"help": "The router configuration, encapsulated in a RouterArguments object."},
    )

@dataclass
class TransferBackendArguments:
    backend_name: Optional[str] = field(
        default="TransferQueue",
        metadata={"help": "The registered backend for transfer. Set to null to disable remote transfer."}
    )
    backend_config: Dict = field(
        default_factory=lambda: {"backend": {"SimpleStorage": {"num_data_storage_units": 16}}},
        metadata={"help": "Configuration dictionary for the backend."}
    )

@dataclass
class BaseConfig(ScheduleConfig):

    exp_name: str = field(
        default=os.path.basename(sys.argv[0])[: -len(".py")],
        metadata={"help": "The name of this experiment (defaults to the file name without the .py extension)."},
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed for initializations."}
    )
    rpc_timeout: int = field(
        default=3600,
        metadata={"help": "Timeout duration for RPC calls in seconds."}
    )
    checkpoint_rpc_timeout: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Timeout for collecting checkpoint worker results in seconds. "
                "None waits for large distributed saves to finish."
            )
        },
    )
    output_dir: str = field(
        default="./output",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    base_dir: str = field(
        default="./output",
        metadata={"help": "The base directory where the model predictions and checkpoints will be written."},
    )
    logging_dir: str = field(
        default="./output/logs",
        metadata={"help": "Directory to store logs."})
    rollout_dump_dir: str = field(
        default=None, metadata={"help": "saving actor_infer rollout to this dir"}
    )
    track_with: str = field(
        default="tensorboard",
        metadata={"help": "The type of tracker to be used for tracking, one of ['wandb', 'tensorboard', 'stdout', 'swanlab']."}
    )
    tracker_kwargs: dict = field(
        default_factory=dict,
        metadata={"help": "Additional keyword arguments to pass to the Tracker class."}
    )
    dump_run_name: str = field(
        default="",
        metadata={"help": "Global run identifier for dump directories. Auto-generated once at init if empty."},
    )
    max_steps: int = field(
        default=500,
        metadata={"help": "If > 0: set total number of pipeline steps"},
    )
    save_steps: int = field(
        default=50,
        metadata={"help": "Save checkpoint every X update steps. Set to 0 to disable checkpointing."}
    )
    max_ckpt_to_keep: int = field(
        default=0,
        metadata={"help": "Maximum number of checkpoints to keep. 0 means keep all checkpoints."}
    )
    logging_steps: int = field(
        default=1,
        metadata={"help": "Number of steps between logging information."}
    )
    eval_steps: int = field(
        default=10,
        metadata={"help": "Run an evaluation every X steps."},
    )
    rollout_batch_size: int = field(
        default=128, metadata={"help": "The number of samples to rollout in each inference batch."}
    )
    max_running_requests: int = field(
        default=128,
        metadata={
            "help": "The maximum number of running requests. Aligned to max(128, max_num_seqs for vllm "
            "or max_running_requests for sglang) when configured in actor_infer strategy_config."
        }
    )
    val_batch_size: int = field(
        default=128,
        metadata={"help": "The number of samples to rollout in each val batch."})
    local_rank: int = field(
        default=-1,
        metadata={"help": "Local rank for distributed training; set to -1 if not applicable."}
    )
    resume_from_checkpoint: Union[bool, str] = field(
        default=False,
        metadata={"help": "load the last checkpoint in *output_dir* as saved by a previous instance or MOS URI."},
    )
    auto_resume: bool = field(
        default=False,
        metadata={"help": "If True, auto-find and resume from the latest checkpoint via get_latest_ckpt()."},
    )
    checkpoint_config: Optional[Dict] = field(
        default_factory=dict,
        metadata={"help": "Configuration checkpoint, this field will be written to worker_config."},
    )
    reward_system_config: Optional[Dict] = field(
        default_factory=dict,
        metadata={"help": "Configuration reward system, this field will be written to worker_config."},
    )
    prompt_length: Optional[int] = field(
        default=1024,
        metadata={"help": "The maximum length of a prompt to be padded."},
    )
    response_length: Optional[int] = field(
        default=None,
        metadata={"help": "The maximum length of the generated tokens to be padded."},
    )
    sequence_length: Optional[int] = field(
        default=None,
        metadata={"help": "The maximum length of the sequence to be padded."},
    )
    val_prompt_length: Optional[int] = field(
        default=None,
        metadata={"help": "The maximum length of a prompt to be padded."},
    )
    val_sequence_length: Optional[int] = field(
        default=None,
        metadata={"help": "The maximum length of the sequence to be padded."},
    )
    profiler_timeline: bool = field(default=False, metadata={"help": "Whether to use profiler mode or not."})
    profiler_memory: bool = field(default=False, metadata={"help": "Whether to use profiler memory or not."})
    report_length_and_rewards: bool = field(default=False, metadata={"help": "Whether to report lengths and rewards of prompts in each epoch."})

    is_offload_states: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to offload model states to CPU to save GPU memory. "
                "Models will be offloaded after each operation and reloaded before the next one. "
                "Reduces GPU memory usage at the cost of CPU-GPU transfer overhead."
            )
        }
    )
    is_offload_optimizer_states_in_train_step: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to offload optimizer states to CPU during training to save GPU memory. "
                "Optimizer states will be offloaded during forward/backward and reloaded for optimizer step. "
                "Reduces GPU memory usage at the cost of CPU-GPU transfer overhead."
            )
        }
    )
    offload_backend: str = field(
        default="local",
        metadata={"help": "Offload store: 'local' (per-rank pinned CPU memory), "
                          "'local_dedup' (1/world pinned chunk per rank + NCCL all-gather, DP-deduplicated)."}
    )

    length_profiler_dir: str = field(
        default='./output/profiler',
        metadata={"help": "directory to write length and rewards metric of prompts"}
    )

    profiler_output_dir: str = field(
        default="./output/profiler", metadata={"help": "Directory to write profiler logs."}
    )
    otlp_output_dir: str = field(
        default="./output/otlp",
        metadata={"help": "The output directory where the OTLP collector saved to."},
    )
    system_envs: dict = field(
        default_factory=dict,
        metadata={"help": "system environment variables."}
    )
    model_update_buffer_size_mb: int = field(
        default=1024,
        metadata={"help": "Buffer size in MB for model update operations (e.g., 1024 = 1GB)."}
    )
    num_nodes: int = field(
        default=1,
        metadata={"help": "Number of nodes available for distributed training."}
    )
    num_gpus_per_node: int = field(
        default=8,
        metadata={
            "help": "Specifies the number of GPUs available per node. When the number of nodes is greater than 1, "
                    "num_gpus_per_node should request the total number of GPUs in the entire node."
                    "Ensure that GPU resource allocation aligns with the request in a multi-node setup."
        }
    )
    model_download_type: Optional[str] = field(
        default=None,
        metadata={"help": "snapshot_download func source type, such as MODELSCOPE, HUGGINGFACE_HUB."},
    )
    rollout_mock: Optional[RolloutMockConfig] = field(
        default=None,
        metadata={"help": "Rollout mock configuration for precision alignment testing."}
    )

    transfer_backend: Optional[TransferBackendArguments] = field(
        default_factory=TransferBackendArguments,
        metadata={"help": "Transfer backend configuration. Defaults to TransferQueue; set backend_name to null to disable."}
    )


    def to_dict(self):
        return dataclasses.asdict(self)

    def __post_init__(self):

        assert self.response_length or self.sequence_length, "response_length or sequence_length must be set"

        if self.sequence_length is None:
            self.sequence_length = self.response_length + self.prompt_length

        for f in dataclasses.fields(self):
            value = getattr(self, f.name, None)
            if isinstance(value, WorkerConfig):
                value._auto_fill_packing_lengths(self.sequence_length)

        if self.response_length is not None:
            self.response_length = None

        if self.val_prompt_length is None:
            assert self.val_sequence_length is None, "val_prompt_length and val_sequence_length must be set simultaneously"
            self.val_prompt_length = self.prompt_length
            self.val_sequence_length = self.sequence_length

        if self.val_prompt_length is not None:
            assert self.val_sequence_length, "val_prompt_length and val_sequence_length must be set simultaneously"


        # Generate a single run timestamp for this pipeline run, shared across all workers.
        # This ensures dump directories are consistent regardless of track_with type.
        run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        if not self.dump_run_name:
            self.dump_run_name = f"{self.exp_name}_{run_timestamp}"

        if self.track_with == "tensorboard":
            self.tracker_kwargs["log_dir"] = os.path.join(
                self.tracker_kwargs.get("log_dir", self.output_dir), self.exp_name, run_timestamp
            )
            logger.info(f"add timestamp to tensorboard log_dir {self.tracker_kwargs['log_dir']}")

        self.logging_dir = os.path.join(self.logging_dir, self.exp_name)
        logger.info(f"add exp_name to logging_dir {self.logging_dir}")
        os.environ["ROLL_LOG_DIR"] = self.logging_dir
        get_logger()

        if self.model_download_type is not None:
            os.environ["MODEL_DOWNLOAD_TYPE"] = self.model_download_type

        upload_type = self.checkpoint_config.get("type", None)
        if upload_type == "file_system":
            output_dir = self.checkpoint_config.get("output_dir")
            self.checkpoint_config["output_dir"] = os.path.join(output_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
            logger.info(f"add timestamp to output_dir {self.checkpoint_config['output_dir']}")
        if self.resume_from_checkpoint is True or self.auto_resume:
            logger.info("ensure async_upload=False when auto_resume=True and auto resume from latest ckpt")
            self.checkpoint_config["async_upload"] = False

        for attribute_name in dir(self):
            attribute = getattr(self, attribute_name)
            if isinstance(attribute, WorkerConfig):
                if hasattr(attribute, "checkpoint_config"):
                    setattr(attribute, "checkpoint_config", self.checkpoint_config)

            if isinstance(attribute, WorkerConfig):
                if hasattr(attribute, "training_args"):
                    setattr(attribute.training_args, "seed", self.seed)

        assert not (
            self.profiler_timeline and self.profiler_memory
        ), f"ensure that only one profiling mode is enabled at a time"

        self.profiler_output_dir = os.path.join(
            self.profiler_output_dir, self.exp_name, datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        self.length_profiler_dir = os.path.join(
            self.length_profiler_dir, self.exp_name, datetime.now().strftime("%Y%m%d-%H%M%S")
        )

        os.environ["PROFILER_OUTPUT_DIR"] = self.profiler_output_dir
        if self.profiler_timeline:
            os.environ["PROFILER_TIMELINE"] = "1"
        if self.profiler_memory:
            os.environ["PROFILER_MEMORY"] = "1"
        if self.rpc_timeout is not None:
            os.environ["roll_RPC_TIMEOUT"] = str(self.rpc_timeout)
        if self.report_length_and_rewards:
            os.environ["REPORT_LENGTH_AND_REWARDS"] = "1"
        os.environ.update(self.system_envs)

        # Auto-generate OTLP endpoint when OTel tracing is enabled
        if self.system_envs.get("ROLL_OTEL_ENABLED") == "1":
            from roll.utils.telemetry import resolve_otel_endpoint
            endpoint = resolve_otel_endpoint()
            self.system_envs["OTEL_EXPORTER_OTLP_ENDPOINT"] = endpoint
            os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = endpoint

        from ..platforms import current_platform
        self.num_gpus_per_node = current_platform.device_count()

        if hasattr(self, 'actor_train') and isinstance(self.actor_train, WorkerConfig):
            self.actor_train.system_envs.update({k: v for k, v in self.system_envs.items() if k not in self.actor_train.system_envs})
        if hasattr(self, 'actor_infer') and isinstance(self.actor_infer, WorkerConfig):
            self.actor_infer.system_envs.update({k: v for k, v in self.system_envs.items() if k not in self.actor_infer.system_envs})
        if hasattr(self, 'reference') and isinstance(self.reference, WorkerConfig):
            self.reference.system_envs.update({k: v for k, v in self.system_envs.items() if k not in self.reference.system_envs})
        if hasattr(self, 'critic') and isinstance(self.critic, WorkerConfig):
            self.critic.system_envs.update({k: v for k, v in self.system_envs.items() if k not in self.critic.system_envs})
        # Also propagate system_envs to _reference_configs for multi-teacher
        if hasattr(self, '_reference_configs'):
            for ref_cfg in self._reference_configs.values():
                if isinstance(ref_cfg, WorkerConfig):
                    ref_cfg.system_envs.update({k: v for k, v in self.system_envs.items() if k not in ref_cfg.system_envs})

        # Validate rollout_batch_size divisibility for Megatron data parallelism
        if hasattr(self, 'actor_train') and isinstance(self.actor_train, WorkerConfig) and self.actor_train.strategy_args is not None:
            strategy_name = self.actor_train.strategy_args.strategy_name

            # Only validate for Megatron strategies
            if 'megatron' in strategy_name.lower():
                try:
                    validate_megatron_batch_size(
                        batch_size=self.rollout_batch_size,
                        num_gpus=len(self.actor_train.device_mapping),
                        strategy_config=self.actor_train.strategy_args.strategy_config,
                    )
                except ValueError as e:
                    logger.error(f"Megatron DP validation failed: {e}")
                    raise
            else:
                logger.debug(
                    f"Skipping DP validation for non-Megatron actor_train strategy: {strategy_name}"
                )

        if hasattr(self, 'actor_infer') and isinstance(self.actor_infer, WorkerConfig) and self.actor_infer.strategy_args is not None:
            strategy_name = self.actor_infer.strategy_args.strategy_name
            assert strategy_name in ["vllm", "vllm_omni", "sglang"]

            engine_max_running_requests = self.actor_infer.strategy_args.strategy_config.get(
                "max_running_requests" if strategy_name == "sglang" else "max_num_seqs"
            )
            if engine_max_running_requests is not None:
                self.max_running_requests = max(max(128, engine_max_running_requests), self.max_running_requests)
                if self.router_args is not None:
                    self.router_args.max_running_requests = self.max_running_requests
                logger.info(f"Set router max_running_requests to {self.max_running_requests} ")
            # Use max_running_requests+1 to reserve extra one for abort_requests.
            # 1000 is ray_constants.DEFAULT_MAX_CONCURRENCY_ASYNC.
            max_concurrency = max(self.max_running_requests + 1, 1000)
            self.actor_infer.max_concurrency = max(self.actor_infer.max_concurrency, max_concurrency)
            logger.info(f"Set max_concurrency of actor_infer to {self.actor_infer.max_concurrency}")

        # the required num nodes
        total_devices = []
        for attribute_name in dir(self):
            attribute = getattr(self, attribute_name)
            if isinstance(attribute, WorkerConfig):
                if attribute.device_mapping is not None:
                    total_devices.extend(attribute.device_mapping)
        # Also include _reference_configs device_mappings for multi-teacher
        if hasattr(self, '_reference_configs'):
            for ref_cfg in self._reference_configs.values():
                if isinstance(ref_cfg, WorkerConfig) and ref_cfg.device_mapping is not None:
                    total_devices.extend(ref_cfg.device_mapping)
        if len(total_devices) > 0:
            max_gpu_num = max(total_devices) + 1
            if max_gpu_num <= self.num_gpus_per_node:
                self.num_nodes = 1
            else:
                self.num_nodes = (max_gpu_num + self.num_gpus_per_node - 1) // self.num_gpus_per_node


    def set_max_steps(self, max_steps: int):
        for attribute_name in dir(self):
            attribute = getattr(self, attribute_name)
            if isinstance(attribute, WorkerConfig):
                if hasattr(attribute, "training_args"):
                    setattr(attribute.training_args, "max_steps", max_steps)

@dataclass
class TrainInferISWeightConfig:
    enabled: bool = field(
        default=False,
        metadata={"help": "Whether to generate train-infer IS weight and store it into batch (train_infer_is_weight)."},
    )
    weight_type: Literal["token", "segment", "geometric", "sequence"] = field(
        default="token",
        metadata={"help": "Granularity for IS weight: token / segment / geometric / sequence."},
    )
    upper_bound: Optional[float] = field(
        default=1.2,
        metadata={"help": "Upper bound (clamp) for IS weight. Set to None to disable clamping."},
    )
    detach: bool = field(
        default=True,
        metadata={"help": "Detach IS weight tensor to prevent gradient flow (recommended)."},
    )


@dataclass
class TrainInferFilterConfig:
    enabled: bool = field(
        default=False,
        metadata={"help": "Whether to enable this filter rule (applied to response_mask)."},
    )
    agg_type: Literal["token", "segment", "geometric", "sequence"] = field(
        default="token",
        metadata={"help": "Aggregation level used for filtering: token / segment / geometric / sequence."},
    )

    ratio_enabled: bool = field(
        default=True,
        metadata={"help": "Whether to apply ratio-based filtering (exp(old_logp - infer_logp))."},
    )
    ratio_low: float = field(default=0.8, metadata={"help": "Lower threshold for ratio filtering."})
    ratio_high: float = field(default=1.2, metadata={"help": "Upper threshold for ratio filtering."})

    diff_enabled: bool = field(
        default=False,
        metadata={"help": "Whether to apply diff-based filtering (exp(old) - exp(infer))."},
    )
    diff_low: float = field(default=-0.2, metadata={"help": "Lower threshold for diff filtering."})
    diff_high: float = field(default=0.2, metadata={"help": "Upper threshold for diff filtering."})


@dataclass
class TrainInferCorrectionConfig:
    is_weight: TrainInferISWeightConfig = field(
        default_factory=TrainInferISWeightConfig,
        metadata={"help": "Config for generating train-infer IS weight (stored in batch)."},
    )
    filters: List[TrainInferFilterConfig] = field(
        default_factory=list,
        metadata={"help": "A list of filter rules applied sequentially to response_mask."},
    )

@dataclass
class PPOConfig(BaseConfig):
    # role related
    pretrain: str = field(default=None, metadata={"help": "Path to pretrain model directory, if available."})
    reward_pretrain: str = field(
        default=None, metadata={"help": "Path to pretrain model directory for the reward model, if available."}
    )
    actor_train: WorkerConfig = field(
        default_factory=WorkerConfig, metadata={"help": "Configuration for the actor's training role."}
    )
    actor_infer: WorkerConfig = field(
        default_factory=WorkerConfig, metadata={"help": "Configuration for the actor's inference role."}
    )
    critic: WorkerConfig = field(
        default_factory=WorkerConfig, metadata={"help": "Configuration for the critic's training role."}
    )
    reference: WorkerConfig = field(
        default_factory=WorkerConfig, metadata={"help": "Configuration for the reference role."}
    )
    reward: WorkerConfig = field(default_factory=WorkerConfig, metadata={"help": "Configuration for reward inference."})

    async_generation_ratio: float = field(
        default=0,
        metadata={
            "help": "The ratio of ahead generation requests in pipeline, 0 means synchronous pipeline."
        },
    )

    # PPO related
    ppo_epochs: int = field(default=1, metadata={"help": "Number of optimisation epochs per batch of samples"})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Maximum norm"})
    l2: float = field(default=0.0, metadata={"help": "L2 regularization"})
    lambd: float = field(default=0.95, metadata={"help": "Lambda parameter for advantage calculation"})
    gamma: float = field(default=1, metadata={"help": "Gamma parameter for advantage calculation"})
    pg_clip: Optional[float] = field(default=0.2, metadata={"help": "Range for clipping in PPO policy gradient loss"})
    use_pg_clip_range: bool = field(default=False, metadata={"help": "Use to change the clipping range of pg_clip"})
    pg_clip_low: Optional[float] = field(
        default=0.2, metadata={"help": "Range for clipping lower in PPO policy gradient loss"}
    )
    pg_clip_high: Optional[float] = field(
        default=0.2, metadata={"help": "Range for clipping higher in PPO policy gradient loss"}
    )

    value_clip: Optional[float] = field(
        default=None, metadata={"help": "Range for clipping values in loss calculation"}
    )
    kl_penalty: Literal["kl", "abs", "mse", "full"] = field(
        default="kl",
        metadata={
            "help": "kl penalty options: 'kl': model_logp - ref_logp, 'abs': abs(kl), 'mse': "
            "mean squared error mse(kl) and 'full': the actual kl for all tokens in the distribution"
        },
    )
    target_kl: Optional[float] = field(default=None, metadata={"help": "Target KL value for adaptive KL control"})
    init_kl_coef: float = field(
        default=0.2, metadata={"help": "Initial KL penalty coefficient (used for adaptive and linear control)"}
    )
    kl_horizon: int = field(default=10000, metadata={"help": "Horizon for adaptive KL control"})
    use_reward_scaling: bool = field(default=False, metadata={"help": "Use reward scaling"})
    add_len_reward: bool = field(default=False)
    reward_clip: float = field(default=None, metadata={"help": "reward clip value."})
    use_reward_norm: bool = field(
        default=False, metadata={"help": "Use reward normalization. Only applicable if use_reward_scaling is True."}
    )
    whiten_rewards: bool = field(default=False, metadata={"help": "Whiten the rewards before compute advantages."})
    whiten_advantages: bool = field(default=False, metadata={"help": "Whiten the advantage."})
    advantage_clip: float = field(default=None, metadata={"help": "advantage_clip value"})
    adv_estimator: Literal["gae", "skip_obs_gae", "reinforce", "grpo", "gigpo", "step_reinforce", "agentic_reinforce", "flowgrpo", "diffnft"] = field(
        default="gae", metadata={"help": "advantage estimator: gae (GAE), skip_obs_gae (SAO-style skip-observation GAE)."}
    )
    norm_mean_type: Literal["batch", "group", "running", None] = field(
        default=None,
        metadata={
            "help": "Mean type for reward normalization: 'batch' (normalize across batch), 'group' (normalize within prompt groups), 'running' (use running statistics), None (without subtracting mean)"
        },
    )
    norm_std_type: Literal["batch", "group", "running", None] = field(
        default=None,
        metadata={
            "help": "Std type for reward normalization: 'batch' (normalize across batch), 'group' (normalize within prompt groups), 'running' (use running statistics), None (without dividing by std)"
        },
    )
    add_token_level_kl: bool = field(default=False, metadata={"help": "Add token level kl penalty"})
    critic_warmup: int = field(
        default=0,
        metadata={"help": "Pre-training step for critic model"},
    )
    critic_epochs: int = field(default=1, metadata={"help": "Number of critic update epochs per policy step."})
    use_kl_loss: bool = field(default=False, metadata={"help": "Use kl loss"})
    kl_loss_coef: float = field(default=0, metadata={"help": "Loss coefficient for kl loss"})
    entropy_loss_coef: float = field(default=0, metadata={"help": "Loss coefficient for entropy loss"})
    loss_agg_mode: Literal["token-mean", "seq-mean-token-sum", "seq-mean-token-mean", "seq-mean-token-sum-norm"] = (
        field(default="seq-mean-token-mean", metadata={"help": "Loss aggregation mode"})
    )
    dual_clip_loss: bool = field(default=False, metadata={"help": "Use dual clip loss"})
    enable_reference: bool = field(
        default=False, metadata={"help": "Whether to enable reference cluster for computing ref_log_probs."}
    )
    enable_old_logprobs_recompute: bool = field(default=False, metadata={"help": "Enable old_logprobs computation optimization for disable caching"})
    force_disable_old_logprobs_recompute: bool = field(default=False, metadata={"help": "Force disable old_logprobs computation optimization for disable caching, priority is higher than enable_old_logprobs_recompute"})

    train_infer_correction: TrainInferCorrectionConfig = field(
        default_factory=TrainInferCorrectionConfig,
        metadata={
            "help": (
                "Train-infer correction config for off-policy/mismatch handling. "
                "Pipeline will compute train_infer_is_weight from old_log_probs vs infer_logprobs "
                "and optionally apply filters to response_mask."
            )
        },
    )

    # OPD (On-Policy Distillation) Configuration
    pure_opd_pipeline_type: Optional[Literal["rlvr", "rlvr_vlm", "agentic"]] = field(
        default=None,
        metadata={"help": "Pipeline type for pure On-Policy Distillation. Used by start_onpolicy_distill_pipeline.py "
                 "to determine which config class and pipeline to use. Only configurable in pure OPD mode; "
                 "defaults to 'rlvr' when unset. "
                 "'rlvr': RLVRConfig + RLVRPipeline, 'rlvr_vlm': RLVRConfig + RLVRVLMPipeline, "
                 "'agentic': AgenticConfig + AgenticPipeline"}
    )
    teacher: Union[Dict[str, WorkerConfig], WorkerConfig] = field(
        default_factory=WorkerConfig,
        metadata={"help": "Teacher model config (OPD mode). Single: WorkerConfig; "
                  "Multi-teacher: Dict[str, WorkerConfig]. "
                  "Dict[str, WorkerConfig] must be placed first in Union for dacite."}
    )
    student_train: WorkerConfig = field(
        default_factory=WorkerConfig,
        metadata={"help": "Configuration for the student training role (used in OPD mode). "
                 "When configured, student_train is mapped to actor_train."}
    )
    student_infer: WorkerConfig = field(
        default_factory=WorkerConfig,
        metadata={"help": "Configuration for the student inference role (used in OPD mode). "
                 "When configured, student_infer is mapped to actor_infer."}
    )
    is_pure_opd: bool = field(
        default=False,
        metadata={"help": "Enable pure On-Policy Distillation mode. "
                 "In this mode, rewards come entirely from Teacher KL divergence. "
                 "Automatically sets: gamma=0, adv_estimator='reinforce', critic_warmup=0. "
                 "This is set by start_onpolicy_distill_pipeline.py automatically."}
    )
    use_opd: bool = field(
        default=False,
        metadata={"help": "Enable mixed OPD mode: add OPD KL penalty to token_level_reward. "
                 "This allows combining RL reward with distillation signal. "
                 "The OPD KL is computed as: reverse_kl = student_logp - teacher_logp, "
                 "and added to token_level_rewards as: reward - opd_kl_coef * reverse_kl"}
    )
    opsd_mode: bool = field(
        default=False,
        metadata={"help": "Enable OPSD (On-Policy Self-Distillation): teacher prompt includes "
                 "reference solution y* as privileged information before evaluating student response. "
                 "Auto-enables is_pure_opd=True (pure self-distillation); set use_opd=True "
                 "explicitly for mixed mode (external rewards + Teacher KL)."}
    )
    opsd_solution_key: str = field(
        default="reference_solution",
        metadata={"help": "non_tensor_batch key for the reference solution (y*, full CoT). "
                 "Must be present in the dataset JSONL."}
    )
    opsd_teacher_template: str = field(
        default=(
            "{problem}\n\n"
            "Here is a reference solution to this problem:\n"
            "=== Reference Solution Begin ===\n{solution}\n=== Reference Solution End ===\n"
            "\n\nAfter reading the reference solution above, make sure you truly understand "
            "the reasoning behind each step — do not copy or paraphrase it. Now, using your "
            "own words and independent reasoning, derive the same final answer to the problem above. "
            "Think step by step, explore different approaches, and don't be afraid to backtrack "
            "or reconsider if something doesn't work out:\n"
        ),
        metadata={"help": "Format string for OPSD teacher user message content. "
                 "Placeholders: {problem} (original problem text), {solution} (reference solution y*). "
                 "The result is wrapped via the chat template (global_template)."}
    )
    opsd_max_solution_length: Optional[int] = field(
        default=None,
        metadata={"help": "Max token length of the reference solution (y*) in OPSD teacher prompt. "
                 "If set, solutions exceeding this are tokenized, truncated, and decoded back to text "
                 "before building the teacher prompt — preserving the template structure. "
                 "If None, no truncation (rely on sequence_length buffer + tokenized fallback)."}
    )
    opd_token_kld_clip: Optional[float] = field(
        default=None,
        metadata={"help": "Per-token KL clip threshold for OPD/OPSD advantage. "
                 "Style tokens (e.g. 'think', 'wait') can have 6-15x higher per-token KL "
                 "and dominate the gradient. When set, total_weighted_kld is clamped to "
                 "[-opd_token_kld_clip, opd_token_kld_clip] before computing advantages. "
                 "Default None = no clipping."}
    )

    def __post_init__(self):
        super().__post_init__()
        assert self.async_generation_ratio == 0 or self.generate_opt_level == 1

        if (
            self.actor_train.model_args.model_name_or_path is None
            or self.actor_infer.model_args.model_name_or_path is None
            or self.reference.model_args.model_name_or_path is None
        ):
            self.actor_train.model_args.model_name_or_path = self.pretrain
            self.actor_infer.model_args.model_name_or_path = self.pretrain
            self.reference.model_args.model_name_or_path = self.pretrain

        if self.critic.model_args.model_name_or_path is None:
            self.critic.model_args.model_name_or_path = self.reward_pretrain

        self.actor_train.training_args.output_dir = self.output_dir
        self.actor_infer.training_args.output_dir = self.output_dir
        self.critic.training_args.output_dir = self.output_dir

        self.actor_infer.name = "actor_infer"
        self.actor_train.name = "actor_train"
        self.reference.name = "reference"
        self.critic.name = "critic"
        if self.use_kl_loss or self.init_kl_coef > 0:
            logger.warning(f"use_kl_loss or init_kl_coef > 0, enable_reference = True")
            self.enable_reference = True
        if self.force_disable_old_logprobs_recompute:
            self.enable_old_logprobs_recompute = False
        elif getattr(self.actor_train, "pg_variant", None) == "sao":
            # SAO uses rollout logprobs as ratio denominator, no need to recompute
            self.enable_old_logprobs_recompute = False
        elif self.adv_estimator in ['step_reinforce', "gigpo", "flowgrpo"]:
            self.enable_old_logprobs_recompute = True
        else:
            self.set_old_logprobs_status()

        if getattr(self.actor_train.router_replay, "mode", "disable") == "R2" and not self.enable_old_logprobs_recompute:
            # R2 records routing during the megatron compute_log_probs forward;
            # skipping recompute means no routing is ever recorded.
            logger.warning("router_replay mode is R2, force enable_old_logprobs_recompute = True")
            self.enable_old_logprobs_recompute = True

        self.use_critic: bool = self.adv_estimator in ("gae", "skip_obs_gae")

        logger.info(f"enable_old_logprobs_recompute: {self.enable_old_logprobs_recompute}\tenable_reference: {self.enable_reference}")

    def _handle_opd_mapping(self):
        """
        Handle OPD (On-Policy Distillation) mode configuration mapping.

        This method is called at the beginning of __post_init__ before normal PPO initialization.

        Teacher is normalized to _reference_configs: Dict[str, WorkerConfig] immediately,
        so all subsequent logic is unified regardless of single vs multi-teacher.
        """
        has_teacher = isinstance(self.teacher, WorkerConfig) and self.teacher.is_configured
        has_teachers = isinstance(self.teacher, dict) and bool(self.teacher)
        has_reference_configured = self.reference.is_configured

        # Mutual exclusion check
        if self.is_pure_opd and self.use_opd:
            raise ValueError(
                "is_pure_opd=True and use_opd=True are mutually exclusive. "
                "Use is_pure_opd=True for pure OPD mode (rewards from Teacher KL only), "
                "or use_opd=True for mixed mode (external rewards + Teacher KL)."
            )

        # OPSD is pure self-distillation by default: auto-enable is_pure_opd so opsd_mode
        # alone is sufficient. Set use_opd=True explicitly to keep the mixed (RL + OPD) mode.
        if self.opsd_mode and not (self.is_pure_opd or self.use_opd):
            logger.info("opsd_mode=True: auto-enabling is_pure_opd=True (pure self-distillation)")
            self.is_pure_opd = True

        # pure_opd_pipeline_type is only consumed by the pure OPD launcher
        # (start_onpolicy_distill_pipeline.py); reject it in non-pure-OPD configs.
        if self.is_pure_opd:
            if self.pure_opd_pipeline_type is None:
                self.pure_opd_pipeline_type = "rlvr"
        elif self.pure_opd_pipeline_type is not None:
            raise ValueError(
                "pure_opd_pipeline_type is only used in pure OPD mode "
                "(launch via examples/start_onpolicy_distill_pipeline.py). "
                "Remove it from this config."
            )

        # OPSD mode: only teacher receives the OPSD transform (y* in prompt).
        # reference would incorrectly receive y* too, so forbid coexistence.
        if self.opsd_mode and has_reference_configured and (has_teacher or has_teachers):
            raise ValueError(
                "OPSD mode (opsd_mode=True) does not support configuring both 'reference' and 'teacher'. "
                "In OPSD mode, only 'teacher' receives the reference solution y* in its prompt. "
                "If you need a reference model without OPSD, use pure OPD mode (opsd_mode=False)."
            )

        # OPSD assumes a single teacher: the OPSD transform (y* in prompt) and the
        # teacher thinking mode are applied once for the whole batch, so multiple
        # teachers with different settings cannot be expressed.
        if self.opsd_mode and has_teachers:
            raise ValueError(
                "opsd_mode=True currently supports a single teacher only: the OPSD transform "
                "(y* in prompt) and teacher enable_thinking are applied once for all references. "
                "Configure a single teacher, or drop opsd_mode for multi-teacher OPD."
            )

        # OPSD is only wired into the RLVR pipeline.
        if self.opsd_mode and self.is_pure_opd and self.pure_opd_pipeline_type != "rlvr":
            raise ValueError(
                "opsd_mode=True is currently only supported with pure_opd_pipeline_type='rlvr'. "
                "The rlvr_vlm and agentic pipelines do not apply the OPSD transform (y* in prompt)."
            )

        # Step 1: Merge reference + teacher into unified _reference_configs dict.
        # reference → "reference", single teacher → "default", dict teacher → by keys.
        self._reference_configs = {}
        if has_reference_configured:
            self._reference_configs["reference"] = self.reference
        if has_teachers:
            if has_reference_configured and "reference" in self.teacher:
                raise ValueError(
                    "Naming collision: 'reference' key exists in both the reference config "
                    "and the teacher dict. Please rename the teacher dict key."
                )
            self._reference_configs.update(self.teacher)
        elif has_teacher:
            self._reference_configs["default"] = self.teacher

        # Step 1a: LoRA auto-reference — no separate teacher/reference config needed,
        # teacher = actor_train with adapter disabled at runtime
        if not self._reference_configs and (self.is_pure_opd or self.use_opd):
            is_lora = (
                self.student_train.is_configured
                and self.student_train.model_args.lora_target is not None
            )
            if is_lora:
                logger.info("OPD + LoRA: no separate teacher, using student_train as reference")
                self._reference_configs = {"reference": self.student_train}

        # Step 1.5: Build tag → teacher_names routing map for multi-teacher OPD
        self._tag_to_teacher_names: Dict[str, List[str]] = {}
        self._teacher_names_for_all_tags: List[str] = []
        for name, ref_cfg in self._reference_configs.items():
            if not ref_cfg.tag_included:
                # Empty tag_included means this teacher handles all tags
                self._teacher_names_for_all_tags.append(name)
            else:
                for tag in ref_cfg.tag_included:
                    if tag not in self._tag_to_teacher_names:
                        self._tag_to_teacher_names[tag] = []
                    self._tag_to_teacher_names[tag].append(name)

        # Step 2: Pure OPD mode
        if self.is_pure_opd:
            if not self._reference_configs:
                raise ValueError("Pure OPD requires teacher or reference config.")
            if not (self.student_train.is_configured and self.student_infer.is_configured):
                raise ValueError(
                    "In pure OPD mode (is_pure_opd=True), 'student_train', 'student_infer' "
                    "and teacher/reference must be configured.\n"
                )
            logger.info(f"Pure OPD mode: mapping student_train to actor_train, "
                        f"student_infer to actor_infer, "
                        f"{len(self._reference_configs)} reference(s) to reference")
            self.actor_train = self.student_train
            self.actor_infer = self.student_infer
            self.reference = next(iter(self._reference_configs.values()))
            self.enable_reference = True
            # Propagate opd_kl_coef default (1.0) to each reference if not explicitly set
            for ref_cfg in self._reference_configs.values():
                if ref_cfg.opd_kl_coef is None:
                    ref_cfg.opd_kl_coef = 1.0

        # Step 3: Mixed OPD mode
        elif self.use_opd:
            if not self._reference_configs:
                raise ValueError("Mixed OPD requires teacher or reference config.")
            logger.info(f"Mixed OPD mode: mapping {len(self._reference_configs)} reference(s) to reference")
            self.reference = next(iter(self._reference_configs.values()))
            self.enable_reference = True
            # Propagate opd_kl_coef default (1.0) to each reference if not explicitly set
            for ref_cfg in self._reference_configs.values():
                if ref_cfg.opd_kl_coef is None:
                    ref_cfg.opd_kl_coef = 1.0

    def set_max_steps(self, max_steps: int):
        actor_backward_batch_size = (
            self.actor_train.training_args.per_device_train_batch_size
            * self.actor_train.training_args.gradient_accumulation_steps
        )
        critic_backward_batch_size = (
            self.critic.training_args.per_device_train_batch_size
            * self.critic.training_args.gradient_accumulation_steps
        )
        # 没有除dp_size，需要在分布式环境初始化后再除
        # 先计算总的训练步数，最后再除以 backward_batch_size
        self.actor_train.training_args.max_steps = max(1, (
            max_steps
            * self.rollout_batch_size
            * self.actor_infer.generating_args.num_return_sequences
            * self.ppo_epochs
            // actor_backward_batch_size
        ))
        self.critic.training_args.max_steps = max(1, (
            max_steps
            * self.rollout_batch_size
            * self.actor_infer.generating_args.num_return_sequences
            // critic_backward_batch_size
        ))

        logger.info(f"pipeline max_steps: {self.max_steps} to {max_steps}")
        logger.info(f"actor train max_steps without dp_size: {self.actor_train.training_args.max_steps}")
        logger.info(f"critic train max_steps without dp_size: {self.critic.training_args.max_steps}")
        self.max_steps = max_steps

    def _get_effective_cp_size_ulysses(self, configured_ulysses_size: Optional[int]) -> int:
        if not configured_ulysses_size or configured_ulysses_size <= 1:
            return 1
        if current_platform.apply_ulysses_patch() is not None:
            return configured_ulysses_size
        return 1

    def set_old_logprobs_status(self):
        batch_size = self.rollout_batch_size * self.actor_infer.generating_args.num_return_sequences
        actor_backward_batch_size = (
            self.actor_train.training_args.per_device_train_batch_size
            * self.actor_train.training_args.gradient_accumulation_steps
        )
        dp_size = 1
        if self.actor_train.strategy_args is not None:
            if self.actor_train.strategy_args.strategy_name in ("fsdp2_train", "fsdp2_infer"):
                configured_ulysses_size = getattr(self.actor_train.model_args, 'ulysses_size', None) or 1
                cp_size = self._get_effective_cp_size_ulysses(configured_ulysses_size)
                dp_size = len(self.actor_train.device_mapping) // cp_size
            elif self.actor_train.strategy_args.strategy_name == "megatron_train":
                strategy_config = self.actor_train.strategy_args.strategy_config
                tp = strategy_config.get('tensor_model_parallel_size', 1)
                pp = strategy_config.get('pipeline_model_parallel_size', 1)
                cp = strategy_config.get('context_parallel_size', 1)
                dp_size = calculate_megatron_dp_size(num_gpus=len(self.actor_train.device_mapping),
                                                     tensor_parallel_size=tp,
                                                     pipeline_parallel_size=pp,
                                                     context_parallel_size=cp)

        # Calculate backward steps per DP rank
        backward_steps_per_rank = (batch_size // dp_size) // actor_backward_batch_size

        # Disable optimization only when multiple backward steps in single training step
        # Multi-epoch training is actually a key scenario for optimization
        if backward_steps_per_rank > 1:
            # Multiple backward steps means model parameters change during training
            # Cannot reuse cached logprobs across backward passes
            self.enable_old_logprobs_recompute = True

        if self.init_kl_coef > 0:
            logger.warning(f"init_kl_coef > 0, enable_old_logprobs_recompute = True")
            self.enable_old_logprobs_recompute = True

    @property
    def async_pipeline(self) -> bool:
        return self.async_generation_ratio > 0

    @property
    def reference_configs(self) -> Dict[str, WorkerConfig]:
        """Always returns Dict[str, WorkerConfig] for unified pipeline usage.
        Single teacher is normalized to {"default": cfg}, multi-teacher to {name: cfg, ...}.
        Reference config is named "reference"."""
        if not hasattr(self, '_reference_configs') or not self._reference_configs:
            self._reference_configs = {"reference": self.reference}
        return self._reference_configs

    @property
    def is_multi_teacher(self) -> bool:
        return len(self.reference_configs) > 1

    @property
    def tag_to_teacher_names(self) -> Dict[str, List[str]]:
        """Map tag -> list of teacher names that should handle it."""
        if not hasattr(self, '_tag_to_teacher_names'):
            self._tag_to_teacher_names = {}
        return self._tag_to_teacher_names

    @property
    def teacher_names_for_all_tags(self) -> List[str]:
        """Teachers with empty tag_included (handle all tags)."""
        if not hasattr(self, '_teacher_names_for_all_tags'):
            self._teacher_names_for_all_tags = []
        return self._teacher_names_for_all_tags

    @property
    def needs_teacher_routing(self) -> bool:
        """Whether any teacher has non-empty tag_included (requires routing logic)."""
        if not hasattr(self, '_tag_to_teacher_names'):
            return False
        return bool(self._tag_to_teacher_names)

    @property
    def is_actor_infer_colocated(self) -> bool:
        """Whether actor_infer are colocated with any other clusters (exclude reward)."""
        return is_actor_infer_overlapping_with_any_cluster(
            actor_infer=self.actor_infer,
            actor_train=self.actor_train,
            reference=self.reference,
            critic=self.critic
        )

    def _apply_opd_config(self):
        """
        Apply OPD-specific parameter overrides.

        This method should be called at the end of __post_init__ in subclasses
        (RLVRConfig, AgenticConfig) to apply OPD-specific settings.

        Note: The mapping of student_*/teacher to actor_*/reference is already
        handled by _handle_opd_mapping(). This method only applies OPD-specific
        parameter overrides like gamma, adv_estimator, etc.
        """
        if not (self.is_pure_opd or self.use_opd):
            return

        # Set teacher worker names
        if len(self._reference_configs) > 1:
            for name, ref_cfg in self._reference_configs.items():
                ref_cfg.name = f"teacher-{name}"
        else:
            self.reference.name = "teacher"

        # Pure OPD mode specific settings
        if self.is_pure_opd:
            self.actor_train.name = "student_train"
            self.actor_infer.name = "student_infer"

            # gamma=0: OPD's token_level_rewards has KL penalty at every token
            # If gamma=1, compute_reinforce_return will accumulate KL values across entire sequence
            self.gamma = 0

            # Use reinforce as default advantage estimator (no GAE, no critic needed)
            logger.warning("Pure OPD mode: set adv_estimator as 'reinforce'")
            self.adv_estimator = "reinforce"

            # No critic warmup needed (reinforce doesn't use critic)
            self.critic_warmup = 0

            # Disable KL loss (OPD handles KL through token_level_rewards)
            self.use_kl_loss = False
            self.add_token_level_kl = False

            logger.info(f"Pure OPD mode configured: gamma={self.gamma}, adv_estimator={self.adv_estimator}")

        elif self.use_opd:
            logger.info(f"Mixed OPD mode configured")
