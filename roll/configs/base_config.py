import dataclasses
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Literal, Optional, Union

from roll.configs.worker_config import WorkerConfig, is_colocated
from roll.utils.config_utils import validate_megatron_batch_size
from roll.utils.logging import get_logger


logger = get_logger()

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


@dataclass
class BaseConfig:

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
    max_steps: int = field(
        default=500,
        metadata={"help": "If > 0: set total number of pipeline steps"},
    )
    save_steps: int = field(
        default=50,
        metadata={"help": "Save checkpoint every X update steps."}
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
        metadata={"help": "The maximum number of running requests."}
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
    checkpoint_config: Optional[Dict] = field(
        default_factory=dict,
        metadata={"help": "Configuration checkpoint, this field will be written to worker_config."},
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
    alive_check_interval: int = field(
        default=10,
        metadata={"help": "The interval of worker alive check."}
    )
    profiler_timeline: bool = field(default=False, metadata={"help": "Whether to use profiler mode or not."})
    profiler_memory: bool = field(default=False, metadata={"help": "Whether to use profiler memory or not."})
    report_length_and_rewards: bool = field(default=False, metadata={"help": "Whether to report lengths and rewards of prompts in each epoch."})

    length_profiler_dir: str = field(
        default='./output/profiler',
        metadata={"help": "directory to write length and rewards metric of prompts"}
    )

    profiler_output_dir: str = field(
        default="./output/profiler", metadata={"help": "Directory to write profiler logs."}
    )
    system_envs: dict = field(
        default_factory=dict,
        metadata={"help": "system environment variables."}
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


    def to_dict(self):
        return dataclasses.asdict(self)

    def __post_init__(self):

        assert self.response_length or self.sequence_length, "response_length or sequence_length must be set"

        if self.sequence_length is None:
            self.sequence_length = self.response_length + self.prompt_length

        if self.response_length is not None:
            self.response_length = None

        if self.val_prompt_length is None:
            assert self.val_sequence_length is None, "val_prompt_length and val_sequence_length must be set simultaneously"
            self.val_prompt_length = self.prompt_length
            self.val_sequence_length = self.sequence_length

        if self.val_prompt_length is not None:
            assert self.val_sequence_length, "val_prompt_length and val_sequence_length must be set simultaneously"


        if self.track_with == "tensorboard":
            self.tracker_kwargs["log_dir"] = os.path.join(
                self.tracker_kwargs.get("log_dir", self.output_dir), self.exp_name, datetime.now().strftime("%Y%m%d-%H%M%S")
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

        from ..platforms import current_platform
        self.num_gpus_per_node = current_platform.device_count()

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

        # the required num nodes
        total_devices = []
        for attribute_name in dir(self):
            attribute = getattr(self, attribute_name)
            if isinstance(attribute, WorkerConfig):
                if attribute.device_mapping is not None:
                    total_devices.extend(attribute.device_mapping)
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
    adv_estimator: Literal["gae", "reinforce", "grpo", "gigpo", "step_reinforce", "agentic_reinforce"] = field(
        default="gae", metadata={"help": "advantage estimator: gae (GAE)."}
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

    def __post_init__(self):
        super().__post_init__()

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
            self.enable_reference = True

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
        self.actor_train.training_args.max_steps = max_steps * (
            self.rollout_batch_size
            * self.actor_infer.generating_args.num_return_sequences
            * self.ppo_epochs
            // actor_backward_batch_size
        )
        self.critic.training_args.max_steps = max_steps * (
            self.rollout_batch_size
            * self.actor_infer.generating_args.num_return_sequences
            // critic_backward_batch_size
        )

        logger.info(f"pipeline max_steps: {self.max_steps} to {max_steps}")
        logger.info(f"actor train max_steps without dp_size: {self.actor_train.training_args.max_steps}")
        logger.info(f"critic train max_steps without dp_size: {self.critic.training_args.max_steps}")
        self.max_steps = max_steps

    @property
    def async_pipeline(self) -> bool:
        return self.async_generation_ratio > 0

    @property
    def is_train_infer_colocated(self) -> bool:
        """Whether actor_train and actor_infer are colocated."""
        return is_colocated(self.actor_train, self.actor_infer)
