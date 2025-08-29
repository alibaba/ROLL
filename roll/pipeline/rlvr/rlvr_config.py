import dataclasses
from dataclasses import dataclass, field
from typing import Optional, Literal, List, Dict, Any

from roll.configs.base_config import BaseConfig, ScheduleConfig
from roll.configs.worker_config import WorkerConfig
from roll.utils.logging import get_logger

logger = get_logger()


@dataclass
class DatasetFilterConfig:
    source: Optional[str] = None
    min_difficulty: Optional[float] = None
    max_difficulty: Optional[float] = None
    num_samples: int = 0

@dataclass
class RewardFilterConfig:
    type: Literal["no_filter", "mean_filter", "std_filter"] = field(
        default="no_filter",
        metadata={"help": "Type of filter to apply to rewards."},
    )
    filter_args: Dict[str, Any] = field(
        default_factory=dict,
        metadata={"help": "Arguments used in `filter_fn`"},
    )


@dataclass
class RewardConfig(WorkerConfig):
    code_url: str = field(
        default=None,
        metadata={"help": "The url of the code."}
    )
    use_local: bool = field(
        default=True,
        metadata={"help": "Whether to use local code instead of downloading from URL."}
    )
    judge_prompt: str = field(
        default=None,
        metadata={"help": "The prompt for judge."}
    )
    judge_model_type: str = field(
        default=None,
        metadata={"help": "api or inference"}
    )
    judge_model_name: str = field(
        default=None,
        metadata={"help": "judge_model_name."}
    )
    judge_api_url: str = field(
        default=None,
        metadata={"help": "judge_api_url."}
    )
    judge_api_key: str = field(
        default=None,
        metadata={"help": "judge_api_key."}
    )
    format_pattern: str = field(
        default=None,
        metadata={"help": "The pattern of the answer format."}
    )
    reward_type: str = field(default=None, metadata={"help": "The type of the reward."})
    response_length_penalty_coef: float = field(default=0.0, metadata={"help": "The coefficient of the response length penalty."})

    tag_included: List[str] = field(default_factory=list, metadata={"help": "The tags of the domain."})
    query_filter_config: RewardFilterConfig = field(
        default_factory=RewardFilterConfig,
        metadata={"help": "Arguments passed to reward query filtering"},)
    response_filter_config: RewardFilterConfig = field(
        default_factory=RewardFilterConfig,
        metadata={"help": "Arguments passed to reward response filtering"},
    )



@dataclass
class RLVRConfig(BaseConfig):
    # global
    global_template: str = field(
        default=None,
        metadata={"help": "The template of the global."})
    dataset_filter: DatasetFilterConfig = field(
        default_factory=DatasetFilterConfig,
        metadata={"help": "Configuration for filtering dataset by source and difficulty"},
    )
    num_return_sequences_in_group: int = field(
        default=1,
        metadata={"help": "The number of return sequences in one group, used in generation_args."}
    )

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
    save_logging_board_dir: str = field(
        default=None, metadata={"help": "saving directory of logging board_metrics"}
    )
    rollout_dump_dir: str = field(
        default=None, metadata={"help": "saving actor_infer rollout to this dir"}
    )

    # role related
    pretrain: str = field(
        default=None,
        metadata={"help": "Path to pretrain model directory, if available."})
    reward_pretrain: str = field(
        default=None,
        metadata={"help": "Path to pretrain model directory for the reward model, if available."}
    )
    validation: WorkerConfig = field(
        default=None,
        metadata={"help": "Configuration for the validation."}
    )
    actor_train: WorkerConfig = field(
        default_factory=WorkerConfig,
        metadata={"help": "Configuration for the actor's training role."}
    )
    actor_infer: WorkerConfig = field(
        default_factory=WorkerConfig,
        metadata={"help": "Configuration for the actor's inference role."}
    )
    critic: WorkerConfig = field(
        default_factory=WorkerConfig,
        metadata={"help": "Configuration for the critic's training role."}
    )
    reference: WorkerConfig = field(
        default_factory=WorkerConfig,
        metadata={"help": "Configuration for the reference role."}
    )
    rewards: Optional[Dict[str, RewardConfig]] = field(
        default_factory=dict,
        metadata={"help": "Configuration for the multi domain rewards."}
    )

    # PPO related
    ppo_epochs: int = field(default=1, metadata={"help": "Number of optimisation epochs per batch of samples"})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Maximum norm"})
    l2: float = field(default=0.0, metadata={"help": "L2 regularization"})
    lambd: float = field(default=0.95, metadata={"help": "Lambda parameter for advantage calculation"})
    gamma: float = field(default=1, metadata={"help": "Gamma parameter for advantage calculation"})
    pg_clip: Optional[float] = field(default=0.2, metadata={"help": "Range for clipping in PPO policy gradient loss"})
    use_pg_clip_range: bool = field(default=False, metadata={"help": "Use to change the clipping range of pg_clip"})
    pg_clip_low: Optional[float] = field(default=0.2, metadata={"help": "Range for clipping lower in PPO policy gradient loss"})
    pg_clip_high: Optional[float] = field(default=0.2, metadata={"help": "Range for clipping higher in PPO policy gradient loss"})
    
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
    difficulty_loss_weight: bool = field(default=False, metadata={"help": "Use difficulty_loss_weight"})
    length_loss_weight: bool = field(default=False, metadata={"help": "Use length_loss_weight"})
    use_reward_norm: bool = field(
        default=False, metadata={"help": "Use reward normalization. Only applicable if use_reward_scaling is True."}
    )
    whiten_rewards: bool = field(default=False, metadata={"help": "Whiten the rewards before compute advantages."})
    whiten_advantages: bool = field(default=False, metadata={"help": "Whiten the advantage."})
    advantage_clip: float = field(default=None, metadata={"help": "advantage_clip value"})
    adv_estimator: Literal["gae", "reinforce", "grpo"] = field(
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
    postive_loss_coef: float = field(
        default=0,
        metadata={"help": "Loss coefficient for SFT loss, used for positive samples"}
    )
    use_topr_neg_loss_coef: float = field(
        default=0.0,
        metadata={"help": "Loss coefficient for TOPR Neg loss"}
    )
    use_policy_loss_type: Literal["PPO", "PG"] = field(
        default="PPO",
        metadata={"help": "whether to use PPO/PG loss"}
    )
    use_topr_loss: bool = field(
        default=False,
        metadata={"help": "whether to use TPRO loss, http://arxiv.org/abs/2503.14286"}
    )
    rl_loss_coef: float = field(
        default=1.0,
        metadata={"help": "Loss coefficient for RL loss"}
    )
    dual_clip_loss: bool = field(default=False, metadata={"help": "Use dual clip loss"})
    loss_agg_mode: Literal["token-mean", "seq-mean-token-sum", "seq-mean-token-mean", "seq-mean-token-sum-norm"] = (
        field(default="seq-mean-token-mean", metadata={"help": "Loss aggregation mode"})
    )
    importance_sampling: Literal["token", "seq"] = (
        field(default="token", metadata={"help": "policy importance sampling"})
    )
    val_greedy: bool = field(default=False, metadata={"help": "Use greedy for validation"})
    val_n_sample: int = field(default=1, metadata={"help": "Number of samples for validation"})
    max_len_mask: bool = field(default=False)
    mask_type: Literal["all", "loss"] = field(default="loss", metadata={"help": "Mask type: 'all' or 'loss'"})
    difficulty_mask: bool = field(default=False)
    balance_length: bool = field(default=False)
    minibatch_data_iter_num: int = field(default=1)
    difficulty_low_threshold: float = field(default=0.0)
    difficulty_high_threshold: float = field(default=1.0)
    error_max_len_clip: bool = field(default=False)
    error_max_len_threshold: int = field(default=9999999999)

    def __post_init__(self):
        super().__post_init__()

        if (
                self.actor_train.model_args.model_name_or_path is None
                or self.actor_infer.model_args.model_name_or_path
                or self.reference.model_args.model_name_or_path is None
        ):
            self.actor_train.model_args.model_name_or_path = self.pretrain
            self.actor_infer.model_args.model_name_or_path = self.pretrain
            self.reference.model_args.model_name_or_path = self.pretrain

        # default worker_cls
        if self.actor_train.worker_cls is None:
            self.actor_train.worker_cls = "roll.pipeline.rlvr.actor_worker.ActorWorker"
        if self.actor_infer.worker_cls is None:
            self.actor_infer.worker_cls = "roll.pipeline.rlvr.actor_worker.ActorWorker"
        if self.reference.worker_cls is None:
            self.reference.worker_cls = "roll.pipeline.rlvr.actor_worker.ActorWorker"
        if self.critic.worker_cls is None:
            self.critic.worker_cls = "roll.pipeline.base_worker.CriticWorker"

        if self.critic.model_args.model_name_or_path is None:
            self.critic.model_args.model_name_or_path = self.reward_pretrain

        self.actor_train.training_args.output_dir = self.output_dir
        self.actor_infer.training_args.output_dir = self.output_dir
        self.critic.training_args.output_dir = self.output_dir

        self.actor_infer.generating_args.num_return_sequences = self.num_return_sequences_in_group

        self.actor_infer.name = "actor_infer"
        self.actor_train.name = "actor_train"
        self.reference.name = "reference"
        self.critic.name = "critic"
        self.domain_2_tag = None
        self.tag_2_domain = None
        if self.rewards is not None:
            self.domain_2_tag = {key: set(worker_config.tag_included) for key, worker_config in self.rewards.items()}
            self.tag_2_domain = {
                tag: key for key, worker_config in self.rewards.items() for tag in worker_config.tag_included
            }
        if self.actor_infer:
            self.actor_infer.generating_args.max_new_tokens = self.sequence_length - self.prompt_length
            logger.warning(f"rewrite actor_infer max_new_tokens: {self.actor_infer.generating_args.max_new_tokens}")
        if self.validation:
            self.validation.generating_args.max_new_tokens = self.val_sequence_length - self.val_prompt_length
            logger.warning(f"rewrite validation max_new_tokens: {self.validation.generating_args.max_new_tokens}")

        # infer the required num nodes
        total_devices = []
        for attribute_name in dir(self):
            attribute = getattr(self, attribute_name)
            if isinstance(attribute, WorkerConfig):
                if attribute.device_mapping is not None:
                    total_devices.extend(attribute.device_mapping)
        for worker_config in self.rewards.values():
            if worker_config.device_mapping is not None:
                total_devices.extend(worker_config.device_mapping)
        if len(total_devices) > 0:
            max_gpu_num = max(total_devices) + 1
            if max_gpu_num <= self.num_gpus_per_node:
                self.num_nodes = 1
            else:
                self.num_nodes = (max_gpu_num + self.num_gpus_per_node - 1) // self.num_gpus_per_node

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

    def to_dict(self):
        return dataclasses.asdict(self)
