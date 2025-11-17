import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from roll.configs.base_config import PPOConfig
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
class RLVRConfig(PPOConfig):
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

    # role related
    validation: WorkerConfig = field(
        default=None,
        metadata={"help": "Configuration for the validation."}
    )
    rewards: Optional[Dict[str, RewardConfig]] = field(
        default_factory=dict,
        metadata={"help": "Configuration for the multi domain rewards."}
    )

    # PPO related
    difficulty_loss_weight: bool = field(default=False, metadata={"help": "Use difficulty_loss_weight"})
    length_loss_weight: bool = field(default=False, metadata={"help": "Use length_loss_weight"})
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
    importance_sampling: Literal["token", "seq"] = (
        field(default="token", metadata={"help": "policy importance sampling"})
    )
    use_rollout_importance_sampling_ratio: bool = field(default=False, metadata={"help": "apply train/infer ratio as token-level loss weight"})
    rollout_importance_sampling_ratio_upper_bound: float = field(default=1.2)

    train_infer_ratio_mask: bool = field(default=False, metadata={"help": "apply train/infer ratio as token-level response mask"})
    train_infer_ratio_threshold_low: float = field(default=0.8)
    train_infer_ratio_threshold_high: float = field(default=1.2)
    train_infer_diff_mask: bool = field(default=False, metadata={"help": "apply train-infer diff as token-level response mask"})
    train_infer_diff_threshold_low: float = field(default=-0.2)
    train_infer_diff_threshold_high: float = field(default=0.2)

    train_infer_ratio_seq_mask: bool = field(default=False, metadata={"help": "apply train/infer ratio as sequence-level response mask"})
    train_infer_ratio_seq_threshold_low: float = field(default=0.8)
    train_infer_ratio_seq_threshold_high: float = field(default=1.2)
    train_infer_diff_seq_mask: bool = field(default=False, metadata={"help": "apply train-infer diff as sequence-level response mask"})
    train_infer_diff_seq_threshold_low: float = field(default=-0.2)
    train_infer_diff_seq_threshold_high: float = field(default=0.2)

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
        self.actor_infer.generating_args.num_return_sequences = self.num_return_sequences_in_group
        super().__post_init__()

        # default worker_cls
        if self.actor_train.worker_cls is None:
            self.actor_train.worker_cls = "roll.pipeline.rlvr.actor_worker.ActorWorker"
        if self.actor_infer.worker_cls is None:
            self.actor_infer.worker_cls = "roll.pipeline.rlvr.actor_worker.ActorWorker"
        if self.reference.worker_cls is None:
            self.reference.worker_cls = "roll.pipeline.rlvr.actor_worker.ActorWorker"
        if self.critic.worker_cls is None:
            self.critic.worker_cls = "roll.pipeline.base_worker.CriticWorker"

        logger.info(f"actor_train.worker_cls: {self.actor_train.worker_cls}")

        self.domain_2_tag = None
        self.tag_2_domain = None
        if self.rewards is not None:
            self.domain_2_tag = {key: set(worker_config.tag_included) for key, worker_config in self.rewards.items()}
            self.tag_2_domain = {
                tag: key for key, worker_config in self.rewards.items() for tag in worker_config.tag_included
            }

        if self.async_pipeline:
            assert self.async_generation_ratio >= 1.0, "async_generation_ratio must be >= 1.0"
            infer_devices = self.actor_infer.device_mapping
            other_worker_devices = set()
            for worker_config in [
                self.actor_train,
                self.critic,
                self.rewards,
                self.reference,
            ]:
                if worker_config is None:
                    continue
                if isinstance(worker_config, dict):
                    for config in worker_config.values():
                        other_worker_devices.update(config.device_mapping or set())
                else:
                    other_worker_devices.update(worker_config.device_mapping or set())

            if infer_devices is not None and len(set(infer_devices).intersection(other_worker_devices)) != 0:
                logger.warning("infer worker are sharing devices with other workers, which may cause performance issue")

            assert self.generate_opt_level == 1, "AsyncRLVRPipeline only support generate_opt_level 1"
            if self.num_return_sequences_in_group > 1 and not self.is_num_return_sequences_expand:
                self.is_num_return_sequences_expand = True
                logger.warning("Async Pipeline must is_num_return_sequences_expand is True when num_return_sequences_in_group > 1")

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

    def to_dict(self):
        return dataclasses.asdict(self)
