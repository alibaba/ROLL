import os
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any, Literal

from omegaconf import DictConfig

from roll.agentic.env import REGISTERED_ENV_CONFIGS
from roll.configs.base_config import BaseConfig
from roll.configs.worker_config import WorkerConfig
from roll.pipeline.rlvr.rlvr_config import RLVRConfig
from roll.utils.logging import get_logger

logger = get_logger()


@dataclass
class RewardNormalizationConfig:
    grouping: str = field(default="state", metadata={"help": "state / batch / inductive"})
    method: str = field(default="identity", metadata={"help": "asym_clip / identity / mean_std"})


@dataclass
class EnvManagerConfig(WorkerConfig):
    env_groups: int = field(default=128, metadata={"help": "Number of environment groups during training."})
    group_size: int = field(
        default=1, metadata={"help": "Under the same group, the env config and env seed are ensured to be equal"}
    )
    tags: List[str] = field(default_factory=lambda: ["SimpleSokoban"], metadata={"help": "Environment tags."})
    n_groups: List[int] = field(
        default_factory=lambda: [128],
        metadata={
            "help": "If not set, all env names divide nums equally. Under the same group, the env config and env seed (prompt) are equal in each generation"
        },
    )
    max_traj_per_env: int = field(
        default=-1, metadata={"help": "The maximum number of trajectories that each environment can rollout."}
    )
    format_penalty: float = field(default=0, metadata={"help": "Format penalty value."})
    worker_cls: Optional[str] = field(
        default="roll.pipeline.agentic.environment_worker.EnvironmentWorker",
        metadata={"help": "The class of the worker."},
    )

    def __post_init__(self):
        """
        根据es config计算world_size
        """
        self.world_size = self.env_groups * self.group_size
        self.env_configs: Optional[Dict[int, Dict]] = None


@dataclass
class AgenticConfig(BaseConfig):
    # agentic related
    custom_envs: Dict[str, Any] = field(default_factory=dict, metadata={"help": "List of environment configurations."})
    train_env_manager: EnvManagerConfig = field(default_factory=EnvManagerConfig)
    val_env_manager: EnvManagerConfig = field(default_factory=EnvManagerConfig)
    enable_response_mask: bool = field(default=True, metadata={"help": "Whether to mask the response."})
    render_save_dir: str = field(default=None, metadata={"help": "Directory to save rendered frames."})
    action_sep: str = field(default="||", metadata={"help": "Action separator."})
    use_turn_scores: bool = field(
        default=False,
        metadata={
            "help": "Important to GAE when applying token-level rewards to token-level advantages. If False, will take the sum of scores as the reward for the last turn."
        },
    )
    enable_think: bool = field(default=True, metadata={"help": "False -> no think RL"})
    reward_normalization: RewardNormalizationConfig = field(
        default_factory=RewardNormalizationConfig, metadata={"help": "Reward normalization configuration."}
    )
    special_token_list: List[str] = field(
        default_factory=lambda: ["<think>", "</think>", "<answer>", "</answer>", "<|im_start|>", "<|im_end|>"],
        metadata={"help": "Special tokens."},
    )
    max_steps_per_traj: int = field(default=10, metadata={"help": "Max steps per trajectory."})

    # role related
    pretrain: str = field(
        default=None,
        metadata={"help": "Path to pretrain model directory, if available."})
    reward_pretrain: str = field(
        default=None,
        metadata={"help": "Path to pretrain model directory for the reward model, if available."}
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

    # PPO related
    ppo_epochs: int = field(default=1, metadata={"help": "Number of optimisation epochs per batch of samples"})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Maximum norm"})
    l2: float = field(default=0.0, metadata={"help": "L2 regularization"})
    lambd: float = field(default=0.95, metadata={"help": "Gamma parameter for advantage calculation"})
    gamma: float = field(default=1, metadata={"help": "Lambda parameter for advantage calculation"})
    pg_clip: Optional[float] = field(default=0.2, metadata={"help": "Range for clipping in PPO policy gradient loss"})
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
    adv_estimator: Literal["gae", "reinforce", "grpo"] = field(
        default="gae", metadata={"help": "advantage estimator: gae (GAE)."}
    )
    reward_norm: Literal["batch", "group", "running", None] = field(
        default=None,
        metadata={
            "help": "Reward normalization type: 'batch' (normalize across batch), 'group' (normalize within prompt groups), 'running' (use running statistics)"
        },
    )
    reward_shift: bool = field(
        default=False, metadata={"help": "Only subtract mean without dividing by std during reward normalization"}
    )
    reward_scale: bool = field(
        default=False, metadata={"help": "Only divide by std without subtracting mean during reward normalization"}
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
        field(default="token-mean", metadata={"help": "Loss aggregation mode"})
    )
    dual_clip_loss: bool = field(default=False, metadata={"help": "Use dual clip loss"})

    def __post_init__(self):
        BaseConfig.__post_init__(self)

        if (
            self.actor_train.model_args.model_name_or_path is None
            or self.actor_infer.model_args.model_name_or_path
            or self.reference.model_args.model_name_or_path is None
        ):
            self.actor_train.model_args.model_name_or_path = self.pretrain
            self.actor_infer.model_args.model_name_or_path = self.pretrain
            self.reference.model_args.model_name_or_path = self.pretrain

        if self.critic.model_args.model_name_or_path is None:
            self.critic.model_args.model_name_or_path = self.reward_pretrain

        # default worker_cls
        if self.actor_train.worker_cls is None:
            self.actor_train.worker_cls = "roll.pipeline.base_worker.ActorWorker"
        if self.actor_infer.worker_cls is None:
            self.actor_infer.worker_cls = "roll.pipeline.base_worker.ActorWorker"
        if self.reference.worker_cls is None:
            self.reference.worker_cls = "roll.pipeline.base_worker.ActorWorker"
        if self.critic.worker_cls is None:
            self.critic.worker_cls = "roll.pipeline.base_worker.CriticWorker"

        self.actor_train.training_args.output_dir = self.output_dir
        self.actor_infer.training_args.output_dir = self.output_dir
        self.critic.training_args.output_dir = self.output_dir

        self.actor_infer.name = "actor_infer"
        self.actor_train.name = "actor_train"
        self.reference.name = "reference"
        self.critic.name = "critic"
        self.train_env_manager.name = "train_env"
        self.val_env_manager.name = "val_env"

        if self.render_save_dir:
            self.render_save_dir = os.path.join(
                self.render_save_dir, self.exp_name, datetime.now().strftime("%Y%m%d-%H%M%S")
            )
        logger.info(f"add timestamp to render_save_dir  {self.render_save_dir}")

        assert self.max_steps > 0, "max_steps must be greater than 0"

        self.train_env_manager.model_args.model_name_or_path = self.pretrain
        self.train_env_manager.generating_args = self.actor_infer.generating_args
        self.val_env_manager.model_args.model_name_or_path = self.pretrain
        self.val_env_manager.generating_args = self.actor_infer.generating_args
        self.custom_envs = DictConfig(self.custom_envs)
        self.make_env_configs(self.train_env_manager)
        self.make_env_configs(self.val_env_manager)

        train_env_num = len(self.train_env_manager.env_configs)
        traj_per_env = (self.rollout_batch_size + train_env_num - 1) // train_env_num
        if self.train_env_manager.max_traj_per_env < 0:
            self.train_env_manager.max_traj_per_env = traj_per_env
        assert self.train_env_manager.max_traj_per_env >= traj_per_env, f"max_traj_per_env must be >= {traj_per_env}"

        val_env_num = len(self.val_env_manager.env_configs)
        traj_per_env = (self.val_batch_size + val_env_num - 1) // val_env_num
        if self.val_env_manager.max_traj_per_env < 0:
            self.val_env_manager.max_traj_per_env = traj_per_env
        assert self.val_env_manager.max_traj_per_env >= traj_per_env, f"max_traj_per_env must be >= {traj_per_env}"

    def make_env_configs(self, env_manager_config: EnvManagerConfig):
        # construct env configs
        env_configs = {}
        done_groups = 0
        env_manager_config.env_configs = {}
        group_seeds = {}
        for tag, n_group in zip(env_manager_config.tags, env_manager_config.n_groups):
            for env_id in range(
                done_groups * env_manager_config.group_size, (done_groups + n_group) * env_manager_config.group_size
            ):
                cfg_template = self.custom_envs[tag]
                env_class = cfg_template.env_type
                max_actions_per_traj = cfg_template.max_actions_per_traj
                if cfg_template.env_config is None:
                    env_config = REGISTERED_ENV_CONFIGS[env_class]()
                else:
                    env_config = REGISTERED_ENV_CONFIGS[env_class](**cfg_template.env_config)
                group_id = env_id // env_manager_config.group_size
                if group_id not in group_seeds:
                    group_seeds[group_id] = random.randint(0, 1000000)
                entry = {
                    "tag": tag,
                    "group_id": env_id // env_manager_config.group_size,
                    "env_id": env_id,
                    "config": env_config,
                    "max_actions_per_traj": max_actions_per_traj,
                    "env_class": env_class,
                    "group_seed": group_seeds[group_id],
                }
                env_configs[env_id] = entry
            done_groups += n_group
        env_manager_config.env_configs = env_configs

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
