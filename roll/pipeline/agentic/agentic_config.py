import os
import random
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any, Literal

from omegaconf import DictConfig

from roll.configs.base_config import BaseConfig
from roll.configs.worker_config import WorkerConfig
from roll.utils.logging import get_logger

logger = get_logger()


@dataclass
class RewardNormalizationConfig:
    grouping: str = field(default="state", metadata={"help": "state / batch / inductive"})
    method: str = field(default="identity", metadata={"help": "asym_clip / identity / mean_std"})

@dataclass
class LLMProxyConfig:
    proxy_type: str = field(default="policy", metadata={"help": "llm proxy type: [policy, openai, random]."})
    proxy_config: Dict = field(default_factory=dict, metadata={"help": "llm proxy config."})


@dataclass
class EnvManagerConfig(WorkerConfig):
    llm_proxy: LLMProxyConfig = field(default_factory=LLMProxyConfig, metadata={"help": "llm proxy config."})
    num_env_groups: int = field(default=128, metadata={"help": "Number of environment groups during training."})
    group_size: int = field(
        default=1, metadata={"help": "Under the same group, the env config and env seed are ensured to be equal"}
    )
    tags: List[str] = field(default_factory=lambda: ["SimpleSokoban"], metadata={"help": "Environment tags."})
    num_groups_partition: List[int] = field(
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
    max_env_num_per_worker: int = field(
        default=0,
        metadata={"help": "The maximum number of envs per worker. one env per thread."}
    )

    def __post_init__(self):
        """
        根据es config计算world_size
        """
        if self.max_env_num_per_worker <= 0:
            self.max_env_num_per_worker = self.num_env_groups * self.group_size
            logger.warning("all env in one worker by default, you can set max_env_num_per_worker to scale env.")
        logger.info(f"max_env_num_per_worker: {self.max_env_num_per_worker}")

        assert self.num_env_groups * self.group_size % self.max_env_num_per_worker == 0
        self.world_size = (self.num_env_groups * self.group_size + self.max_env_num_per_worker - 1) // self.max_env_num_per_worker
        self.env_configs: Optional[Dict[int, Dict[int, Dict]]] = None
        """
        worker_rank: 
            env_id:
                env_config
        """


@dataclass
class AgenticConfig(BaseConfig):
    # agentic related
    custom_envs: Dict[str, Any] = field(default_factory=dict, metadata={"help": "List of environment configurations."})
    train_env_manager: EnvManagerConfig = field(default_factory=EnvManagerConfig)
    val_env_manager: EnvManagerConfig = field(default_factory=EnvManagerConfig)
    render_save_dir: str = field(default=None, metadata={"help": "Directory to save rendered frames."})
    reward_normalization: RewardNormalizationConfig = field(
        default_factory=RewardNormalizationConfig, metadata={"help": "Reward normalization configuration."}
    )

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

    batch_adjust_mode: Literal["copy", "delete", "auto"] = field(
        default="copy", metadata={"help": "batch adjust mode: copy or delete"}
    )
    episode_reward_weight: float = field(default=1.0, metadata={"help": "Episode reward weight, used in GiGPO."})
    step_reward_weight: float = field(default=1.0, metadata={"help": "Step reward weight, used in GiGPO."})
    step_reward_gamma: float = field(default=0.95, metadata={"help": "Gamma parameter for step reward calculation"})

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
    use_reward_norm: bool = field(
        default=False, metadata={"help": "Use reward normalization. Only applicable if use_reward_scaling is True."}
    )
    whiten_rewards: bool = field(default=False, metadata={"help": "Whiten the rewards before compute advantages."})
    whiten_advantages: bool = field(default=False, metadata={"help": "Whiten the advantage."})
    advantage_clip: float = field(default=None, metadata={"help": "advantage_clip value"})
    adv_estimator: Literal["gae", "reinforce", "grpo", "gigpo"] = field(
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

        self.actor_infer.generating_args.num_return_sequences = 1

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

        train_env_num = self.train_env_manager.num_env_groups * self.train_env_manager.group_size
        traj_per_env = (self.rollout_batch_size + train_env_num - 1) // train_env_num
        if self.async_generation_ratio > 0:
            # force set max_traj_per_env when use async training
            self.train_env_manager.max_traj_per_env = traj_per_env
        elif self.train_env_manager.max_traj_per_env < 0:
            self.train_env_manager.max_traj_per_env = traj_per_env
        logger.info(f"train_env_manager.max_traj_per_env: {self.train_env_manager.max_traj_per_env}")
        assert self.train_env_manager.max_traj_per_env >= traj_per_env, f"max_traj_per_env must be >= {traj_per_env}"

        val_env_num = self.val_env_manager.num_env_groups * self.val_env_manager.group_size
        traj_per_env = (self.val_batch_size + val_env_num - 1) // val_env_num
        if self.val_env_manager.max_traj_per_env < 0:
            self.val_env_manager.max_traj_per_env = traj_per_env
        logger.info(f"val_env_manager.max_traj_per_env: {self.val_env_manager.max_traj_per_env}")
        assert self.val_env_manager.max_traj_per_env >= traj_per_env, f"max_traj_per_env must be >= {traj_per_env}"

    def make_env_configs(self, env_manager_config: EnvManagerConfig):
        # construct env configs
        env_configs = defaultdict(defaultdict)
        done_groups = 0
        env_manager_config.env_configs = {}
        group_seeds = {}
        max_env_num_per_worker = env_manager_config.max_env_num_per_worker
        for tag, n_group in zip(env_manager_config.tags, env_manager_config.num_groups_partition):
            for env_id in range(
                done_groups * env_manager_config.group_size, (done_groups + n_group) * env_manager_config.group_size
            ):
                cfg_template = self.custom_envs[tag]
                env_class = cfg_template.env_type

                group_id = env_id // env_manager_config.group_size

                if "env_config" not in cfg_template:
                    cfg_template.env_config = {}
                # cfg_template.env_config["rank"] = group_id
                # cfg_template.env_config["world_size"] = env_manager_config.num_env_groups
                env_config = {**cfg_template.env_config}

                if group_id not in group_seeds:
                    group_seeds[group_id] = random.randint(0, 1000000)
                entry = {}
                entry.update(cfg_template)
                entry.pop("env_config", None)
                entry.update({
                    "tag": tag,
                    "group_id": group_id,
                    "env_id": env_id,
                    "config": env_config,
                    "env_class": env_class,
                    "env_manager_cls": cfg_template.get("env_manager_cls", "roll.pipeline.agentic.env_manager.traj_env_manager.TrajEnvManager"),
                    "group_seed": group_seeds[group_id],
                })
                worker_rank = env_id // max_env_num_per_worker
                env_configs[worker_rank][env_id] = DictConfig(entry)
            done_groups += n_group
        assert done_groups == env_manager_config.num_env_groups
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
