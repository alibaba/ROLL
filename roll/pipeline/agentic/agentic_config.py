import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from omegaconf import DictConfig

from roll.configs.base_config import PPOConfig
from roll.configs.worker_config import WorkerConfig
from roll.utils.logging import get_logger


logger = get_logger()


def _resolve_reward_norm_defaults(method: str, grouping: str) -> Dict[str, Optional[str]]:
    normalized_group = (grouping or "").lower()
    if normalized_group == "batch":
        mean_type = "batch"
        std_type = "batch"
    else:
        if normalized_group not in ["state", "inductive"]:
            logger.warning(
                f"`RewardNormalizationConfig.grouping` 的取值 {normalized_group} 不在 ['batch', 'state', 'inductive'] 中，mean和std的统计范围设置为 'group'， 然后再结合method来做选择norm的方式"
            )
        mean_type = "group"
        std_type = "group"

    if method == "identity":
        return {"norm_mean_type": None, "norm_std_type": None}
    elif method == "mean":
        return {"norm_mean_type": mean_type, "norm_std_type": None}
    elif method in {"mean_std", "asym_clip"}:
        return {"norm_mean_type": mean_type, "norm_std_type": std_type}

    return {"norm_mean_type": None, "norm_std_type": None}


@dataclass
class RewardNormalizationConfig:
    grouping: str = field(default="state", metadata={"help": "state / batch / inductive"})
    method: str = field(
        default="identity",
        metadata={
            "help": "已废弃字段: 取值仅用于推导 norm_mean_type / norm_std_type；请优先直接配置新字段",
            "deprecated": True,
        },
    )
    norm_mean_type: Optional[Literal["batch", "group"]] = field(
        default=None,
        metadata={
            "help": "Mean type for reward normalization: 'batch' (normalize across batch), 'group' (normalize within groups), None (without subtracting mean)"
        },
    )
    norm_std_type: Optional[Literal["batch", "group"]] = field(
        default=None,
        metadata={
            "help": "Std type for reward normalization: 'batch' (normalize across batch), 'group' (normalize within groups), None (without dividing by std)"
        },
    )

    def __post_init__(self):

        if self.method not in {"identity", "mean", "mean_std", "asym_clip"}:
            logger.warning(
                f"`RewardNormalizationConfig.method` 的取值 {self.method!r} 已废弃且无效，将回退为 'identity'。"
            )
            self.method = "identity"

        logger.warning(
            "`RewardNormalizationConfig.method` 已废弃，后续版本将移除；显式配置 `norm_mean_type` / `norm_std_type` 优先级最高，"
            " `method` 仅在`norm_mean_type` / `norm_std_type`字段为空时参与兜底。"
        )

        defaults = _resolve_reward_norm_defaults(self.method, self.grouping)
        if self.norm_mean_type is None:
            logger.info(
                "`norm_mean_type` 未显式配置，将依据 method=%s 与 grouping=%s 推导为 %s。",
                self.method,
                self.grouping,
                defaults["norm_mean_type"],
            )
            self.norm_mean_type = defaults["norm_mean_type"]
        if self.norm_std_type is None:
            logger.info(
                "`norm_std_type` 未显式配置，将依据 method=%s 与 grouping=%s 推导为 %s。",
                self.method,
                self.grouping,
                defaults["norm_std_type"],
            )
            self.norm_std_type = defaults["norm_std_type"]
        logger.info(
            "`RewardNormalizationConfig` 将采用 norm_mean_type=%s, norm_std_type=%s。",
            self.norm_mean_type,
            self.norm_std_type,
        )


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
    group_size_redundancy: int = field(default=0, metadata={"help": "Redundancy num of group size."})
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
        default=0, metadata={"help": "The maximum number of envs per worker. one env per thread."}
    )
    group_filter_cls: str = field(
        default="roll.pipeline.agentic.agentic_pipeline.GroupFilter",
        metadata={"help": "Group level filter function. Return false to filter out group."},
    )

    def __post_init__(self):
        """
        根据es config计算world_size
        """
        if self.max_env_num_per_worker <= 0:
            self.max_env_num_per_worker = self.num_env_groups * self.final_group_size
            logger.warning("all env in one worker by default, you can set max_env_num_per_worker to scale env.")
        logger.info(f"max_env_num_per_worker: {self.max_env_num_per_worker}")

        self.world_size = (
            self.num_env_groups * self.final_group_size + self.max_env_num_per_worker - 1
        ) // self.max_env_num_per_worker
        self.env_configs: Optional[Dict[int, Dict[int, Dict]]] = None
        """
        worker_rank: 
            env_id:
                env_config
        """

    @property
    def final_group_size(self):
        return self.group_size + self.group_size_redundancy


@dataclass
class AgenticConfig(PPOConfig):
    # agentic related
    custom_envs: Dict[str, Any] = field(default_factory=dict, metadata={"help": "List of environment configurations."})
    train_env_manager: EnvManagerConfig = field(default_factory=EnvManagerConfig)
    val_env_manager: EnvManagerConfig = field(default_factory=EnvManagerConfig)
    render_save_dir: str = field(default=None, metadata={"help": "Directory to save rendered frames."})
    reward_normalization: RewardNormalizationConfig = field(
        default_factory=RewardNormalizationConfig, metadata={"help": "Reward normalization configuration."}
    )

    batch_adjust_mode: Literal["copy", "delete", "auto", "random_sample"] = field(
        default="copy", metadata={"help": "batch adjust mode: copy or delete"}
    )
    episode_reward_weight: float = field(default=1.0, metadata={"help": "Episode reward weight, used in GiGPO."})
    step_reward_weight: float = field(default=1.0, metadata={"help": "Step reward weight, used in GiGPO."})
    step_reward_gamma: float = field(default=0.95, metadata={"help": "Gamma parameter for step reward calculation"})
    ratio_type: Literal["token", "segment"] = field(default="token", metadata={"help": "Ratio type: token or segment"})

    def __post_init__(self):
        self.actor_infer.generating_args.num_return_sequences = 1
        super().__post_init__()

        # default worker_cls
        if self.actor_train.worker_cls is None:
            self.actor_train.worker_cls = "roll.pipeline.agentic.agentic_actor_worker.ActorWorker"
        if self.actor_infer.worker_cls is None:
            self.actor_infer.worker_cls = "roll.pipeline.base_worker.ActorWorker"
        if self.reference.worker_cls is None:
            self.reference.worker_cls = "roll.pipeline.base_worker.ActorWorker"
        if self.critic.worker_cls is None:
            self.critic.worker_cls = "roll.pipeline.base_worker.CriticWorker"

        self.train_env_manager.name = "train_env"
        self.val_env_manager.name = "val_env"

        if self.render_save_dir:
            self.render_save_dir = os.path.join(
                self.render_save_dir, self.exp_name, datetime.now().strftime("%Y%m%d-%H%M%S")
            )
        logger.info(f"add timestamp to render_save_dir  {self.render_save_dir}")

        assert self.max_steps > 0 or self.max_steps == -1, "max_steps must be greater than 0 or -1"

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
        if self.val_batch_size < 0:
            self.val_env_manager.max_traj_per_env = sys.maxsize
        else:
            assert (
                self.val_batch_size % val_env_num == 0
            ), f"val_batch_size {self.val_batch_size} must be divisible by val_env_num {val_env_num}, equal best"

            traj_per_env = (self.val_batch_size + val_env_num - 1) // val_env_num
            if self.val_env_manager.max_traj_per_env < 0:
                self.val_env_manager.max_traj_per_env = traj_per_env
        logger.info(f"val_env_manager.max_traj_per_env: {self.val_env_manager.max_traj_per_env}")
        assert self.val_env_manager.max_traj_per_env >= traj_per_env, f"max_traj_per_env must be >= {traj_per_env}"

        self.validate_worker_config()

    def make_env_configs(self, env_manager_config: EnvManagerConfig):
        # construct env configs
        env_configs = defaultdict(defaultdict)
        done_groups = 0
        env_manager_config.env_configs = {}
        group_seeds = {}
        max_env_num_per_worker = env_manager_config.max_env_num_per_worker
        for tag, n_group in zip(env_manager_config.tags, env_manager_config.num_groups_partition):
            for env_id in range(
                done_groups * env_manager_config.final_group_size,
                (done_groups + n_group) * env_manager_config.final_group_size,
            ):
                cfg_template = self.custom_envs[tag]
                env_class = cfg_template.env_type

                group_id = env_id // env_manager_config.final_group_size

                if "env_config" not in cfg_template:
                    cfg_template.env_config = {}
                # cfg_template.env_config["rank"] = group_id
                # cfg_template.env_config["world_size"] = env_manager_config.num_env_groups
                env_config = {**cfg_template.env_config}

                if group_id not in group_seeds:
                    group_seeds[group_id] = random.randint(0, 2**31 - 1)
                entry = {}
                entry.update(cfg_template)
                entry.pop("env_config", None)
                entry.update(
                    {
                        "tag": tag,
                        "group_id": group_id,
                        "env_id": env_id,
                        "config": env_config,
                        "env_class": env_class,
                        "env_manager_cls": cfg_template.get(
                            "env_manager_cls", "roll.pipeline.agentic.env_manager.traj_env_manager.TrajEnvManager"
                        ),
                        "group_seed": group_seeds[group_id],
                    }
                )
                worker_rank = env_id // max_env_num_per_worker
                env_configs[worker_rank][env_id] = DictConfig(entry)
                logger.info(
                    f"[ENV CONFIG] tag: {tag}, group_id: {group_id}, group_seeds: {group_seeds[group_id]}, env_id: {env_id}"
                )
            done_groups += n_group
        assert done_groups == env_manager_config.num_env_groups
        env_manager_config.env_configs = env_configs
