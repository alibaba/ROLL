import copy
import time
from abc import abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass, field
from itertools import zip_longest
from queue import Queue
from threading import Lock, Thread
from typing import Dict, List, Optional, Tuple

import numpy as np
import ray
import torch
from ray.util.queue import Empty
from tensordict import TensorDict
from transformers import PreTrainedTokenizer, AutoTokenizer

from roll.agentic.env import REGISTERED_ENVS
from roll.agentic.env.base import BaseEnv
from roll.agentic.rollout.env_action_limiter import get_global_limiter
from roll.agentic.rollout.token_mask_utils import messages_to_tokens_and_masks
from roll.datasets.chat_template import get_chat_template
from roll.distributed.scheduler.generate_scheduler import RequestScheduler, GlobalCounter
from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.agentic.agentic_config import EnvManagerConfig, AgenticConfig
from roll.utils.constants import RAY_NAMESPACE
from roll.utils.functionals import pad_to_length
from roll.utils.logging import get_logger


@dataclass
class RolloutCache:
    env_id: int
    group_id: int
    tag: str

    history: List[Dict] = field(default_factory=list)   # keys: [state, actions_left, reward, penalty, llm_response, metrics], a dict save each step info
    frames: List = field(default_factory=list)

    truncated: bool = False
    terminated: bool = False
    step: int = 0


class BaseEnvManager:
    def __init__(self, *args, **kwargs):
        self.current_step = -1
        self.running = False

    @abstractmethod
    def run_rollout_loop(self, data: DataProto):
        """
        1. Each time run_rollout_loop is called,
           it will continuously play episodes until it receives a command that data collection is complete.
           The seed needs to be reset to ensure consistency across all groups.
           episode_id is reset to 0.

        Seed update logic:
           group_seed = base_seed + group_id
           episode_seed = group_seed + episode_id

        trajectory_id: f"{group_id}_{episode_id}_{episode_seed}"
        """
        pass

    def reset(self) -> RolloutCache:
        pass

    def step(self, llm_output: DataProto) -> RolloutCache:
        pass

    def make_decision(self, rollout_cache: RolloutCache) -> DataProto:
        pass

    def format_messages(self, history: List[Dict]) -> List[Dict]:
        pass

    def formulate_rollouts(self, rollout_cache: RolloutCache) -> DataProto:
        pass

    def update_step(self, global_step):
        self.current_step = global_step

    def stop(self):
        self.running = False