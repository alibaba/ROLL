"""Geo3K rule-based reward implementation.

The scoring logic is adapted from VeRL's Geo3K reward implementation:
https://github.com/verl-project/verl/blob/bec9ef74768dd201881cd4e54cd0385e87caae27/verl/utils/reward_score/geo3k.py
"""

import re
from typing import TYPE_CHECKING

import torch
from mathruler.grader import extract_boxed_content, grade_answer

from roll.configs.worker_config import WorkerConfig
from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.decorator import Dispatch, register
from roll.distributed.scheduler.protocol import DataProto
from roll.models.model_providers import default_tokenizer_provider

if TYPE_CHECKING:
    from roll.pipeline.rlvr.rlvr_config import RLVRConfig


DEFAULT_FORMAT_SCORE = 0.1
GEO3K_FORMAT_PATTERN = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)


def format_reward(predict_str: str) -> float:
    """Return one when the response follows the Geo3K think-and-box format."""
    return 1.0 if GEO3K_FORMAT_PATTERN.fullmatch(predict_str) else 0.0


def accuracy_reward(predict_str: str, ground_truth: str, use_boxed: bool = True) -> float:
    """Grade a predicted Geo3K answer with MathRuler."""
    answer = extract_boxed_content(predict_str) if use_boxed else predict_str
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def compute_score(
    predict_str: str,
    ground_truth: str,
    use_boxed: bool = True,
    format_score: float = DEFAULT_FORMAT_SCORE,
) -> float:
    """Compute the verl-compatible Geo3K composite reward."""
    accuracy = accuracy_reward(predict_str=predict_str, ground_truth=ground_truth, use_boxed=use_boxed)
    return (1.0 - format_score) * accuracy + format_score * format_reward(predict_str)


class Geo3kRewardWorker(Worker):
    """Compute MathRuler accuracy and verl-compatible format shaping for Geo3K."""

    def __init__(self, worker_config: WorkerConfig) -> None:
        super().__init__(worker_config=worker_config)
        self.rank_info.dp_rank = self.rank_info.rank
        self.rank_info.dp_size = self.rank_info.world_size
        self.tokenizer = None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self, pipeline_config: "RLVRConfig") -> None:
        """Initialize the actor tokenizer used to decode generated responses."""
        super().initialize(pipeline_config)
        self.tokenizer = default_tokenizer_provider(model_args=pipeline_config.actor_train.model_args)

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE, clear_cache=False)
    def compute_rewards(self, data: DataProto) -> DataProto:
        """Return correctness scores and composite sequence-level rewards."""
        if self.tokenizer is None:
            raise RuntimeError("Geo3kRewardWorker must be initialized before computing rewards.")

        response_ids = data.batch["responses"]
        responses = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        ground_truths = data.non_tensor_batch["ground_truth"]

        accuracy_values = []
        format_values = []
        reward_values = []
        for response, ground_truth in zip(responses, ground_truths, strict=True):
            accuracy = accuracy_reward(predict_str=response, ground_truth=ground_truth)
            format_value = format_reward(response)
            reward = (1.0 - DEFAULT_FORMAT_SCORE) * accuracy + DEFAULT_FORMAT_SCORE * format_value
            accuracy_values.append(accuracy)
            format_values.append(format_value)
            reward_values.append(reward)

        scores = torch.tensor(accuracy_values, dtype=torch.float32, device=response_ids.device)
        response_level_rewards = torch.tensor(reward_values, dtype=torch.float32, device=response_ids.device)
        token_level_rewards = torch.zeros_like(response_ids, dtype=torch.float32)

        batch_size = len(reward_values)
        metrics = {
            "geo3k/accuracy_mean": sum(accuracy_values) / batch_size if batch_size else 0.0,
            "geo3k/format_mean": sum(format_values) / batch_size if batch_size else 0.0,
            "geo3k/reward_mean": sum(reward_values) / batch_size if batch_size else 0.0,
        }
        return DataProto.from_dict(
            tensors={
                "scores": scores,
                "response_level_rewards": response_level_rewards,
                "token_level_rewards": token_level_rewards,
            },
            meta_info={"metrics": metrics},
        )
