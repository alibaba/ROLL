# Import necessary libraries and modules
from functools import partial
from typing import Optional, Union, Iterator
import json
import re

import ray
import torch
from codetiming import Timer

from roll.configs.worker_config import WorkerConfig
from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.decorator import Dispatch, register
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.strategy.factory import create_strategy
from roll.distributed.strategy.strategy import InferenceStrategy, TrainStrategy

from roll.models.model_providers import default_reward_model_provider, default_tokenizer_provider

from typing import Union, Dict, List

from roll.utils.logging import get_logger

logger = get_logger()  # Get logger instance


def get_response_length_reward(min_len, max_len):
    def response_length_reward(response):
        if len(response) < min_len:
            return -0.5
        elif len(response) > max_len:
            return 0.0
        else:
            return -0.5 * (max_len - len(response)) / (max_len - min_len)

    return response_length_reward


def get_repetition_penalty_reward(ngram_size: int, max_penalty: float):
    """
    Reference implementation from: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py
    """
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    def zipngram(text: str, ngram_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])

    def repetition_penalty_reward(response, **kwargs) -> float:
        """
        ref implementation: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py
        """

        if response == "" or len(response.split()) < ngram_size:
            return 0.0

        ngrams = set()
        total = 0
        for ng in zipngram(response, ngram_size):
            ngrams.add(ng)
            total += 1

        scaling = 1 - len(ngrams) / total
        reward = scaling * max_penalty
        return reward

    return repetition_penalty_reward


def extract_after_last_think(input_string, end_think="</think>"):
    """
    Extract content after the last "end_think" tag in the input string,
    and remove all newlines at the beginning of the result string.

    Args:
    input_string: Original string.

    Returns:
    Extracted and processed string. Returns empty string if "end_think" tag not found.
    """
    last_index = input_string.rfind(end_think)

    if last_index == -1:
        return input_string  # Or return None or original string as needed

    start_pos = last_index + len(end_think)
    extracted_part = input_string[start_pos:]
    cleaned_part = extracted_part.lstrip("\n")

    return cleaned_part


def crossthinkqa_reward_fn(response, ground_truth, reward_type):

    format_flag = False
    correct_flag = False

    # 1. format
    # Find all \\boxed{} matches
    box_matches = re.findall(r"\\boxed\{([^}]+)\}", response)
    # If no \\boxed{} found, return None
    if not box_matches:
        lower_response = response.lower()
        last_answer_index = lower_response.rfind("answer is")
        if last_answer_index == -1:
            extracted_answer = response
        else:
            extracted_answer = response[last_answer_index + 9 :]
    # Get content of the last \\boxed{}
    else:
        format_flag = True
        extracted_answer = box_matches[-1]

    # 2. correct
    for char in extracted_answer:
        if char.isupper():
            if char == ground_truth[0]:
                correct_flag = True
            break

    if correct_flag and format_flag:
        loose_reward = 1.0
        soft_reward = 1.0
        strict_reward = 1.0
    elif correct_flag and not format_flag:
        loose_reward = 1.0
        soft_reward = 0.5
        strict_reward = -1.0
    elif not correct_flag and format_flag:
        loose_reward = -1.0
        soft_reward = -0.5
        strict_reward = -1.0
    else:
        loose_reward = -1.0
        soft_reward = -1.0
        strict_reward = -1.0

    reward_dict = {"loose": loose_reward, "soft": soft_reward, "strict": strict_reward}

    reward = reward_dict[reward_type]

    return extracted_answer, reward, format_flag, correct_flag


class CrossThinkQARuleRewardWorker(Worker):
    """
    Example Reward Worker for executing ifeval validation and putting each func result into output.tensors.
    In this example, ground_truths string
    """

    def __init__(self, worker_config: WorkerConfig):
        super().__init__(worker_config=worker_config)
        self.rank_info.dp_rank = self.rank_info.rank
        self.rank_info.dp_size = self.rank_info.world_size
        self.tokenizer = default_tokenizer_provider(model_args=self.worker_config.model_args)
        self.strategy: Optional[Union[InferenceStrategy, TrainStrategy]] = None

        self.repetition_penalty_reward_fn = get_repetition_penalty_reward(ngram_size=3, max_penalty=-0.5)
        self.response_length_reward_fn = get_response_length_reward(min_len=100, max_len=400)

        self.reward_type = worker_config.reward_type
        self.response_length_penalty_coef = worker_config.response_length_penalty_coef

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self, pipeline_config):
        pass

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE)
    def compute_rewards(self, data: DataProto):
        response_text_list = self.tokenizer.batch_decode(data.batch["responses"], skip_special_tokens=False)
        batch_size = len(response_text_list)

        prompts = data.non_tensor_batch["prompt"]
        ground_truths = data.non_tensor_batch["ground_truth"]
        tags = data.non_tensor_batch["tag"]

        crossthinkqa_rewards = []
        scores = []
        repetition_penalty_rewards = []
        response_length_rewards = []
        format_values = []  # Format-correct values (strictly require \boxed{})
        correct_values = []  # Answer-correct values (use more lenient extraction rules)

        for i, (resp_tokens, ground_truth, tag, prompt) in enumerate(
            zip(data.batch["responses"], ground_truths, tags, prompts)
        ):
            ori_resp_text = self.tokenizer.decode(resp_tokens, skip_special_tokens=False)
            resp_text_without_sptoken = (
                ori_resp_text.replace("<|endoftext|>", "").replace("<pad>", "").replace("<|im_end|>", "")
            )
            answer_text = extract_after_last_think(resp_text_without_sptoken)

            extracted_answer, crossthinkqa_reward, format_flag, correct_flag = crossthinkqa_reward_fn(
                answer_text, ground_truth, self.reward_type
            )
            repetition_penalty_reward = self.repetition_penalty_reward_fn(resp_text_without_sptoken)
            response_length_reward = self.response_length_reward_fn(resp_tokens)

            format_value = 1 if format_flag else 0
            correct_value = 1 if correct_flag else 0

            # score should be 0 or 1, indicating model response correctness
            if crossthinkqa_reward > 0:
                score = 1.0
            else:
                score = 0.0

            # Store to crossthinkqa_rewards
            crossthinkqa_rewards.append(crossthinkqa_reward)
            scores.append(score)
            repetition_penalty_rewards.append(repetition_penalty_reward)
            response_length_rewards.append(response_length_reward)
            format_values.append(format_value)
            correct_values.append(correct_value)

            try:
                outputs = json.dumps(
                    {
                        "crossthinkqa_reward": crossthinkqa_reward,
                        "repetition_penalty_reward": repetition_penalty_reward,
                        "response_length_reward": response_length_reward,
                        "format_flag": format_flag,
                        "correct_flag": correct_flag,
                        "prompt": str(prompt),
                        "response": str(extracted_answer),
                        "ground_truth": str(ground_truth),
                        "ori_response": str(resp_text_without_sptoken),
                    },
                    ensure_ascii=False,
                )
                self.logger.debug(outputs)
            except Exception as e:
                self.logger.error(f"answer check except: {e}")

        token_level_rewards = torch.zeros_like(data.batch["responses"], dtype=torch.float16)
        crossthinkqa_rewards = torch.tensor(crossthinkqa_rewards, dtype=torch.float16)
        scores = torch.tensor(scores, dtype=torch.float16)
        repetition_penalty_rewards = torch.tensor(repetition_penalty_rewards, dtype=torch.float16)
        response_length_rewards = torch.tensor(response_length_rewards, dtype=torch.float16)

        response_level_rewards = (
            crossthinkqa_rewards
            + repetition_penalty_rewards
            + self.response_length_penalty_coef * response_length_rewards
        )

        format_values = torch.tensor(format_values, dtype=torch.float16)
        correct_values = torch.tensor(correct_values, dtype=torch.float16)

        # 5) Pack these tensors into the same dictionary
        # TODO: Whether different reward workers' outputs need unified output, or if there's an adaptive approach to avoid modifying every worker when adding new monitoring metrics
        output_tensors = {
            "token_level_rewards": token_level_rewards,
            "response_level_rewards": response_level_rewards,
            "scores": scores,
            # "repetition_penalty_rewards": repetition_penalty_rewards,
            # "response_length_rewards": response_length_rewards,
            # "format_values": format_values,
            # "correct_values": correct_values
        }

        # 6) Use DataProto.from_dict(...) to construct return value
        output = DataProto.from_dict(tensors=output_tensors)
        return output
