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


def single_choice_reward(response, ground_truth):
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

    reward = reward_dict["loose"]

    return extracted_answer, reward, format_flag, correct_flag


class GeneralValRuleRewardWorker(Worker):
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

        scores = []
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

            if tag in ["ceval", "race_high", "mmlu_pro", "commonsense_qa"]:
                extracted_answer, reward, format_flag, correct_flag = single_choice_reward(answer_text, ground_truth)
                format_value = 1 if format_flag else 0
                correct_value = 1 if correct_flag else 0
                # score should be 0 or 1, indicating model response correctness
                if reward > 0:
                    score = 1.0
                else:
                    score = 0.0

            # Store to scores
            scores.append(score)
            format_values.append(format_value)
            correct_values.append(correct_value)
            try:
                outputs = json.dumps(
                    {
                        "tag": tag,
                        "score": score,
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

        scores = torch.tensor(scores, dtype=torch.float16)

        format_values = torch.tensor(format_values, dtype=torch.float16)
        correct_values = torch.tensor(correct_values, dtype=torch.float16)

        token_level_rewards = torch.zeros_like(data.batch["responses"], dtype=torch.float16)
        response_level_rewards = torch.zeros_like(scores, dtype=torch.float16)
        # 5) Pack these tensors into the same dictionary
        output_tensors = {
            "scores": scores,
            # "format_values": format_values,
            # "correct_values": correct_values
            "token_level_rewards": token_level_rewards,
            "response_level_rewards": response_level_rewards,
        }

        # 6) Use DataProto.from_dict(...) to construct return value
        output = DataProto.from_dict(tensors=output_tensors)
        return output
