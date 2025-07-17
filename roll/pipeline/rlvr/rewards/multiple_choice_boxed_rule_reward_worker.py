# 导入必要的库和模块
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

logger = get_logger()  # 获取日志记录器实例

def extract_after_last_think(input_string, end_think="</think>"):
    """
    提取输入字符串中最后一个"end_think"标签之后的内容，
    并移除结果字符串开头的所有换行符。

    Args:
    input_string: 原始字符串。

    Returns:
    提取并处理后的字符串。如果未找到"end_think"标签，则返回空字符串。
    """
    last_index = input_string.rfind(end_think)

    if last_index == -1:
        return input_string  # 或者根据需要返回 None 或原始字符串

    start_pos = last_index + len(end_think)
    extracted_part = input_string[start_pos:]
    cleaned_part = extracted_part.lstrip("\n")

    return cleaned_part


def multiple_choice_boxed_reward_fn(response, ground_truth, reward_type=None):

    format_flag = False
    correct_flag = False

    # 1. format
    # 找到所有的 \\boxed{} 匹配项
    box_matches = re.findall(r"\\boxed\{([^}]+)\}", response)
    # 如果没有找到 \\boxed{} 则返回 None
    if not box_matches:
        lower_response = response.lower()
        last_answer_index = lower_response.rfind("answer is")
        if last_answer_index == -1:
            extracted_answer = response
        else:
            extracted_answer = response[last_answer_index + 9 :]
    # 获取最后一个 \\boxed{} 的内容
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
        reward = 1.0
    else:
        reward = 0

    return extracted_answer, reward, format_flag, correct_flag


class MultipleChoiceBoxedRuleRewardWorker(Worker):
    def __init__(self, worker_config: WorkerConfig):
        super().__init__(worker_config=worker_config)
        self.rank_info.dp_rank = self.rank_info.rank
        self.rank_info.dp_size = self.rank_info.world_size
        self.tokenizer = default_tokenizer_provider(model_args=self.worker_config.model_args)
        self.strategy: Optional[Union[InferenceStrategy, TrainStrategy]] = None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self, pipeline_config):
        pass

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE, clear_cache=False)
    def compute_rewards(self, data: DataProto):
        response_text_list = self.tokenizer.batch_decode(data.batch["responses"], skip_special_tokens=True)
        batch_size = len(response_text_list)

        prompts = data.non_tensor_batch["prompt"]
        ground_truths = data.non_tensor_batch["ground_truth"]
        tags = data.non_tensor_batch["tag"]

        multiple_choice_boxed_rewards = []
        scores = []

        for i, (resp_tokens, ground_truth, tag, prompt) in enumerate(
            zip(data.batch["responses"], ground_truths, tags, prompts)
        ):
            ori_resp_text = self.tokenizer.decode(resp_tokens, skip_special_tokens=False)
            resp_text_without_sptoken = (
                ori_resp_text.replace("<|endoftext|>", "").replace("<pad>", "").replace("<|im_end|>", "")
            )
            answer_text = extract_after_last_think(resp_text_without_sptoken)

            extracted_answer, multiple_choice_boxed_reward, format_flag, correct_flag = multiple_choice_boxed_reward_fn(
                answer_text, ground_truth
            )

            # score应该为0或者1，标志模型回复的对错
            if multiple_choice_boxed_reward > 0.5:
                score = 1.0
            else:
                score = 0.0

            # 存到 multiple_choice_boxed_rewards
            multiple_choice_boxed_rewards.append(multiple_choice_boxed_reward)
            scores.append(score)

            try:
                outputs = json.dumps(
                    {
                        "multiple_choice_boxed_reward": multiple_choice_boxed_reward,
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
        scores = torch.tensor(scores, dtype=torch.float16)
        response_level_rewards = torch.zeros_like(scores, dtype=torch.float16)
        # 5) 将这些张量打包进同一个字典
        # TODO: 不同的reward worker的output是否需要统一output，或者有没有自适应的办法，避免在新增监控量时每个worker都需要修改
        output_tensors = {
            "token_level_rewards": token_level_rewards,
            "response_level_rewards": response_level_rewards,
            "scores": scores,
        }

        # 6) 用 DataProto.from_dict(...) 构造返回值
        output = DataProto.from_dict(tensors=output_tensors)
        return output
