from functools import partial
from typing import Optional, Union, Iterator
import json
import re

import ray
import torch
from codetiming import Timer
from tqdm import tqdm
import signal
import multiprocessing

from roll.configs.worker_config import WorkerConfig
from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.decorator import Dispatch, register
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.strategy.factory import create_strategy
from roll.distributed.strategy.strategy import InferenceStrategy, TrainStrategy
from roll.models.model_providers import default_reward_model_provider, default_tokenizer_provider
from roll.utils.context_managers import state_offload_manger
from math_verify import parse, verify

class timeout:
    def __init__(self, seconds=1, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def _hf_verify_math_sample(response, answer, output_queue):
    gold_answer = parse(answer)
    exect_answer = parse(response)
    if gold_answer is None or exect_answer is None:
        output_queue.put((False, exect_answer, gold_answer))
    else:
        ans = verify(gold_answer, exect_answer)
        output_queue.put((ans, exect_answer, gold_answer))


def hf_verify_math_sample(answer_a, answer_b, timeout_sec=5.0):
    """
    在多进程中调用 hf math verify,
    以在超时时间内完不成时返回 False.
    """
    output_queue = multiprocessing.Queue()

    p = multiprocessing.Process(target=_hf_verify_math_sample, args=(answer_a, answer_b, output_queue))
    p.start()
    p.join(timeout_sec)

    if p.is_alive():
        # 超时 -> 杀掉子进程, 返回 False
        p.terminate()
        p.join()
        return False, "", ""

    if not output_queue.empty():
        return output_queue.get()
    else:
        return False, "", ""


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


def long_block_penalty_reward_fn(text: str, max_length: int = 100) -> float:
    max_block_len = max([len(i) for i in text.split(" ")])
    reward = -float(max_block_len > max_length)
    return reward


def format_reward_fn(text: str, pattern: Optional[str] = r"^<think>.*?</think>.*?<answer>.*?</answer>$"):
    # pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    if pattern is None:
        pattern: str = r"^<think>.*?</think>.*?<answer>.*?</answer>$"
    matche = re.match(pattern, text, re.DOTALL | re.MULTILINE)
    reward = 0 if matche else -1
    return reward


class MathRuleRewardWorker(Worker):
    """
    (x)Reward Model 使用 AutoModelForSequenceClassification 协议
    面向math的rule reward model
    """

    def __init__(self, worker_config: WorkerConfig):
        super().__init__(worker_config=worker_config)
        self.rank_info.dp_rank = self.rank_info.rank
        self.rank_info.dp_size = self.rank_info.world_size
        self.tokenizer = default_tokenizer_provider(model_args=self.worker_config.model_args)
        self.strategy: Optional[Union[InferenceStrategy, TrainStrategy]] = None
        self.repetition_penalty_reward_fn = get_repetition_penalty_reward(ngram_size=3, max_penalty=-0.1)
        self.format_pattern = self.worker_config.format_pattern

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self, pipeline_config):
        pass

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE)
    def compute_rewards(self, data: DataProto):
        """
        return DataProto.from_dict(tensors={'rewards': rewards})
        """

        verify_answer = []
        repetition_penalty_rewards = []
        long_block_penalty_rewards = []
        response_length_rewards = []
        format_rewards = []
        # response_text_list = self.tokenizer.batch_decode(data.batch['responses'], skip_special_tokens=True)
        response_text_list = self.tokenizer.batch_decode(data.batch["responses"], skip_special_tokens=False)
        for response, answer in zip(response_text_list, data.non_tensor_batch["ground_truth"]):
            # print(f'answer outpus: {outputs}')
            # verify_answer.append(verify_math_sample(response, answer))
            response = response.replace("<|endoftext|>", "")
            response = response.replace("<|im_end|>", "")
            response = response.replace("<pad>", "")
            try:
                with timeout(5):
                    correct, extracted_response, extracted_ground_truth = hf_verify_math_sample(
                        response, f"${answer}$"
                    )
            except Exception as e:
                self.logger.error(f"timeout answer: {answer}, response: {response}")
                correct = False
                extracted_response = ""
                extracted_ground_truth = ""
            # TODO check Anser
            try:
                outputs = json.dumps(
                    {
                        "correct": correct,
                        "answer": str(answer),
                        "extracted_response": str(extracted_response),
                        "extracted_ground_truth": str(extracted_ground_truth),
                        "response": str(response),
                    }
                )
                self.logger.debug(f"answer check: {outputs}")
            except Exception as e:
                self.logger.error(f"answer check except: {e}")

            if correct:
                verify_answer.append(1)
            else:
                verify_answer.append(0)  # other?
            repetition_penalty_rewards.append(self.repetition_penalty_reward_fn(response))
            format_rewards.append(format_reward_fn(response, self.format_pattern))
            long_block_penalty_rewards.append(long_block_penalty_reward_fn(response))
            response_length_rewards.append(len(response) / 20000)
        token_level_rewards = torch.zeros_like(data.batch["responses"], dtype=torch.float16)
        response_length_rewards = torch.tensor(response_length_rewards, dtype=torch.float16)
        # format
        repetition_penalty_rewards = torch.tensor(repetition_penalty_rewards, dtype=torch.float16)
        long_block_penalty_rewards = torch.tensor(long_block_penalty_rewards, dtype=torch.float16)
        format_rewards = torch.tensor(format_rewards, dtype=torch.float16)

        # for log
        scores = torch.tensor(verify_answer, dtype=torch.float16)

        response_level_rewards = torch.tensor(
            verify_answer, dtype=torch.float16
        )  # + repetition_penalty_rewards + 0.1 * long_block_penalty_rewards #+ 0.5 * format_rewards

        output = DataProto.from_dict(
            tensors={
                "token_level_rewards": token_level_rewards,
                "response_level_rewards": response_level_rewards,
                # "long_block_penalty_rewards": long_block_penalty_rewards,
                # "response_length_rewards": response_length_rewards,
                # "repetition_penalty_reward": repetition_penalty_rewards,
                # "format_reward": format_rewards,
                "scores": scores,
            }
        )

        self.logger.debug(f"reward output: {output}, response_level_rewards: {response_level_rewards}")
        return output
