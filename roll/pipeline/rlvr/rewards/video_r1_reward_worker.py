"""
reference: https://github.com/tulerfeng/Video-R1/blob/main/src/r1-v/src/open_r1/grpo.py
"""

import re
import torch
from collections import defaultdict
from typing import Dict, List

from rouge_score import rouge_scorer

from roll.configs.worker_config import WorkerConfig
from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.decorator import Dispatch, register
from roll.distributed.scheduler.protocol import DataProto
from roll.models.model_providers import default_tokenizer_provider
from roll.utils.logging import get_logger


logger = get_logger()


def extract_boxed_content(text):
    pattern = r"\\boxed{(.*?)}"
    matches = re.findall(pattern, text)
    if len(matches) > 0:
        return matches[-1]
    else:
        return text


def accuracy_reward(completion, solution, question_type="acc"):
    def extract_answer(text):
        pattern = r"<answer>\s*(.*?)\s*</answer>"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text

    def normalize_number(num_str):
        try:
            num_str = num_str.replace(",", "")
            return float(num_str)
        except Exception as e:
            print(f"Error converting '{num_str}' to float: {e}")
            return None

    def wer(reference, hypothesis):
        ref_words = reference.split()
        hyp_words = hypothesis.split()
        m = len(ref_words)
        n = len(hyp_words)
        d = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            d[i][0] = i
        for j in range(n + 1):
            d[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref_words[i - 1] == hyp_words[j - 1]:
                    d[i][j] = d[i - 1][j - 1]
                else:
                    d[i][j] = 1 + min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1])
        return d[m][n] / max(1, m)

    def compute_rouge_score(reference, hypothesis, use_stemmer=True):
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=use_stemmer)
        scores = scorer.score(reference, hypothesis)
        average_fmeasure = (scores["rouge1"].fmeasure + scores["rouge2"].fmeasure + scores["rougeL"].fmeasure) / 3
        return average_fmeasure

    output_ans = extract_answer(completion)
    gt_ans = extract_answer(solution)
    if question_type == "multiple choice":
        output_ans = extract_boxed_content(output_ans)
        gt_ans = extract_boxed_content(gt_ans)
        reward = 1.0 if (output_ans.strip().strip(".")[:1] == gt_ans.strip().strip(".")[:1]) else 0.0
    elif question_type == "free-form":
        score = compute_rouge_score(gt_ans, output_ans)
        reward = max(0.0, min(1.0, score))
    elif question_type == "numerical":
        gt_has_decimal = ("." in gt_ans) or ("," in gt_ans)
        out_has_decimal = ("." in output_ans) or ("," in output_ans)
        if gt_has_decimal != out_has_decimal:
            reward = 0.0
        else:
            gt_number = normalize_number(gt_ans)
            out_number = normalize_number(output_ans)
            if gt_number is None or out_number is None:
                reward = 0.0
            else:
                reward = 1.0 if round(gt_number, 2) == round(out_number, 2) else 0.0
    elif question_type == "OCR":
        error_rate = wer(gt_ans, output_ans)
        reward = 1 - error_rate
        reward = max(0.0, min(1.0, reward))
    elif question_type == "regression":
        gt_number = normalize_number(gt_ans)
        out_number = normalize_number(output_ans)
        if gt_number is None or out_number is None:
            reward = 0.0
        rel_diff = (abs(out_number - gt_number) + 1e-9) / (abs(gt_number) + 1e-9)
        rel_diff = min(1.0, max(0.0, rel_diff))
        reward = 1 - rel_diff
    else:
        reward = 0.0

    return reward


def format_reward(completion):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    match = re.fullmatch(pattern, completion, re.DOTALL)
    return 1.0 if match else 0.0


class VideoR1RewardWorker(Worker):
    def __init__(self, worker_config: WorkerConfig):
        super().__init__(worker_config=worker_config)
        self.rank_info.dp_rank = self.rank_info.rank
        self.rank_info.dp_size = self.rank_info.world_size

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self, pipeline_config):
        super().initialize(pipeline_config)
        self.tokenizer = default_tokenizer_provider(pipeline_config.actor_train.model_args)

    def judge_responses(self, prompts: List[Dict[str, str]]) -> List[Dict[str, float]]:
        """Judge a batch of responses using vLLM"""
        scores = []
        for prompt in prompts:
            model_response = prompt["response"]
            ground_truth = prompt["ground_truth"]
            question_type = prompt["question_type"]
            acc_reward = accuracy_reward(model_response, ground_truth, question_type=question_type)
            fmt_reward = format_reward(model_response)
            overall = acc_reward + fmt_reward
            scores.append({"overall": overall, "acc_reward": acc_reward, "fmt_reward": fmt_reward})
        return scores

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE, clear_cache=False)
    def compute_rewards(self, data: DataProto):
        # some initialization need after work.initialize
        response_ids = data.batch["responses"]
        response_length = data.batch["response_mask"].sum(dim=-1)
        judge_prompts = []
        for i in range(len(data)):
            valid_response_ids = response_ids[i][: response_length[i]]
            # NOTE: `re.fullmatch(pattern)` in format_reward seems not suitable for qwen3-vl
            # thinking model, `<think>` is the prompt ending suffix thus adjust format_reward
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            ground_truth = (
                data.non_tensor_batch["ground_truth"][i]
                if "ground_truth" in data.non_tensor_batch
                else data.non_tensor_batch["answer"][i]
            )
            question_type = data.non_tensor_batch["reward_model"][i]
            judge_prompt = {
                "question_type": question_type,
                "ground_truth": ground_truth,
                "response": response_str,
            }
            judge_prompts.append(judge_prompt)
        scores = self.judge_responses(judge_prompts)
        rewards = []
        reward_metrics = defaultdict(list)
        for i, score in enumerate(scores):
            rewards.append(float(score["overall"]))
            for key, value in score.items():
                reward_metrics[key].append(value)
        scores_tensor = torch.tensor(rewards, dtype=torch.float16)
        token_level_rewards = torch.zeros_like(response_ids, dtype=torch.float16)
        response_level_rewards = scores_tensor
        output = DataProto.from_dict(
            tensors={
                "token_level_rewards": token_level_rewards,
                "response_level_rewards": response_level_rewards,
                "scores": scores_tensor,
            }
        )
        logger.debug(f"{judge_prompt=}, {reward_metrics=}")
        output.meta_info = {"metrics": reward_metrics}
        return output
