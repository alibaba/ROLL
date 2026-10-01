"""Rule-based reward worker for GSM8K and MATH."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch

from roll.configs.worker_config import WorkerConfig
from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.decorator import Dispatch, register
from roll.distributed.scheduler.protocol import DataProto
from roll.models.model_providers import default_tokenizer_provider

if TYPE_CHECKING:
    from roll.pipeline.rlvr.rlvr_config import RLVRConfig


GSM8K_DATA_SOURCE = "openai/gsm8k"
MATH_DATA_SOURCE = "DigitalLearningGmbH/MATH-lighteval"
_SOLUTION_CLIP_CHARS = 300


def extract_gsm8k_solution(solution_str: str) -> str | None:
    """Extract the final numeric answer from a GSM8K response."""
    if len(solution_str) > _SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]

    solutions = re.findall(r"#### (-?[0-9.,]+)", solution_str)
    return None if not solutions else solutions[-1].replace(",", "")


def compute_gsm8k_score(solution_str: str, ground_truth: str) -> float:
    """Return one for an exact GSM8K answer match, otherwise zero."""
    answer = extract_gsm8k_solution(solution_str)
    return float(answer is not None and answer == ground_truth)


def remove_boxed(value: str) -> str:
    """Remove the outer ``boxed`` command from an answer."""
    if "\\boxed " in value:
        prefix = "\\boxed "
        assert value[: len(prefix)] == prefix
        return value[len(prefix) :]

    prefix = "\\boxed{"
    assert value[: len(prefix)] == prefix and value[-1] == "}"
    return value[len(prefix) : -1]


def last_boxed_only_string(value: str) -> str | None:
    """Extract the last complete boxed expression, including nested braces."""
    index = value.rfind("\\boxed")
    if "\\boxed " in value:
        return "\\boxed " + value.split("\\boxed ")[-1].split("$")[0]
    if index < 0:
        index = value.rfind("\\fbox")
        if index < 0:
            return None

    right_brace_index = None
    open_braces = 0
    for position in range(index, len(value)):
        if value[position] == "{":
            open_braces += 1
        elif value[position] == "}":
            open_braces -= 1
            if open_braces == 0:
                right_brace_index = position
                break
    return None if right_brace_index is None else value[index : right_brace_index + 1]


def _fix_fracs(value: str) -> str:
    parts = value.split("\\frac")
    result = parts[0]
    for part in parts[1:]:
        result += "\\frac"
        if part[0] == "{":
            result += part
            continue
        try:
            assert len(part) >= 2
        except AssertionError:
            return value
        numerator, denominator = part[0], part[1]
        remainder = part[2:] if len(part) > 2 else ""
        if denominator != "{":
            result += "{" + numerator + "}{" + denominator + "}" + remainder
        else:
            result += "{" + numerator + "}" + denominator + remainder
    return result


def _fix_a_slash_b(value: str) -> str:
    if len(value.split("/")) != 2:
        return value
    numerator, denominator = value.split("/")
    try:
        numerator, denominator = int(numerator), int(denominator)
        assert value == f"{numerator}/{denominator}"
        return f"\\frac{{{numerator}}}{{{denominator}}}"
    except (AssertionError, ValueError):
        return value


def _remove_right_units(value: str) -> str:
    if "\\text{ " not in value:
        return value
    parts = value.split("\\text{ ")
    assert len(parts) == 2
    return parts[0]


def _fix_sqrt(value: str) -> str:
    if "\\sqrt" not in value:
        return value
    parts = value.split("\\sqrt")
    result = parts[0]
    for part in parts[1:]:
        if part[0] != "{":
            result += "\\sqrt{" + part[0] + "}" + part[1:]
        else:
            result += "\\sqrt" + part
    return result


def normalize_math_answer(value: str) -> str:
    """Normalize common equivalent answer formats used by the MATH dataset."""
    value = value.replace("\n", "").replace("\\!", "").replace("\\\\", "\\")
    value = value.replace("tfrac", "frac").replace("dfrac", "frac")
    value = value.replace("\\left", "").replace("\\right", "")
    value = value.replace("^{\\circ}", "").replace("^\\circ", "").replace("\\$", "")
    value = _remove_right_units(value)
    value = value.replace("\\\\%", "").replace("\\%", "")
    value = value.replace(" .", " 0.").replace("{.", "{0.")
    if not value:
        return value
    if value[0] == ".":
        value = "0" + value
    if len(value.split("=")) == 2 and len(value.split("=")[0]) <= 2:
        value = value.split("=")[1]
    value = _fix_sqrt(value).replace(" ", "")
    value = _fix_fracs(value)
    if value == "0.5":
        value = "\\frac{1}{2}"
    return _fix_a_slash_b(value)


def compute_math_score(solution_str: str, ground_truth: str) -> float:
    """Return one when the final boxed MATH answer matches after normalization."""
    try:
        boxed = last_boxed_only_string(solution_str)
        if boxed is None:
            return 0.0
        answer = remove_boxed(boxed)
        return float(normalize_math_answer(answer) == normalize_math_answer(ground_truth))
    except (AssertionError, IndexError, ValueError):
        return 0.0


def compute_score(solution_str: str, ground_truth: str, data_source: str) -> float:
    """Dispatch a response to its dataset-specific scorer."""
    if data_source == GSM8K_DATA_SOURCE:
        return compute_gsm8k_score(solution_str, ground_truth)
    if data_source == MATH_DATA_SOURCE:
        return compute_math_score(solution_str, ground_truth)
    raise NotImplementedError(f"Unsupported math reward data source: {data_source}")


class GSM8KMathRewardWorker(Worker):
    """Score mixed GSM8K and MATH batches using dataset-specific rules."""

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
        """Return FP32 sequence-level correctness rewards in input order."""
        if self.tokenizer is None:
            raise RuntimeError("GSM8KMathRewardWorker must be initialized before computing rewards.")

        response_ids = data.batch["responses"]
        responses = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        ground_truths = data.non_tensor_batch["ground_truth"]
        data_sources = data.non_tensor_batch["data_source"]

        reward_values = [
            compute_score(solution_str=response, ground_truth=ground_truth, data_source=data_source)
            for response, ground_truth, data_source in zip(responses, ground_truths, data_sources, strict=True)
        ]
        scores = torch.tensor(reward_values, dtype=torch.float32, device=response_ids.device)
        batch_size = len(reward_values)

        return DataProto.from_dict(
            tensors={
                "scores": scores,
                "response_level_rewards": scores.clone(),
                "token_level_rewards": torch.zeros_like(response_ids, dtype=torch.float32),
            },
            meta_info={
                "metrics": {
                    "math/accuracy_mean": sum(reward_values) / batch_size if batch_size else 0.0,
                }
            },
        )
