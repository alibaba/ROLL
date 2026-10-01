from __future__ import annotations

import numpy as np
import pytest
import torch

from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.rlvr.rewards.gsm8k_math_reward_worker import (
    GSM8K_DATA_SOURCE,
    GSM8KMathRewardWorker,
    compute_gsm8k_score,
    compute_math_score,
    compute_score,
    extract_gsm8k_solution,
    normalize_math_answer,
)


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("reasoning\n#### 12", "12"),
        ("reasoning\n#### 1,234", "1234"),
        ("first #### 1\ncorrected #### -2.5", "-2.5"),
        ("the answer is 12", None),
        ("#### .", "."),
    ],
)
def test_extract_gsm8k_solution_strict_format(response: str, expected: str | None) -> None:
    assert extract_gsm8k_solution(response) == expected


def test_extract_gsm8k_solution_only_scans_last_300_characters() -> None:
    assert extract_gsm8k_solution("#### 12" + "x" * 300) is None
    assert extract_gsm8k_solution("x" * 300 + "#### 12") == "12"


def test_compute_gsm8k_score() -> None:
    assert compute_gsm8k_score("work\n#### 1,234", "1234") == 1.0
    assert compute_gsm8k_score("1234", "1234") == 0.0
    assert compute_gsm8k_score("work\n#### 1235", "1234") == 0.0


@pytest.mark.parametrize(
    ("response", "ground_truth"),
    [
        (r"The result is \boxed{\frac{1}{2}}.", "0.5"),
        (r"The result is \boxed{\sqrt3}.", r"\sqrt{3}"),
        (r"First \boxed{1}, finally \boxed{42}.", "42"),
        (r"The result is \boxed{x = 7}.", "7"),
    ],
)
def test_compute_math_score_accepts_normalized_equivalents(response: str, ground_truth: str) -> None:
    assert compute_math_score(response, ground_truth) == 1.0


def test_compute_math_score_rejects_incorrect_or_unboxed_answers() -> None:
    assert compute_math_score(r"The result is \boxed{3}.", "4") == 0.0
    assert compute_math_score("The result is 4.", "4") == 0.0


def test_normalize_math_answer_repairs_common_latex_forms() -> None:
    assert normalize_math_answer(r"\dfrac12") == r"\frac{1}{2}"
    assert normalize_math_answer(r"\left( 2 \right)") == "(2)"


def test_compute_score_dispatches_by_data_source() -> None:
    assert compute_score("#### 8", "8", GSM8K_DATA_SOURCE) == 1.0
    assert compute_score(r"\boxed{8}", "8", "DigitalLearningGmbH/MATH-lighteval") == 1.0
    with pytest.raises(NotImplementedError, match="Unsupported math reward data source"):
        compute_score("8", "8", "unknown")


def test_reward_worker_mixed_batch_contract_and_order() -> None:
    class FakeTokenizer:
        def batch_decode(self, response_ids: torch.Tensor, skip_special_tokens: bool) -> list[str]:
            assert response_ids.shape == (4, 2)
            assert skip_special_tokens is True
            return ["work\n#### 2", r"work \boxed{\frac12}", "malformed 3", r"work \boxed{7}"]

    math_source = "DigitalLearningGmbH/MATH-lighteval"
    data = DataProto.from_dict(
        tensors={"responses": torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])},
        non_tensors={
            "ground_truth": np.asarray(["2", r"\frac{1}{2}", "3", "8"], dtype=object),
            "data_source": np.asarray([GSM8K_DATA_SOURCE, math_source, GSM8K_DATA_SOURCE, math_source], dtype=object),
        },
    )
    worker = object.__new__(GSM8KMathRewardWorker)
    worker.tokenizer = FakeTokenizer()
    compute_rewards = getattr(worker.compute_rewards, "__wrapped__", worker.compute_rewards)
    output = compute_rewards(data)

    assert output.batch["scores"].tolist() == [1.0, 1.0, 0.0, 0.0]
    assert output.batch["response_level_rewards"].tolist() == [1.0, 1.0, 0.0, 0.0]
    assert output.batch["scores"].dtype == torch.float32
    assert output.batch["token_level_rewards"].dtype == torch.float32
    assert output.batch["token_level_rewards"].shape == data.batch["responses"].shape
    assert torch.count_nonzero(output.batch["token_level_rewards"]).item() == 0
    assert output.meta_info["metrics"]["math/accuracy_mean"] == pytest.approx(0.5)
