import numpy as np
import pytest
import torch

from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.rlvr.rewards.general_val_rule_reward_worker import (
    GeneralValRuleRewardWorker,
)


class StubTokenizer:
    """每个 response 只有一个 token，token id 即对应文本的下标。"""

    def __init__(self, texts):
        self.texts = texts

    def decode(self, token_ids, skip_special_tokens=False):
        return self.texts[int(token_ids[0])]

    def batch_decode(self, token_ids, skip_special_tokens=False):
        return [self.decode(ids, skip_special_tokens) for ids in token_ids]


def make_worker(texts):
    worker = GeneralValRuleRewardWorker.__new__(GeneralValRuleRewardWorker)
    worker.tokenizer = StubTokenizer(texts)
    return worker


def make_data(texts, ground_truths, tags):
    return DataProto.from_single_dict(
        {
            "responses": torch.arange(len(texts), dtype=torch.long).reshape(-1, 1),
            "prompt": np.array(["prompt"] * len(texts), dtype=object),
            "ground_truth": np.array(ground_truths, dtype=object),
            "tag": np.array(tags, dtype=object),
        }
    )


def test_compute_rewards_supported_tags():
    texts = ["the answer is \\boxed{A}", "the answer is \\boxed{C}", "no answer here"]
    worker = make_worker(texts)
    data = make_data(texts, ["A", "B", "D"], ["ceval", "mmlu_pro", "race_high"])

    output = worker.compute_rewards(data)

    assert output.batch["scores"].tolist() == [1.0, 0.0, 0.0]


def test_compute_rewards_rejects_unsupported_tag():
    texts = ["the answer is \\boxed{A}"]
    worker = make_worker(texts)
    data = make_data(texts, ["A"], ["gsm8k"])

    with pytest.raises(ValueError, match="gsm8k"):
        worker.compute_rewards(data)


def test_compute_rewards_does_not_reuse_previous_sample_score():
    # 未支持的 tag 不能沿用上一个样本的 score/extracted_answer，
    # 否则 val_correct 指标会被静默污染。
    texts = ["the answer is \\boxed{A}", "I have no idea."]
    worker = make_worker(texts)
    data = make_data(texts, ["A", "Z"], ["ceval", "gsm8k"])

    with pytest.raises(ValueError, match="gsm8k"):
        worker.compute_rewards(data)
