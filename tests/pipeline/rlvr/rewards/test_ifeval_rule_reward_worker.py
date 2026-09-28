import json

import torch

from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.rlvr.rewards.ifeval_rule_reward_worker import (
    GeneralRuleRewardWorker,
    get_repetition_penalty_reward,
)


class _FakeTokenizer:
    """Trivial codepoint-per-id mapping; compute_rewards only decodes ids to text."""

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(int(i)) for i in ids.tolist() if int(i) != 0)

    def batch_decode(self, batch_ids, skip_special_tokens=False):
        return [self.decode(ids, skip_special_tokens) for ids in batch_ids]


def _encode(text, width=8):
    ids = [ord(c) for c in text]
    assert len(ids) <= width
    return torch.tensor(ids + [0] * (width - len(ids)), dtype=torch.long)


def _build_worker():
    # Bypasses __init__'s real tokenizer load, unrelated to compute_rewards' control flow.
    worker = GeneralRuleRewardWorker.__new__(GeneralRuleRewardWorker)
    worker.tokenizer = _FakeTokenizer()
    worker.repetition_penalty_reward_fn = get_repetition_penalty_reward(ngram_size=3, max_penalty=-0.5)
    return worker


def test_missing_func_name_does_not_break_batch_shape():
    # Sample 0 has no usable func_name; samples 1-2 are ordinary ifeval checks.
    worker = _build_worker()
    responses = torch.stack([_encode("aa"), _encode("bb"), _encode("cc")])
    non_tensors = {
        "prompt": ["p0", "p1", "p2"],
        "ground_truth": [
            json.dumps({"func_name": "does_not_exist"}),
            json.dumps({"func_name": "verify_keywords", "keyword_list": ["bb"]}),
            json.dumps({"func_name": "verify_keywords", "keyword_list": ["cc"]}),
        ],
        "tag": ["ifeval", "ifeval", "ifeval"],
    }
    data = DataProto.from_dict(tensors={"responses": responses}, non_tensors=non_tensors)

    output = worker.compute_rewards(data)

    scores = output.batch["scores"].tolist()
    assert scores == [0.0, 1.0, 1.0]


def test_ifeval_function_exception_yields_false_not_a_crash():
    # verify_keyword_frequency needs "N"; omitting it makes the real call raise.
    worker = _build_worker()
    responses = torch.stack([_encode("aa")])
    non_tensors = {
        "prompt": ["p0"],
        "ground_truth": [json.dumps({"func_name": "verify_keyword_frequency", "word": "aa"})],
        "tag": ["ifeval"],
    }
    data = DataProto.from_dict(tensors={"responses": responses}, non_tensors=non_tensors)

    output = worker.compute_rewards(data)

    assert output.batch["scores"].tolist() == [0.0]


def test_unknown_tag_does_not_reuse_previous_result():
    # Sample 0 is a passing ifeval check; sample 1 carries a non-ifeval tag.
    worker = _build_worker()
    responses = torch.stack([_encode("aa"), _encode("zz")])
    non_tensors = {
        "prompt": ["p0", "p1"],
        "ground_truth": [
            json.dumps({"func_name": "verify_keywords", "keyword_list": ["aa"]}),
            json.dumps({}),
        ],
        "tag": ["ifeval", "some_other_task"],
    }
    data = DataProto.from_dict(tensors={"responses": responses}, non_tensors=non_tensors)

    output = worker.compute_rewards(data)

    scores = output.batch["scores"].tolist()
    assert scores == [1.0, 0.0]
