"""Unit tests for RewardDumpMixin (shared diffusion reward dumping).

Covered without any distributed setup, using a fake worker:
- gating truth table (empty type/path/idx, phase and step selection)
- mandatory jsonl schema + per-reward file naming (multi-reward isolation)
- reward-specific extra_columns, including numpy / nested values
- column length mismatch fails loudly instead of silently padding
- reserved keys cannot be shadowed
- directory layout shared with the pipeline helpers
"""

import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from tensordict import TensorDict

from roll.distributed.scheduler.decorator import BIND_WORKER_METHOD_FLAG
from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.diffusion.rewards.dump_mixin import (
    RewardDumpMixin,
    is_dump_enabled,
    resolve_dump_base_dir,
    resolve_dump_step_dir,
    to_jsonable,
)


class _FakeWorker(RewardDumpMixin):
    """Minimal stand-in for a reward Worker: only what the mixin touches."""

    def __init__(self, pipeline_config, reward_name="ocr_reward", rank=0):
        self.pipeline_config = pipeline_config
        self.worker_config = SimpleNamespace(name=reward_name)
        self.rank = rank
        self.logger = SimpleNamespace(
            info=lambda *a, **k: None,
            exception=lambda *a, **k: None,
        )


def _pipeline_config(tmp_path, types=("train",), idx=(0, 1, 2), run_name="run-x"):
    return SimpleNamespace(
        dump_step_output_path=str(tmp_path),
        dump_step_output_type=list(types),
        dump_step_output_idx=list(idx),
        dump_run_name=run_name,
        exp_name="exp",
    )


def _data(global_step=1, is_val=False, batch_size=2, prompts=None, seeds=None):
    non_tensor = {}
    if prompts is not None:
        non_tensor["prompt"] = np.array(prompts, dtype=object)
    if seeds is not None:
        non_tensor["rollout_seed"] = np.array(seeds, dtype=object)
    data = DataProto(
        batch=TensorDict({"dummy": torch.zeros(batch_size)}, batch_size=[batch_size]),
        non_tensor_batch=non_tensor,
    )
    data.meta_info = {"global_step": global_step, "is_val": is_val}
    return data


def _images(n=2):
    return [Image.new("RGB", (4, 4), color=(i, i, i)) for i in range(n)]


def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _flush(worker):
    """Drain the dump queue without going through the RPC-decorated wrapper."""
    if worker._dump_queue is not None:
        worker._dump_queue.join()


def _dump_and_wait(worker, **kwargs):
    worker.maybe_dump_step_output(**kwargs)
    _flush(worker)


# --- gating ---------------------------------------------------------------


@pytest.mark.parametrize(
    "types, idx, step, is_val, expected",
    [
        (["train"], [0, 1], 1, False, True),
        (["train"], [0, 1], 1, True, False),   # phase not selected
        (["val"], [0, 1], 1, True, True),
        (["train", "val"], [0, 1], 1, True, True),
        ([], [0, 1], 1, False, False),         # empty type disables everything
        (["train"], [], 1, False, False),      # empty idx disables everything
        (["train"], [0, 1], 5, False, False),  # step not selected
    ],
)
def test_gating_truth_table(tmp_path, types, idx, step, is_val, expected):
    config = _pipeline_config(tmp_path, types=types, idx=idx)
    assert is_dump_enabled(config, step, is_val) is expected
    worker = _FakeWorker(config)
    assert worker._should_dump(step, is_val) is expected


def test_gating_requires_path(tmp_path):
    config = _pipeline_config(tmp_path)
    config.dump_step_output_path = ""
    assert is_dump_enabled(config, 1, False) is False
    assert resolve_dump_base_dir(config) == ""
    assert resolve_dump_step_dir(config, 1, False) == ""


def test_base_dir_falls_back_to_exp_name(tmp_path):
    config = _pipeline_config(tmp_path, run_name="")
    assert resolve_dump_base_dir(config) == os.path.join(str(tmp_path), "exp_unknown")


def test_step_dir_layout_matches_worker(tmp_path):
    config = _pipeline_config(tmp_path)
    worker = _FakeWorker(config)
    expected = os.path.join(str(tmp_path), "run-x", "step1", "train")
    assert resolve_dump_step_dir(config, 1, False) == expected
    assert worker._get_dump_step_dir(1, False) == expected


# --- schema ---------------------------------------------------------------


def test_mandatory_schema_and_extra_columns(tmp_path):
    worker = _FakeWorker(_pipeline_config(tmp_path))
    _dump_and_wait(
        worker,
        data=_data(prompts=["a photo", "a cat"], seeds=[7, 8]),
        sample_ids=["train-1-0", "train-1-1"],
        images=_images(2),
        reward_scores=[0.25, 0.75],
        extra_columns={"judge_response": ["hello", "world"]},
    )

    step_dir = os.path.join(str(tmp_path), "run-x", "step1", "train")
    records = _read_jsonl(os.path.join(step_dir, "output-ocr_reward-rank0.jsonl"))
    assert len(records) == 2
    first = records[0]
    assert first["key"] == "1-train-1-0"
    assert first["reward_name"] == "ocr_reward"
    assert first["sample_id"] == "train-1-0"
    assert first["img_path"] == "1-train-1-0.png"
    assert first["reward_score"] == pytest.approx(0.25)
    assert first["prompt"] == "a photo"
    assert first["rollout_seed"] == 7
    assert first["judge_response"] == "hello"
    # images land next to the jsonl, named by the same key
    assert os.path.exists(os.path.join(step_dir, "1-train-1-0.png"))
    assert os.path.exists(os.path.join(step_dir, "1-train-1-1.png"))


def test_missing_prompt_and_seed_default_to_empty(tmp_path):
    worker = _FakeWorker(_pipeline_config(tmp_path))
    _dump_and_wait(
        worker,
        data=_data(),
        sample_ids=["s0", "s1"],
        images=_images(2),
        reward_scores=[0.1, 0.2],
    )
    step_dir = os.path.join(str(tmp_path), "run-x", "step1", "train")
    records = _read_jsonl(os.path.join(step_dir, "output-ocr_reward-rank0.jsonl"))
    assert [r["prompt"] for r in records] == ["", ""]
    assert [r["rollout_seed"] for r in records] == [None, None]


def test_numpy_and_nested_values_are_serialized(tmp_path):
    """Geneval-style payloads: numpy scalars and nested dicts must survive."""
    worker = _FakeWorker(_pipeline_config(tmp_path), reward_name="geneval_reward")
    _dump_and_wait(
        worker,
        data=_data(),
        sample_ids=["s0", "s1"],
        images=_images(2),
        reward_scores=[np.float32(0.5), np.float64(1.0)],
        extra_columns={
            "geneval_detail": [
                {"tag": "counting", "correct": np.bool_(True), "score": np.float32(0.5)},
                {"tag": "colors", "correct": np.bool_(False), "score": np.float32(0.0)},
            ],
            "geneval_strict_reward": np.array([0.0, 1.0], dtype=np.float32),
        },
    )
    step_dir = os.path.join(str(tmp_path), "run-x", "step1", "train")
    records = _read_jsonl(os.path.join(step_dir, "output-geneval_reward-rank0.jsonl"))
    assert records[0]["reward_score"] == pytest.approx(0.5)
    assert records[0]["geneval_detail"] == {"tag": "counting", "correct": True, "score": pytest.approx(0.5)}
    assert records[1]["geneval_strict_reward"] == pytest.approx(1.0)


def test_to_jsonable_handles_common_types():
    assert to_jsonable(np.float32(1.5)) == pytest.approx(1.5)
    assert to_jsonable(np.array([1, 2])) == [1, 2]
    assert to_jsonable(torch.tensor([1.0, 2.0])) == [pytest.approx(1.0), pytest.approx(2.0)]
    assert to_jsonable({"a": {"b": np.int64(3)}}) == {"a": {"b": 3}}
    assert sorted(to_jsonable({np.int64(1), np.int64(2)})) == [1, 2]
    assert to_jsonable(None) is None


# --- multi-reward isolation ----------------------------------------------


def test_concurrent_rewards_write_separate_files(tmp_path):
    """Two rewards dumping the same step must not append to one jsonl."""
    config = _pipeline_config(tmp_path)
    for name, score in (("ocr_reward", 0.3), ("geneval_reward", 0.6)):
        worker = _FakeWorker(config, reward_name=name)
        _dump_and_wait(
            worker,
            data=_data(),
            sample_ids=["s0"],
            images=_images(1),
            reward_scores=[score],
        )

    step_dir = os.path.join(str(tmp_path), "run-x", "step1", "train")
    files = sorted(f for f in os.listdir(step_dir) if f.endswith(".jsonl"))
    assert files == ["output-geneval_reward-rank0.jsonl", "output-ocr_reward-rank0.jsonl"]
    for name, score in (("ocr_reward", 0.3), ("geneval_reward", 0.6)):
        records = _read_jsonl(os.path.join(step_dir, f"output-{name}-rank0.jsonl"))
        assert len(records) == 1
        assert records[0]["reward_name"] == name
        assert records[0]["reward_score"] == pytest.approx(score)


def test_rank_is_part_of_filename(tmp_path):
    config = _pipeline_config(tmp_path)
    for rank in (0, 3):
        worker = _FakeWorker(config, rank=rank)
        _dump_and_wait(
            worker,
            data=_data(),
            sample_ids=["s0"],
            images=_images(1),
            reward_scores=[0.5],
        )
    step_dir = os.path.join(str(tmp_path), "run-x", "step1", "train")
    assert os.path.exists(os.path.join(step_dir, "output-ocr_reward-rank0.jsonl"))
    assert os.path.exists(os.path.join(step_dir, "output-ocr_reward-rank3.jsonl"))


# --- failure modes -------------------------------------------------------


def test_column_length_mismatch_raises(tmp_path):
    worker = _FakeWorker(_pipeline_config(tmp_path))
    with pytest.raises(AssertionError, match="length mismatch"):
        worker.maybe_dump_step_output(
            data=_data(),
            sample_ids=["s0", "s1"],
            images=_images(2),
            reward_scores=[0.1, 0.2],
            extra_columns={"judge_response": ["only-one"]},
        )


def test_reserved_key_cannot_be_shadowed(tmp_path):
    worker = _FakeWorker(_pipeline_config(tmp_path))
    with pytest.raises(AssertionError, match="reserved dump keys"):
        worker.maybe_dump_step_output(
            data=_data(),
            sample_ids=["s0", "s1"],
            images=_images(2),
            reward_scores=[0.1, 0.2],
            extra_columns={"reward_score": [1.0, 2.0]},
        )


def test_disabled_dump_writes_nothing(tmp_path):
    worker = _FakeWorker(_pipeline_config(tmp_path, types=[]))
    worker.maybe_dump_step_output(
        data=_data(),
        sample_ids=["s0"],
        images=_images(1),
        reward_scores=[0.5],
    )
    _flush(worker)  # safe even though no thread was ever started
    assert not os.path.exists(os.path.join(str(tmp_path), "run-x"))


def test_flush_dump_is_registered_for_rpc():
    """Cluster._bind_worker_method discovers RPCs via dir()+getattr on the class,
    so the mixin's flush_dump must carry the bind flag through the MRO."""
    assert "flush_dump" in dir(_FakeWorker)
    assert hasattr(getattr(_FakeWorker, "flush_dump"), BIND_WORKER_METHOD_FLAG)


def test_missing_global_step_writes_nothing(tmp_path):
    worker = _FakeWorker(_pipeline_config(tmp_path))
    data = _data()
    data.meta_info.pop("global_step")
    worker.maybe_dump_step_output(
        data=data,
        sample_ids=["s0"],
        images=_images(1),
        reward_scores=[0.5],
    )
    assert not os.path.exists(os.path.join(str(tmp_path), "run-x"))
