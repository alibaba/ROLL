"""Rollout dumps must serialize the TensorDict-backed columns the TransferQueue backend returns."""
import json

import numpy as np
import pytest

pytest.importorskip("tensordict")


def test_dump_writes_nontensor_stack_and_ndarray_columns(tmp_path):
    from tensordict import NonTensorStack

    from roll.pipeline.rlvr.utils import COLUMMNS_CONFIG, write_to_json_process

    data = {
        "id": NonTensorStack("a", "b"),
        "domain": np.array(["x", "y"], dtype=object),
        "responses": ["r1", "r2"],
        "scores": [1.0, 0.0],
        "global_step": [3, 3],
    }
    write_to_json_process(str(tmp_path), data, COLUMMNS_CONFIG)

    row = json.loads((tmp_path / "rollout_dump_data.step_3.jsonl").read_text())
    assert row["id"] == ["a", "b"]
    assert row["domain"] == ["x", "y"]
    assert row["scores"] == [1.0, 0.0]
