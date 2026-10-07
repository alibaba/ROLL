"""Native score RPC results retain a real batch dimension through DP collection."""
import asyncio
import inspect
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.tinker_backend_runtime import workers
from roll.pipeline.tinker_backend_runtime.roll_backend import ROLLRuntimeBackend


@pytest.mark.parametrize("rows", [
    [[None, -0.1, -0.2]],
    [[None, -0.1, -0.2], [None, -0.3, -0.4]],
    [[None, -0.1], [None, -0.3, -0.4, -0.5], [None]],
])
def test_real_sampler_score_output_is_valid_collectable_and_backend_consumable(monkeypatch, rows):
    strategy = object()
    monkeypatch.setattr(workers, "current_platform", SimpleNamespace(device_type="cpu"))
    method = inspect.unwrap(workers.TinkerInferWorker.compute_prompt_logprobs)
    worker = SimpleNamespace(strategy=strategy)

    def score_output(expected_rows):
        data = DataProto.from_dict(tensors={
            "input_ids": torch.ones((len(expected_rows), 4), dtype=torch.long, device="cpu"),
        })

        async def primitive(actual_strategy, actual_data):
            assert actual_strategy is strategy
            assert actual_data is data
            assert actual_data.batch["input_ids"].device.type == "cpu"
            return expected_rows

        monkeypatch.setattr(workers, "compute_prompt_logprobs_with_vllm_strategy", primitive)
        output = asyncio.run(method(worker, data))
        output.check_consistency()
        assert isinstance(output._batch, TensorDict)
        assert output.batch.batch_size == torch.Size([len(expected_rows)])
        assert list(output.batch.keys()) == []
        assert len(output) == len(expected_rows)
        assert [list(row) for row in output.non_tensor_batch["prompt_logprobs"]] == expected_rows
        assert data.batch["input_ids"].device.type == "cpu"
        return output

    output = score_output(rows)
    # A second DP output with another prompt length verifies current ROLL
    # custom object-array concatenation handles both uniform and ragged rows.
    extra_rows = [[None, -1.1, -1.2, -1.3, -1.4]]
    collected = DataProto.concat([output, score_output(extra_rows)])
    collected.check_consistency()
    assert collected.batch.batch_size == torch.Size([len(rows) + 1])
    expected = rows + extra_rows
    calls = []

    def compute_prompt_logprobs(data, blocking):
        calls.append(data)
        assert blocking is True
        return collected

    backend = SimpleNamespace(actor_infer=SimpleNamespace(compute_prompt_logprobs=compute_prompt_logprobs))
    request_data = object()
    entries = [(str(i), [1] * len(row)) for i, row in enumerate(expected)]
    actual = ROLLRuntimeBackend._score_with_actor_infer(backend, request_data, entries)
    assert actual == expected
    assert calls == [request_data]
