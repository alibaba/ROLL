"""Current ROLL and older ROLL input conversion contracts.

Extract actual worker method bodies to avoid CUDA/Ray imports in this CPU test.
"""
import ast
from contextlib import nullcontext
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Dict

import pytest
import torch


class CPUDataProto:
    def __init__(self, batch=None, meta_info=None):
        self.batch = batch
        self.meta_info = {} if meta_info is None else meta_info
        self.devices = []

    def to(self, device):
        self.devices.append(device)
        return self


@pytest.mark.parametrize("method", ["forward_tinker", "forward_backward_accumulate_tinker"])
@pytest.mark.parametrize("has_converter", [False, True])
def test_worker_accepts_current_data_proto_and_optional_legacy_converter(monkeypatch, method, has_converter):
    path = Path(__file__).resolve().parents[2] / "roll/pipeline/tinker_backend_runtime/workers.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    actor_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TinkerActorWorker")
    function = next(node for node in actor_class.body if isinstance(node, ast.FunctionDef) and node.name == method)
    function.decorator_list = []
    namespace = {
        "DataProto": CPUDataProto, "torch": torch, "Dict": Dict,
        "state_offload_manger": lambda **kwargs: nullcontext(),
        "OffloadStateType": SimpleNamespace(model_params=0, other_params=1),
        "current_platform": SimpleNamespace(device_type="cpu"),
        "reduce_metrics": lambda metrics: metrics,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)

    original = CPUDataProto(batch={"input_ids": torch.tensor([[1, 2]])})
    converted = CPUDataProto(batch={"input_ids": torch.tensor([[3, 4]])})
    expected = converted if has_converter else original
    calls = []

    def forward_step(*, batch, forward_func):
        assert batch is expected
        calls.append("forward")
        return None  # Non-output TP rank: exercise the actual worker return path.

    def backward(*, strategy, batch, loss_func):
        assert batch is expected
        calls.append("backward")
        return {"backward_test_metric": 1.0}

    strategy = SimpleNamespace(forward_step=forward_step)
    if has_converter:
        def convert(data):
            assert data is original
            calls.append("convert")
            return converted
        strategy.get_data_input = convert

    package = ModuleType("roll.pipeline.tinker_backend_runtime")
    package.__path__ = []
    primitive = ModuleType("roll.pipeline.tinker_backend_runtime.megatron_primitives")
    primitive.forward_backward_accumulate = backward
    package.megatron_primitives = primitive
    monkeypatch.setitem(sys.modules, "roll.pipeline.tinker_backend_runtime", package)
    monkeypatch.setitem(sys.modules, primitive.__name__, primitive)
    worker = SimpleNamespace(
        strategy=strategy, cluster_name="actor_train",
        worker_config=SimpleNamespace(infer_batch_size=1, use_dynamic_batching_in_train=False,
                                      use_sequence_packing=False),
        rank_info=SimpleNamespace(tp_rank=1, cp_rank=0, is_pipeline_last_stage=True),
        forward_func_tinker_log_probs=object(), tinker_ce_loss_func=object(),
    )
    result = namespace[method](worker, original)
    assert isinstance(result, CPUDataProto)
    assert result.batch is None
    assert expected.devices == ["cpu", "cpu"]
    expected_calls = (["convert"] if has_converter else []) + ["forward" if method == "forward_tinker" else "backward"]
    assert calls == expected_calls
    assert expected.meta_info["loss_mask_keys"] == []
    if method == "forward_tinker":
        assert expected.meta_info["micro_batch_size"] == 1
    else:
        assert expected.meta_info["skip_microbatch_count_check"] is True
        assert result.meta_info["metrics"] == {"backward_test_metric": 1.0}
