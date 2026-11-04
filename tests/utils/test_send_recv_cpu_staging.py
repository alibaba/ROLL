import numpy as np
import torch

from roll.platforms import current_platform
from roll.utils.cuda_ipc_utils import MultiprocessingSerializer
from roll.utils.send_recv_utils import (
    named_tensors_from_bucket,
    serialize_named_weights,
)


def _make_named_weights():
    return [
        ("weight_a", torch.arange(0, 6, dtype=torch.float32).reshape(2, 3)),
        ("weight_b", torch.arange(6, 10, dtype=torch.float32)),
    ]


def test_serialize_named_weights_cpu_staging(monkeypatch):
    """The opt-in CPU-staging transport serializes a numpy payload without CUDA IPC."""
    monkeypatch.setenv("ROLL_WEIGHT_SYNC_USE_CPU", "1")
    named_weights = _make_named_weights()

    serialized = serialize_named_weights(named_weights, infer_strategy="vllm")

    assert isinstance(serialized, bytes)
    payload = MultiprocessingSerializer.deserialize(serialized)
    assert isinstance(payload["bucket"], np.ndarray)

    reconstructed = dict(named_tensors_from_bucket(payload["bucket"], payload["tensors_meta"]))
    assert set(reconstructed) == {"weight_a", "weight_b"}
    assert torch.equal(reconstructed["weight_a"], named_weights[0][1])
    assert torch.equal(reconstructed["weight_b"], named_weights[1][1])


def test_serialize_named_weights_default_path_cpu_only(monkeypatch):
    """Without the opt-in env, CPU tensors take the existing `.to(current_platform.device_type)` path."""
    monkeypatch.delenv("ROLL_WEIGHT_SYNC_USE_CPU", raising=False)
    # Emulate a CPU-only instance so the default path never touches an accelerator.
    monkeypatch.setattr(type(current_platform), "device_type", "cpu")
    named_weights = _make_named_weights()

    serialized = serialize_named_weights(named_weights, infer_strategy="vllm")

    assert isinstance(serialized, bytes)
    payload = MultiprocessingSerializer.deserialize(serialized)
    assert isinstance(payload["bucket"], torch.Tensor)
    assert payload["bucket"].device.type == "cpu"

    reconstructed = dict(named_tensors_from_bucket(payload["bucket"], payload["tensors_meta"]))
    assert torch.equal(reconstructed["weight_a"], named_weights[0][1])
    assert torch.equal(reconstructed["weight_b"], named_weights[1][1])
