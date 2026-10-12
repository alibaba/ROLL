"""Regression tests for bounded, streaming Qwen3.8 DCP payload validation."""
import builtins
import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.distributed.checkpoint import save


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/qwen38/validate_backbone_payloads.py"


def load_validator(name="qwen38_payload_validator"):
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_storage_reader_reuses_file_and_reads_offsets_in_order(tmp_path, monkeypatch):
    path = tmp_path / "rank.distcp"
    payloads = [torch.arange(3), torch.arange(5, 9)]
    blobs = []
    for payload in payloads:
        stream = io.BytesIO()
        torch.save(payload, stream)
        blobs.append(stream.getvalue())
    offsets = [7, 7 + len(blobs[0]) + 11]
    path.write_bytes(b"prefix" + b"X" + blobs[0] + b"Y" * 11 + blobs[1])
    storage = {
        ("late", None): SimpleNamespace(relative_path=path.name, offset=offsets[1],
                                         length=len(blobs[1]), transform_descriptors=None),
        ("early", None): SimpleNamespace(relative_path=path.name, offset=offsets[0],
                                          length=len(blobs[0]), transform_descriptors=None),
    }
    validator = load_validator("qwen38_payload_validator_stream")
    original_open = Path.open
    opened = []

    def tracked_open(self, *args, **kwargs):
        if self == path:
            opened.append(self)
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", tracked_open)
    items = list(validator._iter_storage_items(tmp_path, storage))
    assert [key for key, _ in items] == [("early", None), ("late", None)]
    assert [value.tolist() for _, value in items] == [[0, 1, 2], [5, 6, 7, 8]]
    assert len(opened) == 1


def test_full_backbone_import_does_not_require_lora_helper(monkeypatch):
    original_import = builtins.__import__

    def import_without_lora(name, *args, **kwargs):
        if name == "scripts.qwen38.lora_checkpoint_integrity":
            raise ModuleNotFoundError(name=name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_lora)
    validator = load_validator("qwen38_payload_validator_without_lora")
    assert callable(validator.validate_payloads)


def test_payload_validator_reads_model_and_optimizer_payloads(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    model = checkpoint / "iter_0000001" / "dist_model"
    optimizer = checkpoint / "iter_0000001" / "dist_optimizer"
    save({"weight": torch.arange(4)}, checkpoint_id=model)
    save({"state": {"step": torch.tensor(1), "exp_avg": torch.ones(4)}},
         checkpoint_id=optimizer)
    torch.save({}, model / "common.pt")
    torch.save({}, optimizer / "common.pt")
    result = load_validator("qwen38_payload_validator_dcp").validate_payloads(checkpoint)
    assert result["payloads_valid"] is True
    assert result["storage_items"] > 0
