"""Cold NCCL startup must see released cache before synchronous grad sync."""
import importlib.util
from pathlib import Path

import pytest
import torch


def load_guard():
    path = Path(__file__).resolve().parents[3] / "roll/third_party/megatron/grad_sync_memory.py"
    spec = importlib.util.spec_from_file_location("grad_sync_memory", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FirstGradSyncCacheRelease


def test_releases_cache_before_first_finalize_and_preserves_arguments_and_result(monkeypatch):
    events = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: events.append("release"))
    grads = torch.tensor([1., 2., 3.])
    token_count = torch.tensor(2)

    def finalize(models, num_tokens, *, force_sync):
        events.append("finalize")
        assert models is grads and num_tokens is token_count and force_sync is True
        models.div_(num_tokens)
        return models

    wrapped = load_guard()(finalize)
    assert wrapped(grads, token_count, force_sync=True) is grads
    assert wrapped(grads, token_count, force_sync=True) is grads
    torch.testing.assert_close(grads, torch.tensor([0.25, 0.5, 0.75]))
    assert events == ["release", "finalize", "finalize"]


def test_failed_first_finalize_does_not_mark_communications_initialized(monkeypatch):
    events = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: events.append("release"))

    def fail():
        raise RuntimeError("collective initialization failed")

    wrapped = load_guard()(fail)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="collective initialization failed"):
            wrapped()
    assert events == ["release", "release"]
