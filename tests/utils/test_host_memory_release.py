"""Host cache cleanup must release idle storage without invalidating tensors."""
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from roll.utils import offload_states


@pytest.mark.parametrize("api", ["public", "legacy", "unavailable"])
def test_host_cleanup_supports_torch_api_versions(monkeypatch, api):
    events = []
    platform = SimpleNamespace(synchronize=lambda: events.append("synchronize"),
                               empty_cache=lambda: events.append("device_cache"))
    monkeypatch.setattr(offload_states, "current_platform", platform)
    monkeypatch.setattr(torch.accelerator.memory, "empty_host_cache",
                        lambda: events.append("public"), raising=False)
    monkeypatch.setattr(torch._C, "_host_emptyCache", lambda: events.append("legacy"), raising=False)
    if api != "public":
        monkeypatch.delattr(torch.accelerator.memory, "empty_host_cache")
    if api == "unavailable":
        monkeypatch.delattr(torch._C, "_host_emptyCache")
    live = torch.arange(4096)
    offload_states.clear_memory(clear_host_memory=True)
    torch.testing.assert_close(live, torch.arange(4096), atol=0, rtol=0)
    assert events == ["synchronize", "device_cache"] + ([] if api == "unavailable" else [api])


@pytest.mark.skipif(sys.platform != "linux", reason="Linux glibc allocator regression")
def test_releases_fragmented_free_heap_and_preserves_live_tensors():
    # Isolate allocator tuning and RSS from pytest and other tests. Small live
    # allocations between freed blocks prevent an implicit top-of-heap trim.
    script = r'''
import ctypes, json
from pathlib import Path
from types import SimpleNamespace
import torch
from roll.utils import offload_states
from roll.utils.offload_states import clear_memory
# The CPU-only fixture exercises host allocation; there is no device cache.
offload_states.current_platform = SimpleNamespace(synchronize=lambda: None, empty_cache=lambda: None)
libc = ctypes.CDLL(None)
if not hasattr(libc, "malloc_trim"):
    raise SystemExit(77)
libc.mallopt.argtypes = [ctypes.c_int, ctypes.c_int]
libc.mallopt(-3, 16 * 1024**2)  # M_MMAP_THRESHOLD
libc.mallopt(-1, 1024 * 1024**2)  # M_TRIM_THRESHOLD
libc.malloc.argtypes = [ctypes.c_size_t]
libc.malloc.restype = ctypes.c_void_p
libc.free.argtypes = [ctypes.c_void_p]
live = torch.arange(16384, dtype=torch.float32)
pointer = live.data_ptr()
blocks, guards = [], []
for _ in range(256):
    block = libc.malloc(1024**2)
    guard = libc.malloc(16384)
    assert block and guard
    ctypes.memset(block, 42, 1024**2)
    ctypes.memset(guard, 17, 16384)
    blocks.append(block)
    guards.append(guard)
for block in blocks:
    libc.free(block)
def rss():
    return int(next(l for l in Path('/proc/self/status').read_text().splitlines()
                    if l.startswith('VmRSS:')).split()[1]) * 1024
before = rss()
clear_memory(clear_host_memory=True)
after = rss()
assert live.data_ptr() == pointer
torch.testing.assert_close(live, torch.arange(16384, dtype=torch.float32), atol=0, rtol=0)
assert all(ctypes.string_at(p, 16384) == bytes([17]) * 16384 for p in guards)
for guard in guards:
    libc.free(guard)
print(json.dumps(dict(before=before, after=after, released=before-after,
                     cuda_initialized=torch.cuda.is_initialized())))
assert before - after >= 200 * 1024**2, 'unused fragmented heap remained resident'
assert not torch.cuda.is_initialized()
'''
    result = subprocess.run([sys.executable, "-c", script], text=True, capture_output=True,
                            cwd=Path(__file__).resolve().parents[2], timeout=90,
                            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    if result.returncode == 77:
        pytest.skip("allocator does not expose malloc_trim")
    assert result.returncode == 0, result.stdout + result.stderr
