"""Regression test: local offload backend must keep FSDP2 DTensor params valid.

DTensor is a wrapper subclass — the data lives in the Python object's
`_local_tensor`, not in its TensorImpl storage. Rebinding `param.data` (the old
put/get implementation) leaves the parameter with an invalid storage: any read
faults (CUDA illegal memory access on GPU, segfault on CPU), and the reloaded
GPU buffer is freed because nothing references it.

The correct pattern — also used by FSDP2's own FSDPParam.reset_sharded_param —
is to re-link `_local_tensor` and then let FSDP2 re-derive `_sharded_param_data`
(the all-gather copy-in source).

Run: pytest tests/distributed/store/test_local_backend_dtensor.py -v
"""
import os
import sys

import pytest
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FSDPModule, fully_shard
from torch.distributed.tensor import DTensor

from roll.distributed.store.local.backend import CPUOffloadBackend
from roll.utils.fsdp_utils import iter_fsdp_params, resolve_fsdp_param_groups


def _init_single_process_group(tmp_path):
    if dist.is_initialized():
        return
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{tmp_path}/pg",
        rank=0,
        world_size=1,
    )


def _build_fsdp2_model():
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Linear(8, 4))
    mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("fsdp",))
    fully_shard(model[0], mesh=mesh)
    fully_shard(model[1], mesh=mesh)
    fully_shard(model, mesh=mesh)
    return model


def _reset_sharded_params(model):
    # mirrors FSDP2StrategyBase._reset_fsdp_sharded_params
    for fsdp_param in iter_fsdp_params(model):
        fsdp_param.reset_sharded_param()


def _param_shard(param):
    return param._local_tensor.detach().clone()


@pytest.fixture()
def fsdp2_model(tmp_path):
    _init_single_process_group(tmp_path)
    model = _build_fsdp2_model()
    yield model
    backend_keys = getattr(model, "_test_offload_backend", None)
    if backend_keys is not None:
        backend_keys.clear()


def test_put_get_roundtrip_keeps_dtensor_valid(fsdp2_model):
    backend = CPUOffloadBackend()
    params = [p for _, p in fsdp2_model.named_parameters()]
    assert all(isinstance(p, DTensor) for p in params)
    before = [_param_shard(p) for p in params]

    key = "test_params"
    backend.put_tensors(key, params, replicated_over=False)

    for p in params:
        assert isinstance(p, DTensor)
        assert p._local_tensor.numel() == 0
        assert p._local_tensor.device.type == "cpu"

    backend.get_tensors(key, params, device="cpu", replicated_over=False)

    for p, ref in zip(params, before):
        assert isinstance(p, DTensor), "DTensor identity must survive reload"
        assert p._local_tensor.device.type == "cpu"
        assert torch.equal(_param_shard(p), ref), "reloaded shard data mismatch"
        full = p.full_tensor()
        assert torch.isfinite(full).all()

    backend.delete_tensors(key, replicated_over=False)


def test_reload_keeps_fsdp2_allgather_copyin_alive(fsdp2_model):
    """After reload + reset_sharded_param, FSDP2 must be able to run a forward:
    the all-gather copy-in reads FSDPParam._sharded_param_data, which has to
    share storage with the reloaded _local_tensor."""
    backend = CPUOffloadBackend()
    params = [p for _, p in fsdp2_model.named_parameters()]
    key = "test_params"
    backend.put_tensors(key, params, replicated_over=False)

    inputs = torch.randn(3, 4)

    backend.get_tensors(key, params, device="cpu", replicated_over=False)
    _reset_sharded_params(fsdp2_model)

    for module in fsdp2_model.modules():
        if not isinstance(module, FSDPModule):
            continue
        for group in module._get_fsdp_state()._fsdp_param_groups:
            for fsdp_param in group.fsdp_params:
                local = fsdp_param.sharded_param._local_tensor
                assert local.numel() > 0
                assert (
                    fsdp_param._sharded_param_data.untyped_storage().data_ptr()
                    == local.untyped_storage().data_ptr()
                ), "_sharded_param_data must view the reloaded buffer"

    with torch.no_grad():
        out = fsdp2_model(inputs)
    assert out.shape == (3, 4)
    assert torch.isfinite(out).all()

    backend.delete_tensors(key, replicated_over=False)


def test_repeated_put_get_cycles(fsdp2_model):
    """Offload -> reload -> delete -> offload again must stay consistent
    (mirrors the per-step train cycle)."""
    backend = CPUOffloadBackend()
    params = [p for _, p in fsdp2_model.named_parameters()]
    before = [_param_shard(p) for p in params]

    for cycle in range(3):
        key = f"cycle_{cycle}"
        backend.put_tensors(key, params, replicated_over=False)
        assert all(p._local_tensor.numel() == 0 for p in params)
        backend.get_tensors(key, params, device="cpu", replicated_over=False)
        _reset_sharded_params(fsdp2_model)
        for p, ref in zip(params, before):
            assert torch.equal(_param_shard(p), ref), f"data corrupted at cycle {cycle}"
        backend.delete_tensors(key, replicated_over=False)

        # simulate an optimizer update so the next cycle has new data
        with torch.no_grad():
            for p in params:
                p._local_tensor.add_(1.0)
        before = [_param_shard(p) for p in params]


def test_plain_tensors_still_work(fsdp2_model, tmp_path):
    """Non-DTensor tensors keep the plain .data rebind semantics."""
    backend = CPUOffloadBackend()
    tensors = [torch.nn.Parameter(torch.arange(6, dtype=torch.float32).view(2, 3))]
    key = "plain"
    backend.put_tensors(key, tensors, replicated_over=False)
    assert tensors[0].data.numel() == 0
    backend.get_tensors(key, tensors, device="cpu", replicated_over=False)
    assert torch.equal(tensors[0].data, torch.arange(6, dtype=torch.float32).view(2, 3))
    backend.delete_tensors(key, replicated_over=False)


def test_resolve_fsdp_param_groups_version_layouts():
    """FSDPState exposes param groups differently across torch versions:
    `_fsdp_param_groups` (list, torch >= 2.7) vs `_fsdp_param_group`
    (single, torch <= 2.6). resolve_fsdp_param_groups must handle both."""
    from types import SimpleNamespace

    group = object()
    # torch >= 2.7: plural list wins even if singular is also present
    state = SimpleNamespace(_fsdp_param_groups=[group])
    assert resolve_fsdp_param_groups(state) == [group]
    # torch <= 2.6: only the singular attribute exists
    state = SimpleNamespace(_fsdp_param_group=group)
    assert resolve_fsdp_param_groups(state) == [group]
    # uninitialized state
    assert resolve_fsdp_param_groups(SimpleNamespace()) == []


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        _init_single_process_group(tmp)
        model = _build_fsdp2_model()
        backend = CPUOffloadBackend()
        params = [p for _, p in model.named_parameters()]
        before = [_param_shard(p) for p in params]
        backend.put_tensors("k", params, replicated_over=False)
        backend.get_tensors("k", params, device="cpu", replicated_over=False)
        _reset_sharded_params(model)
        ok = all(torch.equal(_param_shard(p), r) for p, r in zip(params, before))
        out = model(torch.randn(3, 4))
        print("roundtrip ok:", ok, "forward ok:", torch.isfinite(out).all().item())
        backend.delete_tensors("k", replicated_over=False)
        dist.destroy_process_group()
