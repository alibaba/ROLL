"""The accelerated global-plan validator must preserve PyTorch's caller contract."""
import copy
import importlib.util
import sys

import pytest
import torch
from torch.distributed import checkpoint as dcp
from torch.distributed.checkpoint import default_planner
from torch.distributed.checkpoint.metadata import ChunkStorageMetadata, Metadata, TensorProperties, TensorStorageMetadata
from torch.distributed.checkpoint.planner import SavePlan

from mcore_adapter.patcher import patch_torch_validate_global_plan


@pytest.fixture
def native_validator(monkeypatch):
    original = default_planner._validate_global_plan
    monkeypatch.setattr(default_planner, '_validate_global_plan', original)
    # Another test may already have patched the process-global function.
    # Load the installed module in a separate namespace for an independent
    # reference without replacing the planner used by the running test suite.
    name = 'torch.distributed.checkpoint._mca_native_default_planner'
    spec = importlib.util.spec_from_file_location(name, default_planner.__file__)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module._validate_global_plan


def metadata_for(chunks):
    return Metadata({'weight': TensorStorageMetadata(
        properties=TensorProperties(dtype=torch.float32), size=torch.Size([4]),
        chunks=[ChunkStorageMetadata(torch.Size([offset]), torch.Size([size]))
                for offset, size in chunks])})


@pytest.mark.parametrize('chunks', [[(0, 2), (2, 2)], [(0, 3), (2, 2)], [(0, 2), (3, 2)], [(0, 1), (2, 1)]])
def test_validator_retains_native_success_and_error_contract(native_validator, chunks):
    plans = [SavePlan([]), SavePlan([])]
    metadata = metadata_for(chunks)
    expected = native_validator(plans, copy.deepcopy(metadata))
    patch_torch_validate_global_plan()
    patch_torch_validate_global_plan()  # Every ROLL worker strategy calls this.
    actual = default_planner._validate_global_plan(plans, metadata)
    assert type(actual) is type(expected)
    if isinstance(expected, bool):
        assert actual is expected
    else:
        assert bool(actual) == bool(expected)
        assert all(isinstance(error, str) and 'weight' in error for error in actual)


def test_real_checkpoint_save_and_load_after_repeated_patch(native_validator, tmp_path):
    patch_torch_validate_global_plan()
    patch_torch_validate_global_plan()
    source = {'model': torch.arange(12, dtype=torch.float32).reshape(3, 4),
              'optimizer': {'step': torch.tensor(2), 'exp_avg': torch.arange(4, dtype=torch.float32)}}
    destination = {'model': torch.zeros_like(source['model']),
                   'optimizer': {'step': torch.tensor(0), 'exp_avg': torch.zeros(4)}}
    dcp.save(source, checkpoint_id=tmp_path / 'checkpoint', no_dist=True)
    dcp.load(destination, checkpoint_id=tmp_path / 'checkpoint', no_dist=True)
    torch.testing.assert_close(destination, source, atol=0, rtol=0)
