"""Reject empty distributed validation before loading the real model."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
from torch.utils.data import DataLoader


path = Path(__file__).parents[2] / "scripts/qwen38/run_sft_validation.py"
spec = importlib.util.spec_from_file_location("sft_data_preflight", path)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.mark.parametrize("records,dp,ga,infer", [(2, 8, 1, 1), (0, 1, 1, 1), (8, 8, 2, 1), (8, 8, 1, 2)])
def test_preflight_rejects_validation_without_a_complete_batch(records, dp, ga, infer):
    worker = SimpleNamespace(training_args=SimpleNamespace(gradient_accumulation_steps=ga), infer_batch_size=infer)
    assert len(DataLoader(range(records), batch_size=dp * ga * infer, drop_last=True)) == 0
    with pytest.raises(ValueError, match="heldout.*complete validation batch"):
        runner.validate_heldout_batch(records, worker, dp)


@pytest.mark.parametrize("records,dp,ga,infer", [(8, 8, 1, 1), (9, 8, 1, 1), (2, 1, 1, 1), (32, 8, 2, 2)])
def test_preflight_accepts_real_validation_batches(records, dp, ga, infer):
    worker = SimpleNamespace(training_args=SimpleNamespace(gradient_accumulation_steps=ga), infer_batch_size=infer)
    runner.validate_heldout_batch(records, worker, dp)
    assert len(DataLoader(range(records), batch_size=dp * ga * infer, drop_last=True)) > 0
