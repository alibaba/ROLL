import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "capabilities", Path(__file__).parents[1] / "src/mcore_adapter/models/qwen4_exp/capabilities.py"
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
validate_training_capabilities = _module.validate_training_capabilities


def test_sparse_training_requires_kernel_above_budget():
    with pytest.raises(ValueError, match="sparse QSA training kernel"):
        validate_training_capabilities(sequence_length=2049, qsa_training_kernel=False)


def test_dense_fallback_is_explicit_for_short_sequences():
    report = validate_training_capabilities(sequence_length=2048, qsa_training_kernel=False)
    assert report["qsa_mode"] == "dense_equivalent"
