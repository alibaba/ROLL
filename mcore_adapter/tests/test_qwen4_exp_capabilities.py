import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "capabilities", Path(__file__).parents[1] / "src/mcore_adapter/models/qwen4_exp/capabilities.py"
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
validate_training_capabilities = _module.validate_training_capabilities
estimate_full_parameter_memory = _module.estimate_full_parameter_memory


def test_sparse_training_requires_kernel_above_budget():
    with pytest.raises(ValueError, match="sparse QSA training kernel"):
        validate_training_capabilities(sequence_length=2049, qsa_training_kernel=False)


def test_dense_fallback_is_explicit_for_short_sequences():
    report = validate_training_capabilities(sequence_length=2048, qsa_training_kernel=False)
    assert report["qsa_mode"] == "dense_equivalent"


def test_full_parameter_memory_keeps_cpu_adam_state_explicit():
    report = estimate_full_parameter_memory(125_743_653_760, optimizer_cpu_offload=True)
    assert report["gpu_optimizer_state_bytes"] == 0
    assert report["cpu_optimizer_state_bytes"] == 125_743_653_760 * 8
    assert report["gpu_peak_bytes"] == 125_743_653_760 * 6
    assert report["cpu_peak_bytes"] == 125_743_653_760 * 8


def test_full_parameter_memory_rejects_empty_parameter_set():
    with pytest.raises(ValueError, match="positive"):
        estimate_full_parameter_memory(0)
