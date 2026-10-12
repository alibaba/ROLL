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
    assert report["ngram_table_mode"] == "frozen_external"


@pytest.mark.parametrize("qsa_training_kernel", [False, True])
def test_trainable_ngram_table_is_rejected_until_backend_exists(qsa_training_kernel):
    with pytest.raises(NotImplementedError, match="trainable NGram"):
        validate_training_capabilities(
            sequence_length=2048,
            qsa_training_kernel=qsa_training_kernel,
            train_ngram_table=True,
        )


def test_cpu_adam_storage_includes_fp32_master_moments_and_gradient_copy():
    report = estimate_full_parameter_memory(10, optimizer_cpu_offload=True)
    assert report["gpu_parameter_bytes"] == 20
    assert report["gpu_gradient_bytes"] == 20
    assert report["gpu_optimizer_state_bytes"] == 0
    assert report["gpu_master_parameter_bytes"] == 0
    assert report["cpu_master_parameter_bytes"] == 40
    assert report["cpu_optimizer_state_bytes"] == 80
    assert report["cpu_gradient_bytes"] == 40
    assert report["gpu_peak_bytes"] == 40
    assert report["cpu_peak_bytes"] == 160


def test_gpu_adam_storage_includes_master_without_cpu_copies():
    report = estimate_full_parameter_memory(10)
    assert report["gpu_master_parameter_bytes"] == 40
    assert report["gpu_optimizer_state_bytes"] == 80
    assert report["cpu_master_parameter_bytes"] == 0
    assert report["cpu_optimizer_state_bytes"] == 0
    assert report["cpu_gradient_bytes"] == 0
    assert report["gpu_peak_bytes"] == 160
    assert report["cpu_peak_bytes"] == 0


def test_fp32_gpu_gradient_does_not_remove_fp32_cpu_gradient_copy():
    report = estimate_full_parameter_memory(10, gradient_dtype_bytes=4, optimizer_cpu_offload=True)
    assert report["gpu_gradient_bytes"] == 40
    assert report["cpu_gradient_bytes"] == 40
    assert report["gpu_peak_bytes"] == 60
    assert report["cpu_peak_bytes"] == 160


def test_full_parameter_memory_rejects_empty_parameter_set():
    with pytest.raises(ValueError, match="positive"):
        estimate_full_parameter_memory(0)


def test_bounded_rl_statistics_capability_is_explicit_and_reports_contract():
    validator = getattr(_module, "validate_bounded_rl_token_statistics", None)
    assert callable(validator), "bounded RL token-statistics capability gate is missing"
    assert validator(enabled=False) == {"enabled": False}
    report = validator(enabled=True, chunk_size=64)
    assert report == {
        "enabled": True,
        "output": "selected_logprob_and_entropy",
        "dtype": "float32",
        "chunk_size": 64,
    }


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"context_parallel_size": 2}, "CP=1"),
        ({"sequence_packing": True}, "packing"),
        ({"mtp_num_layers": 1}, "MTP"),
        ({"output_head_adapter": True}, "output-head adapters"),
        ({"chunk_size": 0}, "chunk_size"),
    ],
)
def test_bounded_rl_statistics_rejects_unsupported_modes_before_forward(kwargs, message):
    validator = getattr(_module, "validate_bounded_rl_token_statistics", None)
    assert callable(validator), "bounded RL token-statistics capability gate is missing"
    with pytest.raises((ValueError, NotImplementedError), match=message):
        validator(enabled=True, **kwargs)
