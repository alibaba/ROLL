"""Explicit capability checks for Qwen3.8-Flash-Next training modes."""

from __future__ import annotations


def validate_training_capabilities(
    *,
    sequence_length: int,
    qsa_training_kernel: bool,
    token_budget: int = 2048,
    train_ngram_table: bool = False,
) -> dict[str, object]:
    """Return a capability report or fail before constructing an invalid run."""
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if sequence_length > token_budget and not qsa_training_kernel:
        raise ValueError(
            f"sequence_length={sequence_length} exceeds QSA token budget={token_budget}; "
            "a sparse QSA training kernel with backward support is required"
        )
    return {
        "qsa_mode": "sparse_autograd" if qsa_training_kernel else "dense_equivalent",
        "ngram_table_mode": "trainable_row_sharded" if train_ngram_table else "frozen_external",
        "sequence_length": sequence_length,
        "token_budget": token_budget,
    }


def estimate_full_parameter_memory(
    trainable_parameters: int,
    *,
    parameter_dtype_bytes: int = 2,
    gradient_dtype_bytes: int = 4,
    optimizer_state_bytes: int = 8,
    optimizer_cpu_offload: bool = False,
) -> dict[str, int]:
    """Estimate peak model/gradient/Adam bytes before a full-parameter run.

    The estimate is intentionally conservative: BF16 parameters remain on GPU,
    gradients use FP32 accumulation, and Adam keeps two FP32 moments. When CPU
    optimizer offload is enabled, moments are accounted on host memory rather
    than silently omitted from the budget.
    """
    if trainable_parameters <= 0:
        raise ValueError("trainable_parameters must be positive")
    model_bytes = trainable_parameters * parameter_dtype_bytes
    gradient_bytes = trainable_parameters * gradient_dtype_bytes
    optimizer_bytes = trainable_parameters * optimizer_state_bytes
    return {
        "gpu_parameter_bytes": model_bytes,
        "gpu_gradient_bytes": gradient_bytes,
        "gpu_optimizer_state_bytes": 0 if optimizer_cpu_offload else optimizer_bytes,
        "cpu_optimizer_state_bytes": optimizer_bytes if optimizer_cpu_offload else 0,
        "gpu_peak_bytes": model_bytes + gradient_bytes + (0 if optimizer_cpu_offload else optimizer_bytes),
        "cpu_peak_bytes": optimizer_bytes if optimizer_cpu_offload else 0,
    }


__all__ = ["estimate_full_parameter_memory", "validate_training_capabilities"]
