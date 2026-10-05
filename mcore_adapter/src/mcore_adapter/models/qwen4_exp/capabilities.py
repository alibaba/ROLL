"""Explicit capability checks for Qwen3.8-Flash-Next training modes."""

from __future__ import annotations


def validate_bounded_rl_token_statistics(
    *,
    enabled: bool,
    chunk_size: int = 256,
    context_parallel_size: int = 1,
    sequence_packing: bool = False,
    mtp_num_layers: int | None = None,
    output_head_adapter: bool = False,
) -> dict[str, object]:
    """Validate the bounded RL projection before model execution."""
    if not isinstance(enabled, bool):
        raise ValueError("bounded_rl_token_statistics must be a boolean")
    if not enabled:
        return {"enabled": False}
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("bounded RL token statistics require a positive chunk_size")
    if context_parallel_size != 1:
        raise NotImplementedError("bounded RL token statistics currently require CP=1")
    if sequence_packing:
        raise NotImplementedError("bounded RL token statistics do not support sequence packing")
    if mtp_num_layers:
        raise NotImplementedError("bounded RL token statistics do not support MTP")
    if output_head_adapter:
        raise NotImplementedError("bounded RL token statistics do not support output-head adapters")
    return {
        "enabled": True,
        "output": "selected_logprob_and_entropy",
        "dtype": "float32",
        "chunk_size": chunk_size,
    }


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
    if train_ngram_table:
        raise NotImplementedError(
            "trainable NGram tables require a row-sharded training backend, which is not implemented; "
            "use the frozen external NGram table"
        )
    if sequence_length > token_budget and not qsa_training_kernel:
        raise ValueError(
            f"sequence_length={sequence_length} exceeds QSA token budget={token_budget}; "
            "a sparse QSA training kernel with backward support is required"
        )
    return {
        "qsa_mode": "sparse_autograd" if qsa_training_kernel else "dense_equivalent",
        "ngram_table_mode": "frozen_external",
        "sequence_length": sequence_length,
        "token_budget": token_budget,
    }


def estimate_full_parameter_memory(
    trainable_parameters: int,
    *,
    parameter_dtype_bytes: int = 2,
    gradient_dtype_bytes: int = 2,
    optimizer_state_bytes: int = 8,
    optimizer_cpu_offload: bool = False,
) -> dict[str, int]:
    """Estimate model, gradient, and mixed-precision Adam tensor storage.

    Defaults assume BF16 model parameters and GPU gradients, FP32 master
    parameters, and two FP32 Adam moments. Full optimizer CPU offload retains
    the model and its gradients on GPU and stores FP32 masters, moments, and a
    separate FP32 gradient copy on CPU, as in precision-aware Hybrid CPU Adam.
    ``optimizer_state_bytes`` describes the moments only.

    The legacy ``*_peak_bytes`` keys sum only these components; they are not
    conservative runtime peak estimates. Activations, frozen parameters,
    communication buffers, optimizer workspaces/temporary GPU gradient casts,
    allocator overhead, and checkpoint/load copies are excluded. No sharding
    or partial offload is inferred from the parameter count.
    """
    if trainable_parameters <= 0:
        raise ValueError("trainable_parameters must be positive")
    model_bytes = trainable_parameters * parameter_dtype_bytes
    gradient_bytes = trainable_parameters * gradient_dtype_bytes
    optimizer_bytes = trainable_parameters * optimizer_state_bytes
    master_parameter_bytes = trainable_parameters * 4
    cpu_gradient_bytes = trainable_parameters * 4 if optimizer_cpu_offload else 0
    gpu_optimizer_bytes = 0 if optimizer_cpu_offload else optimizer_bytes + master_parameter_bytes
    cpu_optimizer_bytes = optimizer_bytes + master_parameter_bytes if optimizer_cpu_offload else 0
    return {
        "gpu_parameter_bytes": model_bytes,
        "gpu_gradient_bytes": gradient_bytes,
        "gpu_master_parameter_bytes": 0 if optimizer_cpu_offload else master_parameter_bytes,
        "cpu_master_parameter_bytes": master_parameter_bytes if optimizer_cpu_offload else 0,
        "cpu_gradient_bytes": cpu_gradient_bytes,
        "gpu_optimizer_state_bytes": 0 if optimizer_cpu_offload else optimizer_bytes,
        "cpu_optimizer_state_bytes": optimizer_bytes if optimizer_cpu_offload else 0,
        "gpu_peak_bytes": model_bytes + gradient_bytes + gpu_optimizer_bytes,
        "cpu_peak_bytes": cpu_optimizer_bytes + cpu_gradient_bytes,
    }


__all__ = [
    "estimate_full_parameter_memory",
    "validate_bounded_rl_token_statistics",
    "validate_training_capabilities",
]
