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


__all__ = ["validate_training_capabilities"]
