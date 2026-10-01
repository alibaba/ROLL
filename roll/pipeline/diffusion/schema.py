from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import torch

from roll.distributed.scheduler.protocol import DataProto


class DataProtoSchemaError(ValueError):
    """Raised when a DataProto violates the diffusion pipeline contract."""


def _batch_keys(data: DataProto) -> list[str]:
    return sorted(list(data.batch.keys())) if data.batch is not None else []


def _non_tensor_keys(data: DataProto) -> list[str]:
    return sorted(list(data.non_tensor_batch.keys())) if data.non_tensor_batch is not None else []


def _meta_keys(data: DataProto) -> list[str]:
    return sorted(list(data.meta_info.keys())) if data.meta_info is not None else []


def tensor_shapes(data: DataProto, keys: Iterable[str] | None = None) -> dict[str, tuple[int, ...]]:
    """Return tensor shapes for diagnostics without materializing tensor values."""
    if data.batch is None:
        return {}
    selected_keys = list(keys) if keys is not None else list(data.batch.keys())
    return {
        key: tuple(data.batch[key].shape)
        for key in selected_keys
        if key in data.batch and not _is_remote_key(data, key) and torch.is_tensor(data.batch[key])
    }


def _is_remote_key(data: DataProto, key: str) -> bool:
    remote_batch = getattr(data, "_remote_batch", None)
    return remote_batch is not None and key in remote_batch


def _schema_error(data: DataProto, context: str, message: str, shape_keys: Iterable[str] | None = None) -> DataProtoSchemaError:
    return DataProtoSchemaError(
        f"{context}: {message}; "
        f"available batch keys={_batch_keys(data)}; "
        f"available non_tensor keys={_non_tensor_keys(data)}; "
        f"available meta keys={_meta_keys(data)}; "
        f"relevant tensor shapes={tensor_shapes(data, shape_keys)}"
    )


def require_batch_keys(data: DataProto, keys: Sequence[str], context: str) -> None:
    """Fail fast when tensor batch fields required by an adapter are absent."""
    available = set(_batch_keys(data))
    missing = [key for key in keys if key not in available]
    if missing:
        raise _schema_error(data, context, f"missing batch keys={missing}", keys)


def require_non_tensor_keys(data: DataProto, keys: Sequence[str], context: str) -> None:
    """Fail fast when object-array metadata required by reward/training logic is absent."""
    available = set(_non_tensor_keys(data))
    missing = [key for key in keys if key not in available]
    if missing:
        raise _schema_error(data, context, f"missing non_tensor keys={missing}")


def require_meta_keys(data: DataProto, keys: Sequence[str], context: str) -> None:
    """Fail fast when control metadata required by rollout/train boundaries is absent."""
    available = set(_meta_keys(data))
    missing = [key for key in keys if key not in available]
    if missing:
        raise _schema_error(data, context, f"missing meta keys={missing}")


def require_tensor_ndim(data: DataProto, key: str, ndim: int, context: str) -> None:
    """Validate rank explicitly so downstream shape indexing errors are readable."""
    require_batch_keys(data, [key], context)
    tensor = data.batch[key]
    if not torch.is_tensor(tensor):
        raise _schema_error(data, context, f"batch['{key}'] must be a torch.Tensor", [key])
    if tensor.ndim != ndim:
        raise _schema_error(data, context, f"batch['{key}'].ndim must be {ndim}, got {tensor.ndim}", [key])


def require_tensor_min_ndim(data: DataProto, key: str, min_ndim: int, context: str) -> None:
    """Validate minimum rank for tensors whose trailing dimensions are model-specific."""
    require_batch_keys(data, [key], context)
    tensor = data.batch[key]
    if not torch.is_tensor(tensor):
        raise _schema_error(data, context, f"batch['{key}'] must be a torch.Tensor", [key])
    if tensor.ndim < min_ndim:
        raise _schema_error(data, context, f"batch['{key}'].ndim must be >= {min_ndim}, got {tensor.ndim}", [key])


def require_same_dim(data: DataProto, left_key: str, right_key: str, dim: int, context: str) -> None:
    """Validate that two tensors agree on one semantic dimension."""
    require_batch_keys(data, [left_key, right_key], context)
    left = data.batch[left_key]
    right = data.batch[right_key]
    if left.shape[dim] != right.shape[dim]:
        raise _schema_error(
            data,
            context,
            f"batch['{left_key}'].shape[{dim}]={left.shape[dim]} must equal "
            f"batch['{right_key}'].shape[{dim}]={right.shape[dim]}",
            [left_key, right_key],
        )


def require_dim_relation(
    data: DataProto,
    left_key: str,
    right_key: str,
    left_dim: int,
    right_dim: int,
    offset: int,
    context: str,
) -> None:
    """Validate a semantic relation such as trajectory length = step length + 1."""
    require_batch_keys(data, [left_key, right_key], context)
    left = data.batch[left_key]
    right = data.batch[right_key]
    expected = right.shape[right_dim] + offset
    if left.shape[left_dim] != expected:
        raise _schema_error(
            data,
            context,
            f"batch['{left_key}'].shape[{left_dim}]={left.shape[left_dim]} must equal "
            f"batch['{right_key}'].shape[{right_dim}] + {offset}={expected}",
            [left_key, right_key],
        )


def require_non_tensor_length(data: DataProto, key: str, expected_length: int, context: str) -> None:
    """Validate object-array length against the tensor batch size."""
    require_non_tensor_keys(data, [key], context)
    values = data.non_tensor_batch[key]
    if not isinstance(values, np.ndarray):
        raise _schema_error(data, context, f"non_tensor_batch['{key}'] must be a numpy array")
    if len(values) != expected_length:
        raise _schema_error(
            data,
            context,
            f"len(non_tensor_batch['{key}'])={len(values)} must equal batch size={expected_length}",
        )


# -------------------- Rollout batch validation dispatch --------------------


def validate_diffusion_rollout_batch(batch: DataProto, algorithm: str, pipeline_config) -> None:
    """Dispatch to algorithm-specific rollout batch validation."""
    if algorithm == "flowgrpo":
        _flowgrpo_validate_rollout_batch(batch, pipeline_config)
    elif algorithm == "diffnft":
        _diffnft_validate_rollout_batch(batch, pipeline_config)
    else:
        raise ValueError(f"Unsupported algorithm={algorithm!r}")


def _flowgrpo_validate_rollout_batch(batch: DataProto, pipeline_config) -> None:
    context = "FlowGRPO rollout batch validation"

    keys = ["all_latents", "all_timesteps", "rollout_log_probs", "prompt_embeds", "prompt_embeds_mask", "scores"]
    generation_config = batch.meta_info.get("generation_config", {}) if batch.meta_info is not None else {}
    guidance_scale = float(generation_config.get("guidance_scale", 1.0))
    if guidance_scale > 1.0:
        keys.extend(["negative_prompt_embeds", "negative_prompt_embeds_mask"])
    require_batch_keys(batch, keys, context)

    require_non_tensor_keys(batch, ["id", "domain", "ground_truth"], context)
    require_tensor_ndim(batch, "all_timesteps", 2, context)
    require_tensor_ndim(batch, "rollout_log_probs", 2, context)
    require_same_dim(batch, "rollout_log_probs", "all_timesteps", 1, context)

    all_timesteps = batch.batch["all_timesteps"]
    # RowRemoteBatch does not carry tensor shapes. Avoid fetching the large latent
    # trajectory on the driver just for validation; the training worker validates
    # and consumes it after prefetching its local DP shard.
    if not _is_remote_key(batch, "all_latents"):
        require_tensor_min_ndim(batch, "all_latents", 2, context)
        require_dim_relation(batch, "all_latents", "all_timesteps", 1, 1, 1, context)
    require_non_tensor_length(batch, "id", int(all_timesteps.shape[0]), context)


def _diffnft_validate_rollout_batch(batch: DataProto, pipeline_config) -> None:
    context = "DiffNFT rollout batch validation"

    # Same field names as FlowGRPO; rollout still emits log-probs (zeros in ODE mode).
    # The ``all_`` prefix is historical naming: the tensors hold only the SDE-window
    # steps (DiffNFT: final latent x0 only), not the full diffusion trajectory.
    keys = ["all_latents", "all_timesteps", "rollout_log_probs", "prompt_embeds", "prompt_embeds_mask", "scores"]
    generation_config = batch.meta_info.get("generation_config", {}) if batch.meta_info is not None else {}
    guidance_scale = float(generation_config.get("guidance_scale", 1.0))
    if guidance_scale > 1.0:
        keys.extend(["negative_prompt_embeds", "negative_prompt_embeds_mask"])
    require_batch_keys(batch, keys, context)

    require_non_tensor_keys(batch, ["id", "domain", "ground_truth"], context)
    require_tensor_ndim(batch, "all_timesteps", 2, context)
    require_tensor_ndim(batch, "rollout_log_probs", 2, context)
    require_same_dim(batch, "rollout_log_probs", "all_timesteps", 1, context)

    # DiffNFT keeps only the final latent x0, so the trajectory length must be 1
    # (FlowGRPO keeps the full trajectory with length K + 1)
    if not _is_remote_key(batch, "all_latents"):
        require_tensor_min_ndim(batch, "all_latents", 2, context)
        all_latents = batch.batch["all_latents"]
        if all_latents.shape[1] != 1:
            raise _schema_error(
                batch,
                context,
                f"batch['all_latents'].shape[1] must be 1 for DiffNFT (only final latent x0), got {all_latents.shape[1]}",
                ["all_latents"],
            )

    # scores are interpreted as reward probabilities: [B] or [B, K]
    scores = batch.batch["scores"]
    if scores.ndim not in (1, 2):
        raise _schema_error(batch, context, f"batch['scores'].ndim must be 1 or 2, got {scores.ndim}", ["scores"])

    require_non_tensor_length(batch, "id", int(batch.batch["all_timesteps"].shape[0]), context)
