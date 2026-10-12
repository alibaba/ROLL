"""Qwen3.8-Flash-Next runtime defaults for ROLL's vLLM workers.

Flash-Next uses the Qwen3.5 hybrid GDN implementation.  The vLLM FlashInfer
prefill path is numerically different from the training path on Hopper, while
the Triton path follows the model's reference recurrence.  Keep the default
local to this model family and preserve an explicit caller override.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any


def _is_qwen38_flash_next(model_name_or_path: Any) -> bool:
    if not isinstance(model_name_or_path, str):
        return False
    # Only the model directory/repository name identifies this fallback.
    # An ancestor experiment directory can contain unrelated baseline models.
    name = model_name_or_path.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    normalized = "".join(ch for ch in name.lower() if ch.isalnum())
    return normalized.startswith("qwen38flashnext")


def configure_qwen38_gdn_backend(
    config: MutableMapping[str, Any], model_name_or_path: Any
) -> MutableMapping[str, Any]:
    """Apply the safe vLLM GDN prefill backend for Flash-Next.

    ``additional_config`` is consumed by vLLM's Qwen hybrid model.  Existing
    values are intentionally left untouched so experiments can opt into a
    different backend explicitly.  Other model families are not modified.
    Generic checkpoint directory names require an explicit backend setting;
    the helper does not infer model identity from their parent directories.
    """

    if not _is_qwen38_flash_next(model_name_or_path):
        return config

    additional_config = config.get("additional_config")
    if additional_config is None:
        additional_config = {}
        config["additional_config"] = additional_config
    if not isinstance(additional_config, MutableMapping):
        raise TypeError("additional_config must be a mutable mapping")
    additional_config.setdefault("gdn_prefill_backend", "triton")
    return config


__all__ = ["configure_qwen38_gdn_backend"]
