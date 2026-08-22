from __future__ import annotations

import importlib
import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from transformers.configuration_utils import CONFIG_NAME as HF_CONFIG_NAME

from ...constants import (
    ASCEND_MXFP8_QUANT_TYPE,
    FLOAT_QUANT_TYPE,
    QUANT_MODEL_DESCRIPTION_NAME,
)

_QUANT_DESCRIPTION_META_KEYS = frozenset({"quant_method", "group_size", "version", "model_quant_type"})

_TRAINABLE_STATE_DICT_LOADER_ENV = "ROLL_ASCEND_MXFP8_STATE_DICT_LOADER"
_TRAINABLE_STATE_DICT_LOADER_CANDIDATES = (
    "mindspeed.core.transformer.custom_layers.transformer_engine.load_modelslim_mxfp8_state_dict",
    "mindspeed.core.transformer.custom_layers.transformer_engine.load_ascend_mxfp8_state_dict",
)


class AscendMxfp8CheckpointError(ValueError):
    """Raised when an Ascend ModelSlim MXFP8 checkpoint is malformed or unsupported."""


@dataclass(frozen=True)
class AscendMxfp8Weight:
    """Logical ModelSlim MXFP8 weight with its sidecar scale tensor."""

    name: str
    weight: torch.Tensor
    scale: torch.Tensor
    scale_name: str
    quant_type: str


def normalize_quant_description(raw_description: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize flat and nested ModelSlim quant descriptions."""
    if not isinstance(raw_description, Mapping):
        raise TypeError("Ascend MXFP8 quant description must be a mapping.")

    nested_description = raw_description.get("quant_description")
    if isinstance(nested_description, Mapping):
        description = dict(nested_description)
        for key in _QUANT_DESCRIPTION_META_KEYS:
            if key in raw_description:
                description.setdefault(key, raw_description[key])
        return description

    return dict(raw_description)


def is_ascend_mxfp8_quant_description(description: Mapping[str, Any] | None) -> bool:
    """Return True when a quant description represents Ascend MXFP8."""
    if not isinstance(description, Mapping):
        return False
    if str(description.get("quant_method", "")).lower() != "ascend":
        return False

    model_quant_type = description.get("model_quant_type")
    if model_quant_type is None:
        return any(str(value).upper() == ASCEND_MXFP8_QUANT_TYPE for value in description.values())
    return "MXFP8" in str(model_quant_type).upper()


def detect_ascend_mxfp8_quant_description(model_path: str) -> dict[str, Any] | None:
    """Return local Ascend MXFP8 metadata when the checkpoint declares it."""
    description_path = os.path.join(model_path, QUANT_MODEL_DESCRIPTION_NAME)
    if os.path.isfile(description_path):
        with open(description_path, "r", encoding="utf-8") as f:
            description = normalize_quant_description(json.load(f))
        if not is_ascend_mxfp8_quant_description(description):
            raise AscendMxfp8CheckpointError(f"{description_path} is not an Ascend MXFP8 quant description.")
        return description

    config_path = os.path.join(model_path, HF_CONFIG_NAME)
    if not os.path.isfile(config_path):
        return None
    with open(config_path, "r", encoding="utf-8") as f:
        quantization_config = json.load(f).get("quantization_config")
    if not quantization_config:
        return None

    description = normalize_quant_description(quantization_config)
    if is_ascend_mxfp8_quant_description(description):
        return description
    if str(description.get("quant_method", "")).lower() == "ascend":
        raise AscendMxfp8CheckpointError(f"{config_path}.quantization_config is not a supported Ascend MXFP8 format.")
    return None


def load_ascend_mxfp8_quant_description(model_path: str) -> dict[str, Any]:
    """Load required Ascend MXFP8 metadata from a local model directory."""
    description = detect_ascend_mxfp8_quant_description(model_path)
    if description is not None:
        return description

    raise AscendMxfp8CheckpointError(
        f"Ascend MXFP8 checkpoint requires {QUANT_MODEL_DESCRIPTION_NAME} or "
        f"{HF_CONFIG_NAME}.quantization_config with quant_method='ascend'."
    )


def scale_name_candidates(weight_name: str) -> tuple[str, ...]:
    """Return supported ModelSlim scale sidecar names for a quantized weight name."""
    candidates = (f"{weight_name}_scale", f"{weight_name}_scale_inv")
    if not weight_name.endswith(".weight"):
        return candidates
    prefix = weight_name.removesuffix(".weight")
    return (*candidates, f"{prefix}.scale", f"{prefix}.scale_inv")


class AscendMxfp8CheckpointAdapter:
    """Adapter for pairing ModelSlim MXFP8 weights with scale sidecars."""

    def __init__(self, quant_description: Mapping[str, Any], strict: bool = True):
        self.quant_description = normalize_quant_description(quant_description)
        if not is_ascend_mxfp8_quant_description(self.quant_description):
            raise AscendMxfp8CheckpointError("quant_description is not an Ascend MXFP8 description.")
        self.strict = strict

    @classmethod
    def from_model_path(cls, model_path: str, strict: bool = True) -> "AscendMxfp8CheckpointAdapter":
        """Build an adapter from a local ModelSlim checkpoint directory."""
        return cls(load_ascend_mxfp8_quant_description(model_path), strict=strict)

    def quant_type_for(self, name: str) -> str | None:
        """Return the ModelSlim quant type for *name*, if the description declares it."""
        value = self.quant_description.get(name)
        return None if value is None else str(value)

    def is_float_weight(self, name: str) -> bool:
        """Return True when ModelSlim declares *name* as a FLOAT weight."""
        return str(self.quant_description.get(name, "")).upper() == FLOAT_QUANT_TYPE

    def is_quantized_weight(self, name: str) -> bool:
        """Return True when ModelSlim declares *name* as an MXFP8 quantized weight."""
        quant_type = str(self.quant_description.get(name, "")).upper()
        return quant_type not in ("", FLOAT_QUANT_TYPE) and name not in _QUANT_DESCRIPTION_META_KEYS

    def is_scale_name(self, name: str) -> bool:
        """Return True when *name* is a known scale sidecar for any described quantized weight."""
        return any(name in scale_name_candidates(weight_name) for weight_name in self.quantized_weight_names())

    def quantized_weight_names(self) -> list[str]:
        """Return all weight names declared as quantized in the description."""
        return [name for name in self.quant_description if self.is_quantized_weight(name)]

    def find_scale_name(self, weight_name: str, state_dict: Mapping[str, torch.Tensor]) -> str | None:
        """Find the scale sidecar name present in *state_dict* for *weight_name*."""
        return next((name for name in scale_name_candidates(weight_name) if name in state_dict), None)

    def get_quantized_weight(
        self,
        weight_name: str,
        state_dict: Mapping[str, torch.Tensor],
    ) -> AscendMxfp8Weight:
        """Return a paired MXFP8 weight payload from a state dict."""
        if not self.is_quantized_weight(weight_name):
            raise AscendMxfp8CheckpointError(f"{weight_name} is not declared as an MXFP8 weight.")
        if weight_name not in state_dict:
            raise AscendMxfp8CheckpointError(f"Missing MXFP8 weight tensor: {weight_name}")

        scale_name = self.find_scale_name(weight_name, state_dict)
        if scale_name is None:
            raise AscendMxfp8CheckpointError(
                f"Missing MXFP8 scale tensor for {weight_name}; expected one of {scale_name_candidates(weight_name)}."
            )

        weight = state_dict[weight_name]
        scale = state_dict[scale_name]
        if not torch.is_tensor(weight) or not torch.is_tensor(scale):
            raise AscendMxfp8CheckpointError(f"MXFP8 weight and scale must be tensors for {weight_name}.")
        return AscendMxfp8Weight(
            name=weight_name,
            weight=weight,
            scale=scale,
            scale_name=scale_name,
            quant_type=str(self.quant_description[weight_name]),
        )

    def validate_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
        is_needed_name: Callable[[str], bool] | None = None,
    ) -> None:
        """Validate that every needed quantized weight has its scale sidecar."""
        for weight_name in self.quantized_weight_names():
            if (is_needed_name is None or is_needed_name(weight_name)) and weight_name in state_dict:
                self.get_quantized_weight(weight_name, state_dict)


def _load_symbol(reference: str) -> Callable[..., Any] | None:
    module_name, _, attr_name = reference.rpartition(".")
    if not module_name or not attr_name:
        raise ValueError(f"Invalid symbol reference: {reference!r}")
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return None
    value = getattr(module, attr_name, None)
    return value if callable(value) else None


def get_trainable_mxfp8_state_dict_loader() -> Callable[..., dict[str, torch.Tensor]] | None:
    """Discover the configured or built-in MindSpeed-TE MXFP8 state-dict loader."""
    user_loader = os.getenv(_TRAINABLE_STATE_DICT_LOADER_ENV)
    references = (user_loader,) if user_loader else _TRAINABLE_STATE_DICT_LOADER_CANDIDATES
    for reference in references:
        loader = _load_symbol(reference)
        if loader is not None:
            return loader
    return None


def require_trainable_mxfp8_state_dict_loader() -> Callable[..., dict[str, torch.Tensor]]:
    """Return a trainable MXFP8 state-dict loader or raise a clear unsupported error."""
    loader = get_trainable_mxfp8_state_dict_loader()
    if loader is not None:
        return loader

    raise RuntimeError(
        "Ascend ModelSlim MXFP8 fp8_param training requires a MindSpeed-TE loader that can materialize "
        "trainable FP8 parameters with scale metadata. ROLL detected the quantized checkpoint format, "
        "but no compatible loader was found. Set ROLL_ASCEND_MXFP8_STATE_DICT_LOADER to a callable loader "
        "or install a MindSpeed-TE version that exposes one; "
        "ROLL will not silently dequantize to BF16."
    )
