from collections.abc import Mapping
from typing import Any


ASCEND_MXFP8_ONLINE_QUANTIZATION = "ascend_mxfp8"
ASCEND_MXFP8_QUANT_TYPE = "W8A8_MXFP8"
FLOAT_QUANT_TYPE = "FLOAT"

_ATTENTION_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "qkv_proj")
_DENSE_MLP_PROJECTIONS = ("gate_proj", "up_proj", "down_proj", "gate_up_proj")


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    return value


def default_load_format_for_quantization(kwargs: Mapping[str, Any]) -> str:
    return "auto" if kwargs.get("quantization") == "ascend" and not kwargs.get("online_quantization") else "dummy"


def _set_projections(description: dict[str, Any], prefix: str, names: tuple[str, ...], value: str) -> None:
    description.update({f"{prefix}.{name}.weight": value for name in names})


def _is_moe_model(config: Any, options: Mapping[str, Any]) -> bool:
    if "is_moe" in options:
        return bool(options["is_moe"])
    model_type = str(getattr(config, "model_type", "")).lower()
    expert_fields = ("num_experts", "num_local_experts", "n_routed_experts", "moe_intermediate_size")
    return "moe" in model_type or any(getattr(config, name, 0) not in (None, 0, 1) for name in expert_fields)


def _is_moe_layer(config: Any, layer_idx: int) -> bool:
    if layer_idx < int(getattr(config, "first_k_dense_replace", 0) or 0):
        return False
    frequency = getattr(config, "moe_layer_freq", 1)
    if isinstance(frequency, (list, tuple)):
        return bool(frequency[layer_idx])
    return layer_idx % int(frequency or 1) == 0


def build_ascend_mxfp8_quant_description(
    hf_config: Any,
    options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the ModelSlim description needed by vLLM-Ascend for Qwen rollout."""
    options = options or {}
    config = getattr(hf_config, "text_config", hf_config)
    num_layers = options.get("num_hidden_layers", getattr(config, "num_hidden_layers", None))
    if num_layers is None:
        raise ValueError("online_quantization=ascend_mxfp8 requires num_hidden_layers in the HF config.")

    description: dict[str, Any] = {
        "quant_method": "ascend",
        "group_size": int(options.get("group_size", 32)),
        "version": "1.0.0",
        "model_quant_type": ASCEND_MXFP8_QUANT_TYPE,
        "model.embed_tokens.weight": FLOAT_QUANT_TYPE,
        "lm_head.weight": FLOAT_QUANT_TYPE,
    }
    is_moe = _is_moe_model(config, options)
    for layer_idx in range(int(num_layers)):
        layer = f"model.layers.{layer_idx}"
        _set_projections(description, f"{layer}.self_attn", _ATTENTION_PROJECTIONS, ASCEND_MXFP8_QUANT_TYPE)
        mlp = f"{layer}.mlp"
        if is_moe and _is_moe_layer(config, layer_idx):
            _set_projections(description, mlp, ("gate", "router"), FLOAT_QUANT_TYPE)
            _set_projections(
                description, f"{mlp}.experts.0", _DENSE_MLP_PROJECTIONS[:3], ASCEND_MXFP8_QUANT_TYPE
            )
            description[f"{mlp}.experts.weight"] = ASCEND_MXFP8_QUANT_TYPE
            _set_projections(description, f"{mlp}.shared_expert", _DENSE_MLP_PROJECTIONS, ASCEND_MXFP8_QUANT_TYPE)
        else:
            _set_projections(description, mlp, _DENSE_MLP_PROJECTIONS, ASCEND_MXFP8_QUANT_TYPE)

    extra_description = options.get("extra_quant_description", {})
    description.update(_require_mapping(extra_description, "online_quantization_config.extra_quant_description"))
    return description


def apply_online_quantization_config(kwargs: dict[str, Any], hf_config: Any | None = None) -> dict[str, Any] | None:
    online_quantization = kwargs.pop("online_quantization", None)
    options = kwargs.pop("online_quantization_config", None) or {}
    if not online_quantization:
        return None
    if online_quantization != ASCEND_MXFP8_ONLINE_QUANTIZATION:
        raise ValueError(f"Unsupported online_quantization={online_quantization!r}.")
    options = _require_mapping(options, "online_quantization_config")
    if kwargs.get("quantization") not in (None, "ascend"):
        raise ValueError(
            "online_quantization=ascend_mxfp8 requires strategy_config.quantization "
            "to be omitted or set to 'ascend'."
        )

    kwargs.update(quantization="ascend", load_format="dummy")
    if hf_config is None:
        from transformers import AutoConfig

        model = kwargs.get("model")
        if not model:
            raise ValueError("online_quantization=ascend_mxfp8 requires the vLLM model path.")
        hf_config = AutoConfig.from_pretrained(
            model,
            trust_remote_code=bool(kwargs.get("trust_remote_code", True)),
            revision=kwargs.get("revision"),
        )

    hf_overrides = dict(_require_mapping(kwargs.get("hf_overrides") or {}, "hf_overrides"))
    description = {
        **build_ascend_mxfp8_quant_description(hf_config, options),
        **_require_mapping(hf_overrides.get("quantization_config", {}), "hf_overrides.quantization_config"),
    }
    if description.get("quant_method") != "ascend":
        raise ValueError("hf_overrides.quantization_config.quant_method must be 'ascend'.")
    kwargs["hf_overrides"] = {**hf_overrides, "quantization_config": description}
    return description
