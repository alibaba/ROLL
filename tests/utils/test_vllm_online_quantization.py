from types import SimpleNamespace

import pytest

from roll.utils.vllm_online_quantization import (
    ASCEND_MXFP8_QUANT_TYPE,
    FLOAT_QUANT_TYPE,
    apply_online_quantization_config,
    build_ascend_mxfp8_quant_description,
    default_load_format_for_quantization,
)


def test_build_ascend_mxfp8_quant_description_for_dense_qwen_like_model():
    hf_config = SimpleNamespace(model_type="qwen3", num_hidden_layers=2)

    description = build_ascend_mxfp8_quant_description(hf_config)

    assert description["quant_method"] == "ascend"
    assert description["group_size"] == 32
    assert description["model.layers.0.self_attn.q_proj.weight"] == ASCEND_MXFP8_QUANT_TYPE
    assert description["model.layers.0.self_attn.qkv_proj.weight"] == ASCEND_MXFP8_QUANT_TYPE
    assert description["model.layers.1.mlp.gate_proj.weight"] == ASCEND_MXFP8_QUANT_TYPE
    assert description["model.layers.1.mlp.gate_up_proj.weight"] == ASCEND_MXFP8_QUANT_TYPE
    assert description["model.embed_tokens.weight"] == FLOAT_QUANT_TYPE
    assert description["lm_head.weight"] == FLOAT_QUANT_TYPE


def test_build_ascend_mxfp8_quant_description_for_qwen_moe_model():
    hf_config = SimpleNamespace(
        model_type="qwen3_moe",
        num_hidden_layers=1,
        num_experts=8,
        moe_intermediate_size=768,
    )

    description = build_ascend_mxfp8_quant_description(hf_config)

    assert description["model.layers.0.mlp.gate.weight"] == FLOAT_QUANT_TYPE
    assert description["model.layers.0.mlp.router.weight"] == FLOAT_QUANT_TYPE
    assert description["model.layers.0.mlp.experts.0.gate_proj.weight"] == ASCEND_MXFP8_QUANT_TYPE
    assert description["model.layers.0.mlp.experts.0.up_proj.weight"] == ASCEND_MXFP8_QUANT_TYPE
    assert description["model.layers.0.mlp.experts.0.down_proj.weight"] == ASCEND_MXFP8_QUANT_TYPE
    assert description["model.layers.0.mlp.shared_expert.gate_proj.weight"] == ASCEND_MXFP8_QUANT_TYPE


def test_apply_online_quantization_config_injects_hf_overrides():
    kwargs = {
        "model": "Qwen/Qwen3-8B",
        "load_format": "auto",
        "online_quantization": "ascend_mxfp8",
        "online_quantization_config": {"group_size": 64},
        "hf_overrides": {"rope_scaling": {"type": "default"}},
    }
    hf_config = SimpleNamespace(model_type="qwen3", num_hidden_layers=1)

    quant_config = apply_online_quantization_config(kwargs, hf_config=hf_config)

    assert kwargs["quantization"] == "ascend"
    assert kwargs["load_format"] == "dummy"
    assert "online_quantization" not in kwargs
    assert "online_quantization_config" not in kwargs
    assert kwargs["hf_overrides"]["rope_scaling"] == {"type": "default"}
    assert kwargs["hf_overrides"]["quantization_config"] is quant_config
    assert quant_config["quant_method"] == "ascend"
    assert quant_config["group_size"] == 64


def test_apply_online_quantization_config_rejects_conflicting_quantization():
    kwargs = {
        "model": "Qwen/Qwen3-8B",
        "online_quantization": "ascend_mxfp8",
        "quantization": "fp8",
    }

    with pytest.raises(ValueError, match="requires strategy_config.quantization"):
        apply_online_quantization_config(kwargs, hf_config=SimpleNamespace(model_type="qwen3", num_hidden_layers=1))


def test_prequantized_ascend_defaults_to_auto_load_format():
    assert default_load_format_for_quantization({"quantization": "ascend"}) == "auto"


def test_online_ascend_mxfp8_defaults_to_dummy_load_format():
    assert (
        default_load_format_for_quantization(
            {"quantization": "ascend", "online_quantization": "ascend_mxfp8"}
        )
        == "dummy"
    )
