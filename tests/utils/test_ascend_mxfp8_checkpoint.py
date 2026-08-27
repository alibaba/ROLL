import json
import sys
from types import ModuleType

import pytest
import torch

import mcore_adapter.models.checkpoint_loaders.ascend_mxfp8 as ascend_mxfp8
from mcore_adapter.models.checkpoint_loaders.ascend_mxfp8 import (
    ASCEND_MXFP8_QUANT_TYPE,
    FLOAT_QUANT_TYPE,
    AscendMxfp8CheckpointAdapter,
    AscendMxfp8CheckpointError,
    detect_ascend_mxfp8_quant_description,
    load_ascend_mxfp8_quant_description,
    require_trainable_mxfp8_state_dict_loader,
    scale_name_candidates,
)


def _quant_description():
    return {
        "quant_method": "ascend",
        "group_size": 32,
        "model_quant_type": ASCEND_MXFP8_QUANT_TYPE,
        "model.layers.0.self_attn.q_proj.weight": ASCEND_MXFP8_QUANT_TYPE,
        "model.embed_tokens.weight": FLOAT_QUANT_TYPE,
    }


def test_load_quant_description_from_modelslim_file(tmp_path):
    path = tmp_path / "quant_model_description.json"
    path.write_text(json.dumps(_quant_description()), encoding="utf-8")

    description = load_ascend_mxfp8_quant_description(str(tmp_path))

    assert description["quant_method"] == "ascend"
    assert description["model.layers.0.self_attn.q_proj.weight"] == ASCEND_MXFP8_QUANT_TYPE


def test_load_quant_description_from_hf_config(tmp_path):
    config = {
        "model_type": "qwen3",
        "quantization_config": {
            "quant_method": "ascend",
            "quant_description": {
                "model_quant_type": ASCEND_MXFP8_QUANT_TYPE,
                "model.layers.0.mlp.down_proj.weight": ASCEND_MXFP8_QUANT_TYPE,
            },
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")

    description = load_ascend_mxfp8_quant_description(str(tmp_path))

    assert description["quant_method"] == "ascend"
    assert description["model.layers.0.mlp.down_proj.weight"] == ASCEND_MXFP8_QUANT_TYPE


def test_detect_quant_description_returns_none_for_standard_hf_checkpoint(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3"}), encoding="utf-8")

    assert detect_ascend_mxfp8_quant_description(str(tmp_path)) is None


def test_detect_quant_description_rejects_unsupported_ascend_format(tmp_path):
    config = {"quantization_config": {"quant_method": "ascend", "model_quant_type": "W8A8"}}
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(AscendMxfp8CheckpointError, match="not a supported Ascend MXFP8 format"):
        detect_ascend_mxfp8_quant_description(str(tmp_path))


def test_checkpoint_adapter_pairs_quantized_weight_with_scale():
    adapter = AscendMxfp8CheckpointAdapter(_quant_description())
    weight_name = "model.layers.0.self_attn.q_proj.weight"
    state_dict = {
        weight_name: torch.ones(4, 4, dtype=torch.uint8),
        f"{weight_name}_scale": torch.ones(4, dtype=torch.float32),
    }

    payload = adapter.get_quantized_weight(weight_name, state_dict)

    assert payload.name == weight_name
    assert payload.scale_name == f"{weight_name}_scale"
    assert torch.equal(payload.weight, state_dict[weight_name])
    assert torch.equal(payload.scale, state_dict[f"{weight_name}_scale"])


def test_checkpoint_adapter_keeps_float_weights_out_of_quantized_names():
    adapter = AscendMxfp8CheckpointAdapter(_quant_description())

    assert adapter.is_float_weight("model.embed_tokens.weight")
    assert "model.embed_tokens.weight" not in adapter.quantized_weight_names()


def test_checkpoint_adapter_rejects_missing_scale():
    adapter = AscendMxfp8CheckpointAdapter(_quant_description())
    weight_name = "model.layers.0.self_attn.q_proj.weight"

    with pytest.raises(AscendMxfp8CheckpointError, match="Missing MXFP8 scale tensor"):
        adapter.get_quantized_weight(weight_name, {weight_name: torch.ones(4, 4, dtype=torch.uint8)})


def test_scale_name_candidates_include_modelslim_variants():
    candidates = scale_name_candidates("model.layers.0.mlp.down_proj.weight")

    assert "model.layers.0.mlp.down_proj.weight_scale" in candidates
    assert "model.layers.0.mlp.down_proj.weight_scale_inv" in candidates
    assert "model.layers.0.mlp.down_proj.scale" in candidates


def test_trainable_state_dict_loader_missing_is_explicit(monkeypatch):
    monkeypatch.delenv("ROLL_ASCEND_MXFP8_STATE_DICT_LOADER", raising=False)
    monkeypatch.setattr(ascend_mxfp8, "_TRAINABLE_STATE_DICT_LOADER_CANDIDATES", ())

    with pytest.raises(RuntimeError, match="will not silently dequantize to BF16"):
        require_trainable_mxfp8_state_dict_loader()


def test_trainable_state_dict_loader_candidates_stay_minimal():
    candidates = ascend_mxfp8._TRAINABLE_STATE_DICT_LOADER_CANDIDATES

    assert candidates == (
        "mindspeed.core.transformer.custom_layers.transformer_engine.load_modelslim_mxfp8_state_dict",
        "mindspeed.core.transformer.custom_layers.transformer_engine.load_ascend_mxfp8_state_dict",
    )


def test_trainable_state_dict_loader_env_override(monkeypatch):
    module = ModuleType("fake_mxfp8_loader")

    def load_state_dict(**kwargs):
        return {}

    module.load_state_dict = load_state_dict
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setenv("ROLL_ASCEND_MXFP8_STATE_DICT_LOADER", f"{module.__name__}.load_state_dict")
    monkeypatch.setattr(ascend_mxfp8, "_TRAINABLE_STATE_DICT_LOADER_CANDIDATES", ())

    assert ascend_mxfp8.get_trainable_mxfp8_state_dict_loader() is load_state_dict
