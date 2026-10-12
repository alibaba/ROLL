"""QSA indexer distillation temperature must default to the reference scaling.

The HF reference divides the summed relu(qk) indexer score by
sqrt(index_head_dim) before the KL target; the config used to hardcode 1.0,
which made the trained-indexer KL target ~sqrt(head_dim) times sharper than
the reference calibration. See qsa.default_indexer_temperature.
"""
import importlib.util

import pytest
import torch


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("megatron") is None, reason="requires mcore_adapter runtime"
)


def make_config(**kwargs):
    from mcore_adapter.models.qwen4_exp.config_qwen4_exp import Qwen4ExpConfig

    return Qwen4ExpConfig(
        num_layers=1, hidden_size=64, num_attention_heads=4,
        num_moe_experts=4, moe_router_topk=2, moe_ffn_hidden_size=128,
        bf16=True, params_dtype=torch.bfloat16, **kwargs,
    )


def test_default_temperature_resolves_to_sqrt_indexer_head_dim():
    config = make_config(indexer_head_dim=16)
    assert config.qsa_indexer_temperature == 4.0


def test_explicit_temperature_override_is_preserved():
    config = make_config(indexer_head_dim=16, qsa_indexer_temperature=1.0)
    assert config.qsa_indexer_temperature == 1.0


def test_default_temperature_without_qsa_falls_back_to_one():
    config = make_config()
    assert config.indexer_head_dim is None
    assert config.qsa_indexer_temperature == 1.0


def test_nonpositive_explicit_temperature_is_rejected():
    with pytest.raises(ValueError, match="temperature must be positive"):
        make_config(indexer_head_dim=16, qsa_indexer_temperature=0.0)


def test_negative_kl_coef_is_still_rejected():
    with pytest.raises(ValueError, match="KL coefficient must be nonnegative"):
        make_config(indexer_head_dim=16, qsa_indexer_kl_coef=-0.1)
