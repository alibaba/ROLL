"""TrainingArguments must not override Qwen4Exp's fused MoE permute default.

TE's fused permute combines top-k expert outputs with a deterministic FP32
reduction; the unfused BF16 scatter_add rounds after every update. Upstream
added a TrainingArguments.moe_permute_fusion field defaulting to False, which
update_with_args then wrote over the model config's True, silently changing
the numerics of every run that did not set it.
"""
import importlib.util

import pytest
import torch

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("megatron") is None, reason="requires mcore_adapter runtime"
)


def resolve(**overrides):
    from mcore_adapter import TrainingArguments
    from mcore_adapter.models.qwen4_exp.config_qwen4_exp import Qwen4ExpConfig

    args = TrainingArguments(output_dir="unused", bf16=True, use_cpu=True, report_to=[])
    for name, value in overrides.items():
        setattr(args, name, value)
    config = Qwen4ExpConfig(
        num_layers=1, hidden_size=64, num_attention_heads=4, num_moe_experts=4,
        moe_router_topk=2, moe_ffn_hidden_size=128, bf16=True, params_dtype=torch.bfloat16,
    )
    assert config.moe_permute_fusion is True
    config.update_with_args(args, verbose=False)
    return config


def test_default_arguments_keep_the_model_config_value():
    assert resolve().moe_permute_fusion is True


@pytest.mark.parametrize("value", [True, False])
def test_explicit_argument_still_overrides(value):
    assert resolve(moe_permute_fusion=value).moe_permute_fusion is value
