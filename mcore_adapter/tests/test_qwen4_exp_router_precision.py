"""Qwen4's BF16 gate must not inherit the generic large-MoE FP32 override."""
import os
from types import SimpleNamespace

import pytest
import torch


pytestmark = pytest.mark.skipif(
    os.environ.get('RUN_QWEN4_ROUTER_TESTS') != '1',
    reason='requires the Megatron/Transformer Engine environment',
)


def make_config(cls, **kwargs):
    return cls(
        num_layers=1, hidden_size=64, num_attention_heads=4,
        num_moe_experts=64, moe_router_topk=10, moe_ffn_hidden_size=128,
        bf16=True, params_dtype=torch.bfloat16, **kwargs,
    )


def test_default_qwen4_router_preserves_model_precision():
    from mcore_adapter.models.qwen4_exp.config_qwen4_exp import Qwen4ExpConfig

    config = make_config(Qwen4ExpConfig)
    assert config.moe_router_dtype is None


@pytest.mark.parametrize('dtype', ['fp32', 'fp64'])
def test_explicit_router_precision_is_preserved(dtype):
    from mcore_adapter.models.qwen4_exp.config_qwen4_exp import Qwen4ExpConfig

    assert make_config(Qwen4ExpConfig, moe_router_dtype=dtype).moe_router_dtype == dtype


def test_other_models_retain_large_moe_fp32_default():
    from mcore_adapter.models.model_config import McaModelConfig

    assert make_config(McaModelConfig).moe_router_dtype == 'fp32'


@pytest.mark.parametrize('dtype', [None, 'fp32'])
def test_real_router_gate_matches_projection_and_gradients(dtype):
    if not torch.cuda.is_available():
        pytest.skip('real router gating requires CUDA')
    from mcore_adapter.models.qwen4_exp.config_qwen4_exp import Qwen4ExpConfig
    from megatron.core.transformer.moe.router import Router

    config = make_config(Qwen4ExpConfig, moe_router_dtype=dtype)
    gen = torch.Generator(device='cuda').manual_seed(359)
    x = torch.randn(149, 64, generator=gen, device='cuda', dtype=torch.bfloat16).requires_grad_()
    weight = torch.randn(64, 64, generator=gen, device='cuda', dtype=torch.bfloat16).requires_grad_()
    # Exercise the real projection method independently of distributed dispatch.
    gate = SimpleNamespace(weight=weight, bias=None, config=config)
    actual = Router.gating(gate, x)
    reference_x = x.detach().requires_grad_()
    reference_w = weight.detach().requires_grad_()
    projection_dtype = torch.float32 if dtype == 'fp32' else x.dtype
    expected = torch.nn.functional.linear(reference_x.to(projection_dtype), reference_w.to(projection_dtype))
    assert actual.dtype == projection_dtype
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    upstream = torch.randn(actual.shape, generator=gen, device='cuda', dtype=actual.dtype)
    actual_grads = torch.autograd.grad(actual, (x, weight), upstream)
    expected_grads = torch.autograd.grad(expected, (reference_x, reference_w), upstream)
    for got, want in zip(actual_grads, expected_grads):
        torch.testing.assert_close(got, want, atol=0, rtol=0)
