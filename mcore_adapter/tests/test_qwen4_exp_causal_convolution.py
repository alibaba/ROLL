"""Exercise batch isolation, strides, bias and gradients at a real 8K length."""
import os

import pytest
import torch
from torch.nn import functional as F


@pytest.mark.skipif(os.environ.get("RUN_QWEN4_GDN_DECAY_TESTS") != "1", reason="requires CUDA")
@pytest.mark.parametrize("batch,length,channels,bias", [(1, 1, 8, False), (2, 7, 16, True),
                                                       (2, 65, 32, False), (1, 8192, 1280, False)])
def test_causal_convolution_matches_independent_stencil_and_gradients(batch, length, channels, bias):
    from mcore_adapter.models.qwen4_exp.causal_convolution import causal_conv1d

    generator = torch.Generator(device="cuda").manual_seed(8107)
    # Strided batch/time dimensions and a gap between consecutive token rows.
    x = torch.randn(length, batch, channels + 3, generator=generator, device="cuda",
                    dtype=torch.bfloat16)[..., :channels].transpose(0, 1).detach().requires_grad_()
    weight = torch.randn(channels, 4, generator=generator, device="cuda",
                         dtype=torch.bfloat16, requires_grad=True)
    bias_value = torch.randn(channels, generator=generator, device="cuda",
                             dtype=torch.bfloat16, requires_grad=True) if bias else None
    actual = causal_conv1d(x, weight, bias_value)
    # Independent vectorized stencil. Each batch gets its own left zeros.
    windows = F.pad(x.float(), (0, 0, 3, 0)).unfold(1, 4, 1)
    accumulator = (windows * weight.float()).bfloat16().float().sum(-1)
    if bias:
        accumulator = accumulator + bias_value.float()
    expected = F.silu(accumulator).bfloat16()
    assert actual.shape == x.shape and actual.dtype == x.dtype
    torch.testing.assert_close(actual, expected, atol=.02, rtol=.02)
    relative = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert relative < 1e-4, "Detect lost BF16 product casts, even below model-level tolerance"
    dy = torch.randn(actual.shape, generator=generator, device="cuda", dtype=torch.bfloat16)
    inputs = (x, weight, bias_value) if bias else (x, weight)
    actual_grad = torch.autograd.grad(actual, inputs, dy)
    expected_grad = torch.autograd.grad(expected, inputs, dy)
    for got, want in zip(actual_grad, expected_grad):
        assert got.dtype == want.dtype == torch.bfloat16
        assert torch.isfinite(got).all() and torch.count_nonzero(got)
        torch.testing.assert_close(got, want, atol=.02, rtol=.02)
        relative = (got.float() - want.float()).norm() / want.float().norm().clamp_min(1e-9)
        assert relative < .02
