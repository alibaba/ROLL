"""GR normalization must preserve a token when its batch shape changes."""
import importlib.util
from pathlib import Path
import sys
import types

import pytest
import torch


_SOURCE = Path(__file__).parents[1] / 'src/mcore_adapter/models/qwen4_exp/hyperconnection.py'
_PACKAGE = '_gr_normalization_test'
_package = types.ModuleType(_PACKAGE)
_package.__path__ = [str(_SOURCE.parent)]
sys.modules[_PACKAGE] = _package
_SPEC = importlib.util.spec_from_file_location(f'{_PACKAGE}.hyperconnection', _SOURCE)
hc = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(hc)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA reduction regression')
def test_same_bf16_token_has_identical_norm_across_sequence_lengths():
    # CPU-generated public inputs avoid embedding checkpoint weights or traces.
    # Wide magnitudes expose row-count-dependent FP32 reduction rounding.
    generator = torch.Generator().manual_seed(2)
    token = (torch.randn(10240, generator=generator)
             * torch.linspace(-8, 8, 10240).exp2()).to(torch.bfloat16).cuda()
    weight = torch.empty(10240).uniform_(-0.5, 0.5, generator=generator).cuda()
    short = token.reshape(1, 1, -1).repeat(2, 1, 1)
    expected = hc.grouped_gemma_rmsnorm(short, weight, 1e-6, 4)[0, 0]
    for length in (149, 8192):
        longer = token.reshape(1, 1, -1).repeat(length, 1, 1)
        actual = hc.grouped_gemma_rmsnorm(longer, weight, 1e-6, 4)[0, 0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize('shared_weight', [False, True])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_grouped_norm_forward_and_gradients_match_double_precision_equations(shared_weight, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA gradient regression')
    generator = torch.Generator().manual_seed(91)
    groups, width, eps = 4, 17, 1e-6
    x = torch.randn(3, 2, groups * width, generator=generator).transpose(0, 1).to(device).requires_grad_()
    weight = torch.randn(width if shared_weight else groups * width,
                         generator=generator).to(device).requires_grad_()
    upstream = torch.randn(x.shape, generator=generator).to(device)
    output = hc.grouped_gemma_rmsnorm(x, weight, eps, groups)
    output.backward(upstream)

    grouped = x.detach().double().reshape(-1, groups, width)
    grad = upstream.double().reshape_as(grouped)
    affine = 1 + weight.detach().double().reshape(1, 1 if shared_weight else groups, width)
    variance = grouped.square().mean(-1, keepdim=True) + eps
    reciprocal_rms = variance.rsqrt()
    expected = grouped * reciprocal_rms * affine
    weighted_grad = grad * affine
    expected_dx = reciprocal_rms * (
        weighted_grad - grouped * (weighted_grad * grouped).mean(-1, keepdim=True) / variance
    )
    expected_dw = (grad * grouped * reciprocal_rms).sum((0, 1) if shared_weight else 0)
    torch.testing.assert_close(output.double(), expected.reshape_as(x), atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(x.grad.double(), expected_dx.reshape_as(x), atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(weight.grad.double(), expected_dw.reshape_as(weight), atol=3e-6, rtol=3e-6)
