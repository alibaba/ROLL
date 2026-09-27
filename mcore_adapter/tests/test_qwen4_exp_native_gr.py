"""Native GR rounding boundaries and independently computed derivatives."""
import importlib.util
from pathlib import Path
import sys
import types

import pytest
import torch


ROOT = Path(__file__).parents[1] / 'src/mcore_adapter/models/qwen4_exp'
PACKAGE = '_qwen38_native_gr_test'
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package
spec = importlib.util.spec_from_file_location(f'{PACKAGE}.hyperconnection', ROOT / 'hyperconnection.py')
hc = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = hc
spec.loader.exec_module(hc)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='native GR kernels require CUDA')
@pytest.mark.parametrize('rows', [2, 149, 2048])
@pytest.mark.parametrize('operator', ['norm', 'silu', 'mix', 'combine'])
def test_gr_matches_native_bf16_rounding(operator, rows):
    native = pytest.importorskip('vllm.models.qwen3_8_flash_next.nvidia.ops.hc')
    gen = torch.Generator(device='cuda').manual_seed(729)
    groups, width = 4, 2560
    def sample(*shape):
        return torch.randn(shape, device='cuda', generator=gen).bfloat16()
    x = sample(rows, groups * width)
    if operator == 'norm':
        w = sample(groups * width)
        actual = hc.grouped_gemma_rmsnorm(x, w, 1e-6, groups)
        expected = native.grouped_gemma_rmsnorm(x, w, 1e-6, groups)
    elif operator == 'silu':
        x = sample(rows, 320)
        actual, expected = hc.hc_silu(x, groups), native.hc_silu(x, groups)
    elif operator == 'mix':
        g = sample(*x.shape)
        actual, expected = hc.hc_gate_mix(x, g, groups), native.hc_gate_mix(x, g, groups)
    else:
        block, inj = sample(rows, width), sample(rows, groups)
        actual = hc.hc_combine(x, block, inj, groups)
        expected = native.hc_combine(x, block, inj, groups)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='native GR kernels require CUDA')
@pytest.mark.parametrize('rows', [2, 149, 2048])
@pytest.mark.parametrize('shared_weight', [False, True])
def test_combined_residual_and_norm_match_native_fused_boundary(rows, shared_weight):
    native = pytest.importorskip('vllm.models.qwen3_8_flash_next.nvidia.ops.hc')
    gen = torch.Generator(device='cuda').manual_seed(945)
    groups, width = 4, 2560
    def sample(*shape):
        return torch.randn(shape, device='cuda', generator=gen).bfloat16()
    x, block, injection = sample(rows, groups * width), sample(rows, width), sample(rows, groups)
    weight = sample(width if shared_weight else groups * width)
    actual = hc.hc_combine(x, block, injection, groups)
    normed = hc.grouped_gemma_rmsnorm(actual, weight, 1e-6, groups, after_combine=True)
    expected, expected_normed = native.hc_combine_norm(x, block, injection, weight, 1e-6, groups)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(normed, expected_normed, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA derivatives')
@pytest.mark.parametrize('operator', ['norm', 'combined_norm', 'silu', 'mix', 'combine'])
def test_gr_cuda_gradients_match_independent_double_equations(operator):
    gen = torch.Generator().manual_seed(431)
    groups, width = 4, 17
    def sample(*shape):
        return torch.randn(shape, generator=gen).cuda().requires_grad_()
    x = sample(3, 2, groups * width).transpose(0, 1)
    if operator in ('norm', 'combined_norm'):
        values = (x, sample(groups * width))
        def reference(a, w):
            grouped = a.reshape(-1, groups, width)
            y = grouped * (grouped.square().mean(-1, keepdim=True) + 1e-6).rsqrt()
            return (y * (1 + w.reshape(1, groups, width))).reshape_as(a)
        actual = hc.grouped_gemma_rmsnorm(*values, 1e-6, groups, after_combine=operator == 'combined_norm')
    elif operator == 'silu':
        values = (x,)
        def reference(a):
            return (a / groups) * (a / groups).sigmoid()
        actual = hc.hc_silu(x, groups)
    elif operator == 'mix':
        values = (x, sample(*x.shape))
        def reference(a, g):
            return (a * g.sigmoid()).unflatten(-1, (groups, width)).mean(-2)
        actual = hc.hc_gate_mix(*values, groups)
    else:
        values = (x, sample(2, 3, width), sample(2, 3, groups))
        def reference(a, b, inj):
            write = 2 * (inj / groups).sigmoid()
            return (a.unflatten(-1, (groups, width)) + b.unsqueeze(-2) * write.unsqueeze(-1)).flatten(-2)
        actual = hc.hc_combine(*values, groups)
    independent = tuple(v.detach().double().requires_grad_() for v in values)
    expected = reference(*independent)
    grad = torch.randn(actual.shape, generator=gen).cuda()
    torch.testing.assert_close(actual.double(), expected, atol=2e-6, rtol=2e-6)
    observed = torch.autograd.grad(actual, values, grad)
    wanted = torch.autograd.grad(expected, independent, grad.double())
    for got, want in zip(observed, wanted):
        torch.testing.assert_close(got.double(), want, atol=3e-6, rtol=3e-6)
