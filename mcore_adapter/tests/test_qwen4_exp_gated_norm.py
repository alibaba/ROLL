"""GDN normalization must retain FP32 through the output gate."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F


def apply_gated_norm():
    # Exercise the actual method without importing the GPU-only Megatron stack.
    path = Path(__file__).parents[1] / 'src/mcore_adapter/models/qwen4_exp/gated_delta_net.py'
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == '_apply_gated_norm')
    namespace = dict(torch=torch, __package__='mcore_adapter.models.qwen4_exp')
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['_apply_gated_norm']


class StoredNorm(torch.nn.Module):
    """Keep real BF16 parameters and model the existing early output cast."""
    def __init__(self, weight, zero_centered):
        super().__init__()
        self.weight = torch.nn.Parameter(weight)
        self.eps = 1e-6
        self.zero_centered_gamma = zero_centered

    def forward(self, x):
        weight = self.weight.float() + int(self.zero_centered_gamma)
        norm = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return (norm * weight).to(x.dtype)


@pytest.mark.parametrize('activation', ['sigmoid', 'silu'])
@pytest.mark.parametrize('zero_centered', [False, True])
def test_gdn_forward_and_gradients_cast_only_after_gate(activation, zero_centered):
    gen = torch.Generator().manual_seed(430)
    x = torch.randn(2, 7, 3, 128, generator=gen).bfloat16().requires_grad_()
    gate = torch.randn(x.shape, generator=gen).bfloat16().requires_grad_()
    weight = (torch.randn(128, generator=gen) * .1 + (0 if zero_centered else 1)).bfloat16()
    norm = StoredNorm(weight, zero_centered)
    config = SimpleNamespace(gdn_output_gate_type=activation,
                             layernorm_epsilon=norm.eps, layernorm_zero_centered_gamma=zero_centered)
    gdn = SimpleNamespace(out_norm=norm, config=config, act_fn=F.silu)
    actual = apply_gated_norm()(gdn, x, gate)
    xf, gf = x.reshape(-1, 128).float(), gate.reshape(-1, 128).float()
    y = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + norm.eps)
    y = y * (norm.weight.float() + int(zero_centered))
    expected = (y * (torch.sigmoid(gf) if activation == 'sigmoid' else F.silu(gf))).bfloat16()
    assert actual.shape == (42, 128) and actual.dtype == x.dtype
    assert torch.equal(actual, expected), 'GDN rounded normalized activations before applying the gate'
    upstream = torch.randn(actual.shape, generator=gen).bfloat16()
    inputs = x, gate, norm.weight
    observed = torch.autograd.grad(actual, inputs, upstream)
    reference = torch.autograd.grad(expected, inputs, upstream)
    for got, want in zip(observed, reference):
        assert torch.isfinite(got).all() and torch.count_nonzero(got)
        assert torch.equal(got, want)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('rows', [6, 894, 912, 8192])
@pytest.mark.parametrize('zero_centered', [False, True])
def test_gdn_cuda_sigmoid_norm_matches_native_rounding_and_gradients(rows, zero_centered):
    """The reciprocal-sqrt boundary must agree before MoE routing amplifies it."""
    from vllm.third_party.flash_linear_attention.ops.layernorm_guard import layer_norm_fwd

    gen = torch.Generator(device='cuda').manual_seed(430)
    x = (torch.randn(rows, 128, device='cuda', generator=gen) * .01).bfloat16().requires_grad_()
    gate = torch.randn(x.shape, device='cuda', generator=gen).bfloat16().requires_grad_()
    weight = (torch.randn(128, device='cuda', generator=gen) * .1 + (0 if zero_centered else 1)).bfloat16()
    norm = StoredNorm(weight, zero_centered)
    config = SimpleNamespace(gdn_output_gate_type='sigmoid')
    gdn = SimpleNamespace(out_norm=norm, config=config, act_fn=F.silu)
    actual = apply_gated_norm()(gdn, x, gate)
    effective_weight = norm.weight.float() + int(zero_centered)
    native = torch.empty_like(x)
    layer_norm_fwd(x.detach(), effective_weight.detach(), None, norm.eps,
                   z=gate.detach(), out=native, group_size=128, norm_before_gate=True,
                   is_rms_norm=True, activation='sigmoid')
    assert torch.equal(actual, native), f'{torch.count_nonzero(actual != native).item()} rounded elements differ'

    upstream = torch.randn(x.shape, device='cuda', generator=gen).bfloat16()
    observed = torch.autograd.grad(actual, (x, gate, norm.weight), upstream)
    # Independent FP64 algebra checks all three derivatives, including learned
    # zero-centered gamma; native inference itself has no backward oracle.
    reference_inputs = [value.detach().double().requires_grad_() for value in (x, gate, norm.weight)]
    rx, rg, rw = reference_inputs
    reference = rx * torch.rsqrt(rx.square().mean(-1, keepdim=True) + norm.eps)
    reference = reference * (rw + int(zero_centered)) * rg.sigmoid()
    expected = torch.autograd.grad(reference, reference_inputs, upstream.double())
    for got, want in zip(observed, expected):
        assert torch.isfinite(got).all() and torch.count_nonzero(got)
        error = (got.double() - want).norm() / want.norm().clamp_min(1e-30)
        assert error < .005, f'gradient relative L2 {error.item()}'
