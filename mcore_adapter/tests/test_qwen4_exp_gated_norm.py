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
    namespace = dict(torch=torch, GatedDeltaNet=object)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['Qwen4ExpGatedDeltaNet']._apply_gated_norm


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
