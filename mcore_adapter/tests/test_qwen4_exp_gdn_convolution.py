"""Keep the checkpoint's BF16 convolution boundary before the activation."""
import torch
from torch.nn import functional as F

from test_qwen4_exp_gdn_decay import environment
from test_qwen4_exp_model import make_model, tiny_config


def test_real_gdn_rounds_convolution_before_silu_and_preserves_gradients(environment):
    gdn = make_model(tiny_config()).decoder.layers[0].self_attention
    captured = {}

    def projection_hook(module, args, result):
        captured['projection'] = result[0]

    original = gdn.gated_delta_rule

    def recurrence(q, k, v, *args, **kwargs):
        captured['v'] = v
        return original(q, k, v, *args, **kwargs)

    hook = gdn.in_proj.register_forward_hook(projection_hook)
    gdn.gated_delta_rule = recurrence
    hidden = torch.randn(16, 2, 128, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    try:
        gdn(hidden, attention_mask=None)
        raw = captured['projection']
        # No previous state contributes to the first causal token. Derive the
        # result independently of the training convolution implementation.
        x = raw[0, :, :gdn.conv_dim]
        preactivation = (x.float() * gdn.conv1d.weight[:, 0, -1].float()).bfloat16()
        reference = F.silu(preactivation)[:, 2 * gdn.qk_dim:].reshape(2, gdn.num_value_heads, gdn.value_head_dim)
        actual = captured['v'][:, 0]
        torch.testing.assert_close(actual, reference, atol=0, rtol=0)
        fused_reference = F.silu(x.float() * gdn.conv1d.weight[:, 0, -1].float()).bfloat16()
        assert torch.any(F.silu(preactivation) != fused_reference), 'Fixture must detect fused activation rounding'
        inputs = raw, gdn.conv1d.weight
        actual_grad = torch.autograd.grad(actual.float().square().sum(), inputs, retain_graph=True)
        reference_grad = torch.autograd.grad(reference.float().square().sum(), inputs)
        for got, want in zip(actual_grad, reference_grad):
            assert got.dtype == want.dtype == torch.bfloat16
            assert torch.isfinite(got).all() and torch.count_nonzero(got)
            relative = (got.float() - want.float()).norm() / want.float().norm().clamp_min(1e-9)
            assert relative < .02
    finally:
        hook.remove()
        gdn.gated_delta_rule = original
