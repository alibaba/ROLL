"""Match native BF16 products across causal tokens, including training gradients."""
import torch
from torch.nn import functional as F

from test_qwen4_exp_gdn_decay import environment
from test_qwen4_exp_model import make_model, tiny_config


def test_real_gdn_rounds_each_product_before_sum_and_preserves_gradients(environment):
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
        # Unfold an independent depthwise stencil over every token and batch.
        # A first-token-only test cannot distinguish product rounding from
        # rounding the completed sum: there is only one nonzero product.
        x = raw.transpose(0, 1)[..., :gdn.conv_dim].float()
        weight = gdn.conv1d.weight[:, 0].float()
        width = weight.shape[-1]
        windows = F.pad(x, (0, 0, width - 1, 0)).unfold(1, width, 1)
        products = windows * weight
        preactivation = products.bfloat16().float().sum(-1)
        reference = F.silu(preactivation).bfloat16()[..., 2 * gdn.qk_dim:]
        reference = reference.reshape(2, raw.shape[0], gdn.num_value_heads, gdn.value_head_dim)
        actual = captured['v']
        torch.testing.assert_close(actual, reference, atol=1e-7, rtol=1e-5)
        sum_rounded = F.silu(products.sum(-1).bfloat16())
        assert torch.any(sum_rounded[:, 1:] != F.silu(preactivation).bfloat16()[:, 1:]), (
            'Fixture must detect rounding the sum instead of individual products')
        inputs = raw, gdn.conv1d.weight
        # The two paths have independent graphs above these shared inputs;
        # the training path needs a single backward, as in ROLL updates.
        actual_grad = torch.autograd.grad(actual.float().square().sum(), inputs)
        reference_grad = torch.autograd.grad(reference.float().square().sum(), inputs)
        for got, want in zip(actual_grad, reference_grad):
            assert got.dtype == want.dtype == torch.bfloat16
            assert torch.isfinite(got).all() and torch.count_nonzero(got)
            relative = (got.float() - want.float()).norm() / want.float().norm().clamp_min(1e-9)
            assert relative < .02
    finally:
        hook.remove()
        gdn.gated_delta_rule = original
