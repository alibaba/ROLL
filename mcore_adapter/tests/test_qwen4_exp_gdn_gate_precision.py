"""Exercise beta precision through the real GDN forward and FLA backward."""
import importlib.util
import os
from pathlib import Path
import sys
import types

import torch

from test_qwen4_exp_gdn_decay import environment
from test_qwen4_exp_model import make_model, tiny_config


def test_recurrence_beta_keeps_fp32_and_projection_gradient(environment):
    model = make_model(tiny_config())
    gdn = model.decoder.layers[0].self_attention
    if os.environ.get('QWEN4_GDN_FORWARD_SOURCE'):
        path = Path(os.environ['QWEN4_GDN_FORWARD_SOURCE'])
        spec = importlib.util.spec_from_file_location('qwen4_gate_preview', path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        gdn.forward = types.MethodType(module.GatedDeltaNet.forward, gdn)
    captured = {}

    def projection_hook(module, args, result):
        captured['projection'] = result[0]
        result[0].retain_grad()

    original = gdn.gated_delta_rule

    def recurrence(*args, **kwargs):
        captured['beta'] = kwargs['beta']
        captured['beta'].retain_grad()
        return original(*args, **kwargs)

    hook = gdn.in_proj.register_forward_hook(projection_hook)
    gdn.gated_delta_rule = recurrence
    hidden = torch.randn(16, 2, 128, dtype=torch.bfloat16, device='cuda', requires_grad=True)
    try:
        output, _ = gdn(hidden, attention_mask=None)
        beta = captured['beta']
        # Casting an already-rounded sigmoid result back to FP32 must also fail.
        assert beta.dtype == torch.float32, 'GDN beta was rounded before recurrence'
        heads = gdn.num_v_heads_local_tp
        raw = captured['projection'].transpose(0, 1)[..., -2 * heads:-heads]
        expected = torch.sigmoid(raw.float())
        torch.testing.assert_close(beta, expected, atol=0, rtol=0)
        assert torch.any(expected != expected.bfloat16().float())
        output.float().square().mean().backward()
        assert beta.grad is not None and torch.isfinite(beta.grad).all() and beta.grad.norm() > 0
        actual_grad = captured['projection'].grad.transpose(0, 1)[..., -2 * heads:-heads]
        reference_raw = raw.detach().clone().requires_grad_()
        reference_grad = torch.autograd.grad(torch.sigmoid(reference_raw.float()), reference_raw, beta.grad)[0]
        torch.testing.assert_close(actual_grad, reference_grad, atol=0, rtol=0)
        assert hidden.grad is not None and torch.isfinite(hidden.grad).all() and hidden.grad.norm() > 0
        assert all(p.dtype == torch.bfloat16 for p in gdn.parameters())
    finally:
        hook.remove()
        gdn.gated_delta_rule = original
