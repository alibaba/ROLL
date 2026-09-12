import importlib.util
from pathlib import Path

import torch

_ROOT = Path(__file__).parents[1] / "src/mcore_adapter/models/qwen4_exp"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_hc = _load("hyperconnection")
_ple = _load("ple_layer")
hc_combine = _hc.hc_combine
hc_gate_mix = _hc.hc_gate_mix
grouped_gemma_rmsnorm = _hc.grouped_gemma_rmsnorm
PLELayer = _ple.PLELayer


def test_hyperconnection_spec_replaces_each_decoder_layer():
    source = (_ROOT / "hyperconnection_layer.py").read_text()
    assert "for layer_spec in spec.layer_specs" in source
    assert "layer_spec.module = HyperConnectionTransformerLayer" in source


def test_zero_block_hyperconnection_is_identity():
    residual = torch.randn(2, 3, 16, dtype=torch.float32)
    injection = torch.zeros(2, 3, 4)
    output = hc_combine(residual, torch.zeros(2, 3, 4), injection, 4)
    torch.testing.assert_close(output, residual)


def test_hyperconnection_read_gate_uses_mean_and_gemma_norm():
    x = torch.ones(1, 1, 8)
    normed = grouped_gemma_rmsnorm(x, torch.zeros(8), 1e-6, 2)
    torch.testing.assert_close(normed, torch.ones_like(x) / (1 + 1e-6) ** 0.5)
    mixed = hc_gate_mix(torch.ones(1, 1, 8), torch.zeros(1, 1, 8), 2)
    torch.testing.assert_close(mixed, torch.full((1, 1, 4), 0.5))


def test_ple_has_widened_stream_and_dilated_conv():
    layer = PLELayer(hidden_size=8, ple_embed_dim=8, hc_count=4, ngram_size=3, conv_kernel_size=4,
                     heads_per_ngram=2, vocab_size=16, eos_token_id=0)
    assert tuple(layer.key_proj.weight.shape) == (32, 8)
    assert tuple(layer.value_proj.weight.shape) == (8, 8)
    assert tuple(layer.conv1d.weight.shape) == (32, 1, 4)
    assert layer.conv1d.dilation == (3,)


def test_ple_forward_preserves_wide_stream_and_backpropagates():
    table = torch.randn(16, 2)
    layer = PLELayer(hidden_size=8, ple_embed_dim=8, hc_count=4, ngram_size=3, conv_kernel_size=4,
                     heads_per_ngram=2, vocab_size=16, eos_token_id=0, table=table)
    hidden = torch.randn(1, 5, 32, requires_grad=True)
    output = layer(hidden, torch.tensor([[1, 2, 3, 0, 4]]))
    assert output.shape == hidden.shape
    output.square().mean().backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    assert layer.key_proj.weight.grad is not None
