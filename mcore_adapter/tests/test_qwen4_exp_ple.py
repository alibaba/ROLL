"""PLE numeric regressions against the pinned Qwen4 HF equations."""
import copy
import importlib.util
import json
import os
import shutil
from pathlib import Path
import struct
import sys
import types

import pytest
import torch
from torch.nn import functional as F

ROOT = Path(__file__).parents[1] / "src/mcore_adapter/models/qwen4_exp"
package = types.ModuleType("_qwen4_ple_test")
package.__path__ = [str(ROOT)]
sys.modules[package.__name__] = package
spec = importlib.util.spec_from_file_location(f"{package.__name__}.ple_layer", ROOT / "ple_layer.py")
ple = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ple
spec.loader.exec_module(ple)


def make_layer():
    layer = ple.PLELayer(8, 8, 4, 4, 3, 2, 16, 0, table=torch.randn(64, 2))
    layer.ple_embedding.layer_multipliers.copy_(torch.tensor([13, 17, 29]))
    layer.ple_embedding.ngram_heads_vocab_sizes.copy_(torch.tensor([11, 13, 17, 19]))
    layer.ple_embedding.ngram_heads_offsets.copy_(torch.tensor([0, 11, 24, 41]))
    with torch.no_grad():
        for name in ("norm_key", "norm_query", "norm_conv"):
            getattr(layer, name).uniform_(-0.4, 0.7)
    return layer


def reference(layer, hidden, tokens, mask=None):
    def norm(x, w):
        groups = x.float().reshape(*x.shape[:-1], 4, 8)
        groups = groups * (groups.square().mean(-1, keepdim=True) + layer.eps).rsqrt()
        return (groups.flatten(-2) * (w.float() + 1)).to(x.dtype)
    embed = layer.ple_embedding(tokens).to(hidden)
    key = norm(F.linear(embed, layer.key_proj.weight), layer.norm_key).unflatten(-1, (4, 8))
    query = norm(hidden, layer.norm_query).unflatten(-1, (4, 8))
    gate = (query * key).sum(-1, keepdim=True) / 8 ** 0.5
    gate = gate.sign() * gate.abs().clamp_min(1e-6).sqrt()
    value = (gate.sigmoid() * F.linear(embed, layer.value_proj.weight).unsqueeze(-2)).flatten(-2)
    conv_input = norm(value, layer.norm_conv)
    if mask is not None:
        value = value * mask.unsqueeze(-1)
        conv_input = conv_input * mask.unsqueeze(-1)
    # Independent explicit depthwise stencil, including dilation and SiLU.
    convolved = torch.zeros_like(conv_input, dtype=torch.float32)
    for tap in range(4):
        delay = (3 - tap) * 3
        if delay < hidden.shape[1]:
            convolved[:, delay:] += conv_input[:, :hidden.shape[1] - delay].float() * layer.conv1d.weight[:, 0, tap].float()
    if conv_input.dtype == torch.bfloat16:
        # Follow the pinned HF BF16 convolution graph for gradient accumulation;
        # the explicit FP32 stencil above is the independent arithmetic oracle.
        convolved = F.conv1d(F.pad(conv_input.transpose(1, 2), (9, 0)),
                            layer.conv1d.weight, groups=32, dilation=3).transpose(1, 2)
    return value + F.silu(convolved.to(conv_input.dtype))


@pytest.mark.parametrize("padding", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_ple_forward_and_all_gradients_match_reference(padding, dtype):
    torch.manual_seed(713)
    device = os.environ.get("PLE_TEST_DEVICE", "cpu")
    layer = make_layer().to(device=device, dtype=dtype)
    ref = copy.deepcopy(layer)
    x = (torch.randn(2, 13, 32) * torch.tensor([0.1, 1., 3., 10.]).repeat_interleave(8)).to(device=device, dtype=dtype).requires_grad_()
    xr = x.detach().clone().requires_grad_()
    tokens = torch.tensor([[1, 2, 3, 0, 4, 5, 6, 7, 8, 9, 10, 0, 0], [0, 0, 1, 2, 0, 3, 4, 5, 6, 7, 8, 9, 10]], device=device)
    mask = tokens.ne(0) if padding else None
    actual = layer(x, tokens, valid_mask=mask) if padding else layer(x, tokens)
    expected = reference(ref, xr, tokens, mask)
    tolerance = dict(atol=1e-5, rtol=1e-4) if dtype == torch.float32 else dict(atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(actual, expected, **tolerance)
    cotangent = torch.randn_like(actual)
    (actual * cotangent).sum().backward()
    (expected * cotangent).sum().backward()
    torch.testing.assert_close(x.grad, xr.grad, **tolerance)
    for (name, p), (_, rp) in zip(layer.named_parameters(), ref.named_parameters()):
        assert p.grad is not None, name
        torch.testing.assert_close(p.grad, rp.grad, **tolerance, msg=name)


def test_hashing_eos_and_int64_overflow():
    layer = make_layer().ple_embedding
    layer.layer_multipliers.copy_(torch.tensor([2**62 + 1, 2**61 + 17, 2**60 + 9]))
    tokens = torch.tensor([[7, 5, 0, 3, 9], [0, 2, 4, 0, 1]])
    expected = []
    for row in tokens.tolist():
        rows, history = [], []
        for token in row:
            history.append(token)
            ids = []
            for n in (2, 3):
                value = 0
                for back in range(n):
                    t = history[-1-back] if back < len(history) else 0
                    value ^= (t * int(layer.layer_multipliers[back])) & ((1 << 64) - 1)
                if value >= 1 << 63:
                    value -= 1 << 64
                for head in range((n-2)*2, (n-1)*2):
                    ids.append(value % int(layer.ngram_heads_vocab_sizes[head]) + int(layer.ngram_heads_offsets[head]))
            rows.append(ids)
            if token == 0:
                history = []
        expected.append(rows)
    torch.testing.assert_close(layer.compute_ngram_ids(tokens), torch.tensor(expected), rtol=0, atol=0)


def test_table_is_external_and_does_not_follow_module_dtype():
    layer = make_layer()
    layer.double()
    output = layer.ple_embedding(torch.tensor([[1, 2, 3]]))
    assert output.dtype == torch.float32
    assert not any("table" in key for key in layer.state_dict())
    assert not any("table" in key for key, _ in layer.named_parameters())


def test_unattached_table_fails_before_embedding():
    layer = ple.PLELayer(8, 8, 4, 4, 3, 2, 16, 0)
    with pytest.raises(RuntimeError, match="attach|checkpoint"):
        layer(torch.randn(1, 2, 32), torch.tensor([[1, 2]]))


def write_safetensors(path, tensors):
    header, payload = {}, bytearray()
    for key, value in tensors.items():
        raw = value.contiguous().view(torch.uint8).numpy().tobytes()
        header[key] = {"dtype": {torch.bfloat16: "BF16", torch.int64: "I64"}[value.dtype],
                       "shape": list(value.shape), "data_offsets": [len(payload), len(payload)+len(raw)]}
        payload.extend(raw)
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def checkpoint_fixture(path):
    path.mkdir()
    prefix = "model.language_model.layers.1.ple.ple_embedding."
    table = torch.arange(72).reshape(36, 2).to(torch.bfloat16)
    # Deliberately put shard_10 before shard_2 in the index.
    tensors = {prefix+f"ngram_embedding.shard_{i}.weight": table[i*3:(i+1)*3]
               for i in sorted(range(12), key=str)}
    tensors.update({prefix+"layer_multipliers": torch.tensor([13, 17, 29]),
                    prefix+"ngram_heads_vocab_sizes": torch.tensor([5, 7, 11, 13]),
                    prefix+"ngram_heads_offsets": torch.tensor([0, 5, 12, 23])})
    write_safetensors(path / "weights.safetensors", tensors)
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {k: "weights.safetensors" for k in tensors}}))
    return table


def test_mmap_lookup_shard_boundaries_and_manifest_relocation(tmp_path):
    from _qwen4_ple_test.ngram_embedding import MMapNGramStore
    source = tmp_path / "source"
    table = checkpoint_fixture(source)
    store = MMapNGramStore(source, staging_rows=3)
    ids = torch.tensor([[35, 0, 3, 2, 30], [29, 10, 11, 10, 0]])
    torch.testing.assert_close(store.lookup(ids), table[ids], rtol=0, atol=0)
    relocated = tmp_path / "relocated"
    shutil.copytree(source, relocated)
    restored = MMapNGramStore(relocated, expected_manifest=store.manifest)
    torch.testing.assert_close(restored.lookup(ids), table[ids], rtol=0, atol=0)
    with pytest.raises(ValueError, match="outside"):
        store.lookup(torch.tensor([-1]))
    with pytest.raises(ValueError, match="outside"):
        store.lookup(torch.tensor([36]))
    changed = copy.deepcopy(store.manifest)
    changed["index_sha256"] = "changed"
    with pytest.raises(ValueError, match="manifest mismatch"):
        MMapNGramStore(source, expected_manifest=changed)


def test_attach_checkpoint_loads_exact_hash_and_keeps_table_external(tmp_path):
    source = tmp_path / "source"
    table = checkpoint_fixture(source)
    layer = ple.PLELayer(8, 8, 4, 4, 3, 2, 16, 0)
    manifest = layer.ple_embedding.attach_checkpoint(source)
    layer.double()
    ids = torch.tensor([[1, 2, 0, 3]])
    hashes = layer.ple_embedding.compute_ngram_ids(ids)
    torch.testing.assert_close(layer.ple_embedding(ids), table[hashes].flatten(-2), rtol=0, atol=0)
    assert manifest["identity_kind"] == "index_and_header_sha256"
    assert layer.ple_embedding.store.dtype == torch.bfloat16
    assert not any("shard" in key or "table" in key for key in layer.state_dict())
