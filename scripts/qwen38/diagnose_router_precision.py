"""Isolate expert selection precision on a saved real-checkpoint layer input."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch
from torch.nn import functional as F
from safetensors import safe_open

from full_hf_reference_probe import SlicedEmbedding, hfmod


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--mca", type=Path, required=True)
    parser.add_argument("--input-ids", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--length", type=int, default=8192)
    parser.add_argument("--output", type=Path, required=True)
    cli = parser.parse_args()
    torch.cuda.set_device(0)
    torch.set_float32_matmul_precision("highest")
    config = SimpleNamespace(**{"seed": 1234, "norm_topk_prob": True,
                                **json.loads((cli.model / "config.json").read_text())["text_config"]})
    config._attn_implementation = "eager"
    index = json.loads((cli.model / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = f"model.language_model.layers.{cli.layer}."
    with torch.device("meta"):
        layer = hfmod.Qwen4ExpTextDecoderLayer(config, cli.layer)
    if layer.ple is not None:
        layer.ple.ple_embedding.ngram_embedding = SlicedEmbedding(
            cli.model, index, prefix + "ple.ple_embedding.ngram_embedding.shard_", torch.float32)
    with ExitStack() as stack:
        files, weights = {}, {}
        for key, filename in index.items():
            if key.startswith(prefix) and ".ngram_embedding.shard_" not in key:
                if filename not in files:
                    files[filename] = stack.enter_context(safe_open(cli.model / filename, framework="pt"))
                weights[key[len(prefix):]] = files[filename].get_tensor(key)
        layer.load_state_dict(weights, strict=True, assign=True)
    del weights
    layer = layer.to(device="cuda", dtype=torch.float32).eval().requires_grad_(False)
    previous = "embedding" if cli.layer == 0 else str(cli.layer - 1)
    inputs = torch.load(cli.mca / f"hidden-{cli.length}-{previous}.pt", weights_only=True).cuda().float()
    if cli.layer == 0:
        inputs = inputs.repeat(1, 1, config.hc_count)
    expected = torch.load(cli.mca / f"hidden-{cli.length}-{cli.layer}.pt", weights_only=True).cuda().float()
    ids = torch.load(cli.input_ids, weights_only=True)[:, :cli.length].cuda()
    rope = config.rope_parameters
    width = int(config.head_dim * rope["partial_rotary_factor"])
    frequency = 1 / rope["rope_theta"] ** (torch.arange(0, width, 2, device="cuda").float() / width)
    angles = torch.arange(cli.length, device="cuda").float().unsqueeze(-1) * frequency
    angles = torch.cat([angles, angles], dim=-1).unsqueeze(0)
    mask = torch.zeros((1, 1, cli.length, cli.length), device="cuda")
    mask.masked_fill_(torch.ones(cli.length, cli.length, device="cuda", dtype=torch.bool).triu(1),
                      torch.finfo(torch.float32).min)
    measurements, saved = [], {}

    def measure(name, actual, reference):
        error = actual.float() - reference.float()
        result = dict(name=name, relative_l2=float(error.norm()/reference.float().norm().clamp_min(1e-12)),
                      max_abs=float(error.abs().max()), mean_abs=float(error.abs().mean()))
        measurements.append(result)
        print(json.dumps(result), flush=True)

    def capture(_module, args, output):
        saved["hidden"] = args[0].detach()
        saved["logits"], saved["probs"], saved["indices"] = (v.detach() for v in output)

    def forward():
        return layer(inputs, (angles.cos(), angles.sin()), attention_mask=mask, ple_input_ids=ids)

    with torch.no_grad():
        handle = layer.mlp.gate.register_forward_hook(capture)
        original = forward()
        handle.remove()
        measure("hf-original-vs-mca", original, expected)
        measure("hf-repeat-vs-original", forward(), original)
        logits = saved["logits"]
        raw_indices = logits.topk(layer.mlp.gate.top_k, dim=-1).indices
        disagree = (raw_indices.sort(-1).values != saved["indices"].sort(-1).values).any(-1)
        print(json.dumps(dict(name="topk-logits-vs-full-softmax", changed_tokens=int(disagree.sum()),
                              token_positions=disagree.nonzero().flatten().tolist())), flush=True)
        split_logits = torch.cat([F.linear(part, layer.mlp.gate.weight)
                                  for part in saved["hidden"].chunk(2, 0)], 0)
        measure("split-gemm-logits", split_logits, logits)
        split_indices = split_logits.topk(layer.mlp.gate.top_k, dim=-1).indices
        split_disagree = (split_indices.sort(-1).values != raw_indices.sort(-1).values).any(-1)
        print(json.dumps(dict(name="split-gemm-selection", changed_tokens=int(split_disagree.sum()),
                              token_positions=split_disagree.nonzero().flatten().tolist())), flush=True)

        def selected_softmax(router, hidden):
            values = F.linear(hidden.reshape(-1, router.hidden_dim), router.weight)
            selected, indices = values.topk(router.top_k, dim=-1)
            return values, selected.softmax(-1), indices

        layer.mlp.gate.forward = MethodType(selected_softmax, layer.mlp.gate)
        alternative = forward()
        measure("hf-selected-softmax-vs-mca", alternative, expected)
        measure("hf-selected-softmax-vs-original", alternative, original)

        def split_selected_softmax(router, hidden):
            values = torch.cat([F.linear(part, router.weight)
                                for part in hidden.reshape(-1, router.hidden_dim).chunk(2, 0)], 0)
            selected, indices = values.topk(router.top_k, dim=-1)
            return values, selected.softmax(-1), indices

        layer.mlp.gate.forward = MethodType(split_selected_softmax, layer.mlp.gate)
        partitioned = forward()
        measure("hf-split-router-vs-mca", partitioned, expected)
        token_error = (original - expected).abs().amax(-1).flatten()
        print(json.dumps(dict(name="mca-mismatch-positions", threshold=1e-5,
                              token_positions=(token_error > 1e-5).nonzero().flatten().tolist())), flush=True)
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    cli.output.write_text(json.dumps(dict(layer=cli.layer, length=cli.length, measurements=measurements,
                                         selection_changed=int(disagree.sum()),
                                         split_selection_changed=int(split_disagree.sum())), indent=2) + "\n")


if __name__ == "__main__":
    main()
