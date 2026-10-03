"""Stream real checkpoint layers through the independent pinned HF decoder.

One layer is resident on CUDA at a time. HF retains its original GR, GDN,
QSA selection, MoE and N-gram hashing; only the huge table's storage is replaced
by an independent safetensors slice reader. No MCA weight conversion is used.
"""
import argparse
from contextlib import ExitStack
import gc
import json
from pathlib import Path
import time
from types import MethodType, SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F
from safetensors import safe_open

from conversion_parity_probe import hfmod


class SlicedEmbedding(nn.Module):
    def __init__(self, checkpoint, index, prefix, dtype=torch.bfloat16):
        super().__init__()
        self.checkpoint, self.index = checkpoint, index
        self.dtype = dtype
        self.keys = sorted((key for key in index if key.startswith(prefix)),
                           key=lambda key: int(key.split("shard_")[1].split(".")[0]))
        self.ends = []
        total = 0
        for key in self.keys:
            with safe_open(checkpoint / index[key], framework="pt", device="cpu") as source:
                rows, self.width = source.get_slice(key).get_shape()
                total += rows
                self.ends.append(total)

    @property
    def weight(self):
        # HF uses only this property's device to place lookup indices.
        return torch.empty(0, device="cpu")

    def forward(self, ids):
        flat = ids.cpu().flatten()
        unique, inverse = torch.unique(flat, sorted=True, return_inverse=True)
        values = torch.empty((len(unique), self.width), dtype=torch.bfloat16)
        start = 0
        for key, end in zip(self.keys, self.ends):
            positions = ((unique >= start) & (unique < end)).nonzero().flatten()
            if len(positions):
                with safe_open(self.checkpoint / self.index[key], framework="pt", device="cpu") as source:
                    view = source.get_slice(key)
                    for position in positions.tolist():
                        row = int(unique[position]) - start
                        values[position] = view[row:row+1][0]
            start = end
        return values[inverse].reshape(*ids.shape, self.width).to(self.dtype)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--mca-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-ids", type=Path)
    parser.add_argument("--lengths", type=int, nargs="+", default=[128])
    parser.add_argument("--reference-dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--save-layer-states", action="store_true")
    parser.add_argument("--secondary-output", type=Path,
                        help="Compare another independent run with this reference for precision diagnosis")
    parser.add_argument("--router-token-shards", type=int, default=1,
                        help="Diagnostic only: partition HF router GEMM tokens to match sequence parallelism")
    cli = parser.parse_args()
    cli.output.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(0)
    dtype = torch.bfloat16 if cli.reference_dtype == "bf16" else torch.float32
    config = SimpleNamespace(**{"seed": 1234, "norm_topk_prob": True,
                                **json.loads((cli.model / "config.json").read_text())["text_config"]})
    config._attn_implementation = "eager"
    index = json.loads((cli.model / "model.safetensors.index.json").read_text())["weight_map"]
    tick = time.monotonic()
    metrics = []

    def record(event, **values):
        entry = dict(event=event, elapsed=time.monotonic()-tick,
                     gpu_peak=torch.cuda.max_memory_allocated(), **values)
        metrics.append(entry)
        with (cli.output / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(entry) + "\n")
        print(json.dumps(entry), flush=True)

    if cli.router_token_shards < 1:
        raise ValueError("router-token-shards must be positive")
    record("reference_configuration", dtype=cli.reference_dtype,
           router_token_shards=cli.router_token_shards)

    def partitioned_router(router, hidden_states):
        hidden_states = hidden_states.reshape(-1, router.hidden_dim)
        if hidden_states.shape[0] % cli.router_token_shards:
            raise ValueError("router token count must divide evenly into diagnostic shards")
        logits = torch.cat([F.linear(part, router.weight)
                            for part in hidden_states.chunk(cli.router_token_shards, 0)], 0)
        probs = F.softmax(logits, dtype=torch.float32, dim=-1)
        selected, indices = torch.topk(probs, router.top_k, dim=-1)
        if router.norm_topk_prob:
            selected = selected / selected.sum(dim=-1, keepdim=True)
        return logits, selected.to(logits.dtype), indices

    def load_weights(prefix):
        weights = {}
        with ExitStack() as stack:
            files = {}
            for key, filename in index.items():
                if not key.startswith(prefix) or ".ngram_embedding.shard_" in key:
                    continue
                if filename not in files:
                    files[filename] = stack.enter_context(safe_open(cli.model / filename, framework="pt", device="cpu"))
                weights[key[len(prefix):]] = files[filename].get_tensor(key)
        return weights

    def tensor(key):
        with safe_open(cli.model / index[key], framework="pt", device="cpu") as source:
            return source.get_tensor(key).to(device="cuda", dtype=dtype)

    def compare(name, length, actual, expected):
        assert actual.shape == expected.shape, (name, actual.shape, expected.shape)
        difference = actual.float() - expected.float()
        record("comparison", name=name, length=length,
               relative_l2=float(difference.norm()/expected.float().norm().clamp_min(1e-12)),
               max_abs=float(difference.abs().max()), mean_abs=float(difference.abs().mean()))

    streams, tokens, positions, masks = {}, {}, {}, {}
    embedding = tensor("model.language_model.embed_tokens.weight")
    with torch.no_grad():
        for length in cli.lengths:
            ids = (torch.load(cli.input_ids, weights_only=True)[:, :length].cuda() if cli.input_ids
                   else (torch.arange(length, device="cuda") % 1000 + 100).unsqueeze(0))
            tokens[length] = ids
            streams[length] = F.embedding(ids, embedding).repeat(1, 1, config.hc_count)
            rope = config.rope_parameters
            width = int(config.head_dim * rope["partial_rotary_factor"])
            frequency = 1 / rope["rope_theta"] ** (torch.arange(0, width, 2, device="cuda").float() / width)
            angles = torch.arange(length, device="cuda").float().unsqueeze(-1) * frequency
            angles = torch.cat([angles, angles], dim=-1).unsqueeze(0)
            positions[length] = (angles.cos().to(dtype), angles.sin().to(dtype))
            masks[length] = torch.zeros((1, 1, length, length), dtype=dtype, device="cuda")
            masks[length].masked_fill_(torch.ones(length, length, device="cuda", dtype=torch.bool).triu(1),
                                       torch.finfo(dtype).min)
        del embedding
        for number in range(config.num_hidden_layers):
            with torch.device("meta"):
                layer = hfmod.Qwen4ExpTextDecoderLayer(config, number)
            if layer.ple is not None:
                layer.ple.ple_embedding.ngram_embedding = SlicedEmbedding(
                    cli.model, index, f"model.language_model.layers.{number}.ple.ple_embedding.ngram_embedding.shard_", dtype)
            weights = load_weights(f"model.language_model.layers.{number}.")
            layer.load_state_dict(weights, strict=True, assign=True)
            del weights
            layer = layer.to(device="cuda", dtype=dtype).eval()
            if cli.router_token_shards > 1:
                layer.mlp.gate.forward = MethodType(partitioned_router, layer.mlp.gate)
            record("layer_loaded", layer=number)
            for length in cli.lengths:
                streams[length] = layer(streams[length], positions[length], attention_mask=masks[length],
                                        ple_input_ids=tokens[length])
                expected = torch.load(cli.mca_output / f"hidden-{length}-{number}.pt",
                                      weights_only=True, map_location="cuda")
                compare(f"accumulated-layer-{number}", length, expected, streams[length])
                if cli.save_layer_states:
                    torch.save(streams[length].cpu(), cli.output / f"hidden-{length}-{number}.pt")
                if cli.secondary_output:
                    secondary = torch.load(cli.secondary_output / f"hidden-{length}-{number}.pt",
                                           weights_only=True, map_location="cuda")
                    compare(f"secondary-layer-{number}", length, secondary, streams[length])
                    del secondary
                previous = "embedding" if number == 0 else str(number - 1)
                layer_input = torch.load(cli.mca_output / f"hidden-{length}-{previous}.pt",
                                         weights_only=True, map_location="cuda").to(dtype)
                if number == 0:
                    layer_input = layer_input.repeat(1, 1, config.hc_count)
                local = layer(layer_input, positions[length], attention_mask=masks[length],
                              ple_input_ids=tokens[length])
                compare(f"same-input-layer-{number}", length, expected, local)
                del expected, layer_input, local
            del layer
            gc.collect()
            torch.cuda.empty_cache()
        with torch.device("meta"):
            mixer = hfmod.Qwen4ExpTextGatedResidual(config, use_combine=False)
        mixer.load_state_dict(load_weights("model.language_model.hyper_connection_mixer."), strict=True, assign=True)
        mixer = mixer.to(device="cuda", dtype=dtype).eval()
        head = tensor("lm_head.weight")
        for length in cli.lengths:
            hidden = mixer(streams[length])
            assert hidden.shape == (1, length, config.hidden_size)
            labels = tokens[length].roll(-1, -1)
            losses = []
            for offset in range(0, length, 128):
                logits = F.linear(hidden[:, offset:offset+128], head)
                losses.append(F.cross_entropy(logits.float().flatten(0, 1), labels[:, offset:offset+128].flatten(),
                                              reduction="none"))
            loss = torch.cat(losses).unsqueeze(0)
            torch.save(loss.cpu(), cli.output / f"losses-{length}.pt")
            actual = torch.load(cli.mca_output / f"losses-{length}-rank-0.pt", weights_only=True, map_location="cuda")
            compare("token-logprobs", length, actual, loss)
            record("loss", length=length, hf_mean=float(loss.mean()), mca_mean=float(actual.mean()))
    record("complete", layers=config.num_hidden_layers)
    failures = [entry for entry in metrics if entry["event"] == "comparison" and entry["relative_l2"] >= 0.03]
    assert not failures, failures


if __name__ == "__main__":
    main()
