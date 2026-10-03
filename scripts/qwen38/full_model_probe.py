"""Load the real text backbone and record distributed forward memory evidence.

Run under torchrun; this is a smoke probe, not independent HF parity or SFT.
Frozen N-gram tables are loaded through the production from_pretrained hook.
"""
import argparse
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist

from mcore_adapter import TrainingArguments
from mcore_adapter.models import AutoModel


def enable_decoder_streaming(module):
    """Keep this diagnostic's decoder on CPU except the active layer and PLE."""
    if module.training or any(parameter.requires_grad for parameter in module.parameters()):
        raise ValueError("decoder streaming is restricted to frozen evaluation probes")

    def offload_layer(layer, *_unused):
        layer.cpu()
        # The block invokes PLE before the transformer layer's forward hook.
        ple = getattr(layer, "ple", None)
        if ple is not None:
            ple.cuda()

    def load_layer(layer, _inputs):
        layer.cuda()

    handles = []
    for layer in module.decoder.layers:
        offload_layer(layer)
        handles.append(layer.register_forward_pre_hook(load_layer))
        handles.append(layer.register_forward_hook(offload_layer))
    torch.cuda.empty_cache()
    return handles


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[2048, 8192])
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--ep", type=int, default=8)
    parser.add_argument("--diagnose-replicas", action="store_true")
    parser.add_argument("--input-ids", type=Path)
    parser.add_argument("--input-manifest", type=Path)
    parser.add_argument("--save-layer-states", action="store_true")
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--router-dtype", choices=["model", "fp32"], default="model",
                        help="Compare checkpoint-native routing with an explicit FP32 gate")
    parser.add_argument("--compare-router-dtypes", action="store_true",
                        help="Evaluate both gate precisions using the same loaded weights and natural routes")
    parser.add_argument("--stream-decoder", action="store_true",
                        help="Diagnostic only: keep decoder parameters on CPU between layers")
    parser.add_argument("--deterministic-gdn", action="store_true",
                        help="Use Megatron's PyTorch GDN for independent FP32 semantic diagnosis")
    cli = parser.parse_args()
    output = Path(cli.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    args = TrainingArguments(
        output_dir=str(output), bf16=cli.dtype == "bf16", tensor_model_parallel_size=cli.tp,
        pipeline_model_parallel_size=1, context_parallel_size=1,
        expert_model_parallel_size=cli.ep, expert_tensor_parallel_size=1,
        sequence_parallel=cli.tp > 1, moe_grouped_gemm=True,
        moe_token_dispatcher_type="alltoall", report_to=[],
        additional_configs={"perform_initialization": False, "gradient_accumulation_fusion": False,
                            "params_dtype": torch.bfloat16 if cli.dtype == "bf16" else torch.float32,
                            "moe_router_dtype": None if cli.router_dtype == "model" else cli.router_dtype,
                            "deterministic_mode": cli.deterministic_gdn},
    )
    events = output / f"rank-{os.environ['RANK']}.jsonl"

    def record(event: str, **values) -> None:
        entry = dict(event=event, rank=int(os.environ["RANK"]), **values)
        entry.update(gpu_allocated=torch.cuda.memory_allocated(),
                     gpu_peak=torch.cuda.max_memory_allocated(),
                     gpu_reserved=torch.cuda.memory_reserved())
        with events.open("a") as stream:
            stream.write(json.dumps(entry) + "\n")
        print(json.dumps(entry), flush=True)

    started = time.monotonic()
    record("load_start", model=cli.model, tp=cli.tp, ep=cli.ep, etp=1,
           dtype=cli.dtype, deterministic_gdn=cli.deterministic_gdn)
    model = AutoModel.from_pretrained(cli.model, args)
    module = model.get_models()[0].eval()
    record("loaded", seconds=time.monotonic() - started,
           parameters=sum(p.numel() for p in module.parameters()),
           layers=len(module.decoder.layers), router_dtype=module.config.moe_router_dtype)
    assert len(module.decoder.layers) == 48, "real-model acceptance requires all 48 layers"
    module.requires_grad_(False)
    handles = []
    if cli.stream_decoder:
        # This no-grad probe can compare full FP32 semantics without keeping
        # every FP32 layer and its temporary activations on CUDA together.
        # Training and throughput measurements must use the ordinary path.
        handles.extend(enable_decoder_streaming(module))
        record("decoder_offloaded", diagnostic_only=True)
    from megatron.core import parallel_state
    group = parallel_state.get_data_parallel_group()
    source = dist.get_process_group_ranks(group)[0]

    def compare_replicas(name, value):
        reference = value.detach().contiguous().clone()
        dist.broadcast(reference, src=source, group=group)
        error = (value.float() - reference.float())
        record("replica_comparison", name=name,
               relative_l2=float(error.norm() / reference.float().norm().clamp_min(1e-12)),
               max_abs=float(error.abs().max()), shape=list(value.shape))

    current_length = None
    case_output = output
    if cli.save_layer_states:
        def save_layer(name):
            def hook(_module, _args, result):
                value = result[0] if isinstance(result, tuple) else result
                if module.config.sequence_parallel:
                    from megatron.core import tensor_parallel
                    value = tensor_parallel.gather_from_sequence_parallel_region(
                        value, tensor_parallel_output_grad=False)
                if dist.get_rank() == 0:
                    torch.save(value.transpose(0, 1).cpu(), case_output / f"hidden-{current_length}-{name}.pt")
            return hook
        for index, layer in enumerate(module.decoder.layers):
            handles.append(layer.register_forward_hook(save_layer(str(index))))
        handles.append(module.embedding.register_forward_hook(save_layer("embedding")))
    if cli.diagnose_replicas:
        import hashlib
        samples = {}
        for name, value in list(module.named_parameters()) + list(module.named_buffers()):
            if ".experts." in name:
                continue
            flat = value.detach().flatten()
            sample = flat[::max(1, flat.numel() // 1024)].contiguous().cpu()
            samples[name] = hashlib.sha256(sample.view(torch.uint8).numpy().tobytes()).hexdigest()
        peers = [None] * dist.get_world_size(group)
        dist.all_gather_object(peers, samples, group=group)
        record("replica_weight_samples", mismatches=[name for name in samples
               if any(peer.get(name) != samples[name] for peer in peers)])
        record("configuration", values={name: str(getattr(module.config, name, None))
               for name in ("hidden_dropout", "attention_dropout", "moe_router_topk",
                            "moe_router_enable_expert_bias", "moe_router_load_balancing_type",
                            "moe_token_dispatcher_type", "moe_parallel_folding", "layernorm_epsilon")})

        def trace(name):
            def hook(_module, _args, result):
                value = result[0] if isinstance(result, tuple) else result
                compare_replicas(name, value)
            return hook

        for name, child in module.named_modules():
            if name == "embedding" or any(name == f"decoder.layers.{index}.{part}"
                    for index in range(48) for part in ("self_attention", "mlp", "ple")):
                handles.append(child.register_forward_hook(trace(name)))
    cases = [("tokens", cli.input_ids, output)]
    if cli.input_manifest:
        manifest = json.loads(cli.input_manifest.read_text())
        cases = [(name, cli.input_manifest.parent / fixture["tensor_file"], output / name)
                 for name, fixture in manifest["fixtures"].items()]
    modes = ["model", "fp32"] if cli.compare_router_dtypes else [cli.router_dtype]
    cases = [(mode, case, path, output / mode / case if cli.compare_router_dtypes else destination)
             for mode in modes for case, path, destination in cases]
    from megatron.core.transformer.moe.router import Router

    for mode, case, input_path, case_output in cases:
        router_dtype = None if mode == "model" else mode
        for child in module.modules():
            if isinstance(child, Router):
                child.config.moe_router_dtype = router_dtype
        case_output.mkdir(parents=True, exist_ok=True)
        for length in cli.lengths:
            current_length = length
            torch.cuda.reset_peak_memory_stats()
            # Same valid, reproducible token sequence across all TP and EP ranks.
            if input_path:
                ids = torch.load(input_path, map_location="cuda", weights_only=True)[:, :length]
                assert ids.shape == (1, length)
            else:
                ids = (torch.arange(length, device="cuda") % 1000 + 100).unsqueeze(0)
            positions = torch.arange(length, device="cuda").unsqueeze(0)
            dist.barrier()
            started = time.monotonic()
            with torch.no_grad():
                losses = module(ids, positions, torch.ones_like(ids), labels=ids.roll(-1, -1))
            torch.cuda.synchronize()
            finite = bool(torch.isfinite(losses).all())
            record("forward", case=case, router_dtype=router_dtype,
                   sequence_length=length, seconds=time.monotonic() - started,
                   finite=finite, mean_loss=float(losses.float().mean()), shape=list(losses.shape))
            assert finite, "real checkpoint produced nonfinite losses"
            compare_replicas(f"losses-{length}", losses)
            torch.save(losses.cpu(), case_output / f"losses-{length}-rank-{dist.get_rank()}.pt")
            del losses
    for handle in handles:
        handle.remove()
    record("complete")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
