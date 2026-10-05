"""Measure real frozen-table backbone backward and ROLL CPU optimizer steps.

This capacity probe uses synthetic token IDs; it is not the SFT acceptance run.
Run with eight torchrun ranks and explicit localhost rendezvous.
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[2048, 8192])
    parser.add_argument("--backward-only", action="store_true")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--checkpoint-cpu-offload", action="store_true")
    parser.add_argument("--trace-memory", action="store_true")
    cli = parser.parse_args()
    output = Path(cli.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    events = output / f"rank-{os.environ['RANK']}.jsonl"
    started = time.monotonic()

    def record(event, **values):
        memory = {}
        for path, keys in (("/proc/self/status", {"VmRSS", "VmHWM"}),
                           ("/proc/meminfo", {"MemAvailable"})):
            for line in Path(path).read_text().splitlines():
                key, _, value = line.partition(":")
                if key in keys:
                    memory[key] = int(value.split()[0]) * 1024
        entry = dict(event=event, rank=int(os.environ["RANK"]), elapsed=time.monotonic()-started,
                     gpu_allocated=torch.cuda.memory_allocated(),
                     gpu_peak=torch.cuda.max_memory_allocated(),
                     gpu_reserved=torch.cuda.memory_reserved(), **memory, **values)
        with events.open("a") as stream:
            stream.write(json.dumps(entry) + "\n")
        print(json.dumps(entry), flush=True)

    args = TrainingArguments(
        output_dir=str(output), bf16=True, tensor_model_parallel_size=2,
        pipeline_model_parallel_size=1, context_parallel_size=1,
        expert_model_parallel_size=8, expert_tensor_parallel_size=1,
        sequence_parallel=True, moe_grouped_gemm=True, moe_token_dispatcher_type="alltoall",
        accumulate_allreduce_grads_in_fp32=False, report_to=[],
        optimizer_cpu_offload=not cli.backward_only, optimizer_offload_fraction=1.0,
        use_precision_aware_optimizer=not cli.backward_only,
        overlap_cpu_optimizer_d2h_h2d=not cli.backward_only,
        bounded_cpu_grad_staging=not cli.backward_only,
        additional_configs={"perform_initialization": False, "gradient_accumulation_fusion": False,
                            "recompute_granularity": "full", "recompute_method": "uniform",
                            "recompute_num_layers": 1, "qsa_indexer_kl_coef": 0.01,
                            "checkpoint_cpu_offload": cli.checkpoint_cpu_offload},
    )
    record("load_start")
    factory = AutoModel.from_pretrained(cli.model, args)
    model = factory.get_models()[0].train()
    assert len(model.decoder.layers) == 48
    assert model.config.moe_permute_fusion
    assert not any("ngram_embedding" in name for name, _ in model.named_parameters())
    record("loaded", parameters=sum(p.numel() for p in model.parameters()),
           checkpoint_cpu_offload=model.config.checkpoint_cpu_offload,
           rotary_percent=model.config.rotary_percent)
    if cli.trace_memory:
        def memory_hook(name, phase):
            def hook(*_):
                if torch.is_grad_enabled():
                    record("recompute_memory", module=name, phase=phase)
            return hook
        for index, layer in enumerate(model.decoder.layers):
            for kind in ("self_attention", "mlp"):
                child = getattr(layer, kind)
                name = f"layer-{index}.{kind}"
                child.register_forward_pre_hook(memory_hook(name, "start"))
                child.register_forward_hook(memory_hook(name, "end"))

    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.distributed.finalize_model_grads import finalize_model_grads
    from megatron.core.optimizer import OptimizerConfig
    from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler
    from roll.third_party.megatron.optimizer import get_megatron_optimizer

    wrapped = DistributedDataParallel(model.config, DistributedDataParallelConfig(
        grad_reduce_in_fp32=False, use_distributed_optimizer=True, overlap_grad_reduce=False,
        overlap_param_gather=False,
    ), model)
    record("ddp_ready")
    optimizer = None
    if not cli.backward_only:
        from roll.third_party.megatron.optimizer_config import build_optimizer_config
        optimizer_config = build_optimizer_config(OptimizerConfig, {
            "optimizer": "adam", "lr": 1e-6, "bf16": True, "params_dtype": torch.bfloat16,
            "use_distributed_optimizer": True, "clip_grad": 1.0,
        }, args)
        optimizer = get_megatron_optimizer(optimizer_config, [wrapped])
        record("optimizer_ready", kind=type(optimizer).__name__)

    watched_names = ("embedding.word_embeddings.weight", "decoder.layers.0.self_attention.in_proj.weight",
                     "decoder.layers.0.mlp.router.weight", "decoder.layers.0.mlp.experts.linear_fc1.weight0",
                     "decoder.layers.1.ple.key_proj.weight",
                     "decoder.layers.3.self_attention.indexer.index_qk_proj.weight",
                     "decoder.hyper_connection_mixer.hc.input_mix_weight_down.weight")
    watched = {name: param for name, param in model.named_parameters() if name in watched_names}
    assert len(watched) == len(watched_names), set(watched_names) - set(watched)
    for step, length in enumerate(cli.lengths * cli.repeat):
        wrapped.zero_grad_buffer()
        if optimizer is not None:
            optimizer.zero_grad()
        torch.cuda.reset_peak_memory_stats()
        MoEAuxLossAutoScaler.set_loss_scale(torch.ones((), device="cuda"))
        before = {name: p.detach().flatten()[::max(1, p.numel() // 4096)].clone()
                  for name, p in watched.items()}
        ids = (torch.arange(length, device="cuda") % 1000 + 100).unsqueeze(0)
        positions = torch.arange(length, device="cuda").unsqueeze(0)
        valid = torch.ones_like(ids, dtype=torch.bool)
        dist.barrier()
        tick = time.monotonic()
        losses = wrapped(ids, positions, valid, labels=ids.roll(-1, -1), loss_mask=valid)
        loss = losses.float().mean()
        torch.cuda.synchronize()
        record("forward", step=step, length=length, seconds=time.monotonic()-tick, loss=float(loss))
        assert bool(torch.isfinite(loss))
        tick = time.monotonic()
        loss.backward()
        finalize_model_grads([wrapped])
        torch.cuda.synchronize()
        gradient_norms = {name: float(p.main_grad.float().norm()) for name, p in watched.items()}
        record("backward", step=step, length=length, seconds=time.monotonic()-tick,
               gradient_norms=gradient_norms)
        # Vocabulary and expert shards legitimately receive zero gradient when
        # this batch never selects their tokens/experts. Validate the global
        # module, not every local shard independently.
        nonzero = torch.tensor(list(gradient_norms.values()), device="cuda")
        finite = torch.isfinite(nonzero).all().to(torch.int32)
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        dist.all_reduce(nonzero, op=dist.ReduceOp.MAX)
        assert bool(finite) and bool((nonzero > 0).all())
        if optimizer is not None:
            tick = time.monotonic()
            updated, norm, zeros = optimizer.step()
            torch.cuda.synchronize()
            changed = {name: int((p.detach().flatten()[::max(1, p.numel() // 4096)] != before[name]).sum())
                       for name, p in watched.items()}
            record("optimizer_step", step=step, seconds=time.monotonic()-tick, updated=bool(updated),
                   grad_norm=float(norm), changed_samples=changed)
            assert updated and any(changed.values())
        del loss, losses, before
    record("complete", backward_only=cli.backward_only)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
