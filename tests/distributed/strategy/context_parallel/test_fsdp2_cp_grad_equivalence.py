import os
import socket
import tempfile
from typing import Dict

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from roll.utils.context_parallel import get_ulysses_group, set_upg_manager
from roll.utils.context_parallel.autograd_gather import ulysses_gather
from roll.utils.functionals import agg_loss, log_probs_from_logits


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _broadcast_state_dict(module: torch.nn.Module, src: int = 0):
    # Ensure identical initialization across ranks.
    for _, p in module.state_dict().items():
        if torch.is_tensor(p):
            dist.broadcast(p, src=src)


def _ddp_average_grads(module: torch.nn.Module):
    for p in module.parameters():
        if p.grad is None:
            continue
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        p.grad.div_(dist.get_world_size())


def _run_and_save_grads(
    rank: int,
    world_size: int,
    cp_size: int,
    loss_agg_mode: str,
    master_port: int,
    out_path: str,
):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)

    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    set_upg_manager(ulysses_size=cp_size, rank=rank, world_size=world_size)
    group = get_ulysses_group()

    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    vocab = 97
    hidden = 32
    model = torch.nn.Sequential(
        torch.nn.Embedding(vocab, hidden),
        torch.nn.Linear(hidden, vocab, bias=False),
    ).cuda()
    _broadcast_state_dict(model, src=0)

    bs, seqlen = 2, 8
    assert seqlen % max(cp_size, 1) == 0

    if rank == 0:
        input_ids = torch.randint(0, vocab, (bs, seqlen), device="cuda", dtype=torch.long)
        attention_mask = torch.ones((bs, seqlen), device="cuda", dtype=torch.long)
    else:
        input_ids = torch.empty((bs, seqlen), device="cuda", dtype=torch.long)
        attention_mask = torch.empty((bs, seqlen), device="cuda", dtype=torch.long)
    dist.broadcast(input_ids, src=0)
    dist.broadcast(attention_mask, src=0)

    if cp_size > 1:
        cp_rank = rank % cp_size
        shard = seqlen // cp_size
        start = cp_rank * shard
        end = (cp_rank + 1) * shard

        input_ids_local = input_ids[:, start:end]
        logits_local = model(input_ids_local)

        labels = input_ids[:, 1:].clone()
        labels[attention_mask[:, 1:] == 0] = 0
        labels = torch.cat([labels, torch.zeros_like(labels[:, :1])], dim=1)
        labels_local = labels[:, start:end]

        log_probs_local = log_probs_from_logits(logits_local, labels_local)
        log_probs = ulysses_gather(
            log_probs_local,
            gather_dim=1,
            group=group,
            grad_scaler=True,
        )
        log_probs = log_probs[:, :-1] * attention_mask[:, 1:]
    else:
        logits = model(input_ids)
        labels = input_ids[:, 1:].clone()
        labels[attention_mask[:, 1:] == 0] = 0
        labels = torch.cat([labels, torch.zeros_like(labels[:, :1])], dim=1)
        log_probs = log_probs_from_logits(logits, labels)
        log_probs = log_probs[:, :-1] * attention_mask[:, 1:]

    # PPO-style uses negative log-prob as a loss term.
    response_mask = attention_mask[:, 1:].long()
    loss = agg_loss(loss_mat=-log_probs, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    loss.backward()

    # Simulate DP/FSDP gradient averaging across ranks.
    _ddp_average_grads(model)

    if rank == 0:
        grads: Dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            grads[name] = p.grad.detach().cpu()
        torch.save({"loss": float(loss.detach().cpu()), "grads": grads}, out_path)

    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is not available")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires >= 2 CUDA devices")
@pytest.mark.parametrize("loss_agg_mode", ["token-mean", "seq-mean-token-sum"])
def test_fsdp2_cp_grad_equivalence_vs_cp1(loss_agg_mode: str):
    """
    Gradient equivalence test for CP gather semantics.

    We run twice on the same 2-GPU world:
    - baseline: cp_size=1
    - CP:       cp_size=2
    Both runs do a DDP-style gradient averaging across the 2 ranks.

    With autograd-friendly CP gather (slice-only backward + grad scaling),
    the averaged gradients should match the cp_size=1 baseline.
    """
    world_size = 2

    with tempfile.TemporaryDirectory() as td:
        out_cp1 = os.path.join(td, f"grads_cp1_{loss_agg_mode}.pt")
        out_cp2 = os.path.join(td, f"grads_cp2_{loss_agg_mode}.pt")

        port1 = _find_free_port()
        port2 = _find_free_port()

        mp.spawn(
            _run_and_save_grads,
            args=(world_size, 1, loss_agg_mode, port1, out_cp1),
            nprocs=world_size,
            join=True,
        )
        mp.spawn(
            _run_and_save_grads,
            args=(world_size, 2, loss_agg_mode, port2, out_cp2),
            nprocs=world_size,
            join=True,
        )

        ref = torch.load(out_cp1, map_location="cpu")
        cp = torch.load(out_cp2, map_location="cpu")

        assert abs(ref["loss"] - cp["loss"]) < 1e-6

        for k in ref["grads"].keys():
            torch.testing.assert_close(cp["grads"][k], ref["grads"][k], rtol=0, atol=1e-6)
