"""
PumpkinComment:

Why this exists:
- CP ranks typically see the same (replicated) batch, but operate on different sequence shards.
- Downstream loss code often wants full-sequence tensors (e.g., log_probs, entropy).
- A naive gather using torch.distributed.nn.functional.all_gather has a backward that performs
  ReduceScatter(SUM)-like behavior, which interacts poorly with replicated-loss semantics.

- forward: gather shards and concatenate along `gather_dim`
- backward: *slice only* the gradient shard for this rank
- optional `grad_scaler`: multiply grad_output by world_size before slicing, so that if an outer
  data-parallel reduction averages across CP replicas, the effective gradient matches cp_size=1.

Reference: https://github.com/volcengine/verl/blob/main/verl/utils/ulysses.py
"""

from typing import Optional

import torch
import torch.distributed as dist


class _UlyssesGather(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        group: dist.ProcessGroup,
        local_tensor: torch.Tensor,
        gather_dim: int,
        grad_scaler: bool,
    ) -> torch.Tensor:
        # Normalize dim.
        if gather_dim < 0:
            gather_dim = local_tensor.dim() + gather_dim

        world_size = dist.get_world_size(group=group)
        rank = dist.get_rank(group=group)
        part_size = local_tensor.size(gather_dim)

        ctx.group = group
        ctx.gather_dim = gather_dim
        ctx.grad_scaler = grad_scaler
        ctx.world_size = world_size
        ctx.rank = rank
        ctx.part_size = part_size

        # Move gather_dim to leading dim so we can use all_gather_into_tensor on dim0.
        x_perm = local_tensor.movedim(gather_dim, 0).contiguous()
        out_perm = torch.empty(
            (world_size * x_perm.size(0),) + tuple(x_perm.shape[1:]),
            device=x_perm.device,
            dtype=x_perm.dtype,
        )
        dist.all_gather_into_tensor(out_perm, x_perm, group=group)

        full = out_perm.movedim(0, gather_dim).contiguous()
        return full

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # grad_output is the full concatenated tensor on every rank.
        if ctx.grad_scaler:
            grad_output = grad_output * ctx.world_size

        gdim = ctx.gather_dim
        if gdim < 0:
            gdim = grad_output.dim() + gdim

        grad_perm = grad_output.movedim(gdim, 0).contiguous()
        start = ctx.rank * ctx.part_size
        end = (ctx.rank + 1) * ctx.part_size
        grad_local_perm = grad_perm[start:end].contiguous()
        grad_local = grad_local_perm.movedim(0, gdim).contiguous()
        return None, grad_local, None, None


def ulysses_gather(
    x: torch.Tensor,
    *,
    gather_dim: int,
    group: Optional[dist.ProcessGroup],
    grad_scaler: bool = True,
) -> torch.Tensor:
    """
    Gather shards across `group` and concatenate along `gather_dim` with autograd-friendly backward.

    Args:
        x: local shard tensor
        gather_dim: dim to concatenate along
        group: process group (if None or world_size<=1, returns x)
        grad_scaler: whether to scale grad_output by world_size before slicing in backward
    """
    if group is None:
        return x
    if dist.get_world_size(group=group) <= 1:
        return x
    return _UlyssesGather.apply(group, x, gather_dim, grad_scaler)
