# This file is modified from https://github.com/feifeibear/long-context-attention
# Implementation refers to USP Paper: https://arxiv.org/abs/2405.07719

from typing import Any, Tuple

import torch
import torch.distributed as dist

from roll.platforms import current_platform
from roll.utils.context_parallel.globals import get_ulysses_seqlen, set_ulysses_seqlen


# Modified from https://github.com/NVlabs/Long-RL/blob/main/verl/utils/sequence_parallel/all_to_all.py
def all_to_all_4D(
    input: torch.Tensor, scatter_idx: int = 2, gather_idx: int = 1, group=None, use_sync: bool = False
) -> torch.Tensor:
    """
    all-to-all for QKV

    Args:
        input (torch.tensor): a tensor sharded along dim scatter dim
        scatter_idx (int): default 1
        gather_idx (int): default 2
        group : torch process group
        use_sync (bool): whether to synchronize after all-to-all

    Returns:
        torch.tensor: resharded tensor (bs, seqlen/P, hc, hs)
    """
    assert input.dim() == 4, f"input must be 4D tensor, got {input.dim()} and shape {input.shape}"

    seq_world_size = dist.get_world_size(group)
    if scatter_idx == 2 and gather_idx == 1:
        # input (torch.tensor): a tensor sharded along dim 1 (bs, seqlen/P, hc, hs) output: (bs, seqlen, hc/P, hs)
        bs, shard_seqlen, hc, hs = input.shape
        # Pad sequence for multi-modality use case
        ulysses_seqlen = [torch.zeros(1, dtype=torch.int64, device=input.device) for _ in range(seq_world_size)]
        dist.barrier(group=group)
        dist.all_gather(
            ulysses_seqlen,
            torch.tensor([shard_seqlen], device=input.device),
            group=group,
        )
        set_ulysses_seqlen(ulysses_seqlen)

        max_global_length = max(ulysses_seqlen)
        # pad to the second dimension to the longest
        input = torch.nn.functional.pad(input, (0, 0, 0, 0, 0, max_global_length - shard_seqlen))

        shard_seqlen_padded = int(max_global_length.item())
        seqlen_padded = shard_seqlen_padded * seq_world_size
        shard_hc = hc // seq_world_size

        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        # (bs, seqlen/P, hc, hs) -reshape-> (bs, seq_len/P, P, hc/P, hs) -transpose(0,2)-> (P, seq_len/P, bs, hc/P, hs)
        input_t = input.reshape(bs, shard_seqlen_padded, seq_world_size, shard_hc, hs).transpose(0, 2).contiguous()

        output = torch.empty_like(input_t)
        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
        # (P, seq_len/P, bs, hc/P, hs) scatter seqlen -all2all-> (P, seq_len/P, bs, hc/P, hs) scatter head

        if seq_world_size > 1:
            dist.all_to_all_single(output, input_t, group=group)
            if use_sync:
                current_platform.synchronize()
        else:
            output = input_t
        # if scattering the seq-dim, transpose the heads back to the original dimension
        output = output.reshape(seqlen_padded, bs, shard_hc, hs)

        # then we will unpad it back
        output_list = torch.split(output, shard_seqlen_padded, dim=0)
        assert len(output_list) == seq_world_size
        unpadded_output_list = [_output[: _seqlen.item()] for _output, _seqlen in zip(output_list, ulysses_seqlen)]

        # Concatenate the unpadded tensors back together
        output = torch.cat(unpadded_output_list)
        seqlen_actual = int(output.size(0))

        # (seq_len_actual, bs, hc/P, hs) -> (bs, seq_len_actual, hc/P, hs)
        output = output.transpose(0, 1).contiguous().reshape(bs, seqlen_actual, shard_hc, hs)

        return output

    elif scatter_idx == 1 and gather_idx == 2:
        ulysses_seqlen = get_ulysses_seqlen()
        assert ulysses_seqlen is not None, "the second a2a (scatter 1, gather 2) is called at first."
        # input (torch.tensor): a tensor sharded along dim 1 (bs, seqlen, hc/P, hs) output: (bs, seqlen/P, hc, hs)
        bs, _, shard_hc, hs = input.shape
        hc = shard_hc * seq_world_size

        # First we need to recover how to pad
        max_global_length = max(ulysses_seqlen)

        unpadded_input_list = torch.split(input, ulysses_seqlen, dim=1)
        padded_input_list = [
            torch.nn.functional.pad(_unpadded_input, (0, 0, 0, 0, 0, max_global_length - _unpadded_input.shape[1]))
            for _unpadded_input in unpadded_input_list
        ]
        input = torch.cat(padded_input_list, dim=1)

        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        # (bs, seqlen, hc/P, hs) -reshape-> (bs, P, seq_len/P, hc/P, hs) -transpose(0, 3)-> (hc/P, P, seqlen/P, bs, hs) -transpose(0, 1) -> (P, hc/P, seqlen/P, bs, hs)
        input_t = (
            input.reshape(bs, seq_world_size, max_global_length, shard_hc, hs)
            .transpose(0, 3)
            .transpose(0, 1)
            .contiguous()
            .reshape(seq_world_size, shard_hc, max_global_length, bs, hs)
        )

        output = torch.empty_like(input_t)
        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
        # (P, bs x hc/P, seqlen/P, hs) scatter seqlen -all2all-> (P, bs x seq_len/P, hc/P, hs) scatter head
        if seq_world_size > 1:
            dist.all_to_all_single(output, input_t, group=group)
            if use_sync:
                current_platform.synchronize()
        else:
            output = input_t

        # if scattering the seq-dim, transpose the heads back to the original dimension
        output = output.reshape(hc, max_global_length, bs, hs)

        # unpad the output
        self_length = int(ulysses_seqlen[dist.get_rank(group=group)].item())
        output = output[:, :self_length, :, :]

        # (hc, local_seqlen, bs, hs) -> (bs, local_seqlen, hc, hs)
        output = output.transpose(0, 2).contiguous().reshape(bs, self_length, hc, hs)

        return output
    else:
        raise RuntimeError("scatter_idx must be 1 or 2 and gather_idx must be 1 or 2")


class SeqAllToAll4D(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        input: torch.Tensor,
        scatter_idx: int,
        gather_idx: int,
        use_sync: bool = False,
    ) -> torch.Tensor:
        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx
        ctx.use_sync = use_sync
        return all_to_all_4D(input, scatter_idx, gather_idx, group=group, use_sync=use_sync)

    @staticmethod
    def backward(ctx: Any, *grad_output: torch.Tensor) -> Tuple[None, torch.Tensor, None, None]:
        return (
            None,
            SeqAllToAll4D.apply(ctx.group, *grad_output, ctx.gather_idx, ctx.scatter_idx, ctx.use_sync),
            None,
            None,
            None,
        )
