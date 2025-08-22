# This file is modified from https://github.com/feifeibear/long-context-attention
# Implementation refers to USP Paper: https://arxiv.org/abs/2405.07719

import copy
from typing import Any, Callable

import torch
import torch.distributed as dist
from torch import Tensor

from roll.utils.context_parallel.all_to_all import SeqAllToAll4D
from roll.utils.context_parallel.globals import get_ulysses_seqlen, get_ulysses_size


# Modified from https://github.com/NVlabs/Long-RL/blob/main/verl/utils/sequence_parallel/ulysses_attn.py
class _ExpandKVFunction(torch.autograd.Function):
    """
    Repeat the KV head to extend sequence parallel support for Ulysses.

    Args:
        kv: input kv.
        num_repeats: the repeat number of each head.
        head_dim: the dimension of head number.
    """

    @staticmethod
    def forward(ctx, k, v, num_repeats, head_dim):
        kv_shape = k.shape
        num_key_value_heads = kv_shape[head_dim]

        ctx.head_dim = head_dim
        ctx.num_key_value_heads = num_key_value_heads

        # here we construct a repeat index to indicate which dim should copy
        repeat_index = [1] * k.ndim
        repeat_index[head_dim] = num_repeats

        # split the kv into head num splits
        k_splits = torch.chunk(k, chunks=num_key_value_heads, dim=head_dim)
        v_splits = torch.chunk(v, chunks=num_key_value_heads, dim=head_dim)
        k_repeats, v_repeats = [], []
        # for each split, we copy it to num_repeats copys.
        for split in k_splits:
            k_split_repeat = split.repeat(repeat_index)
            k_repeats.append(k_split_repeat)

        for split in v_splits:
            v_split_repeat = split.repeat(repeat_index)
            v_repeats.append(v_split_repeat)

        return torch.cat(k_repeats, dim=head_dim), torch.cat(v_repeats, dim=head_dim)

    @staticmethod
    def backward(ctx, grad_output_k, grad_output_v):
        """
        For backward, we sum the copy head inside a query group.
        """

        head_dim = ctx.head_dim
        num_key_value_heads = ctx.num_key_value_heads

        # we split the grad into query groups splits.
        grad_output_k_splits = torch.chunk(grad_output_k, chunks=num_key_value_heads, dim=head_dim)
        grad_output_v_splits = torch.chunk(grad_output_v, chunks=num_key_value_heads, dim=head_dim)

        grad_output_k_sums, grad_output_v_sums = [], []
        # for each split, we sum the head
        for grad_output_k_split in grad_output_k_splits:
            grad_output_k_sum = grad_output_k_split.sum(dim=head_dim, keepdim=True)
            grad_output_k_sums.append(grad_output_k_sum)

        for grad_output_v_split in grad_output_v_splits:
            grad_output_v_sum = grad_output_v_split.sum(dim=head_dim, keepdim=True)
            grad_output_v_sums.append(grad_output_v_sum)

        # then we concat the split sums on the head_dim dimension.
        grad_k = torch.cat(grad_output_k_sums, dim=head_dim)
        grad_v = torch.cat(grad_output_v_sums, dim=head_dim)

        return grad_k, grad_v, None, None


expandKV = _ExpandKVFunction.apply


class UlyssesAttention(torch.nn.Module):
    """Initialization.

    Arguments:
        local_attention (Module): local attention with q,k,v
        sequence_process_group (ProcessGroup): sequence parallel process group
        scatter_idx (int): scatter_idx for all2all comm
        gather_idx (int): gather_idx for all2all comm
        use_sync (bool): whether to synchronize after all-to-all. This flag can save cuda memory but will slow down the speed.
        attn_type (AttnType): attention type enum
    """

    def __init__(
        self,
        attn_fn: Callable,
        sequence_process_group: dist.ProcessGroup = None,
        scatter_idx: int = 2,
        gather_idx: int = 1,
        use_sync: bool = False,
    ) -> None:
        super(UlyssesAttention, self).__init__()
        self.attn_fn = attn_fn
        self.spg = sequence_process_group
        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx
        self.use_sync = use_sync
        self.ulysses_size = get_ulysses_size()

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor = None,
        dropout_p=0.0,
        softmax_scale=None,
        seqlens_in_batch=None,
        causal=False,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
        *args: Any,
    ) -> Tensor:
        """forward

        Arguments:
            query (Tensor): query input to the layer
            key (Tensor): key input to the layer
            value (Tensor): value input to the layer
            args: other args

        Returns:
            * output (Tensor): context output
        """
        # (bs, seq_len/N, head_cnt, head_size) -> (bs, seq_len, head_cnt/N, head_size)

        # KV Replication for GQA
        head_dim = 1
        num_key_value_heads = key.shape[head_dim]
        if self.ulysses_size > num_key_value_heads:
            assert self.ulysses_size % num_key_value_heads == 0, (
                "Ulysses require num_key_value_heads to be dividable by ulysses_size."
            )
            key, value = expandKV(key, value, self.ulysses_size // num_key_value_heads, head_dim)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # scatter 2, gather 1
        q = SeqAllToAll4D.apply(self.spg, query, self.scatter_idx, self.gather_idx, self.use_sync)
        k = SeqAllToAll4D.apply(self.spg, key, self.scatter_idx, self.gather_idx, self.use_sync)
        v = SeqAllToAll4D.apply(self.spg, value, self.scatter_idx, self.gather_idx, self.use_sync)

        # Gather attention mask
        if attention_mask is not None:
            local_attention_mask = copy.deepcopy(attention_mask)
            shard_seqlen = local_attention_mask.size(1)
            ulysses_seqlen = get_ulysses_seqlen()
            max_global_length = max(ulysses_seqlen)
            global_attention_mask_list = []
            sp_size = dist.get_world_size(self.spg)
            sp_rank = dist.get_rank(self.spg)
            for i in range(sp_size):
                if i == sp_rank:
                    global_attention_mask_list.append(
                        torch.cat(
                            [
                                local_attention_mask,
                                torch.zeros(
                                    (local_attention_mask.size(0), max_global_length - shard_seqlen),
                                    dtype=local_attention_mask.dtype,
                                    device=local_attention_mask.device,
                                ),
                            ],
                            dim=1,
                        )
                    )
                else:
                    global_attention_mask_list.append(
                        torch.zeros(
                            (local_attention_mask.size(0), max_global_length),
                            dtype=local_attention_mask.dtype,
                            device=local_attention_mask.device,
                        )
                    )

            global_attention_mask = torch.stack(global_attention_mask_list, dim=0)
            dist.all_reduce(global_attention_mask, group=self.spg)
            dist.barrier(group=self.spg)
            new_global_attention_mask_list = list(torch.unbind(global_attention_mask, dim=0))
            # Unpad the global attention mask list and concatenate them
            for i in range(len(new_global_attention_mask_list)):
                new_global_attention_mask_list[i] = new_global_attention_mask_list[i][:, : ulysses_seqlen[i]]
            global_attention_mask = torch.cat(new_global_attention_mask_list, dim=1)
            context_layer = self.attn_fn(
                q,
                k,
                v,
                attention_mask=global_attention_mask,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                seqlens_in_batch=seqlens_in_batch,
                causal=causal,
            )
        else:
            context_layer = self.attn_fn(
                q,
                k,
                v,
                *args,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
            )

        if isinstance(context_layer, tuple):
            context_layer = context_layer[0]

        # (bs, seq_len, head_cnt/N, head_size) -> (bs, seq_len/N, head_cnt, head_size)
        # scatter 1, gather 2
        output = SeqAllToAll4D.apply(self.spg, context_layer, self.gather_idx, self.scatter_idx, self.use_sync)

        # out e.g., [s/p::h]
        return output
