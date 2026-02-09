# This file is modified from https://github.com/feifeibear/long-context-attention
# Implementation refers to USP Paper: https://arxiv.org/abs/2405.07719

import copy
import inspect
import os
from typing import Any, Callable, Optional

import torch
import torch.distributed as dist
from flash_attn import flash_attn_func, flash_attn_varlen_func
from flash_attn.bert_padding import pad_input
from torch import Tensor
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import _upad_input
from transformers.utils import is_flash_attn_greater_or_equal

from roll.utils.context_parallel.all_to_all import SeqAllToAll4D
from roll.utils.context_parallel.globals import get_ulysses_group, get_ulysses_seqlen, get_ulysses_size


def _ulysses_attn_varlen_func(
    query_states,
    key_states,
    value_states,
    attention_mask=None,
    dropout_p=0.0,
    softmax_scale=None,
    seqlens_in_batch=None,
    causal=None,
):
    batch_size = query_states.shape[0]

    # overwrite query_length with the actual length of the sequence after SP communciation
    query_length = attention_mask.shape[1]

    query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = _upad_input(
        query_states, key_states, value_states, attention_mask, query_length
    )

    cu_seqlens_q, cu_seqlens_k = cu_seq_lens
    max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

    attn_output_unpad = flash_attn_varlen_func(
        query_states,
        key_states,
        value_states,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_in_batch_q,
        max_seqlen_k=max_seqlen_in_batch_k,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=True,
    )

    attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
    return attn_output


def _flash_attention_forward(
    module: torch.nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor,
    is_causal: bool = True,
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    position_ids: Optional[torch.Tensor] = None,
    sliding_window: Optional[int] = None,
    use_top_left_mask: bool = False,
    softcap: Optional[float] = None,
    deterministic: bool = None,
    output_attentions: bool = False,
    use_cache: bool = False,
):
    """
    Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
    first unpad the input, then computes the attention scores and pad the final attention scores.

    Args:
        query_states (`torch.Tensor`):
            Input query states to be passed to Flash Attention API
        key_states (`torch.Tensor`):
            Input key states to be passed to Flash Attention API
        value_states (`torch.Tensor`):
            Input value states to be passed to Flash Attention API
        attention_mask (`torch.Tensor`):
            The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
            position of padding tokens and 1 for the position of non-padding tokens.
        dropout (`float`):
            Attention dropout
        softmax_scale (`float`, *optional*):
            The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        use_top_left_mask (`bool`, defaults to `False`):
            flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference.
        softcap (`float`, *optional*):
            Softcap for the attention logits, used e.g. in gemma2.
        deterministic (`bool`, *optional*):
            Determines if the deterministic option introduced in flash_attn>=2.4.1 is enabled.
    """

    # Assuming 4D tensors, key_states.shape[1] is the key/value sequence length (source length).
    use_sliding_windows = (
        "window_size" in list(inspect.signature(flash_attn_func).parameters)
        and sliding_window is not None
        and key_states.shape[1] > sliding_window
    )
    flash_kwargs = {"window_size": (sliding_window, sliding_window)} if use_sliding_windows else {}

    if is_flash_attn_greater_or_equal("2.4.1"):
        if deterministic is None:
            deterministic = os.environ.get("FLASH_ATTENTION_DETERMINISTIC", "0") == "1"
        flash_kwargs["deterministic"] = deterministic

    if softcap is not None:
        flash_kwargs["softcap"] = softcap

    seqlens_in_batch = torch.sum(attention_mask, dim=1)

    attn_output = UlyssesAttention(_ulysses_attn_varlen_func, get_ulysses_group())(
        query_states,
        key_states,
        value_states,
        attention_mask=attention_mask,
        dropout_p=dropout,
        softmax_scale=scaling,
        seqlens_in_batch=seqlens_in_batch,  # _get_unpad_data.seqlens_in_batch
    )
    return attn_output, None


def _update_causal_mask(
    self,
    attention_mask: torch.Tensor,
    input_tensor: torch.Tensor,
    cache_position: torch.Tensor,
    past_key_values: Cache,
    output_attentions: bool,
):
    return attention_mask


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
            assert (
                self.ulysses_size % num_key_value_heads == 0
            ), "Ulysses require num_key_value_heads to be dividable by ulysses_size."
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
