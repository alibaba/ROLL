import inspect
import os
from typing import Optional

import torch
from flash_attn import flash_attn_func, flash_attn_varlen_func
from flash_attn.bert_padding import pad_input
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import _upad_input
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model
from transformers.utils import is_flash_attn_greater_or_equal

from roll.utils.context_parallel.globals import get_ulysses_group
from roll.utils.context_parallel.ulysses_attention import UlyssesAttention


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


old_flash_attention_forward = ALL_ATTENTION_FUNCTIONS["flash_attention_2"]
old_update_causal_mask = Qwen2Model._update_causal_mask


def apply_ulysses_patch():
    ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = _flash_attention_forward
    Qwen2Model._update_causal_mask = _update_causal_mask


def unapply_ulysses_patch():
    global old_flash_attention_forward, old_update_causal_mask
    ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = old_flash_attention_forward
    Qwen2Model._update_causal_mask = old_update_causal_mask
