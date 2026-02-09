import inspect
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.distributed as dist

from roll.utils.context_parallel.all_to_all import SeqAllToAll4D, all_to_all_4D
from roll.utils.context_parallel.globals import get_ulysses_group, get_ulysses_size
from roll.utils.context_parallel.ulysses_attention import expandKV
from roll.utils.logging import get_logger

logger = get_logger()

_DTYPE_ID_TO_DTYPE = {
    0: torch.int32,
    1: torch.int64,
    2: torch.bool,
    3: torch.float16,
    4: torch.bfloat16,
    5: torch.float32,
}


def _dtype_to_id(dtype: torch.dtype) -> int:
    for k, v in _DTYPE_ID_TO_DTYPE.items():
        if v == dtype:
            return k
    return -1


def _sync_optional_tensor_meta(
    t: Any,
    *,
    group: dist.ProcessGroup,
    dev: torch.device,
) -> Tuple[bool, Optional[torch.dtype]]:
    """
    Synchronize whether `t` is a tensor across `group` and (if present) its dtype.
    Returns:
      (global_present, global_dtype_if_present)
    """
    present = 1 if torch.is_tensor(t) else 0
    dtype_id = _dtype_to_id(t.dtype) if torch.is_tensor(t) else -1
    meta = torch.tensor([present, dtype_id], device=dev, dtype=torch.int32)
    metas = [torch.empty_like(meta) for _ in range(dist.get_world_size(group))]
    dist.all_gather(metas, meta, group=group)
    meta_stack = torch.stack(metas, dim=0)

    global_present = bool(int(meta_stack[:, 0].max().item()) == 1)
    if not global_present:
        return False, None

    present_mask = meta_stack[:, 0] == 1
    dtype_ids = meta_stack[present_mask][:, 1]
    dtype_min = int(dtype_ids.min().item())
    dtype_max = int(dtype_ids.max().item())
    if dtype_min != dtype_max or dtype_min not in _DTYPE_ID_TO_DTYPE:
        return True, None
    return True, _DTYPE_ID_TO_DTYPE[dtype_min]


_PATCH_STATE: Dict[str, Any] = {
    "patched": False,
    "orig_modeling_flash_attention_forward": None,
    "orig_integrations_flash_attention_forward": None,
}


def _normalize_position_ids_for_fa_varlen(position_ids: Any) -> Any:
    """
    Normalize `position_ids` for HF FlashAttention varlen bookkeeping.

    Some Transformers versions derive FlashAttention varlen `cu_seqlens` by scanning `position_ids == 0`
    to find packed-sequence boundaries. In some pipelines, user-provided `position_ids` starts from 1,
    meaning there are no zeros and boundary detection fails.

    In typical HF attention implementations, RoPE is applied to Q/K before calling the (FlashAttention)
    forward, so `position_ids` passed into `_flash_attention_forward` is used for varlen metadata, not
    for rotary math. Therefore shifting it here is safe for correctness of attention computation.

    Policy:
    - If `position_ids` is an int tensor of shape (seqlen,) or (bs, seqlen) and the first token of each
      sequence is not 0 (e.g. starts from 1), shift each sequence by its first value so it starts at 0.
      This also works when CP-align padding introduces zeros later in the tensor (e.g. rmpad adds [0..pad)).
    - Otherwise return it unchanged.

    Note:
    - This normalization is intentionally applied *after* we gather `position_ids` to the global sequence
      for Ulysses CP so that every rank sees consistent varlen metadata.
    """
    if not torch.is_tensor(position_ids):
        return position_ids
    if position_ids.numel() == 0:
        return position_ids
    if position_ids.dtype not in (torch.int32, torch.int64):
        return position_ids
    if position_ids.dim() not in (1, 2):
        return position_ids

    if position_ids.dim() == 1:
        start_val = position_ids[:1]  # [1]
        if int(start_val.item()) == 0:
            return position_ids
        if int(start_val.item()) < 0:
            return position_ids
        return position_ids - start_val

    # dim == 2: shift each row by its own first token
    start_val = position_ids[:, :1]  # [bs, 1]
    # If all rows already start at 0, leave unchanged.
    if bool(torch.all(start_val == 0).item()):
        return position_ids
    # Avoid shifting for negative/sentinel schemes.
    if bool(torch.any(start_val < 0).item()):
        return position_ids
    return position_ids - start_val


def _pad_to(t: torch.Tensor, target_len: int, *, dim: int = -1, pad_value: int = 0) -> torch.Tensor:
    if dim < 0:
        dim = dim % t.ndim
    if t.size(dim) >= target_len:
        return t
    pad_len = target_len - t.size(dim)
    pad = [0, 0] * t.ndim
    pad[2 * (t.ndim - 1 - dim) + 1] = pad_len
    return torch.nn.functional.pad(t, pad, value=pad_value)


def _gather_sharded_seq_tensor(
    local: torch.Tensor,
    *,
    group: dist.ProcessGroup,
    shard_lens: torch.Tensor,
) -> torch.Tensor:
    world_size = dist.get_world_size(group)
    max_len = int(shard_lens.max().item())

    local_padded = _pad_to(local, max_len, dim=-1, pad_value=0).contiguous()
    gathered = [
        torch.empty(local_padded.shape, device=local_padded.device, dtype=local_padded.dtype)
        for _ in range(world_size)
    ]
    dist.all_gather(gathered, local_padded, group=group)

    pieces = []
    for i, g in enumerate(gathered):
        li = int(shard_lens[i].item())
        pieces.append(g[..., :li])
    return torch.cat(pieces, dim=-1)


def _maybe_repeat_kv_for_ulysses(
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    *,
    ulysses_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # PumpkinComment: (bs, seqlen, n_kv_heads, head_dim)
    n_kv = key_states.size(2)
    if ulysses_size > n_kv:
        assert (
            ulysses_size % n_kv == 0
        ), f"ulysses_size={ulysses_size} must be divisible by num_key_value_heads={n_kv} (or vice versa)."
        repeats = ulysses_size // n_kv
        k = key_states.transpose(1, 2)
        v = value_states.transpose(1, 2)
        k, v = expandKV(k, v, repeats, 1)
        return k.transpose(1, 2), v.transpose(1, 2)
    return key_states, value_states


def make_ulysses_flash_attention_forward(
    original_forward: Callable[..., Any],
) -> Callable[..., Any]:
    """
    Wrap HF `_flash_attention_forward` by inserting Ulysses all-to-all before and after.
    """

    def _wrapped(*args: Any, **kwargs: Any) -> Any:
        ulysses_group = get_ulysses_group()
        ulysses_size = get_ulysses_size() or 1

        # If Ulysses isn't enabled, do nothing.
        if ulysses_group is None or ulysses_size <= 1:
            return original_forward(*args, **kwargs)

        query_states = kwargs.get("query_states", args[0] if len(args) > 0 else None)
        key_states = kwargs.get("key_states", args[1] if len(args) > 1 else None)
        value_states = kwargs.get("value_states", args[2] if len(args) > 2 else None)
        attention_mask = kwargs.get("attention_mask", args[3] if len(args) > 3 else None)
        query_length = kwargs.get("query_length", args[4] if len(args) > 4 else None)
        # Some callers pass `position_ids` positionally (Transformers signature has it after dropout).
        # Handle both forms to avoid silently skipping the CP gather path for packed/varlen attention.
        position_ids = kwargs.get("position_ids", args[7] if len(args) > 7 else None)

        if query_states is None or key_states is None or value_states is None:
            return original_forward(*args, **kwargs)

        if query_states.dim() != 4:
            # Unexpected, fall back.
            return original_forward(*args, **kwargs)

        layout = "bshd"  # (b, s, h, d)
        dev = query_states.device
        attn_present, attn_dtype = _sync_optional_tensor_meta(attention_mask, group=ulysses_group, dev=dev)
        pos_present, pos_dtype = _sync_optional_tensor_meta(position_ids, group=ulysses_group, dev=dev)
        if torch.is_tensor(attention_mask) and attention_mask.dim() == 2:
            seq_len_local = attention_mask.size(1)
            if query_states.size(1) != seq_len_local and query_states.size(2) == seq_len_local:
                layout = "bhsd"
        elif position_ids is not None and torch.is_tensor(position_ids):
            seq_len_local = position_ids.size(-1)
            if query_states.size(1) != seq_len_local and query_states.size(2) == seq_len_local:
                layout = "bhsd"

        if layout == "bhsd":
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)

        key_states, value_states = _maybe_repeat_kv_for_ulysses(key_states, value_states, ulysses_size=ulysses_size)

        q_global = SeqAllToAll4D.apply(ulysses_group, query_states, 2, 1, False)
        k_global = SeqAllToAll4D.apply(ulysses_group, key_states, 2, 1, False)
        v_global = SeqAllToAll4D.apply(ulysses_group, value_states, 2, 1, False)

        # Gather attention_mask / position_ids to global sequence if present.
        # Use shard lengths from the local query sequence (before all2all).
        shard_lens = torch.tensor([query_states.size(1)], device=query_states.device, dtype=torch.int64)
        shard_lens_list = [torch.zeros_like(shard_lens) for _ in range(dist.get_world_size(ulysses_group))]
        dist.all_gather(shard_lens_list, shard_lens, group=ulysses_group)
        shard_lens_cat = torch.cat(shard_lens_list, dim=0)

        attn_mask_global = attention_mask

        # PumpkinComment: (Important for CP > 1 without rmpad)
        # For transformers, it will make attn_mask to none is no pad tokens exists (all_causal)
        # however, if two cp rank, one is fully causal, other is not, the gather process will be hang
        # therefore, we set attn_mask to all ones if not present (fully causal)
        if attn_present:
            if not torch.is_tensor(attention_mask):
                # Dummy local mask (all zeros) so all ranks participate in the same all_gather.
                attention_mask = torch.ones(
                    (query_states.size(0), query_states.size(1)),
                    device=query_states.device,
                    dtype=attn_dtype,
                )
            attn_mask_global = _gather_sharded_seq_tensor(
                attention_mask, group=ulysses_group, shard_lens=shard_lens_cat
            )

        position_ids_global = position_ids
        # PumpkinComment:
        # Transformers sometimes sets `position_ids=None` when not needed, or passes it only in some codepaths.
        # Under Ulysses CP, if one rank enters the gather path and another rank skips it, NCCL will hang.
        if pos_present:
            # Ensure all ranks participate in the gather:
            # - If local `position_ids` is missing, create a dummy 1D tensor.
            # - If local `position_ids` is provided, force it into the HF FlashAttention "1D PE" form.
            local_len = int(query_states.size(1))
            bs = int(query_states.size(0))
            if not torch.is_tensor(position_ids):
                # Create a dummy that matches the query batch size.
                base = torch.arange(local_len, device=dev, dtype=pos_dtype)
                position_ids = base.unsqueeze(0).expand(bs, -1).contiguous()
            else:
                if position_ids.dtype != pos_dtype:
                    position_ids = position_ids.to(dtype=pos_dtype)
                if position_ids.dim() == 1:
                    position_ids = position_ids.unsqueeze(0).expand(bs, -1).contiguous()
                elif position_ids.dim() == 2:
                    if int(position_ids.size(0)) == 1 and bs > 1:
                        position_ids = position_ids.expand(bs, -1).contiguous()
                    assert int(position_ids.size(0)) == bs, (
                        "position_ids batch size must match query batch size under Ulysses CP. "
                        f"position_ids.shape={tuple(position_ids.shape)}, query_bs={bs}"
                    )
                else:
                    raise AssertionError(
                        "Ulysses CP FlashAttention wrapper only supports 1D or 2D `position_ids`. "
                        f"Got shape={tuple(position_ids.shape)}"
                    )

            position_ids_global = _gather_sharded_seq_tensor(
                position_ids, group=ulysses_group, shard_lens=shard_lens_cat
            )
            position_ids_global = _normalize_position_ids_for_fa_varlen(position_ids_global)

        query_length_global = q_global.size(1)

        new_args = list(args)
        if len(new_args) > 0:
            new_args[0] = q_global
        if len(new_args) > 1:
            new_args[1] = k_global
        if len(new_args) > 2:
            new_args[2] = v_global
        if len(new_args) > 3:
            new_args[3] = attn_mask_global
        if len(new_args) > 4:
            new_args[4] = query_length_global

        # Only update kwargs keys that were already provided (do NOT inject new, version-dependent kw names).
        if "query_states" in kwargs:
            kwargs["query_states"] = q_global
        if "key_states" in kwargs:
            kwargs["key_states"] = k_global
        if "value_states" in kwargs:
            kwargs["value_states"] = v_global
        if "attention_mask" in kwargs:
            kwargs["attention_mask"] = attn_mask_global
        if "position_ids" in kwargs:
            kwargs["position_ids"] = position_ids_global
        if "query_length" in kwargs:
            kwargs["query_length"] = query_length_global
        elif len(new_args) <= 4:
            # If query_length isn't positional in this call, pass it iff the original accepts it.
            sig = None
            try:
                sig = inspect.signature(original_forward)
            except Exception:
                sig = None
            if sig is None or "query_length" in sig.parameters:
                kwargs["query_length"] = query_length_global

        out = original_forward(*new_args, **kwargs)

        if isinstance(out, tuple):
            attn_out = out[0]
        else:
            attn_out = out

        if torch.is_tensor(attn_out) and attn_out.dim() == 4:
            local_out = SeqAllToAll4D.apply(ulysses_group, attn_out, 1, 2, False)
            if layout == "bhsd":
                local_out = local_out.transpose(1, 2)
            if isinstance(out, tuple):
                return (local_out,) + out[1:]
            return local_out

        return out

    return _wrapped


def apply_hf_flash_attention_ulysses_patch() -> Dict[str, Any]:
    """
    PumpkinComment: Patch for different versions of Transformers.
    """
    if _PATCH_STATE["patched"]:
        return {"patched": True, "already": True, **_PATCH_STATE}

    patched_any = False
    result: Dict[str, Any] = {"patched": False, "targets": []}

    try:
        import transformers.modeling_flash_attention_utils as mfu

        if hasattr(mfu, "_flash_attention_forward"):
            _PATCH_STATE["orig_modeling_flash_attention_forward"] = mfu._flash_attention_forward
            mfu._flash_attention_forward = make_ulysses_flash_attention_forward(mfu._flash_attention_forward)
            patched_any = True
            result["targets"].append("transformers.modeling_flash_attention_utils._flash_attention_forward")
    except Exception as e:
        logger.warning(f"Failed to patch transformers.modeling_flash_attention_utils._flash_attention_forward: {e}")

    try:
        from transformers.integrations import flash_attention as fa

        if hasattr(fa, "_flash_attention_forward"):
            _PATCH_STATE["orig_integrations_flash_attention_forward"] = fa._flash_attention_forward
            fa._flash_attention_forward = make_ulysses_flash_attention_forward(fa._flash_attention_forward)
            patched_any = True
            result["targets"].append("transformers.integrations.flash_attention._flash_attention_forward")
    except Exception as e:
        logger.warning(f"Failed to patch transformers.integrations.flash_attention._flash_attention_forward: {e}")

    _PATCH_STATE["patched"] = patched_any
    result["patched"] = patched_any
    return result


def unapply_hf_flash_attention_ulysses_patch() -> None:
    if not _PATCH_STATE["patched"]:
        return

    try:
        import transformers.modeling_flash_attention_utils as mfu

        if _PATCH_STATE["orig_modeling_flash_attention_forward"] is not None:
            mfu._flash_attention_forward = _PATCH_STATE["orig_modeling_flash_attention_forward"]
    except Exception:
        pass

    try:
        from transformers.integrations import flash_attention as fa

        if _PATCH_STATE["orig_integrations_flash_attention_forward"] is not None:
            fa._flash_attention_forward = _PATCH_STATE["orig_integrations_flash_attention_forward"]
    except Exception:
        pass

    _PATCH_STATE["patched"] = False
