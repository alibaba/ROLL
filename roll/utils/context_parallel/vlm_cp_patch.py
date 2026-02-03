"""
PumpkinComment:

For many VLMs, slicing `input_ids` before the model builds `inputs_embeds` can break alignment between
visual placeholder tokens and visual features. Instead, keep the full token stream on every CP rank,
build `inputs_embeds`, then slice `inputs_embeds` (and associated tensors) inside the decoder forward.

Reference: https://github.com/volcengine/verl/blob/main/verl/models/transformers/monkey_patch.py
"""

import types
from typing import Any, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
from torch import nn

from roll.utils.context_parallel.globals import get_ulysses_group, get_ulysses_size
from roll.utils.logging import get_logger

logger = get_logger()


def _get_cp_info() -> Tuple[int, int, Optional[dist.ProcessGroup]]:
    group = get_ulysses_group()
    cp_size = int(get_ulysses_size() or 1)
    if group is None or cp_size <= 1:
        return 1, 0, group
    return cp_size, dist.get_rank(group), group


def _slice_seq_dim(x: torch.Tensor, *, start: int, end: int, seq_dim: int) -> torch.Tensor:
    sl = [slice(None)] * x.dim()
    sl[seq_dim] = slice(start, end)
    return x[tuple(sl)].contiguous()


def _slice_position_ids(position_ids: torch.Tensor, *, start: int, end: int) -> torch.Tensor:
    # Common shapes:
    # - (bs, seq)
    # - (C, bs, seq)  (e.g. some multimodal rope layouts)
    # - (C, 1, seq)   (rmpad path with bs==1)
    if position_ids.dim() == 2:
        return position_ids[:, start:end].contiguous()
    if position_ids.dim() == 3:
        return position_ids[..., start:end].contiguous()
    raise ValueError(f"Unexpected position_ids shape: {position_ids.shape}")


def _slice_attention_mask(attention_mask: torch.Tensor, *, start: int, end: int) -> torch.Tensor:
    if attention_mask.dim() == 2:
        return attention_mask[:, start:end].contiguous()
    if attention_mask.dim() == 4 and attention_mask.size(-1) >= end and attention_mask.size(-2) >= end:
        return attention_mask[:, :, start:end, start:end].contiguous()
    raise ValueError(f"Unexpected attention_mask shape: {attention_mask.shape}")


def patch_vlm_decoder_for_cp(
    decoder_module: nn.Module,
    *,
    allow_no_inputs_embeds: bool = True,
    name: str = "",
) -> bool:
    """
    Patch a decoder/text-stack module to slice `inputs_embeds` inside forward under CP.

    This patches ONLY the given module instance (not the global class), to avoid affecting other code paths.
    """
    if getattr(decoder_module, "_roll_vlm_cp_patched", False):
        return True

    original_forward = decoder_module.forward

    def _wrapped_forward(self: nn.Module, *args: Any, **kwargs: Any):
        cp_size, cp_rank, _ = _get_cp_info()
        if cp_size <= 1:
            return original_forward(*args, **kwargs)

        inputs_embeds = kwargs.get("inputs_embeds", None)
        if not torch.is_tensor(inputs_embeds):
            if allow_no_inputs_embeds:
                return original_forward(*args, **kwargs)
            raise RuntimeError("VLM CP patch expects `inputs_embeds` in decoder forward kwargs, but it was missing.")

        # Guard against re-entrancy / nested forwards.
        if not getattr(self, "_roll_vlm_cp_needs_initial_slice", True):
            return original_forward(*args, **kwargs)

        seq_len = inputs_embeds.size(1)
        if seq_len % cp_size != 0:
            # This should not happen if the caller padded to multiple-of-cp, but keep safe.
            raise RuntimeError(f"inputs_embeds seq_len={seq_len} not divisible by cp_size={cp_size}")
        part = seq_len // cp_size
        start = cp_rank * part
        end = (cp_rank + 1) * part

        call_kwargs = dict(kwargs)
        call_kwargs["inputs_embeds"] = _slice_seq_dim(inputs_embeds, start=start, end=end, seq_dim=1)

        # Slice position_ids if present.
        position_ids = call_kwargs.get("position_ids", None)
        if torch.is_tensor(position_ids):
            call_kwargs["position_ids"] = _slice_position_ids(position_ids, start=start, end=end)

        # Slice attention_mask if present (non-rmpad CP path).
        attention_mask = call_kwargs.get("attention_mask", None)
        if torch.is_tensor(attention_mask):
            call_kwargs["attention_mask"] = _slice_attention_mask(attention_mask, start=start, end=end)

        # Qwen3-VL style extras (best-effort).
        visual_pos_masks = call_kwargs.get("visual_pos_masks", None)
        deepstack_visual_embeds = call_kwargs.get("deepstack_visual_embeds", None)
        if torch.is_tensor(visual_pos_masks):
            # visual_pos_masks expected shape: (bs, seq)
            sliced_visual_mask = _slice_seq_dim(visual_pos_masks, start=start, end=end, seq_dim=1)
            call_kwargs["visual_pos_masks"] = sliced_visual_mask

            if isinstance(deepstack_visual_embeds, Sequence) and len(deepstack_visual_embeds) > 0:
                # Compute which visual embeddings belong to this CP shard.
                # We count visual tokens across the whole (replicated) batch.
                with torch.no_grad():
                    visual_start = int(visual_pos_masks[:, :start].sum().item()) if start > 0 else 0
                    visual_end = int(visual_pos_masks[:, :end].sum().item())

                sliced_embeds = []
                for emb in deepstack_visual_embeds:
                    if not torch.is_tensor(emb):
                        sliced_embeds.append(emb)
                        continue
                    if visual_end <= visual_start:
                        sliced_embeds.append(emb[:0])
                    else:
                        sliced_embeds.append(emb[visual_start:visual_end])
                call_kwargs["deepstack_visual_embeds"] = sliced_embeds

        self._roll_vlm_cp_needs_initial_slice = False
        try:
            return original_forward(*args, **call_kwargs)
        finally:
            self._roll_vlm_cp_needs_initial_slice = True

    decoder_module.forward = types.MethodType(_wrapped_forward, decoder_module)
    setattr(decoder_module, "_roll_vlm_cp_patched", True)
    setattr(decoder_module, "_roll_vlm_cp_needs_initial_slice", True)
    if dist.is_available() and dist.is_initialized() and dist.get_rank() == 0:
        logger.info(f"Applied VLM CP decoder slice patch to {name or decoder_module.__class__.__name__}")
    return True
