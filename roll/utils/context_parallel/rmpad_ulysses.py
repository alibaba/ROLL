"""
Reference: https://verl.readthedocs.io/en/latest/_modules/verl/utils/ulysses.html
"""

from typing import Optional, Tuple

import torch
import torch.distributed as dist

from roll.utils.context_parallel.autograd_gather import ulysses_gather
from roll.utils.context_parallel.globals import get_ulysses_group


def ulysses_pad_inputs(
    input_ids_rmpad: torch.Tensor,
    position_ids_rmpad: Optional[torch.Tensor] = None,
    *,
    cp_size: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], int]:
    """
    Pad rmpad token streams so sequence length is divisible by cp_size, without slicing.

    This is used by VLM CP(Ulysses) "slice-after-embedding" paths where we must keep the
    full token stream on every CP rank until the decoder slices `inputs_embeds`.

    Args:
        input_ids_rmpad: shape [1, total_nnz]
        position_ids_rmpad: shape [1, total_nnz] or [C, 1, total_nnz] (e.g. mrope)
        cp_size: context parallel group size

    Returns:
        padded_input_ids_rmpad: shape [1, total_padded]
        padded_position_ids_rmpad: same padding, if provided
        pad_size: how many tokens were padded at the end
    """
    if cp_size <= 1:
        return input_ids_rmpad, position_ids_rmpad, 0

    assert (
        input_ids_rmpad.dim() == 2 and input_ids_rmpad.size(0) == 1
    ), f"Expected input_ids_rmpad shape [1, total_nnz], got {tuple(input_ids_rmpad.shape)}"
    if position_ids_rmpad is not None:
        assert position_ids_rmpad.size(-2) == 1, "position_ids_rmpad must have batch dim==1 for rmpad path"
        assert input_ids_rmpad.size(-1) == position_ids_rmpad.size(-1)

    _, total_seq_len = input_ids_rmpad.shape
    pad_size = (cp_size - (total_seq_len % cp_size)) % cp_size
    if pad_size > 0:
        input_ids_rmpad = torch.nn.functional.pad(input_ids_rmpad, (0, pad_size), value=0)
        if position_ids_rmpad is not None:
            pad_pos = torch.arange(pad_size, device=position_ids_rmpad.device).unsqueeze(0)  # [1, pad]
            if position_ids_rmpad.dim() == 3:
                pad_pos = pad_pos.unsqueeze(0).repeat(position_ids_rmpad.size(0), 1, 1)  # [C, 1, pad]
            position_ids_rmpad = torch.cat((position_ids_rmpad, pad_pos), dim=-1)

    return input_ids_rmpad, position_ids_rmpad, pad_size


def ulysses_pad_and_slice_inputs(
    input_ids_rmpad: torch.Tensor,
    position_ids_rmpad: Optional[torch.Tensor] = None,
    *,
    cp_size: int,
    cp_rank: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], int]:
    """
    Pad and slice rmpad token streams so sequence length is divisible by cp_size.

    Args:
        input_ids_rmpad: shape [1, total_nnz]
        position_ids_rmpad: shape [1, total_nnz] or [C, 1, total_nnz] (e.g. mrope)
        cp_size/cp_rank: context parallel group size/rank

    Returns:
        sliced_input_ids_rmpad: shape [1, total_padded/cp_size]
        sliced_position_ids_rmpad: same slicing, if provided
        pad_size: how many tokens were padded at the end
    """
    if cp_size <= 1:
        return input_ids_rmpad, position_ids_rmpad, 0

    assert (
        input_ids_rmpad.dim() == 2 and input_ids_rmpad.size(0) == 1
    ), f"Expected input_ids_rmpad shape [1, total_nnz], got {tuple(input_ids_rmpad.shape)}"
    if position_ids_rmpad is not None:
        assert position_ids_rmpad.size(-2) == 1, "position_ids_rmpad must have batch dim==1 for rmpad path"
        assert input_ids_rmpad.size(-1) == position_ids_rmpad.size(-1)

    _, total_seq_len = input_ids_rmpad.shape
    pad_size = (cp_size - (total_seq_len % cp_size)) % cp_size
    if pad_size > 0:
        input_ids_rmpad = torch.nn.functional.pad(input_ids_rmpad, (0, pad_size), value=0)
        if position_ids_rmpad is not None:
            pad_pos = torch.arange(pad_size, device=position_ids_rmpad.device).unsqueeze(0)  # [1, pad]
            if position_ids_rmpad.dim() == 3:
                pad_pos = pad_pos.unsqueeze(0).repeat(position_ids_rmpad.size(0), 1, 1)  # [C, 1, pad]
            position_ids_rmpad = torch.cat((position_ids_rmpad, pad_pos), dim=-1)

    total_padded = input_ids_rmpad.size(1)
    part = total_padded // cp_size
    start = cp_rank * part
    end = (cp_rank + 1) * part
    input_ids_rmpad = input_ids_rmpad[:, start:end]
    if position_ids_rmpad is not None:
        position_ids_rmpad = position_ids_rmpad[..., start:end]
    return input_ids_rmpad, position_ids_rmpad, pad_size


def gather_outputs_and_unpad(
    x: torch.Tensor,
    *,
    gather_dim: int,
    unpad_dim: Optional[int] = None,
    padding_size: int = 0,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """
    All-gather tensors across CP ranks and optionally remove padding added by `ulysses_pad_and_slice_inputs`.

    Note: this gathers full tensors onto every CP rank; use only when acceptable.
    """
    group = get_ulysses_group() if group is None else group
    if group is None or dist.get_world_size(group) <= 1:
        if unpad_dim is not None and padding_size:
            sl = [slice(None)] * x.dim()
            sl[unpad_dim] = slice(0, x.size(unpad_dim) - padding_size)
            return x[tuple(sl)]
        return x

    out = ulysses_gather(x, gather_dim=gather_dim, group=group, grad_scaler=True)

    if unpad_dim is not None and padding_size:
        sl = [slice(None)] * out.dim()
        sl[unpad_dim] = slice(0, out.size(unpad_dim) - padding_size)
        out = out[tuple(sl)]
    return out
