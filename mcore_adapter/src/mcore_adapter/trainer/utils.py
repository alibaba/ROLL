from typing import TYPE_CHECKING

import torch
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler


if TYPE_CHECKING:
    from megatron.core.optimizer import MegatronOptimizer

    from ..training_args import TrainingArguments


def get_ltor_masks_and_position_ids(input_ids, build_attention_mask=True, attn_mask_1D=None):
    """Build masks and position id for left to right model."""

    # Extract batch size and sequence length.
    micro_batch_size, seq_length = input_ids.size()
    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
    position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
    if not build_attention_mask:
        return attn_mask_1D, position_ids

    attention_mask = torch.tril(torch.ones((micro_batch_size, seq_length, seq_length), device=input_ids.device)).view(
        micro_batch_size, 1, seq_length, seq_length
    )

    if attn_mask_1D is not None:
        for b in range(micro_batch_size):
            i = torch.sum(attn_mask_1D[b]).long()
            attention_mask[b, 0, (i + 1) :, : (i + 1)] = 0

    # Convert attention mask to binary:
    attention_mask = attention_mask < 0.5
    return attention_mask, position_ids


def get_seqlens_in_batch(attention_mask: "torch.Tensor") -> "torch.Tensor":
    # modified from llama-factory
    r"""
    Gets the sequnce lengths in the current batch.

    e.g.
    ```python
    # input
    [
        [1, 1, 2, 2, 2, 0],
        [1, 2, 2, 3, 3, 3],
    ]
    # output
    [0, 2, 5, 6, 7, 9, 12], 3
    ```
    """
    bsz = attention_mask.size(0)
    dtype, device = attention_mask.dtype, attention_mask.device
    max_num = torch.max(attention_mask).item()
    counts: "torch.Tensor" = torch.zeros((bsz, max_num + 1), dtype=dtype, device=device)
    for i in range(max_num):
        counts[:, i] = torch.sum(attention_mask == (i + 1), dim=-1)
    # TODO: remove paddings
    counts[:, max_num] = torch.sum(attention_mask == 0, dim=-1)

    counts = counts.flatten()
    seqlens = counts[counts.nonzero().squeeze(dim=-1)]
    max_seq_len = seqlens.max()
    seqlens = torch.cumsum(seqlens, dim=-1)
    seqlens = torch.nn.functional.pad(seqlens, (1, 0), value=0)
    return seqlens.to(torch.int32), max_seq_len.to(torch.int32)


def check_pack_seq_aligned(attention_mask: "torch.Tensor", align_size: int):
    r"""
    Check if all sub-sequence is aligned with `align_size` for packed data.

    e.g.
    ```python
    # input
    [
        [1, 1, 2, 2, 2, 0],
        [1, 2, 2, 3, 3, 3],
    ],
    2
    # output
    False
    ```
    """
    max_num = torch.max(attention_mask).item()
    is_valid = True
    for i in range(max_num):
        if not is_valid:
            break
        i_th_seq_lens = torch.sum(attention_mask == (i + 1), dim=-1)
        i_th_seq_valid = (i_th_seq_lens % align_size == 0).all()
        is_valid = is_valid and i_th_seq_valid.item()
    return is_valid


class MegatronLRScheduler(OptimizerParamScheduler):
    _last_lr = None

    def get_lr(self, param_group):
        return super().get_lr(param_group)

    def step(self, increment=1):
        super().step(increment)
        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

    def get_last_lr(self) -> list[float]:
        """Return last computed learning rate by current scheduler."""
        return self._last_lr


def get_megatron_lr_scheduler(args: "TrainingArguments", num_training_steps: int, optimizer: "MegatronOptimizer"):
    scheduler_type_map = {  # hf to megatron
        "constant_with_warmup": "constant",
        "inverse_sqrt": "inverse-square-root",
        "cosine_with_min_lr": "cosine",
        "cosine_warmup_with_min_lr": "cosine",
        "warmup_stable_decay": "WSD",
    }
    lr_scheduler_kwargs = args.lr_scheduler_kwargs or {}
    max_lr = lr_scheduler_kwargs.get("max_lr", args.learning_rate)
    min_lr = lr_scheduler_kwargs.get("min_lr", 0.0)
    init_lr = lr_scheduler_kwargs.get("init_lr", 0.0)
    lr_decay_steps = lr_scheduler_kwargs.get("lr_decay_steps", num_training_steps)
    lr_scheduler_type = getattr(args.lr_scheduler_type, "value", args.lr_scheduler_type)
    lr_decay_style = scheduler_type_map.get(lr_scheduler_type, lr_scheduler_type)
    if lr_decay_style not in ["constant", "cosine", "linear", "inverse-square-root", "WSD"]:
        raise ValueError(f"lr_scheduler_type {lr_scheduler_type} is not supported")
    kwargs = {}
    if lr_decay_style == "WSD":
        wsd_decay_steps = lr_scheduler_kwargs.get("wsd_decay_steps", None)
        lr_wsd_decay_style = lr_scheduler_kwargs.get("lr_wsd_decay_style", None)
        assert wsd_decay_steps is not None, "wsd_decay_steps is required for WSD"
        kwargs = {
            "wsd_decay_steps": wsd_decay_steps,
            "lr_wsd_decay_style": lr_wsd_decay_style,
        }

    return MegatronLRScheduler(
        optimizer,
        init_lr=init_lr,
        max_lr=max_lr,
        min_lr=min_lr,
        lr_warmup_steps=args.get_warmup_steps(num_training_steps),
        lr_decay_steps=lr_decay_steps,
        lr_decay_style=lr_decay_style,
        start_wd=args.weight_decay,  # currently not support weight decay scheduling
        end_wd=args.weight_decay,
        wd_incr_style="constant",
        wd_incr_steps=0,
        **kwargs,
    )
