"""Autograd-safe QSA selection and indexer distillation helpers.

The kernels used by inference engines may replace these helpers, but the
selection contract and KL objective stay shared between SFT and RL training.
"""

from __future__ import annotations

import torch


def select_causal_blocks(
    scores: torch.Tensor, sequence_length: int, block_size: int, token_budget: int
) -> torch.Tensor:
    """Select complete causal blocks and append the final incomplete-block tail.

    ``scores`` is ``[..., num_blocks]`` and represents a single query or a batch
    of queries with identical visibility. The returned indices are the selected
    token positions, ordered by score then token order, followed by the tail.
    """
    if scores.ndim == 1:
        scores = scores.unsqueeze(0)
    complete = sequence_length // block_size
    block_budget = min(token_budget // block_size, complete)
    if block_budget:
        blocks = torch.topk(scores[..., :complete], block_budget, dim=-1, sorted=True).indices
        offsets = torch.arange(block_size, device=scores.device)
        selected = (blocks.unsqueeze(-1) * block_size + offsets).flatten(-2)
    else:
        selected = scores.new_empty((*scores.shape[:-1], 0), dtype=torch.long)
    tail_start = complete * block_size
    tail = torch.arange(tail_start, sequence_length, device=scores.device).expand(*scores.shape[:-1], -1)
    return torch.cat((selected, tail), dim=-1).squeeze(0)


def qsa_indexer_kl_loss(
    scores: torch.Tensor, teacher_probabilities: torch.Tensor, complete_block_mask: torch.Tensor
) -> torch.Tensor:
    """Compute the QSA block KL loss while stopping teacher gradients.

    ``complete_block_mask`` chooses visible complete blocks. Teacher probabilities
    are renormalized over that mask; rows with no valid block contribute zero.
    """
    if scores.shape != teacher_probabilities.shape or scores.shape != complete_block_mask.shape:
        raise ValueError("scores, teacher_probabilities and complete_block_mask must have the same shape")
    mask = complete_block_mask.bool()
    safe_scores = scores.float().masked_fill(~mask, torch.finfo(torch.float32).min)
    log_q = torch.log_softmax(safe_scores, dim=-1)
    teacher = teacher_probabilities.detach().float().masked_fill(~mask, 0)
    teacher = teacher / teacher.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
    per_row = torch.where(mask.any(dim=-1), (teacher * (teacher.clamp_min(1e-12).log() - log_q)).sum(-1), 0)
    return per_row.mean()


__all__ = ["qsa_indexer_kl_loss", "select_causal_blocks"]
