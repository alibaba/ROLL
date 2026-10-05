"""Autograd-safe QSA selection and indexer distillation helpers.

The kernels used by inference engines may replace these helpers, but the
selection contract and KL objective stay shared between SFT and RL training.
"""

from __future__ import annotations

import torch
from dataclasses import dataclass
from functools import lru_cache


def select_causal_blocks(
    scores: torch.Tensor, sequence_length: int, block_size: int, token_budget: int
) -> torch.Tensor:
    """Select complete causal blocks and append the final incomplete-block tail.

    ``scores`` is ``[..., num_blocks]`` and represents a single query or a batch
    of queries with identical visibility. The returned indices are the selected
    token positions, ordered by score then token order, followed by the tail.
    """
    if scores.ndim < 1 or block_size < 1 or token_budget < block_size or sequence_length < 0:
        raise ValueError("invalid QSA score shape, block size, budget or sequence length")
    complete = sequence_length // block_size
    if scores.shape[-1] < complete:
        raise ValueError("scores do not cover every complete visible block")
    block_budget = min(token_budget // block_size, complete)
    if block_budget:
        # Stable ties favor lower block IDs. The selected set for unequal scores
        # agrees with the reference top-k; ties have an explicit replay contract.
        blocks = torch.argsort(scores[..., :complete], dim=-1, descending=True, stable=True)[..., :block_budget]
        offsets = torch.arange(block_size, device=scores.device)
        selected = (blocks.unsqueeze(-1) * block_size + offsets).flatten(-2)
    else:
        selected = scores.new_empty((*scores.shape[:-1], 0), dtype=torch.long)
    tail_start = complete * block_size
    tail = torch.arange(tail_start, sequence_length, device=scores.device).expand(*scores.shape[:-1], -1)
    return torch.cat((selected, tail), dim=-1)


@dataclass
class QSASelection:
    """Per-query complete microblocks plus the causal unfilled tail.

    Storage is [batch, sequence, sequence/block_size], never a token-square mask.
    Visible token ranks allow left/right padding without shifting block geometry.
    Cross-document masks must be handled separately; packing is not supported.
    """
    selected_blocks: torch.Tensor
    block_bitmap: torch.Tensor
    valid: torch.Tensor
    token_rank: torch.Tensor
    complete: torch.Tensor
    block_size: int

    @classmethod
    def from_scores(cls, scores, valid, block_size=4, token_budget=2048):
        if valid.ndim != 2 or valid.dtype != torch.bool:
            raise ValueError("QSA valid mask must be boolean [batch, sequence]")
        if block_size < 1 or token_budget < block_size:
            raise ValueError("QSA budget must contain at least one complete block")
        batch, seq = valid.shape
        if seq < 1 or scores.shape != (batch, seq, seq // block_size):
            raise ValueError("QSA scores must be [batch, sequence, sequence//block_size]")
        if scores.device != valid.device:
            raise ValueError("QSA scores and validity must share a device")
        rank = valid.long().cumsum(-1) - 1
        complete = (rank + 1) // block_size
        num_blocks = scores.shape[-1]
        budget = min(token_budget // block_size, num_blocks)
        block_ids = torch.arange(num_blocks, device=scores.device)
        visible = (block_ids < complete[..., None]) & valid[..., None]
        with torch.no_grad():
            masked = scores.detach().float().masked_fill(~visible, -torch.inf)
            order = torch.argsort(masked, descending=True, stable=True, dim=-1)[..., :budget]
            selected_valid = visible.gather(-1, order)
            bitmap = torch.zeros(batch, seq, max(1, num_blocks), device=scores.device, dtype=torch.bool)
            bitmap.scatter_(-1, order, selected_valid)
            selected = order.masked_fill(~selected_valid, -1)
        return cls(selected, bitmap, valid, rank, complete, block_size)

    def mask_mod(self):
        valid, rank, complete, bitmap, size = self.valid, self.token_rank, self.complete, self.block_bitmap, self.block_size
        seq = valid.shape[1]

        def allowed(b, h, query, key):
            q, k = query.clamp(max=seq-1), key.clamp(max=seq-1)
            block = (rank[b, k].clamp(min=0) // size).clamp(max=bitmap.shape[-1]-1)
            tail = rank[b, k] >= complete[b, q] * size
            return ((query < seq) & (key < seq) & (key <= query) & valid[b, q] & valid[b, k]
                    & (tail | bitmap[b, q, block]))
        return allowed

    def dense_mask_for_testing(self):
        batch, seq = self.valid.shape
        if seq > 256:
            raise ValueError("token-square QSA reference masks are limited to 256 tokens")
        b = torch.arange(batch, device=self.valid.device)[:, None, None]
        q = torch.arange(seq, device=self.valid.device)[None, :, None]
        k = torch.arange(seq, device=self.valid.device)[None, None, :]
        return self.mask_mod()(b, 0, q, k)


@lru_cache(maxsize=1)
def _compiled_flex_attention():
    from torch.nn.attention.flex_attention import flex_attention
    return torch.compile(flex_attention, dynamic=False)


def flex_qsa_attention(query, key, value, selection):
    """Exact selected attention with PyTorch's autograd-capable tiled kernel.

    Microblocks are four tokens but execution tiles are 128 tokens. Every causal
    execution tile is conservatively scheduled and its selected tokens masked.
    This bounds probability storage, but does not promise FLOP savings from the
    fine-grained selection; throughput must be measured against other kernels.
    """
    from torch.nn.attention.flex_attention import AuxRequest, BlockMask
    if query.device.type != "cuda":
        raise ValueError("QSA FlexAttention training requires a CUDA device")
    batch, heads, seq, dim = query.shape
    if key.shape != value.shape or key.shape[0] != batch or key.shape[2:] != (seq, dim):
        raise ValueError("QSA core requires matching self-attention Q/K/V sequence and head dimensions")
    if heads % key.shape[1] or selection.valid.shape != (batch, seq):
        raise ValueError("invalid QSA GQA head ratio or selection shape")
    tile_size = 128
    tiles = (seq + tile_size - 1) // tile_size
    counts = torch.arange(1, tiles+1, dtype=torch.int32, device=query.device).view(1, 1, tiles).expand(batch, 1, tiles).contiguous()
    indices = torch.arange(tiles, dtype=torch.int32, device=query.device).view(1, 1, 1, tiles).expand(batch, 1, tiles, tiles).contiguous()
    block_mask = BlockMask.from_kv_blocks(
        counts, indices, BLOCK_SIZE=tile_size, mask_mod=selection.mask_mod(),
    )
    if seq % tile_size:
        # PyTorch 2.13 requires exact logical sequence lengths. This crop is
        # valid because mask_mod uses absolute positions and already bounds them.
        block_mask = block_mask._adjust(seq, seq)
    output, aux = _compiled_flex_attention()(
        query, key, value, block_mask=block_mask, enable_gqa=True,
        return_aux=AuxRequest(lse=True),
        kernel_options={"BACKEND": "TRITON", "BLOCK_M": 32, "BLOCK_N": 32},
    )
    return output, aux.lse


def selected_qsa_attention(query, key, value, selection):
    """Small independent numerical oracle, never an automatic production fallback."""
    batch, heads, seq, dim = query.shape
    if seq > 256:
        raise ValueError("selected-token reference is limited to 256 tokens")
    groups = heads // key.shape[1]
    block_tokens = visible_block_tokens(selection.valid, selection.block_size)
    output = torch.zeros_like(query)
    lse = torch.full((batch, heads, seq), -torch.inf, dtype=torch.float32, device=query.device)
    for b in range(batch):
        for qpos in range(seq):
            if not bool(selection.valid[b, qpos]):
                continue
            blocks = selection.selected_blocks[b, qpos]
            blocks = blocks[blocks >= 0]
            complete = block_tokens[b].index_select(0, blocks).flatten() if blocks.numel() else block_tokens.new_empty((0,))
            rank = selection.token_rank[b]
            tail = torch.nonzero((rank >= selection.complete[b, qpos] * selection.block_size) &
                                 (rank <= rank[qpos]) & selection.valid[b], as_tuple=False).flatten()
            indices = torch.cat((complete, tail)).unique(sorted=True)
            if indices.numel() == 0:
                continue
            for h in range(heads):
                kvh = h // groups
                logits = (query[b, h, qpos] @ key[b, kvh, indices].transpose(-1, -2)).float() / dim**0.5
                probabilities = logits.softmax(-1)
                output[b, h, qpos] = probabilities.to(value.dtype) @ value[b, kvh, indices]
                lse[b, h, qpos] = logits.logsumexp(-1)
    return output, lse


def visible_block_tokens(valid, block_size=4):
    """Map compressed visible-block IDs back to original token positions."""
    batch, seq = valid.shape
    positions = torch.arange(seq, device=valid.device).expand(batch, -1)
    ordered = positions.masked_fill(~valid, seq).sort(-1).values
    return ordered[:, :seq//block_size*block_size].reshape(batch, seq//block_size, block_size).clamp(max=seq-1)


class QSAIndexerNorm(torch.nn.Module):
    def __init__(self, width, eps, **kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(width, **kwargs))
        self.eps = eps

    def forward(self, value):
        value_fp32 = value.float()
        normalized = value_fp32 * (value_fp32.square().mean(-1, keepdim=True) + self.eps).rsqrt()
        return (normalized * (1 + self.weight.float())).to(value.dtype)


def apply_indexer_rope(value, angles):
    """Apply the same partial rotary angles used by main attention."""
    width = angles.shape[-1]
    if width > value.shape[-1] or width % 2:
        raise ValueError("invalid QSA partial rotary geometry")
    rotary, rest = value[..., :width], value[..., width:]
    a, b = rotary.chunk(2, -1)
    rotated = torch.cat((-b, a), -1)
    return torch.cat((rotary * angles.cos().to(value.dtype) + rotated * angles.sin().to(value.dtype), rest), -1)


class QSAIndexer(torch.nn.Module):
    def __init__(self, hidden_size, num_heads=4, head_dim=128, block_size=4,
                 token_budget=2048, eps=1e-6, device=None, dtype=None):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, head_dim
        self.block_size, self.token_budget = block_size, token_budget
        kw = {"device": device, "dtype": dtype}
        self.index_qk_proj = torch.nn.Linear(hidden_size, (num_heads+1)*head_dim, bias=False, **kw)
        self.q_layernorm = QSAIndexerNorm(head_dim, eps, **kw)
        self.k_layernorm = QSAIndexerNorm(head_dim, eps, **kw)

    def forward(self, hidden, angles, valid):
        """Return selection and differentiable Q/pooled K, in batch-major layout."""
        batch, seq, _ = hidden.shape
        if angles.ndim != 3 or angles.shape[:2] != (batch, seq):
            raise ValueError("QSA indexer angles must be [batch, sequence, rotary_dim]")
        if valid.shape != (batch, seq):
            raise ValueError("QSA indexer requires full-sequence validity")
        qk = self.index_qk_proj(hidden)
        q, k = qk.split((self.num_heads*self.head_dim, self.head_dim), dim=-1)
        q = apply_indexer_rope(self.q_layernorm(q.reshape(batch, seq, self.num_heads, self.head_dim)), angles.unsqueeze(-2))
        block_tokens = visible_block_tokens(valid, self.block_size)
        batch_ids = torch.arange(batch, device=hidden.device)[:, None, None]
        pooled = k[batch_ids, block_tokens].float().mean(-2).to(k.dtype)
        block_angles = angles[torch.arange(batch, device=hidden.device)[:, None], block_tokens[..., 0]]
        pooled = apply_indexer_rope(self.k_layernorm(pooled), block_angles)
        # Selection has no gradient. KL recomputes only selected block scores.
        with torch.no_grad():
            scores = torch.empty(batch, seq, seq//self.block_size, device=hidden.device, dtype=torch.float32)
            for start in range(0, seq, 128):
                dot = torch.einsum("bqhd,bkd->bqhk", q[:, start:start+128].float(), pooled.float())
                scores[:, start:start+128] = dot.relu().sum(-2)
            selection = QSASelection.from_scores(scores, valid, self.block_size, self.token_budget)
        return selection, q, pooled


def default_indexer_temperature(head_dim: int) -> float:
    """QSA indexer softmax temperature absent an explicit config override.

    The official report's Eq.15 shows no temperature, but the HF reference
    divides the summed relu(qk) score by ``sqrt(index_head_dim)`` before the
    top-k selection and the KL target. Top-k is scale-invariant, so omitting
    this only matters for :func:`indexer_distillation_loss`'s KL sharpness --
    callers must resolve an unset temperature to this, not to 1.0.
    """
    if head_dim <= 0:
        raise ValueError("QSA indexer head_dim must be positive")
    return head_dim ** 0.5


def indexer_distillation_loss(index_query, index_key, selection, query, key, lse,
                             *, temperature=1.0, tile_size=16, loss_mask=None, tp_group=None):
    """Train selected indexer blocks against the detached QSA teacher (Eq17–20).

    Teacher probabilities are reconstructed a bounded query tile at a time from
    attention Q/K and its LSE. No token-square probability tensor is retained.
    The student dot products are checkpointed to avoid saving gathered keys for
    all query/block pairs. Tail tokens affect LSE but never enter the block KL.
    ``temperature`` has no reference-matching default here; callers configuring
    real training must resolve it via :func:`default_indexer_temperature`.
    """
    from torch.utils.checkpoint import checkpoint
    if temperature <= 0 or tile_size < 1:
        raise ValueError("QSA indexer temperature and tile_size must be positive")
    batch, seq, index_heads, index_dim = index_query.shape
    if index_key.shape != (batch, seq//selection.block_size, index_dim):
        raise ValueError("QSA compressed index key geometry mismatch")
    active = selection.valid if loss_mask is None else (selection.valid & loss_mask.bool())
    if loss_mask is not None and loss_mask.shape != selection.valid.shape:
        raise ValueError("QSA auxiliary loss mask must have shape [batch, sequence]")
    denominator = active.sum().clamp_min(1)
    selected_count = selection.selected_blocks.shape[-1]
    if selected_count == 0:
        return (index_query.sum() + index_key.sum()) * 0.0
    block_tokens = visible_block_tokens(selection.valid, selection.block_size)
    kv_heads, dim = key.shape[1], key.shape[-1]
    groups = query.shape[1] // kv_heads
    batch_ids = torch.arange(batch, device=query.device)[:, None, None]
    total = index_query.new_zeros((), dtype=torch.float32)

    def student_tile(iq, ik, selected, teacher, row_mask):
        mask = selected.ge(0)
        gathered = ik[batch_ids, selected.clamp_min(0)]
        scores = torch.matmul(iq.float(), gathered.float().transpose(-1, -2)).relu().sum(-2) / temperature
        safe = scores.masked_fill(~mask, torch.finfo(torch.float32).min)
        log_q = safe.log_softmax(-1)
        per_row = (teacher * (teacher.clamp_min(1e-12).log() - log_q)).sum(-1)
        return (per_row * row_mask).sum()

    for start in range(0, seq, tile_size):
        end = min(seq, start+tile_size)
        selected = selection.selected_blocks[:, start:end]
        mask = selected.ge(0)
        with torch.no_grad():
            positions = block_tokens[batch_ids, selected.clamp_min(0)].flatten(-2)
            key_tokens = key.detach().transpose(1, 2)[batch_ids, positions]
            # [B,Q,KV,G,D] @ [B,Q,KV,D,selected_tokens]. No KV head replication.
            q = query.detach()[:, :, start:end].transpose(1, 2).reshape(batch, end-start, kv_heads, groups, dim)
            k = key_tokens.permute(0, 1, 3, 4, 2)
            logits = torch.matmul(q.float(), k.float()) * dim**-0.5
            normalizer = lse.detach()[:, :, start:end].transpose(1, 2).reshape(batch, end-start, kv_heads, groups, 1)
            token_prob = (logits - normalizer).exp().nan_to_num(0.0, posinf=0.0).sum((2, 3))
            if tp_group is not None and torch.distributed.get_world_size(tp_group) > 1:
                torch.distributed.all_reduce(token_prob, group=tp_group)
            block_prob = token_prob.reshape(batch, end-start, selected_count, selection.block_size).amax(-1)
            block_prob = block_prob.masked_fill(~mask, 0)
            teacher = block_prob / block_prob.sum(-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
        total = total + checkpoint(student_tile, index_query[:, start:end], index_key,
                                   selected, teacher, active[:, start:end], use_reentrant=False)
    return total / denominator


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


__all__ = ["default_indexer_temperature", "qsa_indexer_kl_loss", "select_causal_blocks"]
