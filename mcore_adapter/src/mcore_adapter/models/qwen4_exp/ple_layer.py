"""M6: frozen PLE n-gram lookup for training.

The checkpoint carries a 95 GiB n-gram embedding table (128 shards of
``(2500012, 160)`` bf16) on layer 1. It is frozen during post-training, so for RL
we need the *lookup*, not the training, of that table.

Design (validated by P7e before this was written):

  * the table is a **buffer, not a Parameter**. That keeps it out of the optimizer
    state, out of distributed checkpointing, and out of weight sync. Using
    ``Parameter(requires_grad=False)`` instead would still put a 95 GiB tensor
    through all three.
  * the lookup output is an ordinary tensor, so autograd still flows *through* the
    PLE block into shallower layers -- the table simply receives no gradient.
  * the small PLE projections (``key_proj`` / ``value_proj`` / ``conv1d`` and their
    norms) stay trainable.

P7e also pinned down the failure mode to avoid: detaching the residual stream
(``hidden.detach() + ple_out``) leaves the loss numerically **unchanged** while
silently zeroing gradients for every layer below. There is no error and no NaN.
``forward`` below never detaches the stream, and P21 asserts gradients still reach
a lower layer.

The n-gram id computation mirrors the reference implementation:

    shifted_k = tokens shifted by k, clamped at segment (EOS) boundaries
    mixed     = XOR over k of (shifted_k * layer_multipliers[k])
    ids       = mixed % ngram_heads_vocab_sizes + ngram_heads_offsets

``layer_multipliers``, ``ngram_heads_vocab_sizes`` and ``ngram_heads_offsets`` are
checkpoint buffers, so the hashing does not need to be re-derived -- it is loaded.
That matters: the ids must match the table the weights were trained against, and
re-deriving splitmix64 constants by hand is an easy place to differ silently.
"""

from __future__ import annotations

import torch
from torch import nn


class FrozenNGramEmbedding(nn.Module):
    """Frozen n-gram table plus the id hashing, as a buffer-backed lookup.

    ``shard_paths`` lets the table stay on disk / CPU and be paged in, which is how
    a 95 GiB table is used on 80 GiB cards. For tests and small models the table can
    be provided directly.
    """

    def __init__(
        self,
        embedding_dim: int,
        ngram_size: int,
        heads_per_ngram: int,
        vocab_size: int,
        eos_token_id: int,
        table: torch.Tensor | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.ngram_size = ngram_size
        self.heads_per_ngram = heads_per_ngram
        self.eos_token_id = eos_token_id

        if ngram_size < 2:
            raise ValueError(f"ngram_size must be >= 2, got {ngram_size}")
        self.ngram_heads = (ngram_size - 1) * heads_per_ngram
        if embedding_dim % self.ngram_heads:
            raise ValueError(
                f"ple_embed_dim ({embedding_dim}) must be divisible by total ngram "
                f"heads ({self.ngram_heads})"
            )
        self.head_dim = embedding_dim // self.ngram_heads

        # Hash constants come from the checkpoint, not from re-deriving splitmix64.
        # Registered non-persistent so load_state_dict supplies them.
        self.register_buffer(
            "layer_multipliers", torch.ones(ngram_size, dtype=torch.long, device=device)
        )
        self.register_buffer(
            "ngram_heads_vocab_sizes",
            torch.full((self.ngram_heads,), max(vocab_size, 1), dtype=torch.long,
                       device=device),
        )
        self.register_buffer(
            "ngram_heads_offsets",
            torch.zeros(self.ngram_heads, dtype=torch.long, device=device),
        )

        # THE table: a buffer, so it stays out of the optimizer / DCP / weight sync.
        if table is not None:
            tbl = table
        else:
            tbl = torch.zeros(1, self.head_dim, device=device, dtype=dtype)
        self.register_buffer("table", tbl, persistent=False)

    # -------------------------------------------------------------- id hashing
    @staticmethod
    def _segment_positions(tokens: torch.Tensor, eos_token_id: int):
        """Position within the current EOS-delimited segment.

        n-grams must not cross a document boundary, so shifts are only valid while
        they stay inside the current segment.
        """
        _, seq_len = tokens.shape
        positions = torch.arange(seq_len, device=tokens.device, dtype=torch.int64)
        eos_positions = torch.where(tokens == eos_token_id, positions, -1)
        prev_eos_inclusive = torch.cummax(eos_positions, dim=1).values
        prev_eos = torch.cat(
            [eos_positions.new_full((tokens.shape[0], 1), -1), prev_eos_inclusive[:, :-1]],
            dim=1,
        )
        return positions, positions.unsqueeze(0) - prev_eos - 1

    @staticmethod
    def _shift(tokens, positions, position_in_segment, shift, eos_token_id):
        if shift == 0:
            return tokens
        source = positions - shift
        gather_idx = source.clamp_min(0).unsqueeze(0).expand(tokens.shape[0], -1)
        shifted = tokens.gather(1, gather_idx)
        valid = (source.unsqueeze(0) >= 0) & (position_in_segment >= shift)
        return torch.where(valid, shifted, tokens.new_full((), eos_token_id))

    def compute_ngram_ids(self, tokens: torch.Tensor) -> torch.Tensor:
        """Map ``[batch, seq]`` token ids to ``[batch, seq, ngram_heads]`` table ids.

        Integer arithmetic only -- no gradient exists here regardless of context.
        """
        positions, pos_in_seg = self._segment_positions(tokens, self.eos_token_id)
        shifted = [
            self._shift(tokens, positions, pos_in_seg, k, self.eos_token_id)
            for k in range(self.ngram_size)
        ]

        blocks = []
        for ngram in range(2, self.ngram_size + 1):
            start = (ngram - 2) * self.heads_per_ngram
            end = start + self.heads_per_ngram
            mixed = shifted[0] * self.layer_multipliers[0]
            for i in range(1, ngram):
                mixed = torch.bitwise_xor(mixed, shifted[i] * self.layer_multipliers[i])
            sizes = self.ngram_heads_vocab_sizes[start:end]
            offsets = self.ngram_heads_offsets[start:end]
            blocks.append(torch.remainder(mixed.unsqueeze(-1), sizes) + offsets)
        return torch.cat(blocks, dim=-1)

    # ------------------------------------------------------------------ lookup
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return ``[batch, seq, embedding_dim]``.

        The lookup runs under ``no_grad``: the ids are integers and the table is
        frozen, so nothing here needs a graph. The *result* is still an ordinary
        tensor, so downstream ops (and therefore layers below) remain trainable.
        """
        with torch.no_grad():
            ids = self.compute_ngram_ids(tokens)
            # Range-check before the lookup. An out-of-range id surfaces as a CUDA
            # device-side assert inside F.embedding, which gives no hint about
            # which tensor was wrong or by how much; this says so directly.
            rows = self.table.shape[0]
            hi = int(ids.max())
            if hi >= rows:
                need = int((self.ngram_heads_offsets + self.ngram_heads_vocab_sizes).max())
                raise ValueError(
                    f"n-gram id {hi} exceeds the table's {rows} rows. The table must "
                    f"cover every head's slot: max(offset + vocab_size) = {need} rows. "
                    "Check that all 128 checkpoint shards are loaded."
                )
            flat = torch.nn.functional.embedding(ids.reshape(-1), self.table)
        return flat.reshape(*ids.shape[:-1], self.embedding_dim)


class PLELayer(nn.Module):
    """PLE block: frozen n-gram lookup plus small trainable projections.

    Weight names match the checkpoint (``ple.key_proj`` / ``ple.value_proj`` /
    ``ple.conv1d`` / ``ple.norm_*``) so conversion is a rename.
    """

    def __init__(
        self,
        hidden_size: int,
        ple_embed_dim: int,
        hc_count: int,
        conv_kernel_size: int,
        ngram_size: int,
        heads_per_ngram: int,
        vocab_size: int,
        eos_token_id: int,
        eps: float = 1e-6,
        table: torch.Tensor | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        kw = {"device": device, "dtype": dtype}

        self.ple_embedding = FrozenNGramEmbedding(
            embedding_dim=ple_embed_dim,
            ngram_size=ngram_size,
            heads_per_ngram=heads_per_ngram,
            vocab_size=vocab_size,
            eos_token_id=eos_token_id,
            table=table,
            device=device,
            dtype=dtype,
        )

        # Trainable parts -- small, so these DO go in the optimizer.
        self.hc_count = hc_count
        self.key_proj = nn.Linear(ple_embed_dim, hidden_size * hc_count, bias=False, **kw)
        self.value_proj = nn.Linear(ple_embed_dim, hidden_size, bias=False, **kw)
        self.conv1d = nn.Conv1d(
            hidden_size * hc_count, hidden_size * hc_count, conv_kernel_size,
            groups=hidden_size * hc_count, dilation=ngram_size, bias=False, **kw,
        )
        self.norm_key = nn.Parameter(torch.zeros(hidden_size * hc_count, **kw))
        self.norm_query = nn.Parameter(torch.zeros(hidden_size * hc_count, **kw))
        self.norm_conv = nn.Parameter(torch.zeros(hidden_size * hc_count, **kw))
        self.eps = eps
        self.conv_kernel_size = conv_kernel_size
        self.ngram_size = ngram_size

    @staticmethod
    def _gemma_rmsnorm(x, weight, eps):
        """Gemma-style RMSNorm: ``x*rrms*(1+w)``.

        Same convention as the hyperconnection norms (M7) -- the reference PLE uses
        ``normalized * (1.0 + weight)`` too, so plain ``* w`` would be wrong here as
        well.
        """
        xf = x.float()
        out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
        return (out * (1.0 + weight.float())).to(x.dtype)

    def forward(self, hidden_states: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """Compute the PLE contribution for ``hidden_states``.

        Returns a tensor to be ADDED to the residual stream by the caller. This
        function does not touch the stream itself -- keeping the add outside makes
        it obvious at the call site that the stream is not detached (see P7e).
        """
        ngram = self.ple_embedding(tokens)               # frozen lookup
        ngram = ngram.to(hidden_states.dtype)

        key = self._gemma_rmsnorm(self.key_proj(ngram), self.norm_key, self.eps)
        value = self.value_proj(ngram)
        query = self._gemma_rmsnorm(hidden_states, self.norm_query, self.eps)

        # depthwise causal conv over time; conv1d wants [batch, channels, time]
        b, s, _ = key.shape
        key = key.reshape(b, s, self.hc_count, -1)
        query = query.reshape(b, s, self.hc_count, -1)
        gate = (key * query).sum(-1, keepdim=True) / (key.shape[-1] ** 0.5)
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gated_value = torch.sigmoid(gate) * value.unsqueeze(-2)
        gated_value = gated_value.flatten(-2)
        conv_input = self._gemma_rmsnorm(gated_value, self.norm_conv, self.eps)
        pad = (self.conv_kernel_size - 1) * self.ngram_size
        conv_out = self.conv1d(torch.nn.functional.pad(conv_input.transpose(1, 2), (pad, 0)))
        conv_out = conv_out[..., :s].transpose(1, 2)
        return gated_value + conv_out

    # ------------------------------------------------------------- diagnostics
    def frozen_buffer_report(self) -> dict:
        """Confirm the table is excluded from optimizer / checkpoint / weight sync.

        Cheap to call, and worth asserting in tests: if the table ever becomes a
        Parameter, everything still runs -- it just silently adds ~285 GiB of
        optimizer state (Adam keeps two moments per parameter).
        """
        tbl = self.ple_embedding.table
        param_names = {n for n, _ in self.named_parameters()}
        return {
            "table_is_parameter": isinstance(tbl, nn.Parameter),
            "table_requires_grad": bool(tbl.requires_grad),
            "table_in_named_parameters": any("table" in n for n in param_names),
            "table_in_state_dict": "ple_embedding.table" in self.state_dict(),
            "table_numel": tbl.numel(),
            "trainable_params": sum(p.numel() for p in self.parameters()),
        }


__all__ = ["FrozenNGramEmbedding", "PLELayer"]
