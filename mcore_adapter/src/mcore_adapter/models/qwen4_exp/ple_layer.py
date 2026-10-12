"""Qwen4 PLE: grouped gates and a dilated causal convolution.

The external frozen table remains on CPU when this module moves to CUDA.
The returned delta is added by the decoder to the original residual stream.
"""
from __future__ import annotations
import torch
from torch import nn
from .ngram_embedding import FrozenNGramEmbedding


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
        self.hidden_size = hidden_size
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

    def _gemma_rmsnorm(self, x, weight, eps):
        """Gemma-style RMSNorm: ``x*rrms*(1+w)``.

        Same convention as the hyperconnection norms (M7) -- the reference PLE uses
        ``normalized * (1.0 + weight)`` too, so plain ``* w`` would be wrong here as
        well.
        """
        xf = x.float().unflatten(-1, (self.hc_count, self.hidden_size))
        out = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
        return (out.flatten(-2) * (1.0 + weight.float())).to(x.dtype)

    def forward(self, hidden_states: torch.Tensor, tokens: torch.Tensor, *, valid_mask=None) -> torch.Tensor:
        """Compute the PLE contribution for ``hidden_states``.

        Returns a tensor to be ADDED to the residual stream by the caller. This
        function does not touch the stream itself -- keeping the add outside makes
        it obvious at the call site that the stream is not detached (see P7e).
        """
        ngram = self.ple_embedding(tokens)               # frozen lookup
        ngram = ngram.to(hidden_states)

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
        if valid_mask is not None:
            if valid_mask.dtype != torch.bool or valid_mask.shape != tokens.shape:
                raise ValueError("PLE valid_mask must be boolean with input_ids shape")
            gated_value = gated_value * valid_mask.unsqueeze(-1)
            conv_input = conv_input * valid_mask.unsqueeze(-1)
        pad = (self.conv_kernel_size - 1) * self.ngram_size
        conv_out = self.conv1d(torch.nn.functional.pad(conv_input.transpose(1, 2), (pad, 0)))
        conv_out = conv_out[..., :s].transpose(1, 2)
        return gated_value + torch.nn.functional.silu(conv_out)



__all__ = ["FrozenNGramEmbedding", "PLELayer"]
