"""M7: hyperconnection for megatron-core.

Qwen3.8-Flash-Next replaces the ordinary residual connection with a
hyperconnection: the residual stream carries ``hc_count`` parallel copies
(``hidden_size * hc_count`` wide), and each block reads a mixed-down view of it
and writes back a gated broadcast. megatron-core has no equivalent -- searching
the tree for ``hyper_connection`` / ``hc_count`` / ``block_inject`` returns
nothing -- so this module is new.

The arithmetic below is not a paraphrase of the reference: P16 pinned it down by
writing this same maths in plain PyTorch and checking it against the vLLM Triton
kernels on random inputs (all five ops matched to 0.000e+00). Two details were
wrong on a first reading and would have been silent numerical bugs:

  * the per-group RMSNorm is **Gemma-style**, ``x*rrms*(1+w)``, not ``x*rrms*w``
  * ``hc_silu`` divides by ``hc_count`` **before** the SiLU, not after

Checkpoint layout this must match (398 tensors, 48 layers x 2 groups):

    layers.N.attn_hyper_connection.hc_norm.weight                (hidden*hc,)
    layers.N.attn_hyper_connection.input_mix_weight_down.weight  (lowrank, hidden*hc)
    layers.N.attn_hyper_connection.input_mix_weight_up.weight    (hidden*hc, lowrank)
    layers.N.attn_hyper_connection.block_inject_weight.weight    (hc, hidden*hc)
    layers.N.mlp_hyper_connection.<same four>
    hyper_connection_mixer.<three>                               (final mix-down)

Real geometry: hidden=2560, hc_count=4 -> stream width 10240, lowrank 320.

This is a training-side (non-fused) implementation: plain PyTorch ops, so autograd
works and the numerics can be compared against the inference kernels. Fusing it is
a later optimisation, not a correctness requirement.
"""

from __future__ import annotations

import torch
from torch import nn


def grouped_gemma_rmsnorm(
    x: torch.Tensor, weight: torch.Tensor, eps: float, hc_count: int
) -> torch.Tensor:
    """Per-stream RMSNorm with a Gemma ``(1 + w)`` affine.

    ``x`` is ``[..., hidden*hc]``; each of the ``hc`` groups is normalised over its
    own ``hidden`` elements. ``weight`` may be ``[hidden*hc]`` (this checkpoint) or
    ``[hidden]`` (shared across streams).
    """
    *lead, total = x.shape
    assert total % hc_count == 0, f"{total} not divisible by hc_count={hc_count}"
    d = total // hc_count

    xg = x.float().reshape(-1, hc_count, d)
    rrms = torch.rsqrt(xg.pow(2).sum(-1, keepdim=True) / d + eps)
    y = xg * rrms

    wf = weight.float()
    if wf.numel() == d:
        wg = wf.reshape(1, 1, d)
    elif wf.numel() == total:
        wg = wf.reshape(1, hc_count, d)
    else:
        raise ValueError(
            f"hc_norm weight has {wf.numel()} elements; expected {d} or {total}"
        )

    out = y * (1.0 + wg)
    return out.reshape(*lead, total).to(x.dtype)


def hc_silu(x: torch.Tensor, hc_count: int) -> torch.Tensor:
    """SiLU of ``x / hc_count``. The division happens first (verified in P16)."""
    xs = x.float() / hc_count
    return (xs * torch.sigmoid(xs)).to(x.dtype)


def hc_gate_mix(x: torch.Tensor, gate: torch.Tensor, hc_count: int) -> torch.Tensor:
    """Collapse the ``hc`` streams into one block input.

    ``mean_over_streams( sigmoid(gate_s) * x_s )`` -- note the ``1/hc``, which is an
    average rather than a sum. Returns ``[..., hidden]``.
    """
    *lead, total = x.shape
    d = total // hc_count
    xg = x.float().reshape(-1, hc_count, d)
    gg = gate.float().reshape(-1, hc_count, d)
    out = (torch.sigmoid(gg) * xg).sum(1) / hc_count
    return out.reshape(*lead, d).to(x.dtype)


def hc_combine(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    hc_count: int,
) -> torch.Tensor:
    """Write a block's output back onto every stream, gated per stream.

    ``out_s = residual_s + block_output * 2*sigmoid(injection_s / hc)``.
    The ``2*`` and the ``/hc`` inside the sigmoid both matter (verified in P16).
    """
    *lead, total = residual.shape
    d = total // hc_count
    resg = residual.float().reshape(-1, hc_count, d)
    inj = 2.0 * torch.sigmoid(injection_logits.float().reshape(-1, hc_count) / hc_count)
    blk = block_output.float().reshape(-1, d)
    out = resg + blk.unsqueeze(1) * inj.unsqueeze(-1)
    return out.reshape(*lead, total).to(residual.dtype)


class HyperConnection(nn.Module):
    """One hyperconnection block (the checkpoint has two per layer: attn and mlp).

    Usage mirrors the reference:

        # before a block
        residual, block_input, injection = hc.mix(residual)
        block_output = block(block_input)

        # before the next block, folding in the previous block's output
        residual, block_input, injection = hc.combine_and_mix(
            residual, block_output, injection
        )

        # or, to materialise the stream state without preparing a new input
        residual = hc.combine(residual, block_output, injection)

    ``use_combine`` controls whether this module also emits the injection logits
    that the *next* combine consumes. In the reference both per-layer modules set
    it; the final ``hyper_connection_mixer`` does not (it only mixes down).
    """

    def __init__(
        self,
        hidden_size: int,
        hc_count: int,
        lowrank: int,
        eps: float = 1e-6,
        use_combine: bool = True,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        sequence_parallel: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.hc_count = hc_count
        self.lowrank = lowrank
        self.eps = eps
        self.use_combine = use_combine

        kw = {"dtype": dtype, "device": device}
        w = self.hyper_hidden_size

        # Names match the checkpoint so conversion is a rename, not a remap.
        self.hc_norm = nn.Parameter(torch.zeros(w, **kw))
        self.input_mix_weight_down = nn.Linear(w, lowrank, bias=False, **kw)
        self.input_mix_weight_up = nn.Linear(lowrank, w, bias=False, **kw)
        if use_combine:
            self.block_inject_weight = nn.Linear(w, hc_count, bias=False, **kw)
        else:
            self.block_inject_weight = None

        # These matrices are replicated across TP ranks. With sequence
        # parallelism each rank sees different tokens, so Megatron's final grad
        # reduction must SUM every GR parameter, including the mixing matrices.
        for parameter in self.parameters():
            parameter.sequence_parallel = sequence_parallel
            parameter.tensor_model_parallel = False

    @property
    def hyper_hidden_size(self) -> int:
        return self.hidden_size * self.hc_count

    def _mix_from_normed(self, xn: torch.Tensor):
        lora = hc_silu(self.input_mix_weight_down(xn), self.hc_count)
        gate = self.input_mix_weight_up(lora)
        block_input = hc_gate_mix(xn, gate, self.hc_count)
        injection = self.block_inject_weight(xn) if self.use_combine else None
        return block_input, injection

    def mix(self, residual: torch.Tensor):
        """Prepare a block input from the stream state. No pending output to fold."""
        xn = grouped_gemma_rmsnorm(residual, self.hc_norm, self.eps, self.hc_count)
        block_input, injection = self._mix_from_normed(xn)
        return residual, block_input, injection

    def combine_and_mix(
        self,
        residual: torch.Tensor,
        prev_block_output: torch.Tensor,
        prev_injection: torch.Tensor,
    ):
        """Fold the previous block's output in, then prepare the next block input.

        The reference fuses the combine with this module's input RMSNorm; here they
        are two calls, which is numerically the same thing (P16 checked exactly
        this: ``combine_norm == combine followed by grouped RMSNorm``).
        """
        residual = hc_combine(
            residual, prev_block_output, prev_injection, self.hc_count
        )
        xn = grouped_gemma_rmsnorm(residual, self.hc_norm, self.eps, self.hc_count)
        block_input, injection = self._mix_from_normed(xn)
        return residual, block_input, injection

    def combine(
        self,
        residual: torch.Tensor,
        block_output: torch.Tensor,
        injection_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Materialise the stream state without preparing a new block input.

        Needed where something adds directly to the stream -- the PLE layer does
        this -- so any pending combine must be applied first.
        """
        return hc_combine(residual, block_output, injection_logits, self.hc_count)


class HyperConnectionMixer(nn.Module):
    """Final mix-down from the ``hc``-wide stream to a single hidden vector.

    Corresponds to the checkpoint's top-level ``hyper_connection_mixer.*`` (and the
    MTP copy). It has no ``block_inject_weight``: nothing consumes an injection
    after it.
    """

    def __init__(
        self,
        hidden_size: int,
        hc_count: int,
        lowrank: int,
        eps: float = 1e-6,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        sequence_parallel: bool = False,
    ) -> None:
        super().__init__()
        self.hc = HyperConnection(
            hidden_size,
            hc_count,
            lowrank,
            eps=eps,
            use_combine=False,
            dtype=dtype,
            device=device,
            sequence_parallel=sequence_parallel,
        )

    @property
    def hc_norm(self):
        return self.hc.hc_norm

    @property
    def input_mix_weight_down(self):
        return self.hc.input_mix_weight_down

    @property
    def input_mix_weight_up(self):
        return self.hc.input_mix_weight_up

    def forward(
        self,
        residual: torch.Tensor,
        pending_output: torch.Tensor | None = None,
        pending_injection: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if pending_output is not None and pending_injection is not None:
            residual = self.hc.combine(residual, pending_output, pending_injection)
        _, block_input, _ = self.hc.mix(residual)
        return block_input


def expand_to_streams(hidden_states: torch.Tensor, hc_count: int) -> torch.Tensor:
    """Widen an ordinary ``[..., hidden]`` tensor into the ``hc``-wide stream state.

    Used once at the start of the stack: the embedding output is a single vector and
    every stream begins as a copy of it.
    """
    # cat, not repeat_interleave: streams are contiguous blocks of `hidden`
    # (stream s occupies [s*hidden : (s+1)*hidden]), which is the layout every
    # kernel above indexes with `stream * HC_DIM + offset`.
    return torch.cat([hidden_states] * hc_count, dim=-1)


__all__ = [
    "HyperConnection",
    "HyperConnectionMixer",
    "grouped_gemma_rmsnorm",
    "hc_silu",
    "hc_gate_mix",
    "hc_combine",
    "expand_to_streams",
]
