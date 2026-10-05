"""Qwen3.8 sigmoid-gated RMSNorm with native reciprocal-square-root rounding.

FLA's fused norm uses ``1 / sqrt``. Its FP32 result can cross a BF16
rounding boundary relative to native ``rsqrt``, even for identical inputs.
Keep the native forward arithmetic and pass its saved reciprocal RMS to
FLA's fused backward, without changing other models' FLA normalization.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _forward(X, G, W, Y, R, stride_x, stride_g, M, D: tl.constexpr, eps,
             BLOCK_D: tl.constexpr, ROWS: tl.constexpr):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, BLOCK_D)
    mask = (rows[:, None] < M) & (cols[None, :] < D)
    x = tl.load(X + rows[:, None] * stride_x + cols[None, :], mask, 0).to(tl.float32)
    variance = tl.sum(x * x, axis=1) / D
    reciprocal_rms = tl.rsqrt(variance + eps)
    tl.store(R + rows, reciprocal_rms, rows < M)
    weight = tl.load(W + cols, cols < D, 0).to(tl.float32)
    gate = tl.load(G + rows[:, None] * stride_g + cols[None, :], mask, 0).to(tl.float32)
    normalized = x * reciprocal_rms[:, None]
    result = (normalized * weight[None, :]) * tl.sigmoid(gate)
    tl.store(Y + rows[:, None] * D + cols[None, :], result, mask)


class _SigmoidGatedRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gate, weight, eps):
        x, gate, weight = x.contiguous(), gate.contiguous(), weight.contiguous()
        rows, width = x.shape
        output = torch.empty_like(x)
        reciprocal_rms = torch.empty(rows, dtype=torch.float32, device=x.device)
        sm_count = torch.cuda.get_device_properties(x.device).multi_processor_count
        rows_per_block = min(4, triton.next_power_of_2(triton.cdiv(rows, 2 * sm_count)))
        block = triton.next_power_of_2(width)
        _forward[(triton.cdiv(rows, rows_per_block),)](
            x, gate, weight, output, reciprocal_rms, x.stride(0), gate.stride(0), rows, width, eps,
            BLOCK_D=block, ROWS=rows_per_block, num_warps=min(max(block // 256, 1), 8))
        ctx.save_for_backward(x, gate, weight, reciprocal_rms)
        ctx.eps = eps
        return output

    @staticmethod
    def backward(ctx, grad_output):
        from fla.modules.fused_norm_gate import layer_norm_gated_bwd

        x, gate, weight, reciprocal_rms = ctx.saved_tensors
        dx, dgate, dweight, _, _ = layer_norm_gated_bwd(
            dy=grad_output.contiguous(), x=x, g=gate, weight=weight, bias=None,
            activation="sigmoid", eps=ctx.eps, mean=None, rstd=reciprocal_rms,
            has_residual=False, is_rms_norm=True, x_dtype=x.dtype)
        return dx, dgate, dweight, None


def sigmoid_gated_rms_norm(x, gate, weight, eps):
    """Normalize contiguous head rows, retaining FP32 through sigmoid gating."""
    if x.ndim != 2 or gate.shape != x.shape or weight.shape != (x.shape[-1],):
        raise ValueError("Expected [rows, head_dim] inputs and one head_dim weight")
    if not x.is_cuda or not gate.is_cuda or not weight.is_cuda:
        raise ValueError("Qwen3.8 fused gated RMSNorm requires CUDA tensors")
    if x.shape[0] == 0:
        return (x * gate * weight).to(x.dtype)
    return _SigmoidGatedRMSNorm.apply(x, gate, weight, eps)
