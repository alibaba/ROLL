"""Tensor Core linear projections whose partial outputs remain in FP32."""
from __future__ import annotations

import torch
from torch.autograd.function import once_differentiable
from torch.nn import functional as F


class _LinearWithFP32Output(torch.autograd.Function):
    """Supply backward for PyTorch's CUDA mm(out_dtype=float32) overload.

    The row-parallel caller casts the reduced output back to the input dtype.
    Its upstream gradient therefore has that dtype's representable values;
    retain low-precision GEMM operands with FP32 accumulation in backward too.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x, weight)
        output = torch.mm(x.reshape(-1, x.shape[-1]), weight.t(), out_dtype=torch.float32)
        return output.reshape(*x.shape[:-1], weight.shape[0])

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: torch.Tensor):
        x, weight = ctx.saved_tensors
        grad = grad_output.reshape(-1, weight.shape[0]).to(x.dtype)
        dx = dw = None
        if ctx.needs_input_grad[0]:
            dx = torch.mm(grad, weight, out_dtype=torch.float32).to(x.dtype).reshape_as(x)
        if ctx.needs_input_grad[1]:
            dw = torch.mm(grad.t(), x.reshape(-1, x.shape[-1]), out_dtype=torch.float32).to(weight.dtype)
        return dx, dw


def linear_with_fp32_output(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Keep BF16/FP16 GEMM operands and return unrounded FP32 partial sums.

    Use immediately before an FP32 TP sum and a cast to the input dtype. CUDA
    uses Tensor Cores without materializing FP32 copies of the model weights.
    CPU and full-precision inputs use the ordinary differentiable linear path.
    """
    if x.dtype in (torch.bfloat16, torch.float16):
        if x.is_cuda:
            return _LinearWithFP32Output.apply(x, weight)
        return F.linear(x.float(), weight.float())
    return F.linear(x, weight)
