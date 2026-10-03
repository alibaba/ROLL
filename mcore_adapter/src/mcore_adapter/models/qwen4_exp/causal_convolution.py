"""Differentiable Qwen4 convolution with native low-precision products."""
from functools import lru_cache

import torch
from torch.nn import functional as F


def _convolution(x, weight, bias):
    # Match the native BF16 cache path: round each product, accumulate in
    # FP32, apply SiLU in FP32, then cast the result. Rounding the completed
    # sum instead agrees only at the first token (one nonzero product).
    width = weight.shape[-1]
    padded = F.pad(x.float(), (0, 0, width - 1, 0))
    weight = weight.float()
    accumulator = torch.zeros_like(x, dtype=torch.float32)
    if bias is not None:
        accumulator = accumulator + bias.float()
    for tap in range(width):
        product = padded[:, tap:tap + x.shape[1]] * weight[:, tap]
        accumulator = accumulator + product.to(x.dtype).float()
    return F.silu(accumulator).to(x.dtype)


@lru_cache(maxsize=1)
def _compiled_convolution():
    # Inductor normally removes intermediate downcast/upcast pairs. These
    # casts define this model's arithmetic and must survive fusion in both
    # forward and backward. Dynamic token lengths avoid per-length kernels.
    # Keep normal eager fallback when unusually many stride/gradient variants
    # exhaust the compiler cache; fullgraph=True would abort a training job.
    return torch.compile(
        _convolution, dynamic=True,
        options={"emulate_precision_casts": True},
    )


def causal_conv1d(x: torch.Tensor, weight: torch.Tensor, bias=None) -> torch.Tensor:
    """Causal depthwise SiLU convolution for independent [batch, time, dim] rows.

    This training path has no recurrent cache or cross-sample packing. Autograd
    differentiates the explicit casts, including their gradient dtype changes.
    """
    if x.ndim != 3 or weight.ndim != 2 or weight.shape[0] != x.shape[-1]:
        raise ValueError("Expected x [batch, time, dim] and weight [dim, width]")
    if weight.shape[-1] < 1 or (bias is not None and bias.shape != (x.shape[-1],)):
        raise ValueError("Invalid causal convolution width or bias shape")
    function = _compiled_convolution() if x.is_cuda else _convolution
    return function(x, weight, bias)
