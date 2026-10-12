# SPDX-License-Identifier: Apache-2.0
"""Differentiable GR kernels with the native BF16 rounding boundaries.

The forward arithmetic follows vLLM's Qwen3.8-Flash-Next GR operators:
per-stream RMS reduction, Gemma affine FMA, ordered gate accumulation, and
residual FMA. Backward differentiates the underlying real-valued equations.
No inference runtime is needed by the training implementation.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _norm_forward(X, W, Y, R, WIDTH: tl.constexpr, HC: tl.constexpr,
                  SHARED: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    group = row % HC
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * WIDTH + col, col < WIDTH, other=0).to(tl.float32)
    w_col = col if SHARED else group * WIDTH + col
    w = tl.load(W + w_col, col < WIDTH, other=0).to(tl.float32)
    rrms = tl.rsqrt(tl.sum(x * x) / WIDTH + EPS)
    y = x * rrms
    y = tl.fma(y, w, y)
    tl.store(Y + row * WIDTH + col, y, col < WIDTH)
    tl.store(R + row, rrms)


@triton.jit
def _combined_norm_forward(X, W, Y, R, WIDTH: tl.constexpr, HC: tl.constexpr,
                          SHARED: tl.constexpr, EPS: tl.constexpr,
                          TILES: tl.constexpr, BLOCK: tl.constexpr):
    # The inference combine+norm kernel reduces each 512-wide tile first.
    # X already contains the materialized (BF16-rounded) combined residual.
    row = tl.program_id(0)
    group = row % HC
    tiles = tl.arange(0, TILES)
    col = tiles[:, None] * BLOCK + tl.arange(0, BLOCK)[None, :]
    x = tl.load(X + row * WIDTH + col, col < WIDTH, other=0).to(tl.float32)
    rrms = tl.rsqrt(tl.sum(tl.sum(x * x, axis=1), axis=0) / WIDTH + EPS)
    w_col = col if SHARED else group * WIDTH + col
    w = tl.load(W + w_col, col < WIDTH, other=0).to(tl.float32)
    y = x * rrms
    tl.store(Y + row * WIDTH + col, tl.fma(y, w, y), col < WIDTH)
    tl.store(R + row, rrms)


@triton.jit
def _mix_forward(X, G, Y, WIDTH: tl.constexpr, HC: tl.constexpr,
                 BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), tl.float32)
    for group in tl.static_range(HC):
        offset = (row * HC + group) * WIDTH + col
        x = tl.load(X + offset, col < WIDTH, other=0).to(tl.float32)
        g = tl.load(G + offset, col < WIDTH, other=0).to(tl.float32)
        acc = tl.fma(tl.sigmoid(g), x, acc)
    tl.store(Y + row * WIDTH + col, acc / HC, col < WIDTH)


@triton.jit
def _combine_forward(X, B, I, Y, WIDTH: tl.constexpr, HC: tl.constexpr,
                     BLOCK: tl.constexpr, HC_PAD: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    group = tl.arange(0, HC_PAD)
    offset = (row * HC + group[:, None]) * WIDTH + col[None, :]
    mask = (group[:, None] < HC) & (col[None, :] < WIDTH)
    x = tl.load(X + offset, mask, other=0).to(tl.float32)
    b = tl.load(B + row * WIDTH + col, col < WIDTH, other=0).to(tl.float32)
    i = tl.load(I + row * HC + group, group < HC, other=0).to(tl.float32)
    scale = 2.0 * tl.sigmoid(i / HC)
    out = tl.fma(b[None, :], scale[:, None], x)
    tl.store(Y + offset, out, mask)


class GroupedNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps, hc_count, after_combine):
        width = x.shape[-1] // hc_count
        flat = x.contiguous()
        weight = weight.contiguous()
        out = torch.empty_like(flat)
        rrms = torch.empty(x.numel() // width, device=x.device, dtype=torch.float32)
        if after_combine:
            _combined_norm_forward[(rrms.numel(),)](
                flat, weight, out, rrms, width, hc_count, weight.numel() == width,
                eps, triton.next_power_of_2(triton.cdiv(width, 512)), 512,
            )
        else:
            _norm_forward[(rrms.numel(),)](
                flat, weight, out, rrms, width, hc_count, weight.numel() == width,
                eps, triton.next_power_of_2(width),
            )
        ctx.save_for_backward(flat, weight, rrms)
        ctx.hc_count = hc_count
        return out

    @staticmethod
    def backward(ctx, grad):
        x, weight, rrms = ctx.saved_tensors
        hc = ctx.hc_count
        width = x.shape[-1] // hc
        xf = x.float().reshape(-1, hc, width)
        g = grad.float().reshape_as(xf)
        r = rrms.reshape(-1, hc, 1)
        shared = weight.numel() == width
        w = weight.float().reshape(1, 1 if shared else hc, width)
        normalized = xf * r
        dx = dw = None
        if ctx.needs_input_grad[0]:
            weighted = g * (1 + w)
            dx = r * (weighted - normalized * (weighted * normalized).mean(-1, keepdim=True))
            dx = dx.reshape_as(x).to(x.dtype)
        if ctx.needs_input_grad[1]:
            dw = (g * normalized).sum((0, 1) if shared else 0).reshape_as(weight).to(weight.dtype)
        return dx, dw, None, None, None


class GateMix(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gate, hc_count):
        width = x.shape[-1] // hc_count
        x, gate = x.contiguous(), gate.contiguous()
        out = x.new_empty((*x.shape[:-1], width))
        _mix_forward[(out.numel() // width, triton.cdiv(width, 512))](
            x, gate, out, width, hc_count, 512,
        )
        ctx.save_for_backward(x, gate)
        ctx.hc_count = hc_count
        return out

    @staticmethod
    def backward(ctx, grad):
        x, gate = ctx.saved_tensors
        hc = ctx.hc_count
        width = x.shape[-1] // hc
        g = grad.float().reshape(-1, 1, width) / hc
        sigmoid = gate.float().reshape(-1, hc, width).sigmoid()
        dx = dg = None
        if ctx.needs_input_grad[0]:
            dx = (g * sigmoid).reshape_as(x).to(x.dtype)
        if ctx.needs_input_grad[1]:
            dg = g * x.float().reshape(-1, hc, width) * sigmoid * (1 - sigmoid)
            dg = dg.reshape_as(gate).to(gate.dtype)
        return dx, dg, None


class Combine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, residual, block, injection, hc_count):
        width = residual.shape[-1] // hc_count
        residual, block, injection = residual.contiguous(), block.contiguous(), injection.contiguous()
        out = torch.empty_like(residual)
        _combine_forward[(block.numel() // width, triton.cdiv(width, 512))](
            residual, block, injection, out, width, hc_count, 512,
            triton.next_power_of_2(hc_count),
        )
        ctx.save_for_backward(block, injection)
        ctx.hc_count = hc_count
        return out

    @staticmethod
    def backward(ctx, grad):
        block, injection = ctx.saved_tensors
        hc = ctx.hc_count
        width = block.shape[-1]
        g = grad.float().reshape(-1, hc, width)
        sigmoid = (injection.float().reshape(-1, hc) / hc).sigmoid()
        db = di = None
        if ctx.needs_input_grad[1]:
            db = (g * (2 * sigmoid).unsqueeze(-1)).sum(1).reshape_as(block).to(block.dtype)
        if ctx.needs_input_grad[2]:
            di = (g * block.float().reshape(-1, 1, width)).sum(-1)
            di = (di * (2 / hc) * sigmoid * (1 - sigmoid)).reshape_as(injection).to(injection.dtype)
        return grad if ctx.needs_input_grad[0] else None, db, di, None
