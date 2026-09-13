# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.
# This file was copied from https://github.com/Dao-AILab/quack/blob/12f11462f06f8c1d79ac2c3c04e308678c81253c/quack/cross_entropy.py,
# with a minor bug fix applied (see line 156). https://github.com/Dao-AILab/quack/pull/64

import math
from functools import partial
from typing import Literal, Optional, Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import quack.copy_utils as copy_utils
import quack.layout_utils as layout_utils
import quack.utils as utils
import torch
from cutlass import BFloat16, Boolean, Float32, Int32, Int64, const_expr
from quack.compile_utils import make_fake_tensor as fake_tensor
from quack.cute_dsl_utils import torch2cute_dtype_map
from quack.reduce import online_softmax_reduce, row_reduce
from quack.reduction_base import ReductionBase
from torch import Tensor


class CrossEntropy(ReductionBase):
    def __init__(self, dtype: Type[cutlass.Numeric], N: int, online_softmax: bool = True):
        # 2 stages: 1 for max, 1 for sum
        super().__init__(
            dtype,
            N,
            stage=2 if not online_softmax else 1,
            reduction_dtype=Float32 if not online_softmax else Int64,
        )
        self.online_softmax = online_softmax
        self.reload_from = None if N <= 16384 or online_softmax else "smem"

    def _threads_per_row(self):
        N = self.N
        for limit, threads in [(64, 8), (128, 16), (3072, 32), (6144, 64), (16384, 128)]:
            if N <= limit:
                return threads
        return 256

    def _set_cluster_n(self):
        N = self.N
        if const_expr(self.dtype.width == 16):
            thresholds = [(16 * 1024, 1), (32 * 1024, 2), (64 * 1024, 4), (128 * 1024, 8)]
        else:
            thresholds = [(16 * 1024, 1), (64 * 1024, 2), (128 * 1024, 4), (256 * 1024, 8)]
        for limit, cluster in thresholds:
            if N <= limit:
                self.cluster_n = cluster
                return
        self.cluster_n = 16

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,  # (M, N)
        mTarget: cute.Tensor,  # (M,)
        mTargetLogit: Optional[cute.Tensor],  # (M, K) or (M,). If None, we use mX
        mLoss: cute.Tensor,  # (M,)
        mLSE: Optional[cute.Tensor],  # (M,)
        mdX: Optional[cute.Tensor],  # (M, N) - if provided, compute gradient
        ignore_index: Int32,  # Index to ignore in loss computation
        stream: cuda.CUstream,
    ):
        assert mX.element_type == self.dtype
        if const_expr(mTargetLogit is None):
            mTargetLogit = mX
        if const_expr(mdX is not None):
            assert mdX.element_type == self.dtype
        self._set_cluster_n()
        largest_dtype_width = const_expr(mX.element_type.width)
        if const_expr(mdX is not None):
            largest_dtype_width = const_expr(max(largest_dtype_width, mdX.element_type.width))
        vecsize = math.gcd(self.N, 128 // largest_dtype_width)
        tiled_copy, tiler_mn, threads_per_row = self._get_tiled_copy(vecsize=vecsize)
        num_threads = tiled_copy.size
        self.kernel(
            mX,
            mTarget,
            mTargetLogit,
            mLoss,
            mLSE,
            mdX,
            ignore_index,
            tiler_mn,
            tiled_copy,
            threads_per_row,
        ).launch(
            grid=[cute.ceil_div(mX.shape[0], tiler_mn[0]), self.cluster_n, 1],
            block=[num_threads, 1, 1],
            cluster=[1, self.cluster_n, 1] if const_expr(self.cluster_n > 1) else None,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,  # (M, N)
        mTarget: cute.Tensor,  # (M,)
        mTargetLogit: cute.Tensor,  # (M, K) or (M,)
        mLoss: cute.Tensor,  # (M,)
        mLSE: Optional[cute.Tensor],  # (M,)
        mdX: Optional[cute.Tensor],  # (M, N) - if provided, compute gradient
        ignore_index: Int32,  # Index to ignore in loss computation
        tiler_mn: cute.Shape,
        tiled_copy: cute.TiledCopy,
        threads_per_row: cutlass.Constexpr[int],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        cluster_y = const_expr(0) if const_expr(self.cluster_n == 1) else cute.arch.block_idx()[1]
        tv_layout = tiled_copy.layout_tv_tiled

        shape = mX.shape
        idX = cute.make_identity_tensor(shape)
        # slice for CTAs
        gX, cX = [cute.local_tile(mT, tiler_mn, (bidx, cluster_y)) for mT in (mX, idX)]

        smem = cutlass.utils.SmemAllocator()
        sX = smem.allocate_tensor(mX.element_type, cute.make_ordered_layout(tiler_mn, order=(1, 0)), byte_alignment=16)
        reduction_buffer, mbar_ptr = self._allocate_reduction_buffer_and_mbar(smem, tv_layout)

        thr_copy = tiled_copy.get_slice(tidx)

        tXgX = thr_copy.partition_S(gX)
        tXsX = thr_copy.partition_D(sX)
        tXcX = thr_copy.partition_S(cX)[(0, None), None, None]
        tXrX = cute.make_fragment_like(tXgX)

        is_even_N = const_expr(shape[1] == tiler_mn[1] * self.cluster_n)
        tXpX = None if is_even_N else copy_utils.predicate_k(thr_copy.partition_S(cX), limit=shape[1])
        copy = partial(copy_utils.copy, pred=tXpX)

        num_warps = cute.size(tiled_copy) // cute.arch.WARP_SIZE
        self._initialize_cluster(tidx, mbar_ptr, num_warps)

        row = tXcX[0][0]
        target = Int32.zero
        if row < shape[0]:
            target = Int32(mTarget[row])

        if row < shape[0]:
            copy(tXgX, tXsX, is_async=True)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        # Fill OOB values with -inf
        if const_expr(not is_even_N):
            # utils.fill_oob(tXsX, tXpX, -tXsX.element_type.inf)
            utils.fill_oob(tXsX, tXpX, BFloat16(-(2**15)))
        cute.autovec_copy(tXsX, tXrX)
        x = tXrX.load().to(Float32)

        target_logit = Float32.zero
        should_ignore = Boolean(target == ignore_index)
        if row < shape[0] and tXcX[0][1] == 0 and not should_ignore:
            # Only load target logit if not ignoring this index
            if const_expr(cute.rank(mTargetLogit.shape) == 2):
                target_logit = Float32(mTargetLogit[row, target])
            else:
                assert cute.rank(mTargetLogit.shape) == 1
                target_logit = Float32(mTargetLogit[row])

        if const_expr(not self.online_softmax):
            max_x = row_reduce(
                x,
                cute.ReductionOp.MAX,
                threads_per_row,
                reduction_buffer[None, None, 0],
                mbar_ptr + 0 if const_expr(self.cluster_n > 1) else None,
                init_val=-Float32.inf,
                hook_fn=cute.arch.cluster_wait if const_expr(self.cluster_n > 1) else None,
            )
            if const_expr(self.reload_from == "smem"):
                cute.autovec_copy(tXsX, tXrX)
                x = tXrX.load().to(Float32)
            log2_e = math.log2(math.e)
            # This would use ffma instead of fadd then fmul
            exp_x = cute.math.exp2(x * log2_e - (max_x * log2_e), fastmath=False)
            denom = row_reduce(
                exp_x,
                cute.ReductionOp.ADD,
                threads_per_row,
                reduction_buffer[None, None, 1],
                mbar_ptr + 1 if const_expr(self.cluster_n > 1) else None,
                init_val=0.0,
            )
        else:
            max_x, denom, exp_x = online_softmax_reduce(
                x,
                threads_per_row,
                reduction_buffer[None, None, 0],
                mbar_ptr,
                hook_fn=cute.arch.cluster_wait if const_expr(self.cluster_n > 1) else None,
                return_exp_x=const_expr(mdX is not None),
            )

        # Write loss and lse to gmem
        if tXcX[0][1] == 0 and row < shape[0] and (self.cluster_n == 1 or cute.arch.block_idx_in_cluster() == 0):
            lse = max_x + cute.math.log(denom, fastmath=True)
            # Set loss to 0 if this index should be ignored, otherwise compute normally
            loss_val = (lse - target_logit) if not should_ignore else Float32.zero
            mLoss[row] = mLoss.element_type(loss_val)
            if const_expr(mLSE is not None):
                mLSE[row] = lse

        # Compute gradient if mdX is provided
        if const_expr(mdX is not None):
            # Compute probabilities: exp(x) / sum(exp(x))
            # If ignored, gradient should be zero
            denom_inv = (
                # 1.0 / denom
                cute.arch.rcp_approx(denom) if not (denom == 0.0 or denom != denom or should_ignore) else Float32.zero
            )
            probs = exp_x * denom_inv
            gdX = cute.local_tile(mdX, tiler_mn, (bidx, cluster_y))
            tXgdX = thr_copy.partition_D(gdX)
            tXrdX = cute.make_fragment_like(tXgdX)
            tXcFull = thr_copy.partition_S(cX)
            # Compute gradient: probs for all classes, (probs - 1) for target class
            # If ignored, gradient is already zero
            tXrdX_f32 = cute.make_fragment_like(tXrX, Float32)
            tXrdX_f32.store(probs)
            if not should_ignore:
                for i in cutlass.range(cute.size(tXrX), unroll_full=True):
                    tXrdX_f32[i] = tXrdX_f32[i] if tXcFull[i][1] != target else tXrdX_f32[i] - 1.0
            tXrdX.store(tXrdX_f32.load().to(tXrdX.element_type))
            if row < shape[0]:
                copy(tXrdX, tXgdX)


@torch.library.custom_op("quack::cross_entropy_fwd_out", mutates_args={"loss", "lse", "dx"})
def cross_entropy_fwd_out(
    x: Tensor,
    target: Tensor,
    target_logit: Optional[Tensor],
    loss: Tensor,
    lse: Optional[Tensor],
    dx: Optional[Tensor],
    ignore_index: int = -100,
) -> None:
    """Cross entropy forward pass.

    Args:
        x: Input logits tensor of shape (M, N)
        target: Target class indices tensor of shape (M,)
        target_logit: (M, K) or (M,).
            If provided, the target logit will be read from this tensor instead of x.
        loss: Output loss tensor of shape (M,)
        lse: Optional output log-sum-exp tensor of shape (M,)
        dx: Optional output gradient tensor of shape (M, N)
        ignore_index: Index to ignore in loss computation

    Returns:
        None (mutates loss, lse, and optionally dx in-place)
    """
    assert x.dim() == 2, "Input must be 2D"
    assert target.dim() == 1, "Target must be 1D"
    assert x.is_cuda and target.is_cuda, "Tensors must be on CUDA device"
    assert x.dtype in [torch.float16, torch.bfloat16, torch.float32], "Unsupported input dtype"
    assert target.dtype in [torch.int32, torch.int64], "Target must be int32 or int64"
    if target_logit is not None:
        assert target_logit.is_cuda, "Target logits must be on CUDA device"
        assert target_logit.dtype in [torch.float16, torch.bfloat16, torch.float32]
    if dx is not None:
        assert dx.is_cuda, "dx must be on CUDA device"
    N = x.size(1)
    dtype = torch2cute_dtype_map[x.dtype]
    target_dtype = torch2cute_dtype_map[target.dtype]
    target_logit_dtype = torch2cute_dtype_map[target_logit.dtype] if target_logit is not None else None
    compile_key = (
        dtype,
        target_dtype,
        target_logit_dtype,
        N,
        lse is not None,
        dx is not None,
    )
    if compile_key not in cross_entropy_fwd_out.compile_cache:
        batch_sym = cute.sym_int()
        div = math.gcd(128 // dtype.width, N)
        x_cute = fake_tensor(dtype, (batch_sym, N), div)
        dx_cute = fake_tensor(dtype, (batch_sym, N), div) if dx is not None else None
        target_cute = fake_tensor(target_dtype, (batch_sym,))
        if target_logit is not None:
            if target_logit.ndim == 2:
                target_logit_cute = fake_tensor(target_logit_dtype, (batch_sym, cute.sym_int()), div)
            else:
                target_logit_cute = fake_tensor(target_logit_dtype, (batch_sym,))
        else:
            target_logit_cute = None
        loss_cute = fake_tensor(Float32, (batch_sym,))
        lse_cute = fake_tensor(Float32, (batch_sym,)) if lse is not None else None
        # If there's dx, it's faster to not use online softmax since we want the exp(x - max)
        cross_entropy_op = CrossEntropy(dtype, N, online_softmax=dx is None)
        cross_entropy_fwd_out.compile_cache[compile_key] = cute.compile(
            cross_entropy_op,
            x_cute,
            target_cute,
            target_logit_cute,
            loss_cute,
            lse_cute,
            dx_cute,
            Int32(0),  # ignore_index, just for compilation
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    cross_entropy_fwd_out.compile_cache[compile_key](x, target, target_logit, loss, lse, dx, Int32(ignore_index))


cross_entropy_fwd_out.compile_cache = {}


def cross_entropy_fwd(
    x: torch.Tensor,
    target: torch.Tensor,
    target_logit: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    return_lse: bool = False,
    return_dx: bool = False,
    inplace_backward: bool = False,
) -> torch.Tensor | tuple[torch.Tensor]:
    M = x.size(0)
    device = x.device
    loss = torch.empty(M, device=device, dtype=torch.float32)
    lse = torch.empty(M, device=device, dtype=torch.float32) if return_lse else None
    dx = (torch.empty_like(x) if not inplace_backward else x) if return_dx else None
    cross_entropy_fwd_out(x, target, target_logit, loss, lse, dx, ignore_index)
    if return_lse and return_dx:
        return loss, lse, dx
    elif return_lse:
        return loss, lse
    elif return_dx:
        return loss, dx
    else:
        return loss


class CrossEntropyBackward:
    def __init__(self, dtype: Type[cutlass.Numeric], N: int):
        self.dtype = dtype
        self.N = N
        self.vecsize = 128 // dtype.width

    def _threads_per_row(self):
        N = min(self.N, 16384)  # We split by blocks of 16k
        for limit, threads in [(64, 8), (128, 16), (3072, 32), (6144, 64), (16384, 128)]:
            if N <= limit:
                return threads
        return 256

    def _get_tiled_copy(self, vecsize: int):
        assert self.N % vecsize == 0, f"Input N {self.N} is not divisible by vector size {vecsize}"
        N = min(self.N, 16384)
        num_threads = 128 if N <= 16384 else 256
        threads_per_row = self._threads_per_row()
        cols_per_block = num_threads // threads_per_row
        num_blocks_N = cute.ceil_div(N // vecsize, threads_per_row)
        tiler_mn = (cols_per_block, vecsize * num_blocks_N * threads_per_row)
        tiled_copy = copy_utils.tiled_copy_2d(self.dtype, threads_per_row, num_threads, num_copy_elems=vecsize)
        return tiled_copy, tiler_mn, threads_per_row

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mTarget: cute.Tensor,
        mDLoss: cute.Tensor,
        mdX: cute.Tensor,
        mLSE: cute.Tensor,
        ignore_index: Int32,  # Index to ignore in gradient computation
        stream: cuda.CUstream,
    ):
        assert mX.element_type == self.dtype
        assert mdX.element_type == self.dtype
        # e.g. if self.N isn't divisible by 8 for bf16, we might use 64 bits (4 elements) copy
        vecsize = math.gcd(self.N, 128 // self.dtype.width)
        tiled_copy, tiler_mn, threads_per_row = self._get_tiled_copy(vecsize=vecsize)
        num_threads = tiled_copy.size
        # (M,) -> (M, N) with stride 0 in the N dimension
        mDLoss, mTarget, mLSE = [layout_utils.expand(X, dim=1, size=self.N) for X in (mDLoss, mTarget, mLSE)]
        self.kernel(
            mX,
            mTarget,
            mDLoss,
            mdX,
            mLSE,
            ignore_index,
            mX.shape,
            tiler_mn,
            tiled_copy,
            threads_per_row,
        ).launch(
            grid=[
                cute.ceil_div(mX.shape[0], tiler_mn[0]),
                cute.ceil_div(mX.shape[1], tiler_mn[1]),
                1,
            ],
            block=[num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,  # (M, N)
        mTarget: cute.Tensor,  # (M,)
        mDLoss: cute.Tensor,  # (M,)
        mdX: cute.Tensor,  # (M, N)
        mLSE: cute.Tensor,  # (M,)
        ignore_index: Int32,  # Index to ignore in gradient computation
        shape: cute.Shape,
        tiler_mn: cute.Shape,
        tiled_copy: cute.TiledCopy,
        threads_per_row: cutlass.Constexpr[int],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()

        smem = cutlass.utils.SmemAllocator()
        sX = smem.allocate_tensor(mX.element_type, cute.make_ordered_layout(tiler_mn, order=(1, 0)), byte_alignment=16)

        idX = cute.make_identity_tensor(shape)
        gX, gdX, cX = [cute.local_tile(mT, tiler_mn, (bidx, bidy)) for mT in (mX, mdX, idX)]

        thr_copy = tiled_copy.get_slice(tidx)

        tXgX = thr_copy.partition_S(gX)
        tXsX = thr_copy.partition_D(sX)
        tXcX = thr_copy.partition_S(cX)[(0, None), None, None]
        tXcFull = thr_copy.partition_S(cX)
        tXgdX = thr_copy.partition_D(gdX)
        tXrX, tXrdX = [cute.make_fragment_like(thr) for thr in (tXgX, tXgdX)]

        is_even_N = const_expr(shape[1] % tiler_mn[1] == 0)
        tXpX = None if is_even_N else copy_utils.predicate_k(thr_copy.partition_S(cX), limit=shape[1])
        copy = partial(copy_utils.copy, pred=tXpX)

        row = tXcX[0][0]
        if row < shape[0]:
            copy(tXgX, tXsX, is_async=True)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        if const_expr(not is_even_N):
            utils.fill_oob(tXsX, tXpX, -tXsX.element_type.inf)
        cute.autovec_copy(tXsX, tXrX)
        x = tXrX.load().to(Float32)

        target = Int32.zero
        dloss = Float32.zero
        lse = Float32.zero
        if row < shape[0]:
            target = Int32(mTarget[row])
            should_ignore = Boolean(target == ignore_index)
            # Set dloss to 0 if this index should be ignored
            if not should_ignore:
                dloss = Float32(mDLoss[row])
            lse = Float32(mLSE[row])

        log2_e = math.log2(math.e)
        probs = cute.math.exp2(x * log2_e - (lse * log2_e), fastmath=True)
        prob_shifted = probs - 1.0
        mask = cute.make_fragment_like(tXrX, Boolean)
        for i in cutlass.range(cute.size(tXcFull), unroll_full=True):
            mask[i] = tXcFull[i][1] == target
        grad = cute.where(mask.load(), prob_shifted, probs)
        grad = grad * dloss

        tXrdX.store(grad.to(tXrdX.element_type))
        if row < shape[0]:
            copy(tXrdX, tXgdX)


def _cross_entropy_backward(
    x: torch.Tensor,
    target: torch.Tensor,
    dloss: torch.Tensor,
    lse: torch.Tensor,
    dx: torch.Tensor,
    ignore_index=-100,
) -> None:
    """Cross entropy backward pass.
    Args:
        x: Input logits tensor of shape (M, N)
        target: Target class indices tensor of shape (M,)
        dloss: Upstream gradients tensor of shape (M,)
        lse: Log-sum-exp values tensor of shape (M,)
    Returns:
        Input gradients tensor of shape (M, N)
    """
    assert x.dim() == 2, "Input must be 2D"
    assert target.dim() == 1, "Target must be 1D"
    assert dloss.dim() == 1, "dloss must be 1D"
    assert lse.dim() == 1, "lse must be 1D"
    assert x.shape[0] == target.shape[0], "Batch dimensions must match"
    assert x.shape[0] == dloss.shape[0], "Batch dimensions must match"
    assert x.shape[0] == lse.shape[0], "Batch dimensions must match"
    assert x.is_cuda and target.is_cuda and dloss.is_cuda and lse.is_cuda, "Tensors must be on CUDA device"
    assert x.dtype in [torch.float16, torch.bfloat16, torch.float32], "Unsupported input dtype"
    assert target.dtype in [torch.int32, torch.int64], "Target must be int32 or int64"

    N = x.size(1)
    dtype = torch2cute_dtype_map[x.dtype]
    target_dtype = torch2cute_dtype_map[target.dtype]
    compile_key = (dtype, target_dtype, N)
    if compile_key not in _cross_entropy_backward.compile_cache:
        batch_sym = cute.sym_int()
        div = math.gcd(128 // dtype.width, N)
        x_cute, dx_cute = [fake_tensor(dtype, (batch_sym, N), div)] * 2
        target_cute = fake_tensor(target_dtype, (batch_sym,))
        dloss_cute, lse_cute = [fake_tensor(Float32, (batch_sym,))] * 2
        cross_entropy_backward_op = CrossEntropyBackward(dtype, N)
        _cross_entropy_backward.compile_cache[compile_key] = cute.compile(
            cross_entropy_backward_op,
            x_cute,
            target_cute,
            dloss_cute,
            dx_cute,
            lse_cute,
            Int32(0),  # ignore_index, just for compilation
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    _cross_entropy_backward.compile_cache[compile_key](x, target, dloss, dx, lse, Int32(ignore_index))


_cross_entropy_backward.compile_cache = {}


@torch.library.custom_op("quack::cross_entropy_bwd_out", mutates_args={"dx"})
def cross_entropy_bwd_out(
    x: torch.Tensor,
    target: torch.Tensor,
    dloss: torch.Tensor,
    lse: torch.Tensor,
    dx: torch.Tensor,
    ignore_index: int = -100,
) -> None:
    _cross_entropy_backward(x, target, dloss, lse, dx, ignore_index)


def cross_entropy_bwd(
    x: torch.Tensor,
    target: torch.Tensor,
    dloss: torch.Tensor,
    lse: torch.Tensor,
    ignore_index: int = -100,
    inplace_backward: bool = False,
) -> None:
    if inplace_backward and not torch.compiler.is_compiling():
        dx = x
        _cross_entropy_backward(x=x, target=target, dloss=dloss, lse=lse, dx=x, ignore_index=ignore_index)
    else:
        dx = torch.empty_like(x)
        cross_entropy_bwd_out(x=x, target=target, dloss=dloss, lse=lse, dx=dx, ignore_index=ignore_index)
    return dx


class CrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, target, lse_partial=None, ignore_index=-100, inplace_backward=False):
        if lse_partial is None:
            loss, lse = cross_entropy_fwd(x, target, ignore_index=ignore_index, return_lse=True)
        else:
            # if we already compute partial lse, then to compute the final lse we treat
            # @lse_partial as @x and @x as @target_logit
            loss, lse = cross_entropy_fwd(
                lse_partial, target, target_logit=x, ignore_index=ignore_index, return_lse=True
            )
        ctx.save_for_backward(x, target, lse)
        ctx.ignore_index = ignore_index
        ctx.inplace_backward = inplace_backward
        return loss

    @staticmethod
    def backward(ctx, dloss):
        x, target, lse = ctx.saved_tensors
        dx = cross_entropy_bwd(x, target, dloss, lse, ctx.ignore_index, inplace_backward=ctx.inplace_backward)
        return dx, None, None, None, None


def cross_entropy(
    x: torch.Tensor,
    target: torch.Tensor,
    lse_partial: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    reduction: Literal["none", "mean", "sum"] = "mean",
    inplace_backward: bool = False,
) -> torch.Tensor:
    """Cross entropy loss with automatic differentiation support.

    Args:
        x: Input logits tensor of shape (M, N)
        target: Target class indices tensor of shape (M,)
        lse_partial: Optional precomputed log-sum-exp partial results
        reduction: Specifies the reduction to apply to the output:
            'none': no reduction will be applied (default)
            'mean': the sum of the output will be divided by the number of elements
            'sum': the output will be summed
        inplace_backward: Whether to perform backward pass in-place
        ignore_index: Index to ignore in loss computation (loss will be 0 for these indices)

    Returns:
        Cross entropy loss tensor:
            - If reduction='none': tensor of shape (M,) with per-example losses
            - If reduction='mean': scalar tensor with mean loss
            - If reduction='sum': scalar tensor with sum of losses
    """
    loss = CrossEntropyFunction.apply(x, target, lse_partial, ignore_index, inplace_backward)
    if reduction == "mean":
        return loss.sum() / (target != ignore_index).sum().float()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"Invalid reduction mode: {reduction}. Expected one of 'none', 'mean', or 'sum'")
