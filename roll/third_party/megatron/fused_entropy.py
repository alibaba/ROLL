"""
Fused Entropy Kernel Implementation

This module provides CUDA fused kernel implementations for entropy computation,
optimized for tensor parallel size 1 (TP=1). It accelerates the entropy calculation
used in tensor_parallel.py by fusing multiple operations into a single kernel pass.

The entropy forward computation is split into two stages:
- Stage 1: Computes max_x and sum_exp(x - max_x) for each row
- Stage 2: Computes the final entropy value: H = max_x + log(sum_exp) - sum(softmax(x) * x)

Note: Currently only supports TP=1 (single GPU tensor parallelism).
"""

import math
import operator
from functools import partial
from typing import Optional, Tuple, Type, Literal, Callable

import torch
from torch import Tensor

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Float32, Boolean, const_expr, BFloat16

import quack.utils as utils
import quack.copy_utils as copy_utils
import quack.layout_utils as layout_utils
from quack.compile_utils import make_fake_tensor as fake_tensor
from quack.reduce import row_reduce, online_softmax_reduce
from quack.reduction_base import ReductionBase
from quack.cute_dsl_utils import torch2cute_dtype_map


class FusedEntropyForwardStage1(ReductionBase):
    """
    First stage of the fused entropy forward pass.

    This kernel computes the intermediate values needed for entropy computation:
    - max_x: Maximum value in each row (for numerical stability)
    - sum_exp: Sum of exp(x - max_x) for each row (softmax denominator)

    Args:
        dtype: Data type of input tensor (e.g., BFloat16, Float16, Float32)
        N: Number of columns in input tensor (vocabulary size)
        online_softmax: If True, uses online softmax algorithm for better performance
    """

    def __init__(self, dtype: Type[cutlass.Numeric], N: int, online_softmax: bool = True):
        """
        Initialize the first stage of entropy forward pass.
        """
        # 2 stages: 1 for max, 1 for sum when not using online softmax
        # 1 stage when using online softmax (computes both max and sum together)
        super().__init__(
            dtype,
            N,
            stage=2 if not online_softmax else 1,
            reduction_dtype=Float32 if not online_softmax else Int64,
        )
        self.online_softmax = online_softmax
        # For large N (>16384), reload data from shared memory to avoid register spilling
        self.reload_from = None if N <= 16384 or online_softmax else "smem"

    def _threads_per_row(self):
        """
        Determine the optimal number of threads per row for reduction.
        """
        N = self.N
        for limit, threads in [(64, 8), (128, 16), (3072, 32), (6144, 64), (16384, 128)]:
            if N <= limit:
                return threads
        return 256

    def _set_cluster_n(self):
        """
        Determine the optimal cluster size in the N dimension.
        """
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
        mX: cute.Tensor,  # (M, N) in
        mXMax: cute.Tensor,  # (M,) out
        mXSumExp: cute.Tensor,  # (M,) out
        stream: cuda.CUstream,
    ):
        """
        Launch the Stage 1 kernel to compute max and sum_exp for each row.

        Args:
            mX: Input logits tensor of shape (M, N)
            mXMax: Output tensor for max values of shape (M,)
            mXSumExp: Output tensor for sum of exponentials of shape (M,)
            stream: CUDA stream for kernel execution
        """
        assert mX.element_type == self.dtype
        self._set_cluster_n()
        # Calculate vector size for memory loading (max 128 bits per load)
        largest_dtype_width = const_expr(mX.element_type.width)
        vecsize = math.gcd(self.N, 128 // largest_dtype_width) # 8 for bf16/fp16
        tiled_copy, tiler_mn, threads_per_row = self._get_tiled_copy(vecsize=vecsize)
        num_threads = tiled_copy.size
        # Launch kernel with appropriate grid, block, and cluster configuration
        self.kernel(
            mX,
            mXMax,
            mXSumExp,
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
        mX: cute.Tensor,  # (M, N) in
        mXMax: cute.Tensor,  # (M,) out
        mXSumExp: cute.Tensor,  # (M,) out
        tiler_mn: cute.Shape,
        tiled_copy: cute.TiledCopy,
        threads_per_row: cutlass.Constexpr[int],
    ):
        """
        CUDA kernel for Stage 1: Compute max and sum_exp for each row.

        This kernel loads input data, performs reduction operations to compute
        the maximum value and sum of exponentials for each row, and writes
        the results to global memory.

        Args:
            mX: Input logits tensor of shape (M, N)
            mXMax: Output tensor for max values of shape (M,)
            mXSumExp: Output tensor for sum of exponentials of shape (M,)
            tiler_mn: Tiling configuration
            tiled_copy: Tiled copy configuration for efficient memory access
            threads_per_row: Number of threads working on each row
        """
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        cluster_y = const_expr(0) if const_expr(self.cluster_n == 1) else cute.arch.block_idx()[1]
        tv_layout = tiled_copy.layout_tv_tiled

        shape = mX.shape
        idX = cute.make_identity_tensor(shape)
        gX, cX = [cute.local_tile(mT, tiler_mn, (bidx, cluster_y)) for mT in (mX, idX)]

        smem = cutlass.utils.SmemAllocator()
        sX = smem.allocate_tensor(
            mX.element_type, cute.make_ordered_layout(tiler_mn, order=(1, 0)), byte_alignment=16
        )
        reduction_buffer, mbar_ptr = self._allocate_reduction_buffer_and_mbar(smem, tv_layout)

        thr_copy = tiled_copy.get_slice(tidx)

        # Partition tensors for this thread
        tXgX = thr_copy.partition_S(gX)  # Global memory partition
        tXsX = thr_copy.partition_D(sX)  # Shared memory partition
        tXcX = thr_copy.partition_S(cX)[(0, None), None, None]  # Coordinate partition
        tXrX = cute.make_fragment_like(tXgX)  # Register fragment

        # Handle non-divisible N by creating predicates
        is_even_N = const_expr(shape[1] == tiler_mn[1] * self.cluster_n)
        tXpX = None if is_even_N else copy_utils.predicate_k(thr_copy.partition_S(cX), limit=shape[1])
        copy = partial(copy_utils.copy, pred=tXpX)

        # Initialize cluster synchronization
        num_warps = cute.size(tiled_copy) // cute.arch.WARP_SIZE
        self._initialize_cluster(tidx, mbar_ptr, num_warps)

        row = tXcX[0][0]

        # Load input data asynchronously from global to shared memory
        if row < shape[0]:
            copy(tXgX, tXsX, is_async=True)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        # Fill out-of-bounds values with -inf (for numerical stability in max reduction)
        if const_expr(not is_even_N):
            # utils.fill_oob(tXsX, tXpX, -tXsX.element_type.inf)
            utils.fill_oob(tXsX, tXpX, BFloat16(-(2**15)))
        # Copy from shared memory to registers and convert to Float32
        cute.autovec_copy(tXsX, tXrX)
        x = tXrX.load().to(Float32)

        # Compute max and sum_exp using either two-pass or online softmax
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
            # Online softmax: compute both max and sum in a single pass
            max_x, denom, _ = online_softmax_reduce(
                x,
                threads_per_row,
                reduction_buffer[None, None, 0],
                mbar_ptr,
                hook_fn=cute.arch.cluster_wait if const_expr(self.cluster_n > 1) else None,
                return_exp_x=False
            )

        # Write results to global memory (only first thread in row, and first block in cluster)
        if (
            tXcX[0][1] == 0
            and row < shape[0]
            and (self.cluster_n == 1 or cute.arch.block_idx_in_cluster() == 0)
        ):
            mXMax[row] = max_x
            mXSumExp[row] = denom


class FusedEntropyForwardStage2(ReductionBase):
    """
    Second stage of the fused entropy forward pass.

    This kernel computes the final entropy value using the intermediate results
    from Stage 1. The entropy formula is:
        H = max_x + log(sum_exp) - sum(softmax(x) * x)

    Where:
    - max_x: Maximum value from Stage 1
    - sum_exp: Sum of exponentials from Stage 1
    - sum(softmax(x) * x): Expected value under the softmax distribution

    Args:
        dtype: Data type of input tensor (e.g., BFloat16, Float16, Float32)
        N: Number of columns in input tensor (vocabulary size)
    """

    def __init__(self, dtype: Type[cutlass.Numeric], N: int):
        """
        Initialize the second stage of entropy forward pass.
        """
        # 1 stage: for computing sum(softmax * x)
        super().__init__(
            dtype,
            N,
            stage=1,
            reduction_dtype=Float32,
        )
        # For large N (>16384), reload data from shared memory to avoid register spilling
        self.reload_from = None if N <= 16384 else "smem"

    def _threads_per_row(self):
        """
        Determine the optimal number of threads per row for reduction.
        """
        N = self.N
        for limit, threads in [(64, 8), (128, 16), (3072, 32), (6144, 64), (16384, 128)]:
            if N <= limit:
                return threads
        return 256

    def _set_cluster_n(self):
        """
        Determine the optimal cluster size in the N dimension.
        """
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
        mX: cute.Tensor,  # (M, N) in
        mXMax: cute.Tensor,  # (M,) in
        mXSumExp: cute.Tensor,  # (M,) in
        mXSumSoftmaxTimes: cute.Tensor,  # (M,) out
        mEntropy: cute.Tensor,  # (M,) out
        stream: cuda.CUstream,
    ):
        """
        Launch the Stage 2 kernel to compute final entropy values.

        Args:
            mX: Input logits tensor of shape (M, N)
            mXMax: Input tensor with max values from Stage 1 of shape (M,)
            mXSumExp: Input tensor with sum of exponentials from Stage 1 of shape (M,)
            mXSumSoftmaxTimes: Output tensor for sum(softmax * x) of shape (M,)
            mEntropy: Output tensor for entropy values of shape (M,)
            stream: CUDA stream for kernel execution
        """
        assert mX.element_type == self.dtype
        self._set_cluster_n()
        largest_dtype_width = const_expr(mX.element_type.width)
        vecsize = math.gcd(self.N, 128 // largest_dtype_width) # 8 for bf16/fp16
        tiled_copy, tiler_mn, threads_per_row = self._get_tiled_copy(vecsize=vecsize)
        num_threads = tiled_copy.size
        self.kernel(
            mX, mXMax, mXSumExp,
            mXSumSoftmaxTimes, mEntropy,
            tiler_mn, tiled_copy, threads_per_row,
        ).launch(
            grid=[cute.ceil_div(mX.shape[0], tiler_mn[0]), self.cluster_n, 1],
            block=[num_threads, 1, 1],
            cluster=[1, self.cluster_n, 1] if const_expr(self.cluster_n > 1) else None,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,  # (M, N) in
        mXMax: cute.Tensor,  # (M,) in
        mXSumExp: cute.Tensor,  # (M,) in
        mXSumSoftmaxTimes: cute.Tensor,  # (M,) out
        mEntropy: cute.Tensor,  # (M,) out
        tiler_mn: cute.Shape,
        tiled_copy: cute.TiledCopy,
        threads_per_row: cutlass.Constexpr[int],
    ):
        """
        CUDA kernel for Stage 2: Compute final entropy values.

        This kernel computes the entropy using the formula:
            H = max_x + log(sum_exp) - sum(softmax(x) * x)
        """
        # Get thread and block indices
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        cluster_y = const_expr(0) if const_expr(self.cluster_n == 1) else cute.arch.block_idx()[1]
        tv_layout = tiled_copy.layout_tv_tiled

        shape = mX.shape
        idX = cute.make_identity_tensor(shape)
        gX, cX = [cute.local_tile(mT, tiler_mn, (bidx, cluster_y)) for mT in (mX, idX)]

        # Allocate shared memory for input data and reduction buffers
        smem = cutlass.utils.SmemAllocator()
        sX = smem.allocate_tensor(
            mX.element_type, cute.make_ordered_layout(tiler_mn, order=(1, 0)), byte_alignment=16
        )
        # Allocate reduction buffer and memory barrier for cluster synchronization
        reduction_buffer, mbar_ptr = self._allocate_reduction_buffer_and_mbar(smem, tv_layout)

        thr_copy = tiled_copy.get_slice(tidx)

        # Partition tensors for this thread
        tXgX = thr_copy.partition_S(gX)  # Global memory partition
        tXsX = thr_copy.partition_D(sX)  # Shared memory partition
        tXcX = thr_copy.partition_S(cX)[(0, None), None, None]  # Coordinate partition
        tXrX = cute.make_fragment_like(tXgX)  # Register fragment

        # Handle non-divisible N by creating predicates
        is_even_N = const_expr(shape[1] == tiler_mn[1] * self.cluster_n)
        tXpX = None if is_even_N else copy_utils.predicate_k(thr_copy.partition_S(cX), limit=shape[1])
        copy = partial(copy_utils.copy, pred=tXpX)

        num_warps = cute.size(tiled_copy) // cute.arch.WARP_SIZE
        self._initialize_cluster(tidx, mbar_ptr, num_warps)

        row = tXcX[0][0]

        # Load input data asynchronously from global to shared memory
        if row < shape[0]:
            copy(tXgX, tXsX, is_async=True)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        if const_expr(not is_even_N):
            # utils.fill_oob(tXsX, tXpX, -tXsX.element_type.inf)
            utils.fill_oob(tXsX, tXpX, -BFloat16(2**15))
        # Copy from shared memory to registers and convert to Float32
        cute.autovec_copy(tXsX, tXrX)
        x = tXrX.load().to(Float32)

        # Load intermediate results from Stage 1
        max_x = mXMax[row]
        denom = mXSumExp[row]

        # Compute softmax: softmax(x) = exp(x - max_x) / sum_exp
        # rcp_approx: 1.0 / denom (reciprocal approximation for faster division)
        log2_e = math.log2(math.e)
        exp_x = cute.math.exp2(x * log2_e - (max_x * log2_e), fastmath=False)
        softmax_x = exp_x * cute.arch.rcp_approx(denom)

        # Compute sum(softmax * x) via reduction
        sum_softmax_times_x = row_reduce(
            softmax_x * x,
            cute.ReductionOp.ADD,
            threads_per_row,
            reduction_buffer[None, None, 0],
            mbar_ptr if const_expr(self.cluster_n > 1) else None,
            hook_fn=cute.arch.cluster_wait if const_expr(self.cluster_n > 1) else None,
            init_val=0.0,
        )

        # Write results to global memory (only first thread in row, and first block in cluster)
        if (
            tXcX[0][1] == 0
            and row < shape[0]
            and (self.cluster_n == 1 or cute.arch.block_idx_in_cluster() == 0)
        ):
            mXSumSoftmaxTimes[row] = sum_softmax_times_x
            # Entropy formula: H = max_x + log(sum_exp) - sum(softmax * x)
            mEntropy[row] = max_x + cute.math.log(denom, fastmath=True) - sum_softmax_times_x


@torch.library.custom_op("roll_kernel::entropy_fwd_out", mutates_args={"entropy", "x_max", "x_sum_exp", "x_sum_softmax_times"})
def entropy_fwd_out(
    x: Tensor,
    entropy: Tensor,
    x_max: Tensor,
    x_sum_exp: Tensor,
    x_sum_softmax_times: Tensor,
) -> None:
    """
    Fused entropy forward pass with in-place output tensors.

    This function computes the entropy of each row in the input tensor using
    a two-stage fused kernel approach:
    - Stage 1: Computes max_x and sum_exp for each row
    - Stage 2: Computes the final entropy: H = max_x + log(sum_exp) - sum(softmax * x)

    The function uses a compilation cache to avoid recompiling kernels for
    the same configuration (dtype, target_dtype, N).

    Args:
        x: Input logits tensor of shape (M, N), where M is batch size and N is vocabulary size
        entropy: Output tensor for entropy values of shape (M,) [mutated in-place]
        x_max: Output tensor for max values of shape (M,) [mutated in-place]
        x_sum_exp: Output tensor for sum of exponentials of shape (M,) [mutated in-place]
        x_sum_softmax_times: Output tensor for sum(softmax * x) of shape (M,) [mutated in-place]

    Returns:
        None (output tensors are mutated in-place)

    Raises:
        AssertionError: If input validation fails (wrong dimensions, device, or dtype)
    """
    assert x.dim() == 2, "Input must be 2D"
    assert x.is_cuda, "Tensors must be on CUDA device"
    assert x.dtype in [torch.float16, torch.bfloat16, torch.float32], "Unsupported input dtype"
    assert entropy.dtype in [torch.float16, torch.bfloat16, torch.float32], "Unsupported input dtype"

    N = x.size(1)

    dtype = torch2cute_dtype_map[x.dtype]
    target_dtype = torch2cute_dtype_map[entropy.dtype]
    # Create compile key for caching compiled kernels
    compile_key = (dtype, target_dtype, N)
    # Compile kernels if not in cache
    if compile_key not in entropy_fwd_out.stage1_compile_cache:
        # Create symbolic batch size for compilation
        batch_sym = cute.sym_int()
        div = math.gcd(128 // dtype.width, N)
        # Create fake tensors for compilation
        x_cute = fake_tensor(dtype, (batch_sym, N), div)
        entropy_cute = fake_tensor(target_dtype, (batch_sym,))
        x_max_cute = fake_tensor(target_dtype, (batch_sym,))
        x_sum_exp_cute = fake_tensor(target_dtype, (batch_sym,))
        x_sum_softmax_times_cute = fake_tensor(target_dtype, (batch_sym,))

        # Initialize Stage 1 and Stage 2 operators with online softmax enabled
        # Online softmax computes both max and sum in a single pass for better performance
        entropy_stage1_op = FusedEntropyForwardStage1(dtype, N, online_softmax=True)
        entropy_stage2_op = FusedEntropyForwardStage2(dtype, N)

        # Compile Stage 1 kernel (computes max_x and sum_exp)
        entropy_fwd_out.stage1_compile_cache[compile_key] = cute.compile(
            entropy_stage1_op,
            x_cute,
            x_max_cute,
            x_sum_exp_cute,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )

        # Compile Stage 2 kernel (computes entropy)
        entropy_fwd_out.stage2_compile_cache[compile_key] = cute.compile(
            entropy_stage2_op,
            x_cute,
            x_max_cute,
            x_sum_exp_cute,
            x_sum_softmax_times_cute,
            entropy_cute,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )

    # Execute Stage 1: compute max_x and sum_exp
    entropy_fwd_out.stage1_compile_cache[compile_key](
        x, x_max, x_sum_exp
    )
    # Execute Stage 2: compute entropy using results from Stage 1
    entropy_fwd_out.stage2_compile_cache[compile_key](
        x, x_max, x_sum_exp, x_sum_softmax_times, entropy
    )


# Compilation cache for Stage 1 and Stage 2 kernels
entropy_fwd_out.stage1_compile_cache = {}
entropy_fwd_out.stage2_compile_cache = {}


def entropy_fwd(
    x: torch.Tensor,
) -> torch.Tensor | tuple[torch.Tensor]:
    """
    Compute entropy of each row in the input tensor.

    This function allocates output tensors and calls the fused entropy forward pass.
    It returns the entropy value along with intermediate results that are needed
    for the backward pass.

    Args:
        x: Input logits tensor of shape (M, N), where M is batch size and N is vocabulary size

    Returns:
        tuple: A tuple containing:
            - entropy: Tensor of shape (M,) with entropy values
            - x_max: Tensor of shape (M,) with max values (for backward pass)
            - x_sum_exp: Tensor of shape (M,) with sum of exponentials (for backward pass)
            - x_sum_softmax_times: Tensor of shape (M,) with sum(softmax * x) (for backward pass)
    """
    M = x.size(0)
    device = x.device
    calc_type = torch.float32

    # Allocate output tensors
    entropy = torch.zeros(M, dtype=calc_type, device=device)
    x_max = torch.zeros(M, dtype=calc_type, device=device)
    x_sum_exp = torch.zeros(M, dtype=calc_type, device=device)
    x_sum_softmax_times = torch.zeros(M, dtype=calc_type, device=device)

    # Compute entropy using fused kernel
    entropy_fwd_out(x, entropy, x_max, x_sum_exp, x_sum_softmax_times)
    return entropy, x_max, x_sum_exp, x_sum_softmax_times


def test_entropy_fwd(M, N):
    entropy_fwd(x)


class FusedEntropyBackward:
    """
    Backward pass for entropy computation.

    This kernel computes the gradient of entropy with respect to the input logits.
    The gradient formula is derived from the entropy loss:
        dH/dx = -d_entropy * softmax(x) * (x - E[x])

    Where:
    - d_entropy: Gradient of entropy loss (scalar per row)
    - softmax(x): Softmax probabilities
    - E[x]: Expected value under softmax distribution (sum_softmax_times from forward)

    Args:
        dtype: Data type of input tensor (e.g., BFloat16, Float16, Float32)
        N: Number of columns in input tensor (vocabulary size)
    """

    def __init__(self, dtype: Type[cutlass.Numeric], N: int):
        """
        Initialize the entropy backward pass.
        """
        self.dtype = dtype
        self.N = N
        self.vecsize = 128 // dtype.width

    def _threads_per_row(self):
        """
        Determine the optimal number of threads per row for reduction.
        We split by blocks of 16k for large vocabularies.
        """
        N = min(self.N, 16384)  # We split by blocks of 16k
        for limit, threads in [(64, 8), (128, 16), (3072, 32), (6144, 64), (16384, 128)]:
            if N <= limit:
                return threads
        return 256

    def _get_tiled_copy(self, vecsize: int):
        """
        Get tiled copy configuration for efficient memory access.

        Args:
            vecsize: Vector size for memory loading

        Returns:
            tuple: (tiled_copy, tiler_mn, threads_per_row)
        """
        assert self.N % vecsize == 0, f"Input N {self.N} is not divisible by vector size {vecsize}"
        N = min(self.N, 16384)
        num_threads = 128 if N <= 16384 else 256
        threads_per_row = self._threads_per_row()
        rows_per_block = num_threads // threads_per_row
        num_blocks_N = cute.ceil_div(N // vecsize, threads_per_row)
        tiler_mn = (rows_per_block, vecsize * num_blocks_N * threads_per_row)
        tiled_copy = copy_utils.tiled_copy_2d(
            self.dtype, threads_per_row, num_threads, num_copy_elems=vecsize
        )
        return tiled_copy, tiler_mn, threads_per_row

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor, #[M, N]
        mDEntropy: cute.Tensor, #[M, N]
        mXMax: cute.Tensor, #[M]
        mXSumExp: cute.Tensor, #[M]
        mXSumSoftmaxTimes: cute.Tensor, #[M]
        stream: cuda.CUstream,
    ):
        """
        Launch the backward kernel to compute gradients.

        Args:
            mX: Input logits tensor of shape (M, N)
            mDEntropy: Gradient of entropy loss of shape (M,)
            mXMax: Max values from forward pass of shape (M,)
            mXSumExp: Sum of exponentials from forward pass of shape (M,)
            mXSumSoftmaxTimes: Sum(softmax * x) from forward pass of shape (M,)
            stream: CUDA stream for kernel execution
        """
        assert mX.element_type == self.dtype
        # Calculate vector size for memory loading (max 128 bits per load)
        # e.g. if self.N isn't divisible by 8 for bf16, we might use 64 bits (4 elements) copy
        vecsize = math.gcd(self.N, 128 // self.dtype.width)
        tiled_copy, tiler_mn, threads_per_row = self._get_tiled_copy(vecsize=vecsize)
        num_threads = tiled_copy.size
        mDEntropy, mXMax, mXSumExp, mXSumSoftmaxTimes= [
            layout_utils.expand(X, dim=1, size=self.N) for X in (mDEntropy, mXMax, mXSumExp, mXSumSoftmaxTimes)
        ]
        # Launch kernel with appropriate grid and block configuration
        self.kernel(
            mX,
            mDEntropy,
            mXMax,
            mXSumExp,
            mXSumSoftmaxTimes,
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
        mX: cute.Tensor, #[M, N]
        mDEntropy: cute.Tensor, #[M, N]
        mXMax: cute.Tensor, #[M]
        mXSumExp: cute.Tensor, #[M]
        mXSumSoftmaxTimes: cute.Tensor, #[M]
        shape: cute.Shape,
        tiler_mn: cute.Shape,
        tiled_copy: cute.TiledCopy,
        threads_per_row: cutlass.Constexpr[int],
    ):
        """
        CUDA kernel for entropy backward pass.

        This kernel computes the gradient of entropy with respect to the input logits:
            dx = -d_entropy * softmax(x) * (x - E[x])

        Where E[x] is the expected value under the softmax distribution.
        """
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()

        # Allocate shared memory for input data
        smem = cutlass.utils.SmemAllocator()
        sX = smem.allocate_tensor(
            mX.element_type, cute.make_ordered_layout(tiler_mn, order=(1, 0)), byte_alignment=16
        )

        idX = cute.make_identity_tensor(shape)
        gX, cX = [cute.local_tile(mT, tiler_mn, (bidx, bidy)) for mT in (mX, idX)]

        thr_copy = tiled_copy.get_slice(tidx)

        # Partition tensors for this thread
        tXgX = thr_copy.partition_S(gX)  # Global memory partition
        tXsX = thr_copy.partition_D(sX)  # Shared memory partition
        tXcX = thr_copy.partition_S(cX)[(0, None), None, None]  # Coordinate partition
        tXrX = cute.make_fragment_like(tXgX)  # Register fragment

        # Handle non-divisible N by creating predicates
        is_even_N = const_expr(shape[1] % tiler_mn[1] == 0)
        tXpX = (
            None if is_even_N else copy_utils.predicate_k(thr_copy.partition_S(cX), limit=shape[1])
        )
        copy = partial(copy_utils.copy, pred=tXpX)

        row = tXcX[0][0]

        # Load input data asynchronously from global to shared memory
        if row < shape[0]:
            copy(tXgX, tXsX, is_async=True)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        if const_expr(not is_even_N):
            utils.fill_oob(tXsX, tXpX, -tXsX.element_type.inf)
        # Copy from shared memory to registers and convert to Float32
        cute.autovec_copy(tXsX, tXrX)
        x = tXrX.load().to(Float32)

        # Load intermediate results from forward pass
        d_entropy = Float32.zero
        x_max = Float32.zero
        x_sum_exp = Float32.zero
        x_sum_softmax_times = Float32.zero
        if row < shape[0]:
            d_entropy = mDEntropy[row]
            x_max = mXMax[row]
            x_sum_exp = mXSumExp[row]
            x_sum_softmax_times = mXSumSoftmaxTimes[row]

        # Compute softmax for gradient calculation
        log2_e = math.log2(math.e)
        exp_x = cute.math.exp2(x * log2_e - (x_max * log2_e), fastmath=False)
        softmax_x = exp_x * cute.arch.rcp_approx(x_sum_exp)

        # Compute gradient: dx = -d_entropy * softmax(x) * (x - E[x])
        x = -(x - x_sum_softmax_times) * softmax_x * d_entropy

        # Store gradient to registers and write back to global memory
        tXrX.store(x.to(tXrX.element_type))
        if row < shape[0]:
            copy(tXrX, tXgX)


def _entropy_backward(
    x: torch.Tensor,
    d_entropy: torch.Tensor,
    x_max: torch.Tensor,
    x_sum_exp: torch.Tensor,
    x_sum_softmax_times: torch.Tensor,
) -> None:
    """
    Entropy backward pass using fused kernel.

    This function computes the gradient of entropy with respect to the input logits.
    The gradient formula is:
        dx = -d_entropy * softmax(x) * (x - E[x])

    Where E[x] is the expected value under the softmax distribution.

    Args:
        x: Input logits tensor of shape (M, N), where M is batch size and N is vocabulary size
        d_entropy: Gradient of entropy loss of shape (M,)
        x_max: Max values from forward pass of shape (M,)
        x_sum_exp: Sum of exponentials from forward pass of shape (M,)
        x_sum_softmax_times: Sum(softmax * x) from forward pass of shape (M,)

    Returns:
        None (gradient is computed in-place on the input tensor x)

    Raises:
        AssertionError: If input validation fails (wrong dimensions, device, or dtype)
    """
    d_entropy = d_entropy.contiguous()
    # Input validation
    assert x.dim() == 2, "Input must be 2D"
    assert d_entropy.dim() == 1, "d_entropy must be 1D"
    assert x_max.dim() == 1, "x_max must be 1D"
    assert x_sum_exp.dim() == 1, "x_sum_exp must be 1D"
    assert x_sum_softmax_times.dim() == 1, "x_sum_softmax_timesmust be 1D"
    assert x.shape[0] == d_entropy .shape[0], "Batch dimensions must match"
    assert x.shape[0] == x_max.shape[0], "Batch dimensions must match"
    assert x.shape[0] == x_sum_exp.shape[0], "Batch dimensions must match"
    assert x.shape[0] == x_sum_softmax_times.shape[0], "Batch dimensions must match"
    assert x.is_cuda and d_entropy.is_cuda and x_max.is_cuda and x_sum_exp.is_cuda and x_sum_softmax_times.is_cuda, (
        "Tensors must be on CUDA device"
    )
    assert x.dtype in [torch.float16, torch.bfloat16, torch.float32], "Unsupported input dtype"
    assert d_entropy.dtype == torch.float32, "d_entropy must be float32"
    assert x_max.dtype == torch.float32, "max_x must be float32"
    assert x_sum_exp.dtype == torch.float32, "x_sum_exp must be float32"
    assert x_sum_softmax_times.dtype == torch.float32, "x_sum_softmax_times must be float32"

    N = x.size(1)
    dtype = torch2cute_dtype_map[x.dtype]
    calc_dtype = Float32
    # Create compile key for caching compiled kernels
    compile_key = (dtype, calc_dtype, N)
    if compile_key not in _entropy_backward.compile_cache:
        batch_sym = cute.sym_int()
        div = math.gcd(128 // dtype.width, N)
        x_cute= fake_tensor(dtype, (batch_sym, N), div)
        d_entropy_cute, x_max_cute, x_sum_exp_cute, x_sum_softmax_times_cute = [fake_tensor(calc_dtype, (batch_sym,))] * 4
        fused_entropy_backward_op = FusedEntropyBackward(dtype, N)
        # Compile backward kernel
        _entropy_backward.compile_cache[compile_key] = cute.compile(
            fused_entropy_backward_op,
            x_cute, d_entropy_cute,
            x_max_cute, x_sum_exp_cute, x_sum_softmax_times_cute,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    # Execute backward kernel
    _entropy_backward.compile_cache[compile_key](
        x, d_entropy, x_max, x_sum_exp, x_sum_softmax_times,
    )


_entropy_backward.compile_cache = {}


# @torch.library.custom_op("quack::entropy_bwd_out", mutates_args={"x", "d_entropy", "x_max", "x_sum_exp", "x_sum_softmax_times"})
# def entropy_bwd_out(
    # x: torch.Tensor,
    # d_entropy: torch.Tensor,
    # x_max: torch.Tensor,
    # x_sum_exp: torch.Tensor,
    # x_sum_softmax_times: torch.Tensor,
# ) -> None:
    # """
    # Fused entropy backward pass with in-place gradient computation.

    # This function computes the gradient of entropy with respect to the input logits
    # and stores it in the input tensor x.

    # Args:
        # x: Input logits tensor of shape (M, N) [mutated in-place with gradient]
        # d_entropy: Gradient of entropy loss of shape (M,)
        # x_max: Max values from forward pass of shape (M,)
        # x_sum_exp: Sum of exponentials from forward pass of shape (M,)
        # x_sum_softmax_times: Sum(softmax * x) from forward pass of shape (M,)

    # Returns:
        # None (gradient is computed in-place on x)
    # """
    # _entropy_backward(x, d_entropy, x_max, x_sum_exp, x_sum_softmax_times)


def entropy_bwd(
    x: torch.Tensor,
    d_entropy: torch.Tensor,
    x_max: torch.Tensor,
    x_sum_exp: torch.Tensor,
    x_sum_softmax_times: torch.Tensor,
) :
    """
    Compute gradient of entropy with respect to input logits.

    This function computes the gradient and returns the modified input tensor.

    Args:
        x: Input logits tensor of shape (M, N)
        d_entropy: Gradient of entropy loss of shape (M,)
        x_max: Max values from forward pass of shape (M,)
        x_sum_exp: Sum of exponentials from forward pass of shape (M,)
        x_sum_softmax_times: Sum(softmax * x) from forward pass of shape (M,)

    Returns:
        torch.Tensor: The input tensor x with gradient computed in-place
    """
    _entropy_backward(x, d_entropy, x_max, x_sum_exp, x_sum_softmax_times)
    return x


def test_entropy_bwd(M, N):
    """
    Test function for entropy backward pass.
    """
    x = torch.rand((M, N), dtype=torch.bfloat16).cuda()
    d_entropy  = torch.rand((M), dtype=torch.float32).cuda()
    x_max = torch.rand((M), dtype=torch.float32).cuda()
    x_sum_exp = torch.rand((M), dtype=torch.float32).cuda()
    x_sum_softmax_times = torch.rand((M), dtype=torch.float32).cuda()
    entropy_bwd(x, d_entropy, x_max, x_sum_exp, x_sum_softmax_times)


class EntropyFunction(torch.autograd.Function):
    """
    PyTorch autograd function for entropy computation with fused kernels.
    """

    @staticmethod
    def forward(ctx, x):
        entropy, x_max, x_sum_exp, x_sum_softmax_times = entropy_fwd(
            x, target, ignore_index=ignore_index, return_lse=True
        )

        ctx.save_for_backward(x, x_max, x_sum_exp, x_sum_softmax_times)
        return entropy

    @staticmethod
    def backward(ctx, d_entropy):
        x, x_max, x_sum_exp, x_sum_softmax_times = ctx.saved_tensors
        dx = entropy_bwd(
            x, d_entropy, x_max, x_sum_exp, x_sum_softmax_times
        )
        return dx


if __name__ == '__main__':
    test_entropy_fwd(8192, 128000)
    test_entropy_bwd(8192, 128000)
