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


class FusedLogProbsForward(ReductionBase):
    """
    The fused logprobs forward pass.

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
        mTarget: cute.Tensor,  # (M) in
        mLogProbs: cute.Tensor,  # (M,) out
        mXMax: cute.Tensor,  # (M,) out
        mXSumExp: cute.Tensor,  # (M,) out
        stream: cuda.CUstream,
    ):
        """
        Launch the kernel to compute logprobs, max and sum_exp for each row.
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
            mTarget,
            mLogProbs,
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
        mTarget: cute.Tensor,  # (M, N) in
        mLogProbs: cute.Tensor,  # (M,) out
        mXMax: cute.Tensor,  # (M,) out
        mXSumExp: cute.Tensor,  # (M,) out
        tiler_mn: cute.Shape,
        tiled_copy: cute.TiledCopy,
        threads_per_row: cutlass.Constexpr[int],
    ):
        """
        CUDA kernel: Compute logprobs, max and sum_exp for each row.

        This kernel loads input data, performs reduction operations to compute
        the maximum value and sum of exponentials for each row, and writes
        the results to global memory.

        Args:
            mX: Input logits tensor of shape (M, N)
            mTarget: Input targe tensor of shape (M, )
            mLogProbs: Outpu tensor logprobs tensor of shape (M, )
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
            target = Int32(mTarget[row])
            mXMax[row] = max_x
            mXSumExp[row] = denom
            mLogProbs[row] = Float32(mX[row, target]) - max_x - cute.math.log(denom, fastmath=True)


@torch.library.custom_op("roll_kernel::logprobs_fwd_out", mutates_args={"logprobs", "x_max", "x_sum_exp"})
def logprobs_fwd_out(
    x: Tensor,
    target: Tensor,
    logprobs: Tensor,
    x_max: Tensor,
    x_sum_exp: Tensor,
) -> None:
    """
    Fused log_probs forward pass with in-place output tensors.

    Args:
        x: Input logits tensor of shape (M, N), where M is batch size and N is vocabulary size
        logprobs: Output tensor of shape (M,) [mutated in-place]
        x_max: Output tensor for max values of shape (M,) [mutated in-place]
        x_sum_exp: Output tensor for sum of exponentials of shape (M,) [mutated in-place]

    Returns:
        None (output tensors are mutated in-place)

    Raises:
        AssertionError: If input validation fails (wrong dimensions, device, or dtype)
    """
    assert x.dim() == 2, "Input must be 2D"
    assert target.dim() == 1, "Target must be 1D"
    assert x.is_cuda and target.is_cuda, "Tensors must be on CUDA device"
    assert x.dtype in [torch.float16, torch.bfloat16, torch.float32], "Unsupported input dtype"
    # assert target.dtype == torch.int32, "Unsupported input dtype"
    assert logprobs.dtype == torch.float32, "Unsupported input dtype"

    N = x.size(1)

    dtype = torch2cute_dtype_map[x.dtype]
    target_dtype = torch2cute_dtype_map[target.dtype]
    calc_dtype = torch2cute_dtype_map[x_max.dtype]
    # Create compile key for caching compiled kernels
    compile_key = (dtype, target_dtype, calc_dtype, N)
    # Compile kernels if not in cache
    if compile_key not in logprobs_fwd_out.compile_cache:
        # Create symbolic batch size for compilation
        batch_sym = cute.sym_int()
        div = math.gcd(128 // dtype.width, N)
        # Create fake tensors for compilation
        x_cute = fake_tensor(dtype, (batch_sym, N), div)
        target_cute = fake_tensor(target_dtype, (batch_sym,))
        logprobs_cute = fake_tensor(calc_dtype, (batch_sym,))
        x_max_cute = fake_tensor(calc_dtype, (batch_sym,))
        x_sum_exp_cute = fake_tensor(calc_dtype, (batch_sym,))

        # Initialize Stage 1 and Stage 2 operators with online softmax enabled
        # Online softmax computes both max and sum in a single pass for better performance
        online_softmax = True if N > 16384 else False
        logprobs_op = FusedLogProbsForward(dtype, N, online_softmax=online_softmax)

        # Compile Stage 1 kernel (computes max_x and sum_exp)
        logprobs_fwd_out.compile_cache[compile_key] = cute.compile(
            logprobs_op,
            x_cute,
            target_cute,
            logprobs_cute,
            x_max_cute,
            x_sum_exp_cute,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )

    logprobs_fwd_out.compile_cache[compile_key](
        x, target, logprobs, x_max, x_sum_exp
    )


# Compilation cache for Stage 1 and Stage 2 kernels
logprobs_fwd_out.compile_cache = {}


def logprobs_fwd(
    x: torch.Tensor,
    target: torch.Tensor,
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
    logprobs = torch.zeros(M, dtype=calc_type, device=device)
    x_max = torch.zeros(M, dtype=calc_type, device=device)
    x_sum_exp = torch.zeros(M, dtype=calc_type, device=device)

    # Compute entropy using fused kernel
    logprobs_fwd_out(x, target, logprobs, x_max, x_sum_exp)
    return logprobs, x_max, x_sum_exp


def test_logprobs_fwd(M, N):
    x = torch.rand((M, N), dtype=torch.bfloat16).cuda()
    target = torch.randint(low=0, high=N, size=(M,)).cuda()
    logprobs_fwd(x, target)


class FusedLogProbsBackward:
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
        mTarget: cute.Tensor, #[M,]
        mDLogProbs: cute.Tensor, #[M,]
        mXMax: cute.Tensor, #[M]
        mXSumExp: cute.Tensor, #[M]
        stream: cuda.CUstream,
    ):
        """
        Launch the backward kernel to compute gradients.

        Args:
            mX: Input logits tensor of shape (M, N)
            mDLogProbs: Gradient of logprobs of shape (M,)
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
        mDLogProbs, mXMax, mXSumExp = [
            layout_utils.expand(X, dim=1, size=self.N) for X in (mDLogProbs, mXMax, mXSumExp)
        ]
        # Launch kernel with appropriate grid and block configuration
        self.kernel(
            mX,
            mTarget,
            mDLogProbs,
            mXMax,
            mXSumExp,
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
        mTarget: cute.Tensor, #[M,]
        mDLogProbs: cute.Tensor, #[M,]
        mXMax: cute.Tensor, #[M]
        mXSumExp: cute.Tensor, #[M]
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
        tXcFull = thr_copy.partition_S(cX)
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
        target = Int32.zero
        d_log_prob = Float32.zero
        x_max = Float32.zero
        x_sum_exp = Float32.zero
        if row < shape[0]:
            target = Int32(mTarget[row])
            d_log_prob = mDLogProbs[row]
            x_max = mXMax[row]
            x_sum_exp = mXSumExp[row]

        # Compute softmax for gradient calculation
        log2_e = math.log2(math.e)
        exp_x = cute.math.exp2(x * log2_e - (x_max * log2_e), fastmath=False)

        # Compute gradient: dx = d_logprob * (softmax(x) - one_hot(target))
        probs = -exp_x / x_sum_exp  # softmax(x)
        prob_shifted = probs + 1.0
        mask = cute.make_fragment_like(tXrX, Boolean)
        for i in cutlass.range(cute.size(tXcFull), unroll_full=True):
            mask[i] = tXcFull[i][1] == target
        grad = cute.where(mask.load(), prob_shifted, probs)
        grad = grad * d_log_prob

        # Store gradient to registers and write back to global memory
        tXrX.store(grad.to(tXrX.element_type))
        if row < shape[0]:
            copy(tXrX, tXgX)


def _logprobs_backward(
    x: torch.Tensor,
    target: torch.Tensor,
    d_logprobs: torch.Tensor,
    x_max: torch.Tensor,
    x_sum_exp: torch.Tensor,
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
    d_logprobs = d_logprobs.contiguous()
    # Input validation
    assert x.dim() == 2, "Input must be 2D"
    assert d_logprobs.dim() == 1, "d_logprobsmust be 1D"
    assert x_max.dim() == 1, "x_max must be 1D"
    assert x_sum_exp.dim() == 1, "x_sum_exp must be 1D"
    assert x.shape[0] == d_logprobs.shape[0], "Batch dimensions must match"
    assert x.shape[0] == x_max.shape[0], "Batch dimensions must match"
    assert x.shape[0] == x_sum_exp.shape[0], "Batch dimensions must match"
    assert x.is_cuda and target.is_cuda and d_logprobs.is_cuda and x_max.is_cuda and x_sum_exp.is_cuda, (
        "Tensors must be on CUDA device"
    )
    assert x.dtype in [torch.float16, torch.bfloat16, torch.float32], "Unsupported input dtype"
    assert d_logprobs.dtype == torch.float32, "d_entropy must be float32"
    assert x_max.dtype == torch.float32, "max_x must be float32"
    assert x_sum_exp.dtype == torch.float32, "x_sum_exp must be float32"

    N = x.size(1)
    dtype = torch2cute_dtype_map[x.dtype]
    target_dtype = torch2cute_dtype_map[target.dtype]
    calc_dtype = Float32
    # Create compile key for caching compiled kernels
    compile_key = (dtype, target_dtype, calc_dtype, N)
    if compile_key not in _logprobs_backward.compile_cache:
        batch_sym = cute.sym_int()
        div = math.gcd(128 // dtype.width, N)
        x_cute= fake_tensor(dtype, (batch_sym, N), div)
        target_cute = fake_tensor(target_dtype, (batch_sym,))
        d_logprobs_cute, x_max_cute, x_sum_exp_cute = [fake_tensor(calc_dtype, (batch_sym,))] * 3
        fused_logprobs_backward_op = FusedLogProbsBackward(dtype, N)
        # Compile backward kernel
        _logprobs_backward.compile_cache[compile_key] = cute.compile(
            fused_logprobs_backward_op,
            x_cute, target_cute, d_logprobs_cute,
            x_max_cute, x_sum_exp_cute,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    # Execute backward kernel
    _logprobs_backward.compile_cache[compile_key](
        x, target, d_logprobs, x_max, x_sum_exp
    )


_logprobs_backward.compile_cache = {}


def logprobs_bwd(
    x: torch.Tensor,
    target: torch.Tensor,
    d_logprobs: torch.Tensor,
    x_max: torch.Tensor,
    x_sum_exp: torch.Tensor,
) :
    """
    Compute gradient of entropy with respect to input logits.

    This function computes the gradient and returns the modified input tensor.

    Args:
        x: Input logits tensor of shape (M, N)
        d_logprobs: Gradient of logprobs of shape (M,)
        x_max: Max values from forward pass of shape (M,)
        x_sum_exp: Sum of exponentials from forward pass of shape (M,)

    Returns:
        torch.Tensor: The input tensor x with gradient computed in-place
    """
    _logprobs_backward(x, target, d_logprobs, x_max, x_sum_exp)
    return x


def test_logprobs_bwd(M, N):
    """
    Test function for entropy backward pass.
    """
    x = torch.rand((M, N), dtype=torch.bfloat16).cuda()
    target = torch.randint(low=0, high=N, size=(M,)).cuda()
    d_logprobs = torch.rand((M), dtype=torch.float32).cuda()
    x_max = torch.rand((M), dtype=torch.float32).cuda()
    x_sum_exp = torch.rand((M), dtype=torch.float32).cuda()
    logprobs_bwd(x, target, d_logprobs, x_max, x_sum_exp)


if __name__ == '__main__':
    # test_logprobs_fwd(8192, 128000)
    test_logprobs_bwd(8192, 128000)
