from functools import lru_cache

import torch
import torch.distributed as dist
from megatron.core import parallel_state as mpu

from roll.utils.logging import get_logger

try:
    from .fused_entropy import entropy_fwd, entropy_bwd

    FUSED_KERNEL_AVAILABLE = True
except ImportError:
    FUSED_KERNEL_AVAILABLE = False

logger = get_logger()


@torch.compile(dynamic=True)
def _mul_reduce(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a * b).sum(dim=-1, keepdim=True)


@lru_cache(maxsize=None)
def _warn_fused_kernel_disabled(tp_world_size: int, fused_kernel_available: bool) -> None:
    logger.warning(
        "Disabling fused entropy kernel because "
        f"{tp_world_size=} and {fused_kernel_available=}."
    )


class _VocabParallelEntropy(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        vocab_parallel_logits: torch.Tensor,
        used_fp32: bool = True,
        use_fused_kernel: bool = True,
    ) -> torch.Tensor:
        tp_world_size = mpu.get_tensor_model_parallel_world_size()

        if use_fused_kernel and (tp_world_size != 1 or not FUSED_KERNEL_AVAILABLE):
            _warn_fused_kernel_disabled(tp_world_size, FUSED_KERNEL_AVAILABLE)
            use_fused_kernel = False

        if use_fused_kernel:
            vocab_parallel_logits_2d = vocab_parallel_logits.view(-1, vocab_parallel_logits.shape[-1])

            # Use fused kernel implementation (only for TP=1)
            entropy, x_max, x_sum_exp, x_sum_softmax_times = entropy_fwd(vocab_parallel_logits_2d)

            # Convert output back to original shape
            if vocab_parallel_logits.dim() == 3:
                entropy = entropy.view(vocab_parallel_logits.shape[:-1])

            # Save for backward: vocab_parallel_logits and intermediate results
            ctx.save_for_backward(vocab_parallel_logits, x_max, x_sum_exp, x_sum_softmax_times)
            ctx.use_fused_kernel = True
            return entropy

        ctx.input_dtype = vocab_parallel_logits.dtype

        if used_fp32:
            vocab_parallel_logits = vocab_parallel_logits.float()

        logits_max = vocab_parallel_logits.max(dim=-1, keepdim=True).values
        if tp_world_size > 1:
            dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=mpu.get_tensor_model_parallel_group())
        normalized_vocab_parallel_logits = vocab_parallel_logits - logits_max
        normalized_exp_logits = normalized_vocab_parallel_logits.exp_()
        normalized_sum_exp_logits = normalized_exp_logits.sum(dim=-1, keepdim=True)
        if tp_world_size > 1:
            dist.all_reduce(normalized_sum_exp_logits, group=mpu.get_tensor_model_parallel_group())
        softmax_logits = normalized_exp_logits.div_(normalized_sum_exp_logits)
        sum_softmax_times_logits = _mul_reduce(softmax_logits, vocab_parallel_logits)
        if tp_world_size > 1:
            dist.all_reduce(sum_softmax_times_logits, group=mpu.get_tensor_model_parallel_group())
        entropy = logits_max + normalized_sum_exp_logits.log() - sum_softmax_times_logits
        ctx.save_for_backward(vocab_parallel_logits, softmax_logits, sum_softmax_times_logits)
        ctx.use_fused_kernel = False

        return entropy.squeeze(dim=-1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        if ctx.use_fused_kernel:
            # Use fused kernel backward with recomputation (only for TP=1)
            vocab_parallel_logits, x_max, x_sum_exp, x_sum_softmax_times = ctx.saved_tensors

            vocab_parallel_logits_2d = vocab_parallel_logits.view(-1, vocab_parallel_logits.shape[-1])

            # Call fused backward kernel (performs recomputation internally)
            grad_input_2d = entropy_bwd(
                vocab_parallel_logits_2d,
                grad_output.view(-1),
                x_max,
                x_sum_exp,
                x_sum_softmax_times
            )

            return grad_input_2d.view_as(vocab_parallel_logits), None, None

        vocab_parallel_logits, softmax_logits, sum_softmax_times_logits = ctx.saved_tensors
        # Reuse saved tensors to avoid allocating an additional gradient buffer.
        vocab_parallel_logits.sub_(sum_softmax_times_logits)
        softmax_logits.mul_(vocab_parallel_logits)
        softmax_logits.mul_(grad_output.unsqueeze(dim=-1))
        vocab_parallel_logits.add_(sum_softmax_times_logits)
        softmax_logits.mul_(-1)

        return softmax_logits.to(ctx.input_dtype), None, None


def vocab_parallel_entropy(
    vocab_parallel_logits: torch.Tensor,
    used_fp32: bool = True,
    use_fused_kernel: bool = True,
) -> torch.Tensor:
    """
    ref: https://github.com/volcengine/verl/blob/78532923368aeb058f62201489546d013df47710/verl/utils/megatron/tensor_parallel.py#L109
    Compute entropy when the logits are sharded in tp ranks

    Args:
        vocab_parallel_logits: (total_nnz, vocab_size // tp_size)
        use_fused_kernel: whether to use fused kernel implementation (default: True)

    Returns: (total_nnz,)

    """
    return _VocabParallelEntropy.apply(vocab_parallel_logits, used_fp32, use_fused_kernel)
