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


class _VocabParallelEntropy(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        vocab_parallel_logits: torch.Tensor,
        used_fp32: bool = True,
        use_fused_kernel: bool = True
    ) -> torch.Tensor:

        # Only use fused kernel when TP=1 and use_fused_kernel is True
        if use_fused_kernel:
            # Get tensor parallel world size
            tp_world_size = mpu.get_tensor_model_parallel_world_size()
            if tp_world_size != 1 or not FUSED_KERNEL_AVAILABLE:
                logger.warning(f"Disable use_fused_kernel because {tp_world_size=} and {FUSED_KERNEL_AVAILABLE=}.")
                use_fused_kernel = False

        if use_fused_kernel:
            vocab_parallel_logits_2d = vocab_parallel_logits.view(-1, vocab_parallel_logits.shape[-1])

            # Use fused kernel implementation (only for TP=1)
            entropy, x_max, x_sum_exp, x_sum_softmax_times = entropy_fwd(vocab_parallel_logits_2d)

            # Convert output back to original shape
            if vocab_parallel_logits.dim() == 3:
                batch_size, seq_len, vocab_size = vocab_parallel_logits.shape
                entropy = entropy.view(batch_size, seq_len)

            # Save for backward: vocab_parallel_logits and intermediate results
            ctx.save_for_backward(vocab_parallel_logits, x_max, x_sum_exp, x_sum_softmax_times)
            ctx.use_fused_kernel = True
            # ctx.original_shape = original_shape
            return entropy
        else:
            # Original implementation for TP>1 or when fused kernel is not available
            @torch.compile(dynamic=True)
            def mul_reduce(a, b):
                return (a * b).sum(dim=-1, keepdim=True)

            ctx.input_dtype = vocab_parallel_logits.dtype

            if used_fp32:
                vocab_parallel_logits = vocab_parallel_logits.float()

            logits_max = vocab_parallel_logits.max(dim=-1, keepdim=True).values
            dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=mpu.get_tensor_model_parallel_group())
            normalized_vocab_parallel_logits = vocab_parallel_logits - logits_max
            normalized_exp_logits = normalized_vocab_parallel_logits.exp_()
            normalized_sum_exp_logits = normalized_exp_logits.sum(dim=-1, keepdim=True)
            dist.all_reduce(normalized_sum_exp_logits, group=mpu.get_tensor_model_parallel_group())
            softmax_logits = normalized_exp_logits.div_(normalized_sum_exp_logits)
            sum_softmax_times_logits = mul_reduce(softmax_logits, vocab_parallel_logits)
            dist.all_reduce(sum_softmax_times_logits, group=mpu.get_tensor_model_parallel_group())
            entropy = logits_max + normalized_sum_exp_logits.log() - sum_softmax_times_logits
            ctx.save_for_backward(vocab_parallel_logits, softmax_logits, sum_softmax_times_logits)
            ctx.use_fused_kernel = False

            return entropy.squeeze(dim=-1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
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

            grad_input = grad_input_2d.view_as(vocab_parallel_logits)

            return grad_input, None, None
        else:
            # Original implementation for TP>1
            vocab_parallel_logits, softmax_logits, sum_softmax_times_logits = ctx.saved_tensors
            # reuse softmax_logits as grad
            vocab_parallel_logits.sub_(sum_softmax_times_logits)
            softmax_logits.mul_(vocab_parallel_logits)
            softmax_logits.mul_(grad_output.unsqueeze(dim=-1))
            # recover vocab_parallel_logits
            vocab_parallel_logits.add_(sum_softmax_times_logits)
            softmax_logits.mul_(-1)
            softmax_logits = softmax_logits.to(ctx.input_dtype)

            return softmax_logits, None, None

def vocab_parallel_entropy(vocab_parallel_logits: torch.Tensor, used_fp32=True, use_fused_kernel: bool = True) -> torch.Tensor:
    """
    ref: https://github.com/volcengine/verl/blob/78532923368aeb058f62201489546d013df47710/verl/utils/megatron/tensor_parallel.py#L109
    Compute entropy when the logits are sharded in tp ranks

    Args:
        vocab_parallel_logits: (total_nnz, vocab_size // tp_size)
        use_fused_kernel: whether to use fused kernel implementation (default: True)

    Returns: (total_nnz,)

    """
    return _VocabParallelEntropy.apply(vocab_parallel_logits, used_fp32, use_fused_kernel)
