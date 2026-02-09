from typing import Sequence, Tuple

import torch
import torch.distributed as dist
from megatron.core import mpu

from ..utils import divide


class VocabUtility:
    # copy from megatron
    """Split the vocabulary into `world_size` chunks and return the first
    and last index of the vocabulary belonging to the `rank`
    partition: Note that indices in [fist, last)

    """

    @staticmethod
    def vocab_range_from_per_partition_vocab_size(
        per_partition_vocab_size: int, rank, world_size: int
    ) -> Sequence[int]:
        index_f = rank * per_partition_vocab_size
        index_l = index_f + per_partition_vocab_size
        return index_f, index_l

    @staticmethod
    def vocab_range_from_global_vocab_size(global_vocab_size: int, rank: int, world_size: int) -> Sequence[int]:
        per_partition_vocab_size = divide(global_vocab_size, world_size)
        return VocabUtility.vocab_range_from_per_partition_vocab_size(per_partition_vocab_size, rank, world_size)


class _VocabParallelHelper:
    @staticmethod
    def calculate_predicted_logits(
        vocab_parallel_logits: "torch.Tensor", target: "torch.Tensor", logits_max: "torch.Tensor"
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        "copy from megatron"
        # In-place subtraction reduces memory pressure.
        vocab_parallel_logits -= logits_max.unsqueeze(dim=-1)

        # Get the partition's vocab indices
        get_vocab_range = VocabUtility.vocab_range_from_per_partition_vocab_size
        partition_vocab_size = vocab_parallel_logits.size()[-1]
        rank = mpu.get_tensor_model_parallel_rank()
        world_size = mpu.get_tensor_model_parallel_world_size()
        vocab_start_index, vocab_end_index = get_vocab_range(partition_vocab_size, rank, world_size)

        # Create a mask of valid vocab ids (1 means it needs to be masked).
        target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
        masked_target = target.clone() - vocab_start_index
        masked_target[target_mask] = 0

        # Get predicted-logits = logits[target].
        # For Simplicity, we convert logits to a 2-D tensor with size
        # [*, partition-vocab-size] and target to a 1-D tensor of size [*].
        logits_2d = vocab_parallel_logits.view(-1, partition_vocab_size)
        masked_target_1d = masked_target.view(-1)
        arange_1d = torch.arange(start=0, end=logits_2d.size()[0], device=logits_2d.device)
        predicted_logits_1d = logits_2d[arange_1d, masked_target_1d]
        predicted_logits_1d = predicted_logits_1d.clone().contiguous()
        predicted_logits = predicted_logits_1d.view_as(target)
        predicted_logits[target_mask] = 0.0

        exp_logits = vocab_parallel_logits
        torch.exp(vocab_parallel_logits, out=exp_logits)
        sum_exp_logits = exp_logits.sum(dim=-1)

        return target_mask, masked_target_1d, predicted_logits, sum_exp_logits, exp_logits


class _VocabParallelLogProbs(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vocab_parallel_logits: "torch.Tensor", target: "torch.Tensor"):
        vocab_parallel_logits = vocab_parallel_logits.float()
        logits_max = torch.max(vocab_parallel_logits, dim=-1)[0]

        dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=mpu.get_tensor_model_parallel_group())

        (
            target_mask,
            masked_target_1d,
            predicted_logits,
            sum_exp_logits,
            exp_logits,
        ) = _VocabParallelHelper.calculate_predicted_logits(vocab_parallel_logits, target, logits_max)

        dist.all_reduce(sum_exp_logits, op=dist.ReduceOp.SUM, group=mpu.get_tensor_model_parallel_group())
        dist.all_reduce(predicted_logits, op=dist.ReduceOp.SUM, group=mpu.get_tensor_model_parallel_group())

        predicted_logprobs = predicted_logits - torch.log(sum_exp_logits)

        # Save for backward
        ctx.save_for_backward(exp_logits, target_mask, sum_exp_logits, masked_target_1d)

        return predicted_logprobs

    @staticmethod
    def backward(ctx, grad_output: "torch.Tensor"):
        exp_logits, target_mask, sum_exp_logits, masked_target_1d = ctx.saved_tensors

        exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))
        exp_logits.neg_()
        grad_input = exp_logits
        grad_2d = grad_input.view(-1, grad_input.size()[-1])
        arange_1d = torch.arange(start=0, end=grad_2d.size()[0], device=grad_input.device)
        grad_2d[arange_1d, masked_target_1d] += 1 - target_mask.view(-1).float()
        grad_input.mul_(grad_output.unsqueeze(dim=-1))

        return grad_input, None


def vocab_parallel_logprobs(vocab_parallel_logits, target) -> "torch.Tensor":
    """
    Get logprobs when logits are split across tensor parallel ranks

    Args:
        vocab_parallel_logits: logits split across tensor parallel ranks
                               dimension is [batch_size, sequence_length, vocab_size/num_parallel_ranks]

        target: correct vocab ids of dimension [batch_size, sequence_length]
    Returns:
        logprobs: logprobs of dimension [batch_size, sequence_length]

    (It's fine to change the order of sequence_length and batch_size in dimension)
    """
    return _VocabParallelLogProbs.apply(vocab_parallel_logits, target)


def vocab_parallel_target_rank(vocab_parallel_logits: "torch.Tensor", target: "torch.Tensor") -> "torch.Tensor":
    """
    Get target id's rank index when logits are split across tensor parallel ranks

    Args:
        vocab_parallel_logits: logits split across tensor parallel ranks
                           dimension is [batch_size, sequence_length, vocab_size // tensor_model_parallel_size]

        target: correct vocab ids of dimension [batch_size, sequence_length]
    Returns:
        target_rank: target id's rank id of dimension [batch_size, sequence_length]

    """
    batch_size, sequence_length, partition_vocab_size = vocab_parallel_logits.size()

    vocab_parallel_logits = vocab_parallel_logits.float()
    # Get the partition's vocab indices
    get_vocab_range = VocabUtility.vocab_range_from_per_partition_vocab_size
    rank = mpu.get_tensor_model_parallel_rank()
    world_size = mpu.get_tensor_model_parallel_world_size()
    vocab_start_index, vocab_end_index = get_vocab_range(partition_vocab_size, rank, world_size)

    # Create a mask of valid vocab ids (1 means it needs to be masked).
    target_mask = (target >= vocab_start_index) & (target < vocab_end_index)
    masked_target = target[target_mask].clone() - vocab_start_index

    # Get each rank's local target_logits
    masked_target_logits = torch.gather(vocab_parallel_logits[target_mask], dim=1, index=masked_target.unsqueeze(-1))
    target_logits = torch.zeros(
        (batch_size, sequence_length, 1), dtype=vocab_parallel_logits.dtype, device=vocab_parallel_logits.device
    )
    target_logits[target_mask] = masked_target_logits

    # All-reduce across all ranks to get the global target_logits.
    dist.all_reduce(target_logits, op=dist.ReduceOp.SUM, group=mpu.get_tensor_model_parallel_group())

    # Calculate target's ranking idx across all vocab_size for each rank.
    mask = vocab_parallel_logits > target_logits
    target_rank = torch.sum(mask, dim=-1)

    # All-reduce across all ranks to get the global target's ranking idx.
    dist.all_reduce(target_rank, op=dist.ReduceOp.SUM, group=mpu.get_tensor_model_parallel_group())
    return target_rank
