import torch
from megatron.core import mpu


def _gather_first_dim_size(local_tensor, world_size, group):
    local_size = torch.tensor(local_tensor.shape[0], device=local_tensor.device)
    all_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    torch.distributed.all_gather(all_sizes, local_size, group=group)
    return [size.item() for size in all_sizes]


def _gather_and_reorder_along_first_dim(
    local_tensor,
    split_plan,
    all_output_lengths,
    rank,
    world_size,
    group,
    all_sizes=None,
):
    if world_size == 1:
        return local_tensor

    # 1: gather all sizes
    if all_sizes is None:
        all_sizes = _gather_first_dim_size(local_tensor, world_size, group)

    # 2: gather all tensors
    max_size = max(all_sizes)
    gathered_tensors_padded = [local_tensor.new_empty((max_size, *local_tensor.shape[1:])) for _ in range(world_size)]
    gathered_tensors_padded[rank][: all_sizes[rank]] = local_tensor
    torch.distributed.all_gather(gathered_tensors_padded, gathered_tensors_padded[rank], group=group)
    gathered_tensors = [gathered_tensors_padded[i][: all_sizes[i]] for i in range(world_size)]

    # 3: reorder tensors
    reordered_items = [None] * len(all_output_lengths)
    for r, plan in enumerate(split_plan):
        current_pos = 0
        gpu_output_tensor = gathered_tensors[r]
        for _, original_index in plan:
            output_len = all_output_lengths[original_index]
            item_output = gpu_output_tensor[current_pos : current_pos + output_len]
            reordered_items[original_index] = item_output
            current_pos += output_len

    # 4: concat tensors
    full_output = torch.cat(reordered_items, dim=0)

    return full_output


class _GatherFromEncoderSequenceParallelRegion(torch.autograd.Function):
    """
    An encoder sequence parallel region gather autograd.Function for:
    1. Forward: gather and reorder tensors with different sizes.
    2. Backward: scatter the gradients back to the original GPUs.
    """

    @staticmethod
    def symbolic(graph, local_tensor, split_plan, all_output_lengths):
        rank = mpu.get_tensor_and_context_parallel_rank()
        world_size = mpu.get_tensor_and_context_parallel_world_size()
        group = mpu.get_tensor_and_context_parallel_group()
        return _gather_and_reorder_along_first_dim(
            local_tensor, split_plan, all_output_lengths, rank, world_size, group
        )

    @staticmethod
    def forward(ctx, local_tensor, split_plan, all_output_lengths):
        """
        Args:
            local_tensor: tensor on the current GPU.
            split_plan: load-balance plan for each GPU
            all_output_lengths: original tensor sizes
        """
        # --- gather all sizes ---
        rank = mpu.get_tensor_and_context_parallel_rank()
        world_size = mpu.get_tensor_and_context_parallel_world_size()
        group = mpu.get_tensor_and_context_parallel_group()
        all_sizes = _gather_first_dim_size(local_tensor, world_size, group)
        # --- save for backward ---
        ctx.split_plan = split_plan
        ctx.all_output_lengths = all_output_lengths
        ctx.rank = rank
        return _gather_and_reorder_along_first_dim(
            local_tensor,
            split_plan,
            all_output_lengths,
            rank,
            world_size,
            group,
            all_sizes,
        )

    @staticmethod
    def backward(ctx, grad_output):
        """
        Args:
            grad_output: gradient of the forward output
        """
        split_plan = ctx.split_plan
        all_output_lengths = ctx.all_output_lengths
        rank = ctx.rank

        grad_by_item = grad_output.split(all_output_lengths)
        local_grad_by_item = [grad_by_item[original_index] for _, original_index in split_plan[rank]]
        local_grad = torch.cat(local_grad_by_item, dim=0).contiguous()
        return local_grad, None, None


class _GatherFromEncoderSmallBatchSize(torch.autograd.Function):
    """Gather the input from sequence parallel region and concatinate."""

    @staticmethod
    def symbolic(graph, input_):
        return input_

    @staticmethod
    def forward(ctx, input_):
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        world_size = mpu.get_tensor_model_parallel_world_size()
        grad_output = grad_output.div_(world_size)
        return grad_output


def encoder_sequence_parallel_gather(input_, split_plan, input_size_list):
    return _GatherFromEncoderSequenceParallelRegion.apply(input_, split_plan, input_size_list)


def encoder_small_batch_size_gather(input_):
    return _GatherFromEncoderSmallBatchSize.apply(input_)
