"""Bounded-memory vocabulary projection with unreduced TP cross entropy.

Only a token chunk's logits/softmax exist at once. Backward recomputes each
chunk and returns ordinary hidden and weight gradients, so Megatron DDP's
AccumulateGrad hook runs once and owns ``main_grad`` accumulation. In particular,
this does not write ``main_grad`` and return ``None`` for the head gradient.
"""

import torch
import torch.distributed as dist
from torch.autograd.function import once_differentiable


def _probabilities_and_loss(hidden, weight, labels, tp_group, ignore_index):
    # Match the stock BF16 projection followed by FP32 vocabulary cross entropy.
    probabilities = (hidden @ weight.T).float()
    maximum = probabilities.amax(dim=-1)
    world_size = dist.get_world_size(tp_group) if tp_group is not None else 1
    rank = dist.get_rank(tp_group) if tp_group is not None else 0
    if world_size > 1:
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=tp_group)
    probabilities.sub_(maximum.unsqueeze(-1))
    local_labels = labels - rank * weight.shape[0]
    belongs = ((local_labels >= 0) & (local_labels < weight.shape[0])
               & (labels != ignore_index))
    local_labels = local_labels.masked_fill(~belongs, 0)
    rows = torch.arange(labels.numel(), device=labels.device)
    predicted = probabilities[rows, local_labels].masked_fill_(~belongs, 0)
    probabilities.exp_()
    denominator = probabilities.sum(dim=-1)
    if world_size > 1:
        dist.all_reduce(predicted, group=tp_group)
        dist.all_reduce(denominator, group=tp_group)
    loss = denominator.log() - predicted
    loss.masked_fill_(labels == ignore_index, 0)
    probabilities.div_(denominator.unsqueeze(-1))
    return probabilities, loss, belongs, local_labels


def _entropy(probabilities, tp_group):
    # Zero probabilities contribute zero entropy. Clamping before log avoids
    # 0 * -inf; subnormal probabilities contribute below FP32 rounding error.
    log_probabilities = probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()
    entropy = -(probabilities * log_probabilities).sum(dim=-1)
    if tp_group is not None and dist.get_world_size(tp_group) > 1:
        dist.all_reduce(entropy, group=tp_group)
    return entropy, log_probabilities


class _ChunkedVocabProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, labels, chunk_size, tp_group, ignore_index, statistics):
        ctx.save_for_backward(hidden, weight, labels)
        ctx.chunk_size, ctx.tp_group, ctx.ignore_index = chunk_size, tp_group, ignore_index
        ctx.statistics = statistics
        flat_hidden = hidden.reshape(-1, hidden.shape[-1])
        flat_labels = labels.reshape(-1)
        shape = (*flat_labels.shape, 2) if statistics else flat_labels.shape
        losses = torch.empty(shape, device=hidden.device, dtype=torch.float32)
        for start in range(0, flat_labels.numel(), chunk_size):
            stop = start + chunk_size
            probabilities, loss, _, _ = _probabilities_and_loss(
                flat_hidden[start:stop], weight, flat_labels[start:stop], tp_group, ignore_index)
            if statistics:
                entropy, log_probabilities = _entropy(probabilities, tp_group)
                entropy.masked_fill_(flat_labels[start:stop] == ignore_index, 0)
                losses[start:stop, 0].copy_(-loss)
                losses[start:stop, 1].copy_(entropy)
                del entropy, log_probabilities
            else:
                losses[start:stop].copy_(loss)
            del probabilities, loss
        return losses.view(*labels.shape, 2) if statistics else losses.view_as(labels)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_loss):
        hidden, weight, labels = ctx.saved_tensors
        flat_hidden = hidden.reshape(-1, hidden.shape[-1])
        flat_labels = labels.reshape(-1)
        flat_upstream = grad_loss.reshape(-1, 2) if ctx.statistics else grad_loss.reshape(-1)
        grad_hidden = torch.empty_like(flat_hidden) if ctx.needs_input_grad[0] else None
        # A single ordinary gradient buffer keeps tied-weight/DDP accumulation
        # under the autograd engine; its size does not grow with sequence length.
        grad_weight = torch.zeros_like(weight) if ctx.needs_input_grad[1] else None
        for start in range(0, flat_labels.numel(), ctx.chunk_size):
            stop = start + ctx.chunk_size
            chunk_hidden, chunk_labels = flat_hidden[start:stop], flat_labels[start:stop]
            probabilities, _, belongs, local_labels = _probabilities_and_loss(
                chunk_hidden, weight, chunk_labels, ctx.tp_group, ctx.ignore_index)
            rows = torch.arange(chunk_labels.numel(), device=labels.device)
            if ctx.statistics:
                entropy, log_probabilities = _entropy(probabilities, ctx.tp_group)
                upstream = flat_upstream[start:stop].masked_fill(
                    (chunk_labels == ctx.ignore_index).unsqueeze(-1), 0)
                # d log p_y / dz = one_hot(y) - p
                # d H / dz = -p * (log(p) + H), including the global TP entropy.
                log_probabilities.add_(entropy.unsqueeze(-1))
                log_probabilities.mul_(upstream[:, 1].unsqueeze(-1))
                log_probabilities.add_(upstream[:, 0].unsqueeze(-1))
                probabilities.mul_(log_probabilities).neg_()
                probabilities[rows, local_labels] += upstream[:, 0] * belongs
                del entropy, log_probabilities
            else:
                probabilities[rows, local_labels] -= belongs.to(probabilities.dtype)
                upstream = flat_upstream[start:stop].masked_fill(chunk_labels == ctx.ignore_index, 0)
                probabilities.mul_(upstream.unsqueeze(-1))
            grad_logits = probabilities.to(weight.dtype)
            if grad_hidden is not None:
                torch.mm(grad_logits, weight, out=grad_hidden[start:stop])
            if grad_weight is not None:
                grad_weight.addmm_(grad_logits.T, chunk_hidden)
            del probabilities, grad_logits
        return (grad_hidden.view_as(hidden) if grad_hidden is not None else None,
                grad_weight, None, None, None, None, None)


def chunked_vocab_parallel_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    *,
    chunk_size: int = 256,
    tp_group=None,
    sequence_parallel: bool = False,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Return FP32 token losses for a bias-free, vocabulary-sharded linear head.

    ``hidden`` is ``[sequence, batch, hidden]`` (or ``[tokens, hidden]``),
    ``weight`` is ``[local_vocab, hidden]``, and ``labels`` has the global token
    shape ``hidden.shape[:-1]`` after gathering the SP sequence dimension. Each
    TP rank supplies the same labels and upstream loss weights. Pass the head's
    explicit TP group; ``None`` means a local, unsharded vocabulary.

    The head is a plain weight tensor: callers must not use this path for a
    biased head or adapters that alter the output projection. A frozen head is
    supported, including LoRA in the transformer. Ignored labels have zero loss
    and gradient. This implements first-order derivatives only.
    """
    return _chunked_vocab_projection(hidden, weight, labels, chunk_size, tp_group,
                                     sequence_parallel, ignore_index, statistics=False)


def chunked_vocab_parallel_logprobs_and_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    *,
    chunk_size: int = 256,
    tp_group=None,
    sequence_parallel: bool = False,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Return FP32 ``[..., 2]`` selected-token logprobs and full-vocab entropy.

    Inputs and TP/SP contracts match ``chunked_vocab_parallel_cross_entropy``.
    Both channels support independently weighted first-order gradients without
    retaining full-sequence logits. Ignored targets have zero in both channels.
    This requires a plain bias-free head and does not apply head adapters.
    """
    return _chunked_vocab_projection(hidden, weight, labels, chunk_size, tp_group,
                                     sequence_parallel, ignore_index, statistics=True)


def _chunked_vocab_projection(hidden, weight, labels, chunk_size, tp_group,
                             sequence_parallel, ignore_index, statistics):
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if hidden.ndim < 2 or weight.ndim != 2 or hidden.shape[-1] != weight.shape[-1]:
        raise ValueError("hidden and vocabulary weight must have matching hidden dimensions")
    if hidden.dtype != weight.dtype or hidden.device != weight.device:
        raise ValueError("hidden and vocabulary weight must share dtype and device")
    if labels.dtype != torch.long or labels.device != hidden.device:
        raise ValueError("labels must be int64 on the hidden-state device")
    world_size = dist.get_world_size(tp_group) if tp_group is not None else 1
    expected_shape = list(hidden.shape[:-1])
    if sequence_parallel:
        expected_shape[0] *= world_size
    if tuple(labels.shape) != tuple(expected_shape):
        raise ValueError("labels must match the global sequence and batch dimensions")
    invalid = (labels != ignore_index) & ((labels < 0) | (labels >= weight.shape[0] * world_size))
    if bool(invalid.any()):
        raise ValueError("labels contain an index outside the global vocabulary")
    if world_size > 1:
        from megatron.core.tensor_parallel.mappings import (
            copy_to_tensor_model_parallel_region,
            gather_from_sequence_parallel_region,
        )
        if sequence_parallel:
            hidden = gather_from_sequence_parallel_region(
                hidden, tensor_parallel_output_grad=True, group=tp_group)
        else:
            hidden = copy_to_tensor_model_parallel_region(hidden, group=tp_group)
    return _ChunkedVocabProjection.apply(hidden, weight, labels, chunk_size, tp_group, ignore_index, statistics)
