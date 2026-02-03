import logging
import sys
from typing import Any, Mapping

import torch
import torch.distributed as dist
from transformers.trainer_pt_utils import atleast_1d


def get_logger(name: str) -> logging.Logger:
    r"""
    Gets a standard logger with a stream hander to stdout.
    """
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    return logger


def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)


def distributed_concat(tensor: Any, group=None) -> Any:
    if isinstance(tensor, (tuple, list)):
        return type(tensor)(distributed_concat(t, group) for t in tensor)
    if isinstance(tensor, Mapping):
        return type(tensor)({k: distributed_concat(t, group) for k, t in sorted(tensor.items())})
    tensor = atleast_1d(tensor).contiguous()
    output_tensors = [tensor.clone() for _ in range(dist.get_world_size(group=group))]
    dist.all_gather(output_tensors, tensor, group=group)
    concat = torch.cat(output_tensors, dim=0)

    return concat


def distributed_reduce(tensor: Any, group=None, op=dist.ReduceOp.SUM):
    if isinstance(tensor, (tuple, list)):
        return type(tensor)(distributed_reduce(t, group, op) for t in tensor)
    if isinstance(tensor, Mapping):
        return type(tensor)({k: distributed_reduce(t, group, op) for k, t in sorted(tensor.items())})
    tensor = atleast_1d(tensor).contiguous()
    dist.all_reduce(tensor, group=group, op=op)
    return tensor


def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, "{} is not divisible by {}".format(numerator, denominator)


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


def is_megatron_llama():
    """
    Check if the installed package is megatron-llama-core rather than megatron-core.
    Use cached_value to avoid re-checking the package.
    """
    if not hasattr(is_megatron_llama, "cached_value"):
        from importlib.metadata import distributions
        is_megatron_llama.cached_value = any(
            dist.metadata.get('Name') == 'megatron-llama-core' for dist in distributions()
        )
    return is_megatron_llama.cached_value
