"""Utilities for calculating data parallel sizes for Megatron strategies."""

from typing import Dict, List, Optional

from roll.utils.logging import get_logger

logger = get_logger()


def calculate_megatron_dp_size(
    num_gpus: int,
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    context_parallel_size: int = 1,
) -> int:
    """
    Calculate the data parallel size for Megatron.
    
    Megatron uses: DP = Total_GPUs / (TP × PP × CP × EP)
    
    Args:
        num_gpus: Total number of GPUs
        tensor_parallel_size: Tensor parallel size (TP)
        pipeline_parallel_size: Pipeline parallel size (PP)
        context_parallel_size: Context parallel size (CP)

    Returns:
        The calculated data parallel size
        
    Raises:
        ValueError: If num_gpus is not divisible by model parallel size
    """
    model_parallel_size = (
        tensor_parallel_size * 
        pipeline_parallel_size * 
        context_parallel_size
    )
    
    if num_gpus % model_parallel_size != 0:
        raise ValueError(
            f"Total GPUs ({num_gpus}) must be divisible by model parallel size "
            f"({model_parallel_size} = TP:{tensor_parallel_size} × PP:{pipeline_parallel_size} × "
            f"CP:{context_parallel_size})"
        )
    
    dp_size = num_gpus // model_parallel_size
    
    logger.debug(
        f"Megatron DP calculation: {num_gpus} GPUs / "
        f"(TP:{tensor_parallel_size} × PP:{pipeline_parallel_size} × "
        f"CP:{context_parallel_size}) = DP:{dp_size}"
    )
    
    return dp_size


def validate_megatron_batch_size(
    batch_size: int,
    num_gpus: int,
    strategy_config: Dict,
) -> None:
    """
    Validate that a batch size is divisible by the data parallel size for Megatron.
    
    Args:
        batch_size: The batch size to validate
        num_gpus: Total number of GPUs
        strategy_config: Megatron strategy configuration dict
        
    Raises:
        ValueError: If batch_size is not divisible by DP size when DP > 1
    """
    # Extract parallelism dimensions from config
    tp = strategy_config.get('tensor_model_parallel_size', 1)
    pp = strategy_config.get('pipeline_model_parallel_size', 1)
    cp = strategy_config.get('context_parallel_size', 1)

    # Calculate DP size
    dp_size = calculate_megatron_dp_size(
        num_gpus=num_gpus,
        tensor_parallel_size=tp,
        pipeline_parallel_size=pp,
        context_parallel_size=cp,
    )
    
    # Validate divisibility
    if dp_size > 1 and batch_size % dp_size != 0:
        detail = (
            f"  Total GPUs: {num_gpus}\n"
            f"  Model Parallelism: TP={tp} × PP={pp} × CP={cp} = {tp*pp*cp}\n"
            f"  Data Parallel Size: {dp_size}\n"
            f"  Rollout Batch Size: {batch_size}"
        )
        
        raise ValueError(
            f"Rollout Batch Size ({batch_size}) must be divisible by data parallel size ({dp_size}) "
            f"to ensure equal distribution across DP workers.\n{detail}"
        )
    
    logger.info(
        f"Megatron DP validation passed: Rollout Batch Size={batch_size}, DP size={dp_size}"
    )