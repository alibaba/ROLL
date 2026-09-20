import gc
import itertools
import inspect
import logging
from typing import Callable, Dict, List, Optional, Tuple

import torch
from megatron.core import mpu

from megatron.core.optimizer import OptimizerConfig, MegatronOptimizer, _get_param_groups_and_buffers, \
    _get_megatron_optimizer_based_on_param_groups, ChainedOptimizer
from megatron.core.transformer import MegatronModule
from megatron.core.utils import log_single_rank
from roll.third_party.megatron.optimizer_config import validate_bounded_cpu_grad_staging

logger = logging.getLogger(__name__)


def _enable_bounded_cpu_grad_staging(optimizer, hybrid_cls):
    """Enable every Hybrid below Megatron's chained/distributed wrappers."""
    if isinstance(optimizer, hybrid_cls):
        optimizer.bounded_cpu_grad_staging = True
    elif hasattr(optimizer, "chained_optimizers"):
        for child in optimizer.chained_optimizers:
            _enable_bounded_cpu_grad_staging(child, hybrid_cls)
    elif getattr(optimizer, "optimizer", None) is not None:
        _enable_bounded_cpu_grad_staging(optimizer.optimizer, hybrid_cls)


def get_megatron_optimizer(
    config: OptimizerConfig,
    model_chunks: List[MegatronModule],
    no_weight_decay_cond: Optional[Callable] = None,
    scale_lr_cond: Optional[Callable] = None,
    lr_mult: float = 1.0,
) -> MegatronOptimizer:
    """
    copy from megatron/core/optimizer/__init__.py
    to add buffer/model_chunks to optimizer


    Retrieve the Megatron optimizer for model chunks.

    We use separate optimizers for expert parameters and non-expert parameters.

    Args:
        config (OptimizerConfig): optimizer configuration object.
        model_chunks (List[MegatronModule]): model chunks to get optimizer for.
        no_weight_decay_cond (func, optional): function to determine whether a parameter
            should not perform weight decay. Defaults to None.
        scale_lr_cond (func, optional): function to determine whether a parameter
            should have a scaled learning rate. Defaults to None.
        lr_mult (float, optional): learning rate multiplier for parameters that
            satisfy scale_lr_cond. Defaults to 1.0.

    Returns:
        Instance of MegatronOptimizer.
    """

    bounded_hybrid_cls = None
    if validate_bounded_cpu_grad_staging(config):
        patch_error = (
            "bounded_cpu_grad_staging requires the Megatron dependency patch; run "
            "scripts/qwen38/patch_megatron_cpu_grad_staging.py on the task Megatron checkout"
        )
        try:
            from megatron.core.optimizer.cpu_offloading import HybridDeviceOptimizer
        except ImportError as error:
            raise RuntimeError(patch_error) from error
        if "bounded_cpu_grad_staging" not in inspect.signature(HybridDeviceOptimizer).parameters:
            raise RuntimeError(patch_error)
        bounded_hybrid_cls = HybridDeviceOptimizer

    log_single_rank(logger, logging.INFO, f'Setting up optimizer with config {config}')

    # Separate out first model chunk if overlapping param AG with optimizer step.
    if config.overlap_param_gather_with_optimizer_step:
        all_dense_model_chunks = [[model_chunks[0]], model_chunks[1:]]
        overlap_param_gather_with_optimizer_step_flags = [True, False]
    else:
        all_dense_model_chunks = [model_chunks]
        overlap_param_gather_with_optimizer_step_flags = [False]
    model_parallel_rank = torch.distributed.get_rank(mpu.get_model_parallel_group())

    if torch.distributed.get_world_size(
        mpu.get_data_parallel_group(with_context_parallel=True, partial_data_parallel=False)
    ) > torch.distributed.get_world_size(
        mpu.get_data_parallel_group(with_context_parallel=True, partial_data_parallel=True)
    ):
        distributed_optimizer_instance_id = torch.distributed.get_rank(
            mpu.get_inter_partial_data_parallel_group()
        )
    else:
        distributed_optimizer_instance_id = 0

    optimizers = []
    model_chunk_offset = 0
    kwargs = {}
    if "config_overrides" in inspect.signature(_get_param_groups_and_buffers).parameters:
        # config_overrides is required in mcore-core>=0.16
        kwargs = {"config_overrides": None}
    for dense_model_chunks, overlap_param_gather_with_optimizer_step in zip(
        all_dense_model_chunks, overlap_param_gather_with_optimizer_step_flags
    ):
        param_groups, buffers = _get_param_groups_and_buffers(
            dense_model_chunks,
            model_chunk_offset=model_chunk_offset,
            config=config,
            filter_fn=lambda g: not g['is_expert_parallel'],
            buffer_name='buffers',
            **kwargs,
        )
        for model_chunk in dense_model_chunks:
            model_chunk.overlap_param_gather_with_optimizer_step = (
                overlap_param_gather_with_optimizer_step
            )
        optimizers.append(
            _get_megatron_optimizer_based_on_param_groups(
                config,
                model_chunks=dense_model_chunks,
                param_groups=param_groups,
                per_model_buffers=buffers,
                model_parallel_group=mpu.get_model_parallel_group(),
                data_parallel_group=mpu.get_data_parallel_group(
                    with_context_parallel=True, partial_data_parallel=True
                ),
                data_parallel_group_gloo=mpu.get_data_parallel_group_gloo(
                    with_context_parallel=True, partial_data_parallel=True
                ),
                data_parallel_group_idx=model_parallel_rank,
                distributed_optimizer_instance_id=distributed_optimizer_instance_id,
            )
        )
        if not hasattr(optimizers[-1], "buffers"):
            setattr(optimizers[-1], "buffers", list(itertools.chain(*buffers.values())))
            setattr(optimizers[-1], "model_chunks", dense_model_chunks)
        model_chunk_offset += 1

    moe_param_groups, moe_buffers = _get_param_groups_and_buffers(
        model_chunks,
        model_chunk_offset=0,
        config=config,
        filter_fn=lambda g: g['is_expert_parallel'],
        buffer_name='expert_parallel_buffers',
        **kwargs,
    )
    if len(moe_param_groups) > 0:
        model_parallel_rank = torch.distributed.get_rank(
            mpu.get_expert_tensor_model_pipeline_parallel_group()
        )
        optimizers.append(
            _get_megatron_optimizer_based_on_param_groups(
                config,
                model_chunks=model_chunks,
                param_groups=moe_param_groups,
                per_model_buffers=moe_buffers,
                model_parallel_group=mpu.get_expert_tensor_model_pipeline_parallel_group(),
                data_parallel_group=mpu.get_expert_data_parallel_group(),
                data_parallel_group_gloo=mpu.get_expert_data_parallel_group_gloo(),
                data_parallel_group_idx=model_parallel_rank,
            )
        )
        if not hasattr(optimizers[-1], "buffers"):
            setattr(optimizers[-1], "buffers", list(itertools.chain(*moe_buffers.values())))
            setattr(optimizers[-1], "model_chunks", model_chunks)

    optimizer = optimizers[0] if len(optimizers) == 1 else ChainedOptimizer(optimizers)
    if bounded_hybrid_cls is not None:
        _enable_bounded_cpu_grad_staging(optimizer, bounded_hybrid_cls)
    if config.optimizer_cpu_offload and config.use_distributed_optimizer:
        # Megatron replaces each full-parameter Hybrid with one owning DP
        # shards. CPU step hooks form cycles in the retired Hybrid, retaining
        # its FP32 masters until GC. Reclaim those owners before allocating
        # Adam moments during the first step or checkpoint restoration.
        gc.collect()
    return optimizer
