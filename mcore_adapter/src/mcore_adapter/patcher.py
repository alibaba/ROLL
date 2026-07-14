import math
import sys
from bisect import bisect_right, insort
from typing import Optional

import megatron.core.transformer.moe.router
import torch
from megatron.core.transformer.moe.moe_utils import (
    get_tokens_per_expert_and_token_count,
    save_to_aux_losses_tracker,
    switch_load_balancing_loss_func,
)
from torch.distributed._shard.metadata import ShardMetadata
from torch.distributed._shard.sharding_spec._internals import _check_shard_metadata_pair_overlap
from torch.distributed.checkpoint.default_planner import (
    _check_box_bounds,
    _check_box_overlap,
)
from torch.distributed.checkpoint.metadata import (
    BytesStorageMetadata,
    Metadata,
)
from torch.distributed.checkpoint.planner import SavePlan

from .utils import get_logger


logger = get_logger(__name__)


def patch_torch_find_nd_overlapping_shards():
    """
    Ref: https://github.com/pytorch/pytorch/issues/166941
         https://github.com/pytorch/pytorch/pull/167073
    """

    def _find_nd_overlapping_shards(shards: list[ShardMetadata], sharded_dims: list[int]) -> Optional[tuple[int, int]]:
        """Find overlapping shards using sweep-line algorithm."""
        if len(shards) <= 1:
            return None

        dims = len(sharded_dims)
        if dims == 0:
            return None

        sweep_dim_idx = 0
        if dims > 1:
            max_size = 0
            for i, dim in enumerate(sharded_dims):
                dim_size = shards[0].shard_offsets[dim] + shards[0].shard_sizes[dim]
                if dim_size > max_size:
                    max_size = dim_size
                    sweep_dim_idx = i
        sweep_dim = sharded_dims[sweep_dim_idx]

        sorted_indices = sorted(
            range(len(shards)),
            key=lambda idx: (
                shards[idx].shard_offsets[sweep_dim],
                *(shards[idx].shard_offsets[d] for d in sharded_dims if d != sweep_dim),
            ),
        )
        active: list[tuple[int, int]] = []

        for idx in sorted_indices:
            current = shards[idx]
            start = current.shard_offsets[sweep_dim]
            end = start + current.shard_sizes[sweep_dim]

            cutoff = bisect_right(active, (start, sys.maxsize))
            if cutoff:
                del active[:cutoff]

            for _, other_idx in active:
                other = shards[other_idx]

                if _check_shard_metadata_pair_overlap(current, other):
                    return (other_idx, idx)
            insort(active, (end, idx))
        return None

    torch.distributed._shard.sharding_spec._internals._find_nd_overlapping_shards = _find_nd_overlapping_shards


def patch_torch_validate_global_plan():
    """
    Related: https://github.com/pytorch/pytorch/issues/163548
             https://github.com/pytorch/pytorch/pull/166820
    """

    def _validate_global_plan(global_plan: list[SavePlan], metadata: Metadata) -> bool:
        all_good = True
        for key, value in metadata.state_dict_metadata.items():
            if isinstance(value, BytesStorageMetadata):
                continue
            if len(value.size) == 0:
                continue
            chunks = value.chunks
            chunks_volume = 0
            for chunk in chunks:
                # Compute the volume
                if not _check_box_bounds(value.size, chunk):
                    logger.warning(
                        """
                            key:%s has out of bounds chunk:
                            tensor-size:%s chunk: %s
                        """,
                        key,
                        value.size,
                        chunk,
                    )
                    all_good = False
                chunks_volume += math.prod(chunk.sizes)

            if len(chunks) > 1:
                dims = len(value.size)
                # sweep_dim = max(range(dims), default=0, key=lambda d: value.size[d])
                sweep_dim = 0  # use default sweep_dim, avoid degarding to O(N^2)
                sorted_indices = sorted(
                    range(len(chunks)),
                    key=lambda idx: (
                        chunks[idx].offsets[sweep_dim],
                        *(chunks[idx].offsets[d] for d in range(dims)),
                    ),
                )
                active: list[tuple[int, int]] = []
                for idx in sorted_indices:
                    current = chunks[idx]
                    start = current.offsets[sweep_dim]
                    end = start + current.sizes[sweep_dim]

                    cutoff = bisect_right(active, (start, sys.maxsize))
                    if cutoff:
                        del active[:cutoff]

                    for _, other_idx in active:
                        other = chunks[other_idx]
                        if _check_box_overlap(current, other):
                            logger.warning(
                                "key:%s has overlapping chunks: %s %s",
                                key,
                                current,
                                other,
                            )
                            all_good = False

                    insort(active, (end, idx))

            # Check whether combined chunk cover the whole tensor
            tensor_volume = math.prod(value.size)
            if len(global_plan) > 1 and chunks_volume != tensor_volume:
                logger.warning(
                    """
                        key:%s invalid fill tensor-volume:
                        %s chunks-volume: %s
                    """,
                    key,
                    tensor_volume,
                    chunks_volume,
                )
                all_good = False

        return all_good

    torch.distributed.checkpoint.default_planner._validate_global_plan = _validate_global_plan


def patch_hybrid_optimizer():
    def _update_fp32_params_by_new_state(self):
        if not self.param_update_in_fp32 or not self.param_to_fp32_param:
            return
        for param, v in self.state.items():
            fp32_param = self.param_to_fp32_param[param]
            fp32_param.data.copy_(v["master_param"])

    from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import HybridDeviceOptimizer

    HybridDeviceOptimizer._update_fp32_params_by_new_state = _update_fp32_params_by_new_state


def patch_megatron_preload_tensors_non_blocking(non_blocking: bool = False):
    """
    Patch FileSystemWriterAsync.get_save_function_and_args to control the
    non_blocking parameter passed to preload_tensors during async checkpoint D2H.

    On certain GPU+CPU hardware combinations, async D2H (non_blocking=True) in
    Megatron's checkpoint saving can cause segmentation faults. This patch
    allows controlling the non_blocking behavior via McoreAdapter's args.

    This patch is compatible with Megatron-Core versions (>=0.13.0),
    as the get_save_function_and_args API signature is identical across versions.

    Args:
        non_blocking (bool): Whether to use non_blocking D2H transfer.
            Default is False (synchronous, safe for all hardware).
    """
    if non_blocking:
        logger.info("Skip patch mcore preload_tensors when non_blocking=True...")
        return

    try:
        from megatron.core.dist_checkpointing.strategies.filesystem_async import (
            FileSystemWriterAsync,
        )
    except ImportError:
        logger.warning("megatron.core.dist_checkpointing not available, skipping preload_tensors patch")
        return

    import inspect
    from functools import partial

    _original_get_save_function_and_args = FileSystemWriterAsync.get_save_function_and_args

    def patched_get_save_function_and_args(self):
        result = _original_get_save_function_and_args(self)

        if len(result) != 3:
            logger.warning(
                f"The return vals of get_save_function_and_args is not 3, skipping preload_tensors patch, check the mcore version."
            )
            return result

        save_fn, _preload_fn, args = result
        if _preload_fn is None:
            return result

        params = list(inspect.signature(_preload_fn.func).parameters.keys())
        if "non_blocking" not in params:
            logger.warning("preload_tensors no longer has 'non_blocking' parameter, skipping patch")
            return result

        # Override preload_fn with configured non_blocking value
        new_args = list(_preload_fn.args)
        non_blocking_idx = params.index("non_blocking")
        new_args[non_blocking_idx] = non_blocking
        preload_fn = partial(_preload_fn.func, *new_args)
        return (save_fn, preload_fn, args)

    logger.info(
        f"Patched FileSystemWriterAsync.get_save_function_and_args with non_blocking={non_blocking} for D2H transfers"
    )
    FileSystemWriterAsync.get_save_function_and_args = patched_get_save_function_and_args


def patch_apply_aux_loss():
    def _apply_aux_loss(
        self,
        probs: torch.Tensor,
        scores_for_aux_loss: torch.Tensor,
        routing_map: torch.Tensor,
        with_padding_mask: bool = False,
    ):
        """Apply the auxiliary loss for the given scores and routing map."""
        aux_loss_coeff = self.get_aux_loss_coeff("aux_loss")
        if aux_loss_coeff == 0:
            return probs

        global_tokens_per_expert, local_num_tokens, total_num_tokens = get_tokens_per_expert_and_token_count(
            routing_map=routing_map,
            reduce_group=self.tp_cp_group,
            topk=self.topk,
            with_padding_mask=with_padding_mask,
        )
        tokens_per_expert_for_statistics = global_tokens_per_expert.detach().clone()
        num_layers = self.config.num_layers
        if self.config.mtp_num_layers is not None:
            num_layers += self.config.mtp_num_layers
        save_to_aux_losses_tracker(
            "expert_distributed_std",
            torch.std(tokens_per_expert_for_statistics.to(torch.float), unbiased=False),
            self.layer_number,
            num_layers,
        )
        save_to_aux_losses_tracker(
            "max_token_per_expert", torch.max(tokens_per_expert_for_statistics), self.layer_number, num_layers
        )
        aux_loss = switch_load_balancing_loss_func(
            probs=scores_for_aux_loss,
            tokens_per_expert=global_tokens_per_expert,
            total_num_tokens=total_num_tokens,
            topk=self.topk,
            num_experts=self.config.num_moe_experts,
            moe_aux_loss_coeff=aux_loss_coeff,
            fused=self.config.moe_router_fusion,
        )
        probs = self.attach_and_log_load_balancing_loss(
            probs,
            aux_loss_coeff,
            aux_loss,
            "load_balancing_loss",
            self.tp_cp_group,
            valid_token_count=local_num_tokens,
        )
        return probs

    megatron.core.transformer.moe.router.TopKRouter._apply_aux_loss = _apply_aux_loss
