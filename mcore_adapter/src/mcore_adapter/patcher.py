import math
import sys
from typing import Optional

import torch
from torch.distributed._shard.metadata import ShardMetadata
from torch.distributed._shard.sharding_spec._internals import (
    _check_shard_metadata_pair_overlap
)
from torch.distributed.checkpoint.default_planner import (
    _check_box_bounds, 
    _check_box_overlap,
)
from torch.distributed.checkpoint.metadata import (
    BytesStorageMetadata,
    Metadata,
)
from torch.distributed.checkpoint.planner import SavePlan
from bisect import bisect_right, insort

from .utils import get_logger


logger = get_logger(__name__)


def patch_torch_find_nd_overlapping_shards():
    """
    Ref: https://github.com/pytorch/pytorch/issues/166941
         https://github.com/pytorch/pytorch/pull/167073
    """
    def _find_nd_overlapping_shards(
        shards: list[ShardMetadata], sharded_dims: list[int]
    ) -> Optional[tuple[int, int]]:
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
                sweep_dim = 0 # use default sweep_dim, avoid degarding to O(N^2)
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
