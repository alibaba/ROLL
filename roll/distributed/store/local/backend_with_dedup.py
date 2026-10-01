"""Offload to local pinned CPU memory with DP deduplication (offload_backend: local_dedup).

replicated_over=False (or no replication group): every rank stores independently,
same pinned HostMemPool path as CPUOffloadBackend.
replicated_over=True with a multi-rank replication group: the flat tensor is split
into world_size chunks and each rank keeps only its own chunk (~1/world of the
data). On get, each rank uploads its chunk H2D and the full tensor is rebuilt
with a single all-gather over that group.

replicated_over may also be a ProcessGroup: the tensors dedup over that group
instead of the backend's default (e.g. Megatron dense params over DP×CP vs
expert params over expert-DP). The group is recorded per key, so get/delete
never take it.
"""
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import Tensor

from roll.distributed.store.base import as_flats, copy_flat_slice, slice_flats
from roll.distributed.store.local.backend import CPUOffloadBackend
from roll.platforms import current_platform


def _is_dedup_group(group) -> bool:
    return group is not None and dist.is_initialized() and dist.get_world_size(group) > 1


def _chunk_range(total_numel: int, world: int, rank: int) -> Tuple[int, int, int]:
    """rank's [start, end) slice of a flat dedup tensor, plus the padded
    per-rank chunk size required by all_gather_into_tensor."""
    chunk = (total_numel + world - 1) // world
    start = min(rank * chunk, total_numel)
    end = min(start + chunk, total_numel)
    return start, end, chunk


class CPUDedupOffloadBackend(CPUOffloadBackend):
    """Offload to local pinned CPU memory, with DP deduplication (chunk per rank + all-gather)."""

    def __init__(self, dp_rank: int = 0, dp_group=None):
        super().__init__()
        self._dp_rank = dp_rank
        self._dp_group = dp_group
        # Present on every rank for dedup keys: key -> (shape, dtype, group, rank)
        self._dedup_meta: Dict[str, Tuple[torch.Size, torch.dtype, object, int]] = {}

    @property
    def supports_direct_checkpoint(self) -> bool:
        # A dedup key stores only this rank's chunk. Reconstructing the local
        # DTensor shard requires an accelerator all-gather and a new buffer.
        return False

    def _put(self, key: str, tensor: Union[Tensor, List[Tensor]], replicated_over=True) -> None:
        if replicated_over is True:
            group, rank = self._dp_group, self._dp_rank
        elif replicated_over:  # a ProcessGroup: dedup over it
            group = replicated_over
            rank = dist.get_rank(group)
        else:
            group = None
        if not _is_dedup_group(group):
            self._dedup_meta.pop(key, None)
            super()._put(key, tensor)
            return

        if isinstance(tensor, Tensor):
            shape = tensor.shape
            flats = [tensor.detach().reshape(-1)]
        else:
            flats = as_flats(tensor)
            shape = torch.Size([sum(f.numel() for f in flats)])
        dtype = flats[0].dtype
        self._dedup_meta[key] = (shape, dtype, group, rank)

        start, end, _ = _chunk_range(shape.numel(), dist.get_world_size(group), rank)
        buf = self._host_buf(key, end - start, dtype)
        mine = slice_flats(flats, start, end - start)
        if copy_flat_slice(buf, mine, 0, buf.numel()):
            current_platform.current_stream().synchronize()
        self._store[key] = buf

    def _dedup_allgather(self, key: str, shape: torch.Size, dtype: torch.dtype,
                         gpu_device: torch.device, group, rank: int) -> Tensor:
        """Rebuild a dedup key: each rank uploads its pinned chunk H2D (direct
        DMA), then one collective all-gather over the key's group assembles the
        full tensor on every rank."""
        numel = shape.numel()
        world = dist.get_world_size(group)
        _, _, chunk = _chunk_range(numel, world, rank)
        gathered = torch.empty(chunk * world, dtype=dtype, device=gpu_device)
        host = self._store[key]
        if current_platform.is_cuda() or current_platform.is_rocm():
            # H2D straight into this rank's slot, then NCCL in-place all-gather
            # (sendbuff == recvbuff + rank*count): no separate chunk staging
            # tensor. In-place aliasing is guaranteed by NCCL/RCCL only.
            mine = gathered.narrow(0, rank * chunk, chunk)
            if host.numel() > 0:
                mine.narrow(0, 0, host.numel()).copy_(host, non_blocking=True)
        else:
            # Other CCLs (HCCL, vendor stacks) may not support send/recv
            # buffer aliasing: stage this rank's chunk separately.
            mine = torch.zeros(chunk, dtype=dtype, device=gpu_device)
            if host.numel() > 0:
                mine.narrow(0, 0, host.numel()).copy_(host, non_blocking=True)

        dist.all_gather_into_tensor(gathered, mine, group=group)
        return gathered.narrow(0, 0, numel).view(shape)

    def _get(self, key: str, device: Optional[Union[torch.device, str]] = None) -> Tensor:
        meta = self._dedup_meta.get(key)
        if meta is None:
            return super()._get(key, device=device)

        if device is None:
            device = torch.device(f"{current_platform.device_type}:{current_platform.current_device()}")
        if isinstance(device, str):
            device = torch.device(device)
        shape, dtype, group, rank = meta
        # All-gather rides the collective library, so stage through the GPU
        # even for CPU targets.
        gpu_device = device if device.type != "cpu" else torch.device(
            f"{current_platform.device_type}:{current_platform.current_device()}")
        result = self._dedup_allgather(key, shape, dtype, gpu_device, group, rank)
        if device.type == "cpu":
            result = result.cpu()
        return result

    def delete(self, key: str, replicated_over: bool = True) -> None:
        self._dedup_meta.pop(key, None)
        super().delete(key, replicated_over=replicated_over)

    def has_key(self, key: str) -> bool:
        return key in self._store or key in self._dedup_meta

    def clear(self) -> None:
        super().clear()
        self._dedup_meta.clear()
