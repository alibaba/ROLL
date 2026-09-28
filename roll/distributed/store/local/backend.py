"""Offload to local pinned CPU memory (offload_backend: local). Each rank stores independently.
"""
from typing import Dict, List, Optional, Union

import torch
from torch import Tensor

from roll.distributed.store.base import OffloadBackend, as_flats, copy_flat_slice
from roll.distributed.store.host_mem_pool import shared_host_mem_pool
from roll.platforms import current_platform


class CPUOffloadBackend(OffloadBackend):
    """Offload to local pinned CPU memory, allocated and recycled through the shared HostMemPool."""

    def __init__(self):
        super().__init__()
        self._store: Dict[str, Tensor] = {}
        self._pool = shared_host_mem_pool()

    @property
    def supports_direct_checkpoint(self) -> bool:
        return True

    def _host_buf(self, key: str, numel: int, dtype: torch.dtype) -> Tensor:
        return self._pool.alloc(numel, dtype)

    def _put(self, key: str, tensor: Union[Tensor, List[Tensor]], replicated_over=True) -> None:
        # replicated_over (bool or ProcessGroup) is a dedup-only concept; every rank
        # stores independently here.
        if isinstance(tensor, Tensor):
            src = tensor.detach()
            if src.device.type == "cpu":
                self._store[key] = src
                return
            buf = self._host_buf(key, src.numel(), src.dtype).view(src.shape)
            flats, flat_buf = [src.reshape(-1)], buf.reshape(-1)
        else:
            flats = as_flats(tensor)
            total_numel = sum(f.numel() for f in flats)
            buf = self._host_buf(key, total_numel, flats[0].dtype)
            flat_buf = buf

        if copy_flat_slice(flat_buf, flats, 0, flat_buf.numel()):
            current_platform.current_stream().synchronize()
        self._store[key] = buf

    def _get(self, key: str, device: Optional[Union[torch.device, str]] = None) -> Tensor:
        if device is None:
            device = torch.device(f"{current_platform.device_type}:{current_platform.current_device()}")
        if isinstance(device, str):
            device = torch.device(device)

        if key not in self._store:
            raise KeyError(f"Key '{key}' not found in CPUOffloadBackend")
        tensor = self._store[key]
        if device.type == "cpu":
            return tensor
        return tensor.to(device, non_blocking=True)

    def delete(self, key: str, replicated_over: bool = True) -> None:
        buf = self._store.pop(key, None)
        if buf is not None:
            self._pool.free(buf)

    def has_key(self, key: str) -> bool:
        return key in self._store

    def get_stats(self) -> dict:
        return self._pool.stats()

    def clear(self) -> None:
        for buf in self._store.values():
            self._pool.free(buf)
        self._store.clear()
