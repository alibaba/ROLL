"""Host memory pool for offload stores: exact-size pinned CPU buffers, recycled.

Pinned buffers are expensive to create (cudaHostRegister page-locking), so
they are recycled through (numel, dtype) free lists. Offload cycles re-put
the same sizes every step, making size-bucketed reuse exact in practice.

"""
from typing import Dict, List, Optional, Set, Tuple

import torch
from torch import Tensor

from roll.platforms import current_platform


class HostMemPool:
    def __init__(self):
        # ptr -> flat base tensor. Strong refs: a garbage-collected
        # cudaHostRegister range is UB, and ptr identity must stay unambiguous.
        self._owned: Dict[int, Tensor] = {}
        self._free: Dict[Tuple[int, torch.dtype], List[Tensor]] = {}
        self._free_ptrs: Set[int] = set()
        self.pinned_bytes = 0
        self.pinned_free_bytes = 0

    def _register(self, buf: Tensor) -> None:
        nbytes = buf.nbytes
        err = torch.cuda.cudart().cudaHostRegister(buf.data_ptr(), nbytes, 0)
        if err != 0:
            raise RuntimeError(f"cudaHostRegister failed (cudaError {err}) for {nbytes / (1 << 30):.2f}GB")

    def alloc(self, numel: int, dtype: torch.dtype) -> Tensor:
        """Exact-size pinned flat CPU tensor; raises if registration fails."""
        bucket = self._free.get((numel, dtype))
        if bucket:
            buf = bucket.pop()
            self._free_ptrs.discard(buf.data_ptr())
            self.pinned_free_bytes -= buf.nbytes
            return buf

        buf = torch.empty(numel, dtype=dtype, device="cpu")
        if buf.numel() == 0:
            return buf
        if current_platform.is_cuda() or current_platform.is_rocm():
            self._register(buf)
        else:
            # No cudaHostRegister outside CUDA/ROCm (e.g. NPU): use the
            # platform's pinned allocator; pageable when no accelerator.
            try:
                buf = buf.pin_memory()
            except RuntimeError:
                pass
        self._owned[buf.data_ptr()] = buf
        self.pinned_bytes += buf.nbytes
        return buf

    def free(self, buf: Tensor) -> bool:
        """Recycle a pool-owned buffer (views accepted: the owned base tensor
        is what gets recycled). Returns False (no-op) for foreign tensors;
        double-free is ignored."""
        base = self._owned.get(buf.data_ptr())
        if base is None:
            return False
        if base.data_ptr() in self._free_ptrs:
            return True
        self._free.setdefault((base.numel(), base.dtype), []).append(base)
        self._free_ptrs.add(base.data_ptr())
        self.pinned_free_bytes += base.nbytes
        return True

    def stats(self) -> dict:
        return {
            "host_pinned_gb": self.pinned_bytes / (1 << 30),
            "host_pinned_free_gb": self.pinned_free_bytes / (1 << 30),
            "host_buffers": len(self._owned),
        }


_pool: Optional[HostMemPool] = None


def shared_host_mem_pool() -> HostMemPool:
    global _pool
    if _pool is None:
        _pool = HostMemPool()
    return _pool
