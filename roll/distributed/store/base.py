"""
Offload store abstraction.

Offload backends are equal-level implementations selected by config —
no sequential layering.

Data flow:
    offload: backend.put_tensors(key, params) → GPU memory freed
    reload:  backend.get_tensors(key, params, device=gpu) → params rebuilt on GPU


"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor
from torch.distributed.tensor import DTensor

def as_flats(tensor: Union[Tensor, Sequence[Tensor]]) -> List[Tensor]:
    """Normalize a put() source into a list of flat views (logical concatenation)."""
    if isinstance(tensor, Tensor):
        return [tensor.detach().reshape(-1)]
    return [t.detach().reshape(-1) for t in tensor]


def copy_flat_slice(dst: Tensor, flats: Sequence[Tensor], start: int, numel: int) -> bool:
    """Copy the logical range [start, start+numel) of the concatenation of
    `flats` into dst[0:numel] (non-blocking). Returns True if any source
    slice was on GPU (caller decides whether/how to synchronize)."""
    end = start + numel
    offset = 0
    from_gpu = False
    for flat in flats:
        n = flat.numel()
        lo, hi = max(start, offset), min(end, offset + n)
        if lo < hi:
            dst.narrow(0, lo - start, hi - lo).copy_(
                flat.narrow(0, lo - offset, hi - lo), non_blocking=True)
            from_gpu |= flat.device.type != "cpu"
        offset += n
    return from_gpu


def slice_flats(flats: Sequence[Tensor], start: int, numel: int) -> List[Tensor]:
    """Views covering the logical range [start, start+numel) of the concatenation of `flats`."""
    end = start + numel
    out: List[Tensor] = []
    offset = 0
    for flat in flats:
        n = flat.numel()
        lo, hi = max(start, offset), min(end, offset + n)
        if lo < hi:
            out.append(flat.narrow(0, lo - offset, hi - lo))
        offset += n
    return out


def shrink_cuda_storage(data: Tensor) -> None:
    """Free the device memory even when untracked views still alias it (mcore
    keeps cached buffer-shard views that would pin the allocation)."""
    if data.device.type != "cpu" and data.untyped_storage().size() > 0:
        data.untyped_storage().resize_(0)


def relink_dtensor_local(param: Tensor, local: Optional[Tensor]) -> None:
    """Point a DTensor parameter at a new local shard (its data lives in
    `_local_tensor`, not .data). Flat-buffer views are valid local shards;
    checkpoint staging must use their real storage, not the wrapper's fake
    storage. None swaps in an empty CPU placeholder."""
    if local is None:
        local = torch.empty(0, dtype=param.dtype, device="cpu")
    param._local_tensor = local


class OffloadBackend(ABC):
    """Abstract offload backend. Implementations are equal-level and selected by config."""

    def __init__(self):
        # key -> per-tensor (device_mesh, placements, local_shape) recorded by
        # put_tensors; mesh/placements are None for plain (non-DTensor) tensors.
        self._tensor_meta: Dict[str, List[Tuple]] = {}

    @property
    def supports_direct_checkpoint(self) -> bool:
        """Whether get_state() can expose the stored CPU buffer without reconstruction."""
        return False

    @abstractmethod
    def _put(self, key: str, tensor: Union[Tensor, List[Tensor]],
             replicated_over: Union[bool, "torch.distributed.ProcessGroup"] = True) -> None:
        """Storage hook: store a tensor or list of tensors as one flat concatenation.
        replicated_over: False = per-rank copy, True = default replication group,
        ProcessGroup = that group. All source copies must be complete on return."""
        ...

    @abstractmethod
    def _get(self, key: str, device: Optional[Union[torch.device, str]] = None) -> Tensor:
        """Storage hook: retrieve the flat tensor on the target device (default: current GPU)."""
        ...

    @abstractmethod
    def delete(self, key: str, replicated_over: bool = True) -> None:
        """Delete a stored tensor. When replicated_over=True, only dp_rank=0 deletes."""
        ...

    @abstractmethod
    def has_key(self, key: str) -> bool:
        """Check if key exists."""
        ...

    @abstractmethod
    def clear(self) -> None:
        """Clear all stored tensors."""
        ...

    def barrier(self) -> None:
        """Synchronize across DP ranks. No-op for local backends."""
        pass

    def get_state(self, key: str) -> Tuple[Tensor, List[Tuple]]:
        """Return the stored CPU flat buffer and metadata for checkpointing."""
        if not self.supports_direct_checkpoint:
            raise NotImplementedError(f"{type(self).__name__} does not support direct checkpoint reads")
        if key not in self._tensor_meta or not self.has_key(key):
            raise KeyError(f"Key '{key}' not found in offload store")
        return self._get(key, device="cpu"), self._tensor_meta[key]

    def get_stats(self) -> Optional[dict]:
        """Return backend stats for logging. None for backends without stats."""
        return None

    def put_tensors(self, key: str, tensors: List[Tensor],
                    replicated_over: Union[bool, "torch.distributed.ProcessGroup"] = True) -> Optional[str]:
        """Offload a group of tensors under one key (flat concatenation of local
        shards) and free their GPU memory. replicated_over: False = per-rank
        copy, True/ProcessGroup = dedup over that group. Barriers after put
        when deduplicating. Returns the key (None if empty)."""
        if not tensors:
            return None
        if self.has_key(key):
            raise KeyError(
                f"put_tensors: key '{key}' already exists; delete stale keys before "
                f"re-putting (silent overwrite would leak the previous buffer)"
            )

        metas, locals_ = [], []
        for t in tensors:
            if isinstance(t, DTensor):
                local = t._local_tensor
                metas.append((t.device_mesh, t.placements, local.shape))
                locals_.append(local)
            else:
                d = t.data
                if isinstance(d, DTensor):
                    raise NotImplementedError(
                        "offload of a plain parameter whose .data is a DTensor is not supported; "
                        "expected the parameter itself to be DTensor-wrapped (FSDP2 layout)"
                    )
                metas.append((None, None, d.shape))
                locals_.append(d)
        self._tensor_meta[key] = metas
        self._put(key, locals_, replicated_over=replicated_over)

        for t, local in zip(tensors, locals_):
            if isinstance(t, DTensor):
                relink_dtensor_local(t, None)
            else:
                t.data = torch.empty(0, dtype=t.dtype, device="cpu")
            shrink_cuda_storage(local)

        if replicated_over:
            self.barrier()

        return key

    def get_tensors(self, key: str, tensors: List[Tensor],
                    device: torch.device, replicated_over: bool = True) -> None:
        """Restore tensors as views into a single flat buffer, without shard copies.
        Barriers only when replicated_over=True."""
        flat = self._get(key, device=device)

        offset = 0
        for t, (mesh, _, local_shape) in zip(tensors, self._tensor_meta[key]):
            numel = local_shape.numel()
            local = flat.narrow(0, offset, numel).view(local_shape)
            if isinstance(t, DTensor):
                relink_dtensor_local(t, local)
            elif mesh is not None:
                raise NotImplementedError(
                    "reload parameter type mismatch: DTensor-wrapped at put time but "
                    "plain Tensor now; expected the parameter itself to be "
                    "DTensor-wrapped (FSDP2 layout)"
                )
            else:
                t.data = local
            offset += numel
        del flat

        if replicated_over:
            self.barrier()

    def delete_tensors(self, key: str, replicated_over: bool = True) -> None:
        """Delete tensors by key. Barriers only when replicated_over=True."""
        self.delete(key, replicated_over=replicated_over)
        self._tensor_meta.pop(key, None)
        if replicated_over:
            self.barrier()
