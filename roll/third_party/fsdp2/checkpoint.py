"""DCP async staging for DTensors whose local shards alias flat buffers."""

from concurrent.futures import Future
import inspect
from typing import Any, Dict, Optional, Set, Tuple

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.tensor import DTensor

try:
    from torch.distributed.checkpoint._state_dict_stager import StateDictStager
except ImportError:
    # Older PyTorch uses tensor-level CPU staging rather than the affected
    # storage-level stager. Keep that path and its async_save signature intact.
    StateDictStager = None

try:
    from torch.distributed.checkpoint.filesystem import FileSystemWriter
except ImportError:
    FileSystemWriter = None


if StateDictStager is not None:
    class _DTensorStateDictStager(StateDictStager):
        """Stage real local storage, never a DTensor wrapper's fake storage."""

        # stage() blocks until CPU copies finish; only disk I/O is asynchronous.
        should_synchronize_after_execute = False

        def __init__(self, *args, passthrough_storages: Optional[Set[Tuple[int, int]]] = None, **kwargs):
            super().__init__(*args, **kwargs)
            self._passthrough_storages = passthrough_storages or set()

        def _offload_tensor(
            self, value: torch.Tensor, memo: Dict[int, Any], non_blocking: bool = False
        ) -> torch.Tensor:
            if tensor_storage_key(value) in self._passthrough_storages:
                # FSDP2 keeps this store buffer alive until the async writer
                # finishes, so no second CPU snapshot is needed.
                memo[id(value)] = value
                return value
            if not isinstance(value, DTensor):
                return super()._offload_tensor(value, memo, non_blocking=non_blocking)

            local = self.deepcopy_with_tensor_offload(value._local_tensor, memo, non_blocking=non_blocking)
            # Reuse DCP's storage memo: multiple local views copy their shared
            # flat storage to CPU only once, preserving offsets and aliases.
            # from_local() would move this CPU tensor back to the mesh device.
            # Use the internal constructor, as DTensor operators do, to retain
            # the original mesh/placements/global shape while keeping CPU data.
            staged = DTensor(local, value._spec, requires_grad=False)
            memo[id(value)] = staged
            return staged

        def synchronize_staging(self) -> None:
            """No-op: stage() completes the snapshot before returning."""
            pass


def tensor_storage_key(value: torch.Tensor) -> Optional[Tuple[int, int]]:
    """Stable identity for a CPU tensor's real storage, including DTensor locals."""
    local = value._local_tensor if isinstance(value, DTensor) else value
    if local.device.type != "cpu" or local.numel() == 0:
        return None
    storage = local.untyped_storage()
    return storage.data_ptr(), storage.nbytes()


def supports_passthrough_staging() -> bool:
    """Whether this torch version can let DCP write protected CPU storage directly."""
    return StateDictStager is not None


def _close_stager(stager: Any) -> None:
    """Close storage caches when supported by the installed PyTorch version."""
    close = getattr(stager, "close", None)
    if close is not None:
        close()


if FileSystemWriter is not None:
    class _DTensorFileSystemWriter(FileSystemWriter):
        """PyTorch 2.8 bridge: FileSystemWriter also owns the async stager."""

        def __init__(self, checkpoint_id: str, stager: Any):
            super().__init__(checkpoint_id)
            self._dtensor_stager = stager

        @property
        def should_synchronize_after_execute(self) -> bool:
            return False

        def stage(self, state_dict):
            return self._dtensor_stager.stage(state_dict)

        def synchronize_staging(self) -> None:
            self._dtensor_stager.synchronize_staging()


def async_save_dtensor(
    state_dict: Dict[str, Any],
    checkpoint_id: str,
    process_group: Optional[dist.ProcessGroup] = None,
    passthrough_storages: Optional[Set[Tuple[int, int]]] = None,
) -> Future:
    """Stage local DTensor storage, then save through DCP asynchronously.

    Model, optimizer and scheduler state keep the existing checkpoint schema.
    Selected immutable CPU store buffers can bypass the snapshot copy.
    Each save owns its staging cache, so a later save cannot overwrite an active
    snapshot. The cache is closed on completion or failure. No GPU shard copies
    or additional collectives are introduced by staging.
    """
    if StateDictStager is None:
        return dcp.async_save(state_dict=state_dict, checkpoint_id=checkpoint_id, process_group=process_group)

    stager = _DTensorStateDictStager(
        pin_memory=False,
        share_memory=False,
        passthrough_storages=passthrough_storages,
    )
    try:
        if "async_stager" in inspect.signature(dcp.async_save).parameters:
            future = dcp.async_save(
                state_dict=state_dict,
                checkpoint_id=checkpoint_id,
                process_group=process_group,
                async_stager=stager,
            )
        else:
            if FileSystemWriter is None:
                raise RuntimeError("PyTorch async_save has no async_stager support or FileSystemWriter")
            writer = _DTensorFileSystemWriter(checkpoint_id, stager)
            future = dcp.async_save(
                state_dict=state_dict,
                storage_writer=writer,
                process_group=process_group,
            )
    except BaseException:
        _close_stager(stager)
        raise
    future.add_done_callback(lambda _: _close_stager(stager))
    return future
