"""Native NCCL communicator-memory offload.

This module keeps every PyTorch ProcessGroup intact.  When ``offload_nccl``
is enabled, it releases only NCCL's dynamic GPU allocations through
``ncclCommSuspend(NCCL_SUSPEND_MEM)`` and restores them through
``ncclCommResume``.  Both calls are collective and therefore must be issued
in the same order on every rank while the communicators are idle.

NCCL added these APIs in 2.29.7.  Loading the same ``libnccl`` already mapped
by PyTorch is important: NCCL communicator handles cannot be passed to a
different copy of the library.
"""

from __future__ import annotations

import ctypes
import os
from collections.abc import Iterable
from typing import Any

import psutil
import torch
import torch.distributed as dist

from roll.utils.logging import get_logger


logger = get_logger()

NCCL_SUSPEND_MEM = 0x01
NCCL_SUCCESS = 0
NCCL_MIN_VERSION = 22907  # 2.29.7

_nccl_library: ctypes.CDLL | None = None
_nccl_library_path: str | None = None
_nccl_runtime_version: int | None = None
_nccl_load_error: str | None = None
_nccl_load_attempted = False
_suspended_handles_by_pid: dict[int, list[tuple[str, int]]] = {}


class NcclCommSuspendUnavailable(RuntimeError):
    """Raised when ``offload_nccl`` is enabled with an unsupported NCCL."""


def _find_loaded_nccl_library() -> str | None:
    """Return the real libnccl path mapped by this process, if one exists."""
    for memory_map in psutil.Process().memory_maps():
        path = memory_map.path
        if "libnccl.so" in path.lower() and os.path.exists(path):
            return path
    return None


def _get_nccl_library() -> ctypes.CDLL | None:
    """Check the mapped NCCL runtime version and native suspend/resume symbols.

    Like VERL PR #6408, use the library already mapped by PyTorch and require
    both native symbols. Also query ncclGetVersion: package versions and
    torch.cuda.nccl.version() need not describe the loaded shared library.
    """
    global _nccl_library, _nccl_library_path, _nccl_load_attempted
    global _nccl_runtime_version, _nccl_load_error

    if _nccl_load_attempted:
        return _nccl_library
    _nccl_load_attempted = True

    nccl_path = _find_loaded_nccl_library()
    if nccl_path is None:
        _nccl_load_error = "libnccl is not mapped in this process"
        logger.warning("NCCL suspend/resume unavailable: %s", _nccl_load_error)
        return None
    _nccl_library_path = nccl_path

    try:
        library = ctypes.CDLL(nccl_path)
    except OSError as exc:
        _nccl_load_error = f"Failed to load mapped NCCL library {nccl_path}: {exc}"
        logger.warning(_nccl_load_error)
        return None

    missing = [name for name in ("ncclGetVersion", "ncclCommSuspend", "ncclCommResume")
               if not hasattr(library, name)]
    if missing:
        _nccl_load_error = f"NCCL library {nccl_path} is missing {', '.join(missing)}"
        logger.warning(_nccl_load_error)
        return None

    library.ncclGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
    library.ncclGetVersion.restype = ctypes.c_int
    version = ctypes.c_int()
    result = library.ncclGetVersion(ctypes.byref(version))
    _nccl_runtime_version = version.value
    if result != NCCL_SUCCESS or version.value < NCCL_MIN_VERSION:
        _nccl_load_error = (
            f"ncclGetVersion returned {version.value}, status={result}; "
            "requires NCCL >= 2.29.7"
        )
        logger.warning(_nccl_load_error)
        return None

    library.ncclCommSuspend.argtypes = [ctypes.c_void_p, ctypes.c_int]
    library.ncclCommSuspend.restype = ctypes.c_int
    library.ncclCommResume.argtypes = [ctypes.c_void_p]
    library.ncclCommResume.restype = ctypes.c_int

    _nccl_library = library
    logger.info("Native NCCL communicator suspend/resume enabled via %s (runtime=%d)", nccl_path, version.value)
    return _nccl_library


def is_nccl_comm_suspend_supported() -> bool:
    """Whether the mapped runtime is NCCL >= 2.29.7 and exports the APIs."""
    return _get_nccl_library() is not None


def require_nccl_comm_suspend_support() -> None:
    """Raise an actionable error instead of falling back to group recreation."""
    if is_nccl_comm_suspend_supported():
        return
    mapped = _nccl_library_path or _find_loaded_nccl_library() or "no mapped libnccl"
    raise NcclCommSuspendUnavailable(
        "offload_nccl=True requires NCCL >= 2.29.7 with ncclCommSuspend/Resume "
        f"(loaded: {mapped}, runtime: {_nccl_runtime_version}, reason: {_nccl_load_error}). "
        "Provide a compatible NCCL runtime or set offload_nccl=False. "
        "ROLL no longer falls back to destroying and recreating ProcessGroups."
    )


def _as_handle(value: Any) -> int:
    """Convert the native representation of ``ncclComm_t`` to int."""
    if isinstance(value, ctypes.c_void_p):
        return int(value.value or 0)
    raw_value = getattr(value, "value", value)
    return int(raw_value or 0)


def suspend_nccl_comm(handle: Any) -> None:
    """Suspend one communicator without changing its identity."""
    require_nccl_comm_suspend_support()
    comm_handle = _as_handle(handle)
    if comm_handle == 0:
        raise ValueError("Cannot suspend a null NCCL communicator handle")
    result = _nccl_library.ncclCommSuspend(ctypes.c_void_p(comm_handle), NCCL_SUSPEND_MEM)
    if result != NCCL_SUCCESS:
        raise RuntimeError(f"ncclCommSuspend(handle=0x{comm_handle:x}) failed with NCCL error {result}")


def resume_nccl_comm(handle: Any) -> None:
    """Resume one communicator previously suspended by ``suspend_nccl_comm``."""
    require_nccl_comm_suspend_support()
    comm_handle = _as_handle(handle)
    if comm_handle == 0:
        raise ValueError("Cannot resume a null NCCL communicator handle")
    result = _nccl_library.ncclCommResume(ctypes.c_void_p(comm_handle))
    if result != NCCL_SUCCESS:
        raise RuntimeError(f"ncclCommResume(handle=0x{comm_handle:x}) failed with NCCL error {result}")


def _iter_group_values(value: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        yield from ((str(key), group) for key, group in value.items())
    elif isinstance(value, (list, tuple)):
        yield from ((str(index), group) for index, group in enumerate(value))
    else:
        yield "", value


def _add_process_group_handle(
    handles: list[tuple[str, int]], seen_handles: set[int], label: str, process_group: Any
) -> None:
    if process_group is None:
        return
    try:
        backend = process_group._get_backend(torch.device("cuda"))
        handle = _as_handle(backend._comm_ptr())
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return
    if handle == 0 or handle in seen_handles:
        return
    seen_handles.add(handle)
    handles.append((label, handle))


def _collect_megatron_process_groups(handles: list[tuple[str, int]], seen_handles: set[int]) -> None:
    """Collect named Megatron groups first for stable cross-rank call order."""
    try:
        from megatron.core import parallel_state
    except ImportError:
        return

    try:
        initialized = parallel_state.model_parallel_is_initialized()
    except Exception:
        initialized = False
    if not initialized:
        return

    for attribute_name in sorted(dir(parallel_state)):
        if not attribute_name.startswith("_") or "GROUP" not in attribute_name or "GLOO" in attribute_name:
            continue
        value = getattr(parallel_state, attribute_name, None)
        for suffix, process_group in _iter_group_values(value):
            label = f"megatron/{attribute_name.lstrip('_')}"
            if suffix:
                label = f"{label}[{suffix}]"
            _add_process_group_handle(handles, seen_handles, label, process_group)


def _collect_torch_process_groups(handles: list[tuple[str, int]], seen_handles: set[int]) -> None:
    """Collect remaining live NCCL process groups without monkey-patching torch."""
    if not dist.is_available() or not dist.is_initialized():
        return
    distributed_c10d = getattr(dist, "distributed_c10d", None)
    world = getattr(distributed_c10d, "_world", None)
    process_group_map = getattr(world, "pg_map", None)
    if not isinstance(process_group_map, dict):
        return

    from roll.utils.collective import collective

    # Weight-update groups span actor and inference roles, which are not idle
    # together. Suspending only the actor side of this collective would hang.
    # Keep these groups resident until a cross-role lifecycle exists.
    cross_role_groups = {
        id(group) for name, group in collective._group_mgr._name_group_map.items()
        if name.startswith("model_update/")
    }

    # pg_map preserves collective group-creation order. Keep it unchanged: the
    # native calls themselves are collective and every rank must call them in
    # the same order.
    for index, process_group in enumerate(process_group_map):
        if id(process_group) in cross_role_groups:
            continue
        _add_process_group_handle(handles, seen_handles, f"torch/process_group[{index}]", process_group)


def collect_nccl_communicator_handles() -> list[tuple[str, int]]:
    """Return each live, initialized PyTorch NCCL communicator exactly once."""
    handles: list[tuple[str, int]] = []
    seen_handles: set[int] = set()
    _collect_megatron_process_groups(handles, seen_handles)
    _collect_torch_process_groups(handles, seen_handles)
    return handles


def _synchronize_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def suspend_nccl_communicators() -> list[tuple[str, int]]:
    """Suspend all live PyTorch NCCL communicators in this process.

    The operation is idempotent.  It deliberately does not destroy a process
    group, patch ``torch.distributed``, or recreate any communicator.
    """
    pid = os.getpid()
    if pid in _suspended_handles_by_pid:
        logger.debug("NCCL communicators are already suspended in pid %s", pid)
        return _suspended_handles_by_pid[pid]

    handles = collect_nccl_communicator_handles()
    if not handles:
        logger.info("No live NCCL communicators to suspend in pid %s", pid)
        return []

    require_nccl_comm_suspend_support()
    _synchronize_cuda()
    suspended: list[tuple[str, int]] = []
    try:
        for name, handle in handles:
            suspend_nccl_comm(handle)
            suspended.append((name, handle))
    except Exception:
        # Best-effort rollback keeps the process usable when a communicator
        # fails halfway through the collective sequence.
        for _, handle in reversed(suspended):
            try:
                resume_nccl_comm(handle)
            except Exception:
                logger.exception("Failed to roll back NCCL communicator 0x%x", handle)
        raise

    _suspended_handles_by_pid[pid] = handles
    logger.info("Suspended %d NCCL communicator(s) in pid %s", len(handles), pid)
    return handles


def resume_nccl_communicators() -> list[tuple[str, int]]:
    """Resume communicators suspended by ``suspend_nccl_communicators``."""
    pid = os.getpid()
    handles = _suspended_handles_by_pid.get(pid)
    if not handles:
        logger.debug("NCCL communicators are not suspended in pid %s", pid)
        return []

    require_nccl_comm_suspend_support()
    _synchronize_cuda()
    for _, handle in handles:
        resume_nccl_comm(handle)
    _suspended_handles_by_pid.pop(pid, None)
    logger.info("Resumed %d NCCL communicator(s) in pid %s", len(handles), pid)
    return handles
