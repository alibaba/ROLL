import copy
import os
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, Literal, List, Optional, Tuple

import psutil
import torch
import torch.distributed as dist
import logging

from codetiming import Timer
from ray._private import profiling

from roll.platforms import current_platform
from roll.utils.nccl_suspend import resume_nccl_communicators, suspend_nccl_communicators
from roll.utils.offload_states import OffloadStateType
from roll.utils.logging import get_logger, is_roll_debug_mode


logger = get_logger()

memory_log_print_limits = 20


def log_gpu_memory_usage(head: str, logger: logging.Logger = None, rank: int = 0):
    global memory_log_print_limits
    if memory_log_print_limits < 0:
        return
    memory_log_print_limits -= 1
    if (not dist.is_initialized()) or (rank is None) or (dist.get_rank() == rank):
        memory_allocated = current_platform.memory_allocated() / 1024**3
        memory_reserved = current_platform.memory_reserved() / 1024**3
        memory_reserved_max = current_platform.max_memory_reserved() / 1024**3
        memory_device_used = current_platform.device_memory_used() / 1024**3
        rss = cpu_memory_info().rss / 1024**3
        message = (
            f"{head}, memory allocated (GB): {memory_allocated}, memory reserved (GB): {memory_reserved}, "
            f"memory max reserved (GB): {memory_reserved_max}, rss (GB): {rss} memory device used (GB): {memory_device_used}"
        )
        logger.info(msg=message)


MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT: int = 100000


@contextmanager
def local_profiler():
    PROFILER_TIMELINE = int(os.environ.get("PROFILER_TIMELINE", "0"))
    PROFILER_MEMORY = int(os.environ.get("PROFILER_MEMORY", "0"))
    rank = int(os.environ.get("RANK", "0"))
    func_name = os.environ.get("roll_EXEC_FUNC_NAME", None)

    if (PROFILER_MEMORY or PROFILER_TIMELINE) and rank == 0:
        worker_name = os.environ.get("WORKER_NAME", "DRIVER")
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

        profiler_output_dir = os.path.join(
            os.environ.get("PROFILER_OUTPUT_DIR", "./output/profiler"), f"{worker_name}", func_name
        )
        os.makedirs(profiler_output_dir, exist_ok=True)
        logger.info(f"Profiler output directory {profiler_output_dir}")
        if PROFILER_TIMELINE:
            with torch.profiler.profile(
                on_trace_ready=torch.profiler.tensorboard_trace_handler(profiler_output_dir),
                record_shapes=False,
                profile_memory=False,
                with_stack=True,
            ) as prof:
                yield

            return

        elif PROFILER_MEMORY:
            current_platform.memory._record_memory_history(max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT, stacks="python")

            yield

            current_platform.memory._dump_snapshot(os.path.join(profiler_output_dir, f"snapshot_{current_time}.pickle"))
            current_platform.memory._record_memory_history(enabled=None)
    else:
        yield


@contextmanager
def gpu_memory_offload_profiler(metrics, metric_infix, stage):
    memory_start_offload = current_platform.device_memory_used() / 1024**3
    yield
    memory_end_offload = current_platform.device_memory_used() / 1024**3
    metrics[f"memory/{metric_infix}/{stage}"] = abs(memory_end_offload - memory_start_offload)


def get_load_exclude_kwargs(load_kwargs):
    assert load_kwargs.get("include", None) is not None
    exclude_kwargs = copy.deepcopy(load_kwargs)
    exclude_kwargs["include"] = list(
        {OffloadStateType.model_params, OffloadStateType.other_params, OffloadStateType.optimizer_states}
        - set(load_kwargs.get("include"))
    )
    return exclude_kwargs


def cpu_memory_info():
    pid = os.getpid()
    process = psutil.Process(pid)
    memory_info = process.memory_info()
    return memory_info


_offload_debug_logged: set = set()


def log_offload_debug(head: str, store_stats: Optional[dict] = None, once_id: Optional[str] = None) -> None:
    """Log process RSS (and offload-store stats) for offload debugging.

    Only logs in roll debug mode. once_id logs at most once per process.
    """
    if not is_roll_debug_mode():
        return
    if once_id is not None:
        if once_id in _offload_debug_logged:
            return
        _offload_debug_logged.add(once_id)
    try:
        rss_gb = cpu_memory_info().rss / 1024**3
    except Exception:
        return
    msg = f"[OffloadDebug] {head}: rss={rss_gb:.2f}GB"
    if store_stats:
        msg += f", store={store_stats}"
    get_logger().info(msg)


def _get_gpu_memory_metrics(metric_infix: str, stage: str, with_max_frac: bool = False) -> Dict:
    if not is_roll_debug_mode():
        return {}

    metrics = {}
    for device_id in range(current_platform.device_count()):
        metrics[f"memory/{metric_infix}/{stage}/allocated/{device_id}"] = (
            current_platform.memory_allocated(device_id) / 1024**3
        )
        metrics[f"memory/{metric_infix}/{stage}/reserved/{device_id}"] = (
            current_platform.memory_reserved(device_id) / 1024**3
        )
        metrics[f"memory/{metric_infix}/{stage}/max_allocated/{device_id}"] = (
            current_platform.max_memory_allocated(device_id) / 1024**3
        )
        metrics[f"memory/{metric_infix}/{stage}/max_reserved/{device_id}"] = (
            current_platform.max_memory_reserved(device_id) / 1024**3
        )

        if with_max_frac:
            total_cuda_memory = current_platform.mem_get_info(device_id)[1]
            metrics[f"memory/{metric_infix}/{stage}/max_allocated_frac/{device_id}"] = (
                current_platform.max_memory_allocated(device_id) / total_cuda_memory
            )
            metrics[f"memory/{metric_infix}/{stage}/max_reserved_frac/{device_id}"] = (
                current_platform.max_memory_reserved(device_id) / total_cuda_memory
            )
    return metrics


def _get_cpu_memory_metrics(metric_infix: str, stage: str) -> Dict:
    if not is_roll_debug_mode():
        return {}
    memory_info = cpu_memory_info()
    metrics = {
        f"memory/cpu/{metric_infix}/{stage}/rss": memory_info.rss / 1024**3,
        f"memory/cpu/{metric_infix}/{stage}/vms": memory_info.vms / 1024**3,
    }
    container_usage_gb, container_limit_gb = get_container_memory_gb()
    if container_usage_gb is not None:
        metrics[f"memory/container/{metric_infix}/{stage}/usage"] = container_usage_gb
        metrics[f"memory/container/{metric_infix}/{stage}/limit"] = container_limit_gb
        metrics[f"memory/container/{metric_infix}/{stage}/util_frac"] = (
            container_usage_gb / container_limit_gb if container_limit_gb > 0 else 0.0
        )
    return metrics


def get_container_memory_gb() -> Tuple[Optional[float], Optional[float]]:
    """Read container cgroup memory usage and limit in GB.

    Supports both cgroup v2 (memory.current / memory.max) and cgroup v1
    (memory.usage_in_bytes / memory.limit_in_bytes). Returns (None, None) if
    cgroup files are not available.
    """
    usage_bytes: Optional[int] = None
    limit_bytes: Optional[int] = None

    # cgroup v2
    try:
        with open("/sys/fs/cgroup/memory.current", "r") as f:
            usage_bytes = int(f.read().strip())
        with open("/sys/fs/cgroup/memory.max", "r") as f:
            content = f.read().strip()
            limit_bytes = None if content == "max" else int(content)
    except FileNotFoundError:
        pass

    # cgroup v1 fallback
    if usage_bytes is None or limit_bytes is None:
        try:
            if usage_bytes is None:
                with open("/sys/fs/cgroup/memory/memory.usage_in_bytes", "r") as f:
                    usage_bytes = int(f.read().strip())
            if limit_bytes is None:
                with open("/sys/fs/cgroup/memory/memory.limit_in_bytes", "r") as f:
                    limit_bytes = int(f.read().strip())
        except FileNotFoundError:
            pass

    if usage_bytes is None:
        return None, None
    usage_gb = usage_bytes / 1024**3
    limit_gb = limit_bytes / 1024**3 if limit_bytes is not None else None
    return usage_gb, limit_gb


def log_container_memory_usage(head: str, logger: logging.Logger = None, rank: int = 0):
    """Log container cgroup memory usage if available."""
    _logger = logger or get_logger()
    if (not dist.is_initialized()) or (rank is None) or (dist.get_rank() == rank):
        usage_gb, limit_gb = get_container_memory_gb()
        if usage_gb is not None:
            if limit_gb is not None:
                _logger.info(
                    f"{head}, container memory usage: {usage_gb:.2f} GB / {limit_gb:.2f} GB "
                    f"({usage_gb / limit_gb:.2%})"
                )
            else:
                _logger.info(f"{head}, container memory usage: {usage_gb:.2f} GB")


@contextmanager
def state_offload_manger(strategy, metrics: Dict, metric_infix: str, is_offload_states=True, load_kwargs={}):
    """
    strategy.load_states()
    strategy.offload_states()
    为metrics埋点
    """
    previous_exec_name = os.environ.get("roll_EXEC_FUNC_NAME")
    os.environ["roll_EXEC_FUNC_NAME"] = metric_infix
    try:
        with Timer(name=f"{metric_infix}_total") as timer, local_profiler():
            with Timer(name=f"{metric_infix}_onload") as onload_timer, profiling.profile("load_states"):
                for device_id in range(current_platform.device_count()):
                    current_platform.reset_max_memory_allocated(device_id)
                    current_platform.reset_max_memory_cached(device_id)
                    current_platform.reset_peak_memory_stats(device_id)

                metrics.update(_get_gpu_memory_metrics(metric_infix, "start/offload"))

                log_gpu_memory_usage(head=f"{metric_infix}_start_offload", logger=logger, rank=None)
                log_container_memory_usage(head=f"{metric_infix}_start_offload", logger=logger, rank=None)
                strategy.load_states(**load_kwargs)
                if load_kwargs.get("include", None) is not None:
                    strategy.offload_states(**get_load_exclude_kwargs(load_kwargs))
                if strategy.offload_nccl:
                    with Timer(f"{metric_infix}_resume_nccl") as resume_nccl_timer, gpu_memory_offload_profiler(
                        metrics, metric_infix, "resume_nccl"
                    ):
                        resume_nccl_communicators()
                    metrics[f"time/{metric_infix}/resume_nccl"] = resume_nccl_timer.last
                log_gpu_memory_usage(head=f"{metric_infix}_start_onload", logger=logger, rank=None)
                log_container_memory_usage(head=f"{metric_infix}_start_onload", logger=logger, rank=None)

                metrics.update(_get_gpu_memory_metrics(metric_infix, "start/onload"))
                metrics.update(_get_cpu_memory_metrics(metric_infix, "start"))

            try:
                with Timer(name=f"{metric_infix}_execute") as execute_timer, profiling.profile("execute"):
                    yield

            finally:
                with Timer(name=f"{metric_infix}_offload") as offload_timer, profiling.profile("offload_states"):
                    metrics.update(_get_gpu_memory_metrics(metric_infix, "end/onload", with_max_frac=True))

                    log_gpu_memory_usage(head=f"{metric_infix}_end_onload", logger=logger, rank=None)
                    log_container_memory_usage(head=f"{metric_infix}_end_onload", logger=logger, rank=None)
                    if is_offload_states:
                        current_platform.clear_cublas_workspaces()
                        strategy.offload_states()
                        if strategy.offload_nccl:
                            with Timer(f"{metric_infix}_suspend_nccl") as suspend_nccl_timer, gpu_memory_offload_profiler(
                                metrics, metric_infix, "suspend_nccl"
                            ):
                                suspend_nccl_communicators()
                            metrics[f"time/{metric_infix}/suspend_nccl"] = suspend_nccl_timer.last
                    log_gpu_memory_usage(head=f"{metric_infix}_end_offload", logger=logger, rank=None)
                    log_container_memory_usage(head=f"{metric_infix}_end_offload", logger=logger, rank=None)

                    metrics.update(_get_gpu_memory_metrics(metric_infix, "end/offload"))
                    metrics.update(_get_cpu_memory_metrics(metric_infix, "end"))

        metrics[f"time/{metric_infix}/total"] = timer.last
        if is_roll_debug_mode():
            metrics[f"time/{metric_infix}/execute"] = execute_timer.last
            metrics[f"time/{metric_infix}/onload"] = onload_timer.last
            metrics[f"time/{metric_infix}/offload"] = offload_timer.last
    finally:
        if previous_exec_name is None:
            os.environ.pop("roll_EXEC_FUNC_NAME", None)
        else:
            os.environ["roll_EXEC_FUNC_NAME"] = previous_exec_name



@contextmanager
def disable_gradients(models: List[torch.nn.Module]):
    param_require_grad = {}
    if not torch.is_grad_enabled():
        for model in models:
            for param in model.parameters():
                param_require_grad[param] = param.requires_grad
                param.requires_grad_(False)
    try:

        yield

    finally:
        if not torch.is_grad_enabled():
            for model in models:
                for param in model.parameters():
                    param.requires_grad_(param_require_grad[param])
