# Copyright (c) 2025, ALIBABA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
import os
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, Literal, List

import psutil
import torch
import torch.distributed as dist
import logging

from codetiming import Timer
from ray._private import profiling

from roll.utils.offload_states import OffloadStateType
from roll.utils.logging import get_logger

logger = get_logger()

memory_log_print_limits = 20


def log_gpu_memory_usage(head: str, logger: logging.Logger = None, rank: int = 0):
    global memory_log_print_limits
    if memory_log_print_limits < 0:
        return
    memory_log_print_limits -= 1
    if (not dist.is_initialized()) or (rank is None) or (dist.get_rank() == rank):
        memory_allocated = torch.cuda.memory_allocated() / 1024**3
        memory_reserved = torch.cuda.memory_reserved() / 1024**3
        memory_reserved_max = torch.cuda.max_memory_reserved() / 1024**3
        rss = cpu_memory_info().rss / 1024**3
        message = (
            f"{head}, memory allocated (GB): {memory_allocated}, memory reserved (GB): {memory_reserved}, "
            f"memory max reserved (GB): {memory_reserved_max}, rss (GB): {rss}"
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
            torch.cuda.memory._record_memory_history(max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT, stacks="python")

            yield

            torch.cuda.memory._dump_snapshot(os.path.join(profiler_output_dir, f"snapshot_{current_time}.pickle"))
            torch.cuda.memory._record_memory_history(enabled=None)
    else:
        yield


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


@contextmanager
def state_offload_manger(strategy, metrics: Dict, metric_infix: str, is_offload_states=True, load_kwargs={}):
    """
    strategy.load_states()
    strategy.offload_states()
    为metrics埋点
    """
    os.environ["roll_EXEC_FUNC_NAME"] = metric_infix
    with Timer(name=f"{metric_infix}_total") as timer, local_profiler():
        with Timer(name=f"{metric_infix}_onload") as onload_timer, profiling.profile("load_states"):
            for device_id in range(torch.cuda.device_count()):
                torch.cuda.reset_max_memory_allocated(device_id)
                torch.cuda.reset_max_memory_cached(device_id)
                torch.cuda.reset_peak_memory_stats(device_id)
                metrics[f"memory/{metric_infix}/start/offload/allocated/{device_id}"] = (
                    torch.cuda.memory_allocated(device_id) / 1024**3
                )
                metrics[f"memory/{metric_infix}/start/offload/reserved/{device_id}"] = (
                    torch.cuda.memory_reserved(device_id) / 1024**3
                )
                metrics[f"memory/{metric_infix}/start/offload/max_allocated/{device_id}"] = (
                    torch.cuda.max_memory_allocated(device_id) / 1024**3
                )
                metrics[f"memory/{metric_infix}/start/offload/max_reserved/{device_id}"] = (
                    torch.cuda.max_memory_reserved(device_id) / 1024**3
                )

            log_gpu_memory_usage(head=f"{metric_infix}_start_offload", logger=logger, rank=None)
            strategy.load_states(**load_kwargs)
            if load_kwargs.get("include", None) is not None:
                strategy.offload_states(**get_load_exclude_kwargs(load_kwargs))
            log_gpu_memory_usage(head=f"{metric_infix}_start_onload", logger=logger, rank=None)

            for device_id in range(torch.cuda.device_count()):
                metrics[f"memory/{metric_infix}/start/onload/allocated/{device_id}"] = (
                    torch.cuda.memory_allocated(device_id) / 1024**3
                )
                metrics[f"memory/{metric_infix}/start/onload/reserved/{device_id}"] = (
                    torch.cuda.memory_reserved(device_id) / 1024**3
                )
                metrics[f"memory/{metric_infix}/start/onload/max_allocated/{device_id}"] = (
                    torch.cuda.max_memory_allocated(device_id) / 1024**3
                )
                metrics[f"memory/{metric_infix}/start/onload/max_reserved/{device_id}"] = (
                    torch.cuda.max_memory_reserved(device_id) / 1024**3
                )

            memory_info = cpu_memory_info()
            metrics[f"memory/cpu/{metric_infix}/start/rss"] = memory_info.rss / 1024**3
            metrics[f"memory/cpu/{metric_infix}/start/vms"] = memory_info.vms / 1024**3

        with Timer(name=f"{metric_infix}_execute") as execute_timer, profiling.profile("execute"):
            yield

        with Timer(name=f"{metric_infix}_offload") as offload_timer, profiling.profile("offload_states"):
            for device_id in range(torch.cuda.device_count()):
                total_cuda_memory = torch.cuda.mem_get_info(device_id)[1]
                metrics[f"memory/{metric_infix}/end/onload/allocated/{device_id}"] = (
                    torch.cuda.memory_allocated(device_id) / 1024**3
                )
                metrics[f"memory/{metric_infix}/end/onload/reserved/{device_id}"] = (
                    torch.cuda.memory_reserved(device_id) / 1024**3
                )
                metrics[f"memory/{metric_infix}/end/onload/max_allocated/{device_id}"] = (
                    torch.cuda.max_memory_allocated(device_id) / 1024**3
                )
                metrics[f"memory/{metric_infix}/end/onload/max_reserved/{device_id}"] = (
                    torch.cuda.max_memory_reserved(device_id) / 1024**3
                )
                metrics[f"memory/{metric_infix}/end/onload/max_allocated_frac/{device_id}"] = (
                    torch.cuda.max_memory_allocated(device_id) / total_cuda_memory
                )
                metrics[f"memory/{metric_infix}/end/onload/max_reserved_frac/{device_id}"] = (
                    torch.cuda.max_memory_reserved(device_id) / total_cuda_memory
                )

            log_gpu_memory_usage(head=f"{metric_infix}_end_onload", logger=logger, rank=None)
            if is_offload_states:
                torch._C._cuda_clearCublasWorkspaces()
                strategy.offload_states()
            log_gpu_memory_usage(head=f"{metric_infix}_end_offload", logger=logger, rank=None)

            for device_id in range(torch.cuda.device_count()):
                metrics[f"memory/{metric_infix}/end/offload/allocated/{device_id}"] = (
                    torch.cuda.memory_allocated(device_id) / 1024**3
                )
                metrics[f"memory/{metric_infix}/end/offload/reserved/{device_id}"] = (
                    torch.cuda.memory_reserved(device_id) / 1024**3
                )
                metrics[f"memory/{metric_infix}/end/offload/max_allocated/{device_id}"] = (
                    torch.cuda.max_memory_allocated(device_id) / 1024**3
                )
                metrics[f"memory/{metric_infix}/end/offload/max_reserved/{device_id}"] = (
                    torch.cuda.max_memory_reserved(device_id) / 1024**3
                )

            memory_info = cpu_memory_info()
            metrics[f"memory/cpu/{metric_infix}/end/rss"] = memory_info.rss / 1024**3
            metrics[f"memory/cpu/{metric_infix}/end/vms"] = memory_info.vms / 1024**3

    metrics[f"time/{metric_infix}/total"] = timer.last
    metrics[f"time/{metric_infix}/execute"] = execute_timer.last
    metrics[f"time/{metric_infix}/onload"] = onload_timer.last
    metrics[f"time/{metric_infix}/offload"] = offload_timer.last
    del os.environ["roll_EXEC_FUNC_NAME"]


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
