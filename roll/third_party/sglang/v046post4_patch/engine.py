import os
import multiprocessing as mp

import sglang.srt.entrypoints.engine as engine_module
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import (
    maybe_set_triton_cache_manager,
    set_prometheus_multiproc_dir,
    set_ulimit,
)


# Remove signal handler. singla.signal in python can only run in MainThread which fails when using Ray Async Actor.
def _set_envs_and_config(server_args: ServerArgs):
    # Set global environments
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["NCCL_CUMEM_ENABLE"] = "0"
    os.environ["NCCL_NVLS_ENABLE"] = str(int(server_args.enable_nccl_nvls))
    os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "4"
    os.environ["CUDA_MODULE_LOADING"] = "AUTO"

    # Set prometheus env vars
    if server_args.enable_metrics:
        set_prometheus_multiproc_dir()

    # Set ulimit
    set_ulimit()

    # Fix triton bugs
    if server_args.tp_size * server_args.dp_size > 1:
        # FIXME: remove this after https://github.com/triton-lang/triton/pull/4295 is used as a dependency.
        maybe_set_triton_cache_manager()

    # Set mp start method
    mp.set_start_method("spawn", force=True)

def run_scheduler_process(*args, **kwargs):
    from roll.third_party.sglang import fp8
    fp8.monkey_patch_fp8()

    from sglang.srt.managers.scheduler import run_scheduler_process
    return run_scheduler_process(*args, **kwargs)

def run_data_parallel_controller_process(*args, **kwargs):
    import sys
    sys.modules['sglang.srt.managers.data_parallel_controller'].__dict__['run_scheduler_process'] = run_scheduler_process

    from sglang.srt.managers.data_parallel_controller import run_data_parallel_controller_process
    return run_data_parallel_controller_process(*args, **kwargs)

class _roll_launch_subprocesses(object):
    def __init__(self, _launch_subprocesses):
        self._launch_subprocesses = _launch_subprocesses
    
    def __call__(self, *args, **kwargs):
        import sys

        sys.modules['sglang.srt.entrypoints.engine'].__dict__['_set_envs_and_config'] = _set_envs_and_config
        sys.modules['sglang.srt.entrypoints.engine'].__dict__['run_scheduler_process'] = run_scheduler_process
        sys.modules['sglang.srt.entrypoints.engine'].__dict__['run_data_parallel_controller_process'] = run_data_parallel_controller_process
        return self._launch_subprocesses(*args, **kwargs)


engine_module._launch_subprocesses = _roll_launch_subprocesses(engine_module._launch_subprocesses)
