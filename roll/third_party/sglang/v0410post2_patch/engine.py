import os
import multiprocessing as mp

import sglang.srt.entrypoints.engine as engine_module
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import (
    set_prometheus_multiproc_dir,
    set_ulimit,
)


# Remove signal handler. singla.signal in python can only run in MainThread which fails when using Ray Async Actor.
def _set_envs_and_config(server_args: ServerArgs):
    # Set global environments
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["NCCL_CUMEM_ENABLE"] = str(int(server_args.enable_symm_mem))
    if not server_args.enable_symm_mem:
        os.environ["NCCL_NVLS_ENABLE"] = str(int(server_args.enable_nccl_nvls))
    os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "4"
    os.environ["CUDA_MODULE_LOADING"] = "AUTO"

    # Set prometheus env vars
    if server_args.enable_metrics:
        set_prometheus_multiproc_dir()

    # Set ulimit
    set_ulimit()

    # Set mp start method
    mp.set_start_method("spawn", force=True)


class _roll_launch_subprocesses(object):
    def __init__(self, _launch_subprocesses):
        self._launch_subprocesses = _launch_subprocesses
    
    def __call__(self, *args, **kwargs):
        import sys

        sys.modules['sglang.srt.entrypoints.engine'].__dict__['_set_envs_and_config'] = _set_envs_and_config
        return self._launch_subprocesses(*args, **kwargs)


engine_module._launch_subprocesses = _roll_launch_subprocesses(engine_module._launch_subprocesses)
