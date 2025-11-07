from .platform import Platform
from ..utils.logging import get_logger

import torch

logger = get_logger()


class UnknownPlatform(Platform):
    device_name: str = "UNKNOWN"
    device_type: str = "cuda"
    dispatch_key: str = "CUDA"
    ray_device_key: str = "GPU"
    device_control_env_var: str = "CUDA_VISIBLE_DEVICES"
    ray_experimental_noset: str = "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"
    communication_backend: str = "nccl"

    @classmethod
    def clear_cublas_workspaces(cls) -> None:
        torch._C._cuda_clearCublasWorkspaces()

    @classmethod
    def set_allocator_settings(cls, env: str) -> None:
        torch.cuda.memory._set_allocator_settings(env)

    @classmethod
    def get_custom_env_vars(cls) -> dict:
        env_vars = {
            # This is a following temporiary fix for starvation of plasma lock at
            # https://github.com/ray-project/ray/pull/16408#issuecomment-861056024.
            # When the system is overloaded (rpc queueing) and can not pull Object from remote in a short period
            # (e.g. DynamicSampliningScheduler.report_response using ray.get inside Threaded Actor), the minimum
            # 1000ms batch timeout can still starve others (e.g. Release in callback of PinObjectIDs, reported here
            # https://github.com/ray-project/ray/pull/16402#issuecomment-861222140), which in turn, will exacerbates
            # queuing of rpc.
            # So we set a small timeout for PullObjectsAndGetFromPlasmaStore to avoid holding store_client lock
            # too long.
            "RAY_get_check_signal_interval_milliseconds": "1",
            "VLLM_ALLOW_INSECURE_SERIALIZATION":"1",
            "TORCHINDUCTOR_COMPILE_THREADS": "2",
            "RAY_CGRAPH_get_timeout": '600',
        }
        return env_vars
    
    @classmethod
    def get_vllm_worker_class(cls):
        try:
            from vllm import envs

            if envs.VLLM_USE_V1:
                from vllm.v1.worker.gpu_worker import Worker

                logger.info("Successfully imported vLLM V1 Worker.")
                return Worker
            else:
                from vllm.worker.worker import Worker

                logger.info("Successfully imported vLLM V0 Worker.")
                return Worker
        except ImportError as e:
            logger.error("Failed to import vLLM Worker. Make sure vLLM is installed correctly: %s", e)
            raise RuntimeError("vLLM is not installed or not properly configured.") from e

    @classmethod
    def get_vllm_run_time_env_vars(cls, gpu_rank:str) -> dict:
        env_vars = {
            "PYTORCH_CUDA_ALLOC_CONF" : "",
            "VLLM_ALLOW_INSECURE_SERIALIZATION":"1",
        }
        return env_vars
    
    @classmethod
    def apply_ulysses_patch(cls) -> None:
        from roll.utils.context_parallel import apply_ulysses_patch
        return apply_ulysses_patch()
