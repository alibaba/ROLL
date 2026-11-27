import torch

from ..utils.logging import get_logger
from .platform import Platform

logger = get_logger()


class XpuPlatform(Platform):
    device_name: str = "INTEL"
    device_type: str = "xpu"
    dispatch_key: str = "XPU"
    ray_device_key: str = "GPU"
    device_control_env_var: str = "ZE_AFFINITY_MASK"
    ray_experimental_noset: str = "RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR"
    communication_backend: str = "xccl"
    supports_tensor_ipc: bool = False

    @classmethod
    def is_xpu(cls) -> bool:
        return True

    @classmethod
    def clear_cublas_workspaces(cls) -> None:
        return

    @classmethod
    def set_allocator_settings(cls, env: str) -> None:
        return

    @classmethod
    def reset_max_memory_allocated(cls, device=None) -> None:
        torch.xpu.reset_peak_memory_stats(device)

    @classmethod
    def reset_max_memory_cached(cls, device=None) -> None:
        torch.xpu.reset_peak_memory_stats(device)

    @classmethod
    def get_custom_env_vars(cls) -> dict:
        env_vars = {
            **Platform.get_common_envs(),
            "RAY_get_check_signal_interval_milliseconds": "1",
            "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
            "RAY_CGRAPH_get_timeout": "600",
        }
        return env_vars

    @classmethod
    def get_vllm_worker_class(cls):
        try:
            from vllm.v1.worker.xpu_worker import XPUWorker

            logger.info("Successfully imported vLLM XPU Worker.")
            return XPUWorker
        except ImportError as e:
            logger.error("Failed to import vLLM Worker. Make sure vLLM is installed correctly: %s", e)
            raise RuntimeError("vLLM is not installed or not properly configured.") from e

    @classmethod
    def get_vllm_run_time_env_vars(cls, gpu_rank: str) -> dict:
        env_vars = {
            "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
            "ZE_AFFINITY_MASK": f"{gpu_rank}",
            "RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR": "1",
        }
        return env_vars

    @classmethod
    def apply_ulysses_patch(cls) -> None:
        return
