from .platform import Platform
from ..utils.logging import get_logger

import torch

logger = get_logger()


class RocmPlatform(Platform):
    device_name: str = "AMD"
    device_type: str = "cuda"
    dispatch_key: str = "CUDA"
    ray_device_key: str = "GPU"
    device_control_env_var: str = "HIP_VISIBLE_DEVICES"
    ray_experimental_noset: str = "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"
    communication_backend: str = "nccl"

    @classmethod
    def is_rocm(cls) -> bool:
        return True

    @classmethod
    def clear_cublas_workspaces(cls) -> None:
        torch._C._cuda_clearCublasWorkspaces()

    @classmethod
    def set_allocator_settings(cls, env: str) -> None:
        torch.cuda.memory._set_allocator_settings(env)

    @classmethod
    def get_custom_env_vars(cls) -> dict:
        env_vars = {
            "RAY_get_check_signal_interval_milliseconds": "1",
            "RAY_CGRAPH_get_timeout": '600',
            "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
            # These VLLM related enviroment variables are related to backend. maybe used afterwards.
            # "VLLM_USE_TRITON_FLASH_ATTN":"0",
            # "VLLM_ROCM_USE_AITER":"1",
            # "VLLM_ROCM_USE_AITER_MOE":"1",
            # "VLLM_ROCM_USE_AITER_ASMMOE":"1",
            # "VLLM_ROCM_USE_AITER_PAGED_ATTN":"1",
            # "RAY_DEBUG": "legacy",
            "VLLM_USE_V1": "1",
            "TORCHINDUCTOR_COMPILE_THREADS": "2",
            "PYTORCH_HIP_ALLOC_CONF": "expandable_segments:True",
            "SAFETENSORS_FAST_GPU":"1",
            "VLLM_ROCM_USE_AITER_MHA":"0",
            "VLLM_ALLOW_LONG_MAX_MODEL_LEN":"1",
            # "NCCL_DEBUG_SUBSYS":"INIT,COLL",
            # "NCCL_DEBUG":"INFO",
            # "NCCL_DEBUG_FILE":"rccl.%h.%p.log",
        }
        return env_vars

    @classmethod
    def update_env_vars_for_visible_devices(cls, env_vars: dict, gpu_ranks: list) -> None:
        """
        Update environment variables to control device visibility.

        Args:
            env_vars (dict): Dictionary of current environment variables to modify.
            gpu_ranks (list): List of device IDs to expose to the process.

        Behavior:
            - Sets the platform-specific visibility environment variable.
            - Sets the corresponding Ray experimental flag if needed.
        """
        visible_devices_env_vars = {
            "HIP_VISIBLE_DEVICES": ",".join(map(str, gpu_ranks)),
            "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES": "1",
            "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES": "1",
        }
        env_vars.update(visible_devices_env_vars)

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
    def get_vllm_run_time_env_vars(cls, gpu_rank: str) -> dict:
        env_vars = {
            "PYTORCH_CUDA_ALLOC_CONF": "",
            "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
            "HIP_VISIBLE_DEVICES": f"{gpu_rank}",
            "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES": "1",
            "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES": "1",
            # "NCCL_DEBUG_SUBSYS":"INIT,COLL",
            # "NCCL_DEBUG":"INFO",
            # "NCCL_DEBUG_FILE":"rccl.%h.%p.log",
            # "NCCL_P2P_DISABLE":"1",
        }
        return env_vars
    
    @classmethod
    def apply_ulysses_patch(cls) -> None:
        from roll.utils.context_parallel import apply_ulysses_patch
        return apply_ulysses_patch()
