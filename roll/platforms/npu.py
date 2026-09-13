from importlib import import_module

import torch

from .platform import Platform
from ..utils.logging import get_logger

logger = get_logger()


class NpuPlatform(Platform):
    device_name: str = "ASCEND"
    device_type: str = "npu"
    dispatch_key: str = "PrivateUse1"
    ray_device_key: str = "NPU"
    device_control_env_var: str = "ASCEND_RT_VISIBLE_DEVICES"
    ray_experimental_noset: str = "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES"
    communication_backend: str = "hccl"

    @classmethod
    def is_npu(cls) -> bool:
        return True

    @classmethod
    def clear_cublas_workspaces(cls) -> None:
        return

    @classmethod
    def set_allocator_settings(cls, env: str) -> None:
        return

    @classmethod
    def get_custom_env_vars(cls) -> dict:
        env_vars = {
            **Platform.get_common_envs(),
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
            "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
            "RAY_CGRAPH_get_timeout": '600',
        }
        return env_vars

    @classmethod
    def get_vllm_worker_class(cls):
        def import_worker(candidate_modules):
            errors = []
            for module_name in candidate_modules:
                try:
                    module = import_module(module_name)
                    worker = getattr(module, "NPUWorker")
                except (ImportError, AttributeError) as e:
                    errors.append(f"{module_name}: {e}")
                    continue
                logger.info("Successfully imported vLLM NPU Worker from %s.", module_name)
                return worker
            raise ImportError("; ".join(errors))

        try:
            from vllm import envs

            # VLLM_USE_V1 is deprecated in vllm>=0.11.1
            if not hasattr(envs, "VLLM_USE_V1") or envs.VLLM_USE_V1:
                return import_worker(
                    [
                        "vllm_ascend.worker.worker_v1",
                        "vllm_ascend.worker.worker",
                    ]
                )
            else:
                return import_worker(["vllm_ascend.worker.worker"])
        except ImportError as e:
            logger.error("Failed to import vLLM Worker. Make sure vLLM is installed correctly: %s", e)
            raise RuntimeError("vLLM is not installed or not properly configured.") from e

    @classmethod
    def get_vllm_run_time_env_vars(cls, gpu_rank: str) -> dict:
        env_vars = {
            "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
            "ASCEND_RT_VISIBLE_DEVICES": f"{gpu_rank}",
            "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
            # vLLM-Ascend graph/NPU worker initialization is more stable with
            # task queue mode 1; broader training jobs may use mode 2.
            "TASK_QUEUE_ENABLE": "1",
            "VLLM_ASCEND_ENABLE_NZ": "0",
            # vLLM-Ascend's memory pool is incompatible with expandable
            # segments, even if the broader NPU test job enables them.
            "PYTORCH_NPU_ALLOC_CONF": "",
        }
        return env_vars
    
    @classmethod
    def apply_ulysses_patch(cls) -> None:
        return

    @classmethod
    def device_memory_used(cls) -> int:
        free, total = torch.npu.mem_get_info()
        return total - free
