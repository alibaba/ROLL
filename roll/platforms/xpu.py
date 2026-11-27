import os

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
    assigned_devices_env_var: str = "ROLL_ASSIGNED_XPU_IDS"
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
    def update_env_vars_for_visible_devices(cls, env_vars: dict, gpu_ranks: list) -> None:
        # Keep node-local XPUs visible so XCCL ranks share a consistent device view; each worker selects its XPU.
        visible_devices_env_vars = {
            cls.assigned_devices_env_var: ",".join(map(str, gpu_ranks)),
            # Compatibility for components that derive their device from LOCAL_RANK.
            "LOCAL_RANK": str(gpu_ranks[0]),
            cls.ray_experimental_noset: "1",
        }
        env_vars.update(visible_devices_env_vars)

    @classmethod
    def get_visible_gpus(cls) -> list:
        return os.environ.get(cls.assigned_devices_env_var, "").split(",")

    @classmethod
    def select_assigned_devices(cls) -> None:
        assigned = os.environ.get(cls.assigned_devices_env_var)
        if not assigned:
            return
        if os.environ.get(cls.device_control_env_var):
            raise RuntimeError(
                f"{cls.device_control_env_var}={os.environ[cls.device_control_env_var]} is not supported with "
                "explicit XPU placement because ROLL cannot verify consistent XPU visibility across ranks; "
                f"unset it to use the assigned XPUs ({cls.assigned_devices_env_var}={assigned})"
            )
        device = int(assigned.split(",")[0])
        torch.xpu.set_device(device)
        logger.info(
            f"XPU placement: rank={os.environ.get('RANK')} assigned={assigned} "
            f"current_device={device} uuid={torch.xpu.get_device_properties(device).uuid}"
        )

    @classmethod
    def get_vllm_device_kwargs(cls, engine_kwargs: dict) -> dict:
        assigned = os.environ.get(cls.assigned_devices_env_var)
        if not assigned:
            return {}
        import vllm
        from packaging.version import Version

        if Version(vllm.__version__) < Version("0.31.0"):
            raise RuntimeError(f"vLLM {vllm.__version__} ignores device_ids on XPU; vLLM >= 0.31 is required")
        device_ids = [int(i) for i in assigned.split(",")]
        if int(engine_kwargs.get("data_parallel_size") or 1) > 1:
            # vLLM's XPU worker picks its slot in device_ids from the DP rank; place the assigned XPUs in that slot.
            device_count = torch.xpu.device_count()
            slots = device_count // len(device_ids)
            start = int(engine_kwargs.get("data_parallel_rank") or 0) % slots * len(device_ids)
            others = [i for i in range(device_count) if i not in device_ids]
            device_ids = others[:start] + device_ids + others[start:]
        return {"device_ids": device_ids}

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
