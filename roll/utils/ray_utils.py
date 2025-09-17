import os

import ray

from roll.utils.logging import get_logger
from roll.utils.gpu_utils import GPUUtils, DeviceType

@ray.remote(num_gpus=1)
def get_visible_gpus():
    return ray.get_gpu_ids()


@ray.remote(num_gpus=1)
def get_node_rank():
    return int(os.environ.get("NODE_RANK", "0"))

class RayUtils:
    @staticmethod
    def get_custom_env_env_vars(
        device_type: DeviceType | None = None) -> dict:
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
        }
        if device_type is None:
            device_type = GPUUtils.get_device_type()
        if DeviceType.NVIDIA == device_type:
            env_vars.update({
                # "RAY_DEBUG": "legacy",
                "TORCHINDUCTOR_COMPILE_THREADS": "2",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "NCCL_CUMEM_ENABLE": "0",   # https://github.com/NVIDIA/nccl/issues/1234
                "NCCL_NVLS_ENABLE": "0",
            })
        elif DeviceType.AMD == device_type:
            env_vars.update({
                # These VLLM related enviroment variables are related to backend. maybe used afterwards.
                # "VLLM_USE_TRITON_FLASH_ATTN":"0",
                # "VLLM_ROCM_USE_AITER":"1",
                # "VLLM_ROCM_USE_AITER_MOE":"1",
                # "VLLM_ROCM_USE_AITER_ASMMOE":"1",
                # "VLLM_ROCM_USE_AITER_PAGED_ATTN":"1",
                # "RAY_DEBUG": "legacy",
                "VLLM_USE_V1":"0",
                "TORCHINDUCTOR_COMPILE_THREADS": "2",
                "PYTORCH_HIP_ALLOC_CONF": "expandable_segments:True",
                # "NCCL_DEBUG_SUBSYS":"INIT,COLL",
                # "NCCL_DEBUG":"INFO",
                # "NCCL_DEBUG_FILE":"rccl.%h.%p.log",
            })
        elif DeviceType.UNKNOWN == device_type:
            env_vars.update({
                "TORCHINDUCTOR_COMPILE_THREADS": "2",
                "HGGC_ENABLE_KERNEL_COPY": "0",
                "NCCL_PF_U2MM_HOST": "0",
            })

        # used for debug
        # env_vars["RAY_DEBUG"] = "legacy"

        get_logger().info(f"gpu is {device_type}, ray custom env_vars: {env_vars}")
        return env_vars

    @staticmethod
    def update_env_vars_for_visible_devices(
        env_vars: dict, gpu_ranks: list, device_type: DeviceType | None = None):
        visible_devices_env_vars = {}
        if device_type is None:
            device_type = GPUUtils.get_device_type()
        if DeviceType.AMD == device_type:
            visible_devices_env_vars = {
                "HIP_VISIBLE_DEVICES": ",".join(map(str, gpu_ranks)),
                "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES": "1",
                "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES": "1",
            }
        else:
            visible_devices_env_vars = {
                "CUDA_VISIBLE_DEVICES": ",".join(map(str, gpu_ranks)),
                "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
            }
        get_logger().info(f"gpu is {device_type}, update ray env_vars: {visible_devices_env_vars}")
        env_vars.update(visible_devices_env_vars)
        
    @staticmethod    
    def get_visible_gpus(device_type: DeviceType | None = None) -> list:
        if device_type is None:
            device_type = GPUUtils.get_device_type()
        if DeviceType.AMD == device_type:
            return os.environ.get("HIP_VISIBLE_DEVICES", "").split(",")
        if DeviceType.NVIDIA == device_type or DeviceType.UNKNOWN == device_type:
            return os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        return []
    
    @staticmethod
    def get_vllm_run_time_env_vars(
        gpu_rank:str,
        device_type: DeviceType | None = None) -> dict:
        env_vars = {
            "PYTORCH_CUDA_ALLOC_CONF" : "",
            "VLLM_ALLOW_INSECURE_SERIALIZATION":"1",
        }
        if device_type is None:
            device_type = GPUUtils.get_device_type()
        if DeviceType.NVIDIA == device_type:
            env_vars.update({
                    "CUDA_VISIBLE_DEVICES": f"{gpu_rank}",
                    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                })
        elif DeviceType.AMD == device_type:
            env_vars.update({
                    "HIP_VISIBLE_DEVICES": f"{gpu_rank}",
                    "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES": "1",
                    "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES": "1",
                    # "NCCL_DEBUG_SUBSYS":"INIT,COLL",
                    # "NCCL_DEBUG":"INFO",
                    # "NCCL_DEBUG_FILE":"rccl.%h.%p.log",
                    # "NCCL_P2P_DISABLE":"1",
            })
        get_logger().info(f"gpu is {device_type}, ray custom runtime env_vars: {env_vars}")
        return env_vars