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
        env_vars = {}
        if device_type is None:
            device_type = GPUUtils.get_device_type()
        if DeviceType.NVIDIA == device_type:
            env_vars = {
                # "RAY_DEBUG": "legacy"
                "TORCHINDUCTOR_COMPILE_THREADS": "2",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "NCCL_CUMEM_ENABLE": "0",   # https://github.com/NVIDIA/nccl/issues/1234
                "NCCL_NVLS_ENABLE": "0",
            }
        elif DeviceType.AMD == device_type:
            env_vars = {
                "TORCHINDUCTOR_COMPILE_THREADS": "2",
                "PYTORCH_HIP_ALLOC_CONF": "expandable_segments:True",
            }
        elif DeviceType.UNKNOWN == device_type:
            env_vars = {
                "TORCHINDUCTOR_COMPILE_THREADS": "2",
            }
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