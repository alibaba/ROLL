from roll.utils.ray_utils import RayUtils
from roll.utils.gpu_utils import DeviceType
from roll.utils.logging import get_logger

def test_get_custom_env_env_vars():
    device_type = DeviceType.NVIDIA
    env_vars = RayUtils.get_custom_env_env_vars(device_type)
    get_logger().info(f"when use {device_type}, env_vars: {env_vars}")
    assert "PYTORCH_CUDA_ALLOC_CONF" in env_vars

    device_type = DeviceType.AMD
    env_vars = RayUtils.get_custom_env_env_vars(device_type)
    get_logger().info(f"when use {device_type}, env_vars: {env_vars}")
    assert "PYTORCH_HIP_ALLOC_CONF" in env_vars

def test_update_env_vars_for_visible_devices():
    device_type = DeviceType.AMD
    env_vars = {"ENV_ALREADY_SET": "1"}
    gpu_ranks = [0, 1]
    RayUtils.update_env_vars_for_visible_devices(env_vars, gpu_ranks, device_type)
    get_logger().info(f"when use {device_type}, env_vars: {env_vars}")
    assert "ENV_ALREADY_SET" in env_vars
    assert "HIP_VISIBLE_DEVICES" in env_vars

if __name__ == "__main__":
    test_update_env_vars_for_visible_devices()