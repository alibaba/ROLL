from roll.utils.gpu_utils import GPUUtils, DeviceType
from roll.utils.logging import get_logger

def test_get_device_tpye():
    device_type = GPUUtils.get_device_type()
    get_logger().info(f"GPU device type: {device_type}")
    assert isinstance(device_type, DeviceType)

if __name__ == "__main__":
    test_get_device_tpye()