import torch
from enum import Enum

class DeviceType(Enum):
    NONE = "NONE"
    NVIDIA = "NVIDIA"
    AMD = "AMD"
    UNKNOWN = "UNKNOWN"

class GPUUtils:
    @staticmethod
    def get_device_type() -> DeviceType:
        if not torch.cuda.is_available():
            return DeviceType.NONE
        device_name = torch.cuda.get_device_name().upper()
        if "NVIDIA" in device_name:
            return DeviceType.NVIDIA
        if "AMD" in device_name:
            return DeviceType.AMD
        return DeviceType.UNKNOWN
    