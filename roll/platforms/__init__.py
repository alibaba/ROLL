import torch

from .platform import Platform
from .cuda import CudaPlatform
from .npu import NpuPlatform
from .rocm import RocmPlatform
from .unknown import UnknownPlatform
from .cpu import CpuPlatform

from ..utils.logging import get_logger


logger = get_logger()


def _init_platform() -> Platform:
    """
    Detect and initialize the appropriate platform based on available devices.

    Priority:
    1. CUDA (NVIDIA / AMD ROCm)
    2. NPU (if torch_npu is installed)
    3. CPU (fallback)

    Returns:
        An instance of a subclass of Platform corresponding to the detected hardware.
    """
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name().upper()
        logger.debug(f"Detected CUDA device: {device_name}")
        if "NVIDIA" in device_name:
            logger.debug("Initializing CUDA platform (NVIDIA).")
            return CudaPlatform()
        elif "AMD" in device_name:
            logger.debug("Initializing ROCm platform (AMD).")
            return RocmPlatform()
        logger.warning("Unrecognized CUDA device. Falling back to UnknownPlatform.")
        return UnknownPlatform()
    else:
        try:
            import torch_npu  # noqa: F401

            logger.debug("Detected torch_npu. Initializing NPU platform.")
            return NpuPlatform()
        except ImportError:
            logger.debug("No supported accelerator detected. Initializing CPU platform.")
            return CpuPlatform()


# Global singleton representing the current platform in use.
current_platform: Platform = _init_platform()

__all__ = [
    "Platform",
    "current_platform",
]
