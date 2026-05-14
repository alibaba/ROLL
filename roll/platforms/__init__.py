import torch

from .platform import Platform
from .corex import CorexPlatform
from .cuda import CudaPlatform
from .npu import NpuPlatform
from .rocm import RocmPlatform
from .unknown import UnknownPlatform
from .cpu import CpuPlatform

from ..utils.logging import get_logger


logger = get_logger()


def _is_corex_device_name(device_name: str) -> bool:
    normalized = device_name.upper()
    return any(keyword in normalized for keyword in ("ILUVATAR", "COREX", "BI-V"))


def _init_platform() -> Platform:
    """
    Detect and initialize the appropriate platform based on available devices.

    Priority:
    1. CUDA (NVIDIA / AMD ROCm / CoreX-like stacks)
    2. NPU (if torch_npu is installed)
    3. CPU (fallback)

    Returns:
        An instance of a subclass of Platform corresponding to the detected hardware.
    """
    cuda_device_count = 0
    try:
        cuda_device_count = torch.cuda.device_count()
    except Exception as exc:
        logger.warning("Failed to query CUDA device count. Falling back to CPU/NPU detection. Error: %s", exc)

    if cuda_device_count > 0:
        try:
            device_name = torch.cuda.get_device_name(0).upper()
        except Exception as exc:
            logger.warning("Failed to query CUDA device name. Falling back to UnknownPlatform. Error: %s", exc)
            return UnknownPlatform()
        logger.debug(f"Detected CUDA device: {device_name}")
        if "NVIDIA" in device_name:
            logger.debug("Initializing CUDA platform (NVIDIA).")
            return CudaPlatform()
        elif "AMD" in device_name:
            logger.debug("Initializing ROCm platform (AMD).")
            return RocmPlatform()
        elif _is_corex_device_name(device_name):
            logger.debug("Initializing CoreX platform.")
            return CorexPlatform()
        logger.warning("Unrecognized CUDA device. Falling back to UnknownPlatform.")
        return UnknownPlatform()

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
