import ctypes
import ctypes.util
import os
from threading import Lock

import torch

from ..utils.logging import get_logger

logger = get_logger()
_device_memory_used_fallback_warned = set()
_device_memory_used_compat_warned = set()
_nvml_compat_lib = None
_nvml_compat_lib_name = None
_nvml_compat_lib_lock = Lock()


class _NvmlMemoryInfo(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


def _nvml_compatible_library_candidates():
    candidates = []

    env_library = os.environ.get("ROLL_NVML_COMPAT_LIB")
    if env_library:
        candidates.append(env_library)

    for library_name in (
        ctypes.util.find_library("nvidia-ml"),
        "libnvidia-ml.so.1",
        ctypes.util.find_library("ixml"),
        "libixml.so",
    ):
        if library_name and library_name not in candidates:
            candidates.append(library_name)

    return candidates


def _load_nvml_compatible_library():
    global _nvml_compat_lib, _nvml_compat_lib_name

    if _nvml_compat_lib is not None:
        return _nvml_compat_lib, _nvml_compat_lib_name

    with _nvml_compat_lib_lock:
        if _nvml_compat_lib is not None:
            return _nvml_compat_lib, _nvml_compat_lib_name

        errors = []
        for library_name in _nvml_compatible_library_candidates():
            try:
                library = ctypes.CDLL(library_name)
            except OSError as exc:
                errors.append(f"{library_name}: {exc}")
                continue

            init_fn = getattr(library, "nvmlInit_v2", None) or getattr(library, "nvmlInit", None)
            if init_fn is None:
                errors.append(f"{library_name}: missing nvmlInit_v2/nvmlInit")
                continue

            init_fn.restype = ctypes.c_int
            ret = init_fn()
            if ret != 0:
                error_string = _nvml_error_string(library, ret)
                errors.append(f"{library_name}: nvmlInit failed with {ret} ({error_string})")
                continue

            _nvml_compat_lib = library
            _nvml_compat_lib_name = library_name
            return _nvml_compat_lib, _nvml_compat_lib_name

        error_msg = "; ".join(errors) if errors else "no candidate libraries"
        raise RuntimeError(f"failed to load a NVML-compatible library: {error_msg}")


def _nvml_error_string(library, retcode: int) -> str:
    error_fn = getattr(library, "nvmlErrorString", None)
    if error_fn is None:
        return "unknown"

    error_fn.argtypes = [ctypes.c_int]
    error_fn.restype = ctypes.c_char_p
    try:
        value = error_fn(retcode)
    except Exception:
        return "unknown"

    if not value:
        return "unknown"

    try:
        return value.decode("utf-8", errors="replace")
    except Exception:
        return str(value)


def _map_visible_device_index(device: int, env_var: str) -> int:
    visible_devices = os.environ.get(env_var, "")
    if not visible_devices:
        return device

    parts = [part.strip() for part in visible_devices.split(",") if part.strip()]
    if not parts or device < 0 or device >= len(parts):
        return device

    try:
        return int(parts[device])
    except ValueError:
        return device


def _nvml_compatible_device_memory_used(device: int, env_var: str) -> tuple[int, str]:
    library, library_name = _load_nvml_compatible_library()
    physical_device = _map_visible_device_index(device, env_var)

    handle = ctypes.c_void_p()
    get_handle = getattr(library, "nvmlDeviceGetHandleByIndex_v2", None) or getattr(
        library, "nvmlDeviceGetHandleByIndex", None
    )
    if get_handle is None:
        raise RuntimeError(f"{library_name} does not export nvmlDeviceGetHandleByIndex")

    get_handle.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)]
    get_handle.restype = ctypes.c_int
    ret = get_handle(physical_device, ctypes.byref(handle))
    if ret != 0:
        raise RuntimeError(
            f"{library_name} nvmlDeviceGetHandleByIndex({physical_device}) failed with {ret} "
            f"({_nvml_error_string(library, ret)})"
        )

    memory_info = _NvmlMemoryInfo()
    get_memory_info = getattr(library, "nvmlDeviceGetMemoryInfo", None)
    if get_memory_info is None:
        raise RuntimeError(f"{library_name} does not export nvmlDeviceGetMemoryInfo")

    get_memory_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(_NvmlMemoryInfo)]
    get_memory_info.restype = ctypes.c_int
    ret = get_memory_info(handle, ctypes.byref(memory_info))
    if ret != 0:
        raise RuntimeError(
            f"{library_name} nvmlDeviceGetMemoryInfo({physical_device}) failed with {ret} "
            f"({_nvml_error_string(library, ret)})"
        )

    return int(memory_info.used), library_name


class Platform:
    """
    A unified abstraction for different hardware platforms (e.g., GPU, NPU).

    Design Overview
    ----------------
    1. Device-Agnostic Abstraction
       Hardware platforms differ in how they are registered with PyTorch or Ray.
       This class standardizes platform metadata such as `dispatch_key`, `ray_device_key`,
       and device visibility control variables to simplify cross-platform scheduling.

    2. Lazy Attribute Access
       Dynamically delegates unknown attributes to the `torch.<device_type>` submodule.
       This provides clean access to device-specific PyTorch APIs without redundancy.

    3. Extensible Interface
       Subclasses must implement:
       - `clear_cublas_workspaces`: to release or reuse low-level library workspaces.
       - `get_vllm_worker_class`: to specify the vLLM Ray worker class.
       - `set_allocator_settings`: to configure platform-specific memory allocators.
    """

    # High-level platform name, used for readability and logging.
    # Examples: "NVIDIA", "AMD", "ASCEND"
    device_name: str

    # Corresponding torch module name
    # Examples: "cuda", "npu"
    device_type: str

    # available dispatch keys:
    # check https://github.com/pytorch/pytorch/blob/313dac6c1ca0fa0cde32477509cce32089f8532a/torchgen/model.py#L134 # noqa
    # use "CPU" as a fallback for platforms not registered in PyTorch
    # Examples: "CUDA", "PrivateUse1"
    dispatch_key: str

    # available ray device keys:
    # https://github.com/ray-project/ray/blob/10ba5adadcc49c60af2c358a33bb943fb491a171/python/ray/_private/ray_constants.py#L438 # noqa
    # empty string means the device does not support ray
    # Examples: "GPU", "NPU"
    ray_device_key: str

    # platform-agnostic way to specify the device control environment variable,
    # .e.g. CUDA_VISIBLE_DEVICES for CUDA.
    # hint: search for "get_visible_accelerator_ids_env_var" in
    # https://github.com/ray-project/ray/tree/master/python/ray/_private/accelerators # noqa
    # Examples: "CUDA_VISIBLE_DEVICES", "ASCEND_RT_VISIBLE_DEVICES"
    device_control_env_var: str

    # Optional Ray experimental config
    # Some accelerators require specific flags in Ray start parameters;
    # leave blank if not needed
    # Example: "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"
    ray_experimental_noset: str

    # Communication backend for distributed training
    # Examples: "nccl", "hccl"
    communication_backend: str

    def __getattr__(self, key: str):
        """Fallback attribute accessor for device-specific Torch modules.

        This method is called when the requested attribute `key` is not found
        in the current instance. It attempts to retrieve the attribute from
        the corresponding `torch.<device_type>` module (e.g., torch.cuda, torch.xpu).

        If the attribute exists on the device module, it returns it.
        Otherwise, it logs a warning and returns None.

        Args:
            key (str): The name of the attribute to access.

        Returns:
            Any: The requested attribute from the Torch device module if found;
                otherwise, None.
        """
        device = getattr(torch, self.device_type, None)
        if device is not None and hasattr(device, key):
            return getattr(device, key)
        else:
            logger.warning("Current platform %s does not have '%s' attribute.", self.device_type, key)
            return None

    @classmethod
    def device_memory_used(cls, device=None) -> int:
        device_module = getattr(torch, cls.device_type, None)
        if device_module is None:
            logger.warning("Current platform %s does not expose torch.%s.", cls.device_name, cls.device_type)
            return 0

        if device is None:
            try:
                device_count = device_module.device_count()
                device = device_module.current_device() if device_count > 0 else 0
            except Exception:
                device = 0

        try:
            return int(device_module.device_memory_used(device))
        except Exception as exc:
            compat_used = cls._nvml_compatible_device_memory_used(device=device, primary_exc=exc)
            if compat_used is not None:
                return compat_used
            return cls._fallback_device_memory_used(device_module=device_module, device=device, primary_exc=exc)

    @classmethod
    def _nvml_compatible_device_memory_used(cls, device: int, primary_exc: Exception):
        if cls.device_type != "cuda":
            return None

        try:
            used, library_name = _nvml_compatible_device_memory_used(device, cls.device_control_env_var)
        except Exception:
            return None

        warning_key = (cls.device_name, library_name)
        if warning_key not in _device_memory_used_compat_warned:
            _device_memory_used_compat_warned.add(warning_key)
            logger.warning(
                "torch.%s.device_memory_used is unavailable on platform %s (device %s, error: %s). "
                "Using NVML-compatible library %s instead.",
                cls.device_type,
                cls.device_name,
                device,
                primary_exc,
                library_name,
            )

        return used

    @classmethod
    def _fallback_device_memory_used(cls, device_module, device: int, primary_exc: Exception) -> int:
        fallback_candidates = (
            ("mem_get_info", lambda: _mem_get_info_used(device_module, device)),
            ("memory_reserved", lambda: int(device_module.memory_reserved(device))),
            ("memory_allocated", lambda: int(device_module.memory_allocated(device))),
        )

        errors = []
        for fallback_name, fallback_fn in fallback_candidates:
            if not hasattr(device_module, fallback_name):
                continue
            try:
                value = fallback_fn()
            except Exception as fallback_exc:
                errors.append(f"{fallback_name}: {fallback_exc}")
                continue

            cls._warn_device_memory_used_fallback_once(
                device=device,
                fallback_name=fallback_name,
                primary_exc=primary_exc,
            )
            return value

        logger.warning(
            "Failed to query device memory usage for platform %s on device %s. Primary error: %s. "
            "Fallback errors: %s",
            cls.device_name,
            device,
            primary_exc,
            "; ".join(errors) if errors else "none",
        )
        raise primary_exc

    @classmethod
    def _warn_device_memory_used_fallback_once(cls, device: int, fallback_name: str, primary_exc: Exception) -> None:
        warning_key = (cls.device_name, fallback_name)
        if warning_key in _device_memory_used_fallback_warned:
            return

        _device_memory_used_fallback_warned.add(warning_key)
        logger.warning(
            "device_memory_used is unavailable on platform %s (device %s, error: %s). "
            "Falling back to torch.%s.",
            cls.device_name,
            device,
            primary_exc,
            fallback_name,
        )

    @classmethod
    def is_cuda(cls) -> bool:
        return False

    @classmethod
    def is_npu(cls) -> bool:
        return False

    @classmethod
    def is_rocm(cls) -> bool:
        return False

    @classmethod
    def clear_cublas_workspaces(cls) -> None:
        raise NotImplementedError

    @classmethod
    def set_allocator_settings(cls, env: str) -> None:
        """Configure memory allocator settings based on the device type."""
        raise NotImplementedError

    @classmethod
    def get_common_envs(cls) -> dict:
        return {
            "TORCH_EXTENSIONS_DIR": ""
        }

    @classmethod
    def get_custom_env_vars(cls) -> dict:
        """
        Return custom environment variables specific to the platform.

        Returns:
            dict: A dictionary of environment variable key-value pairs.
        """
        raise NotImplementedError

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
            cls.device_control_env_var: ",".join(map(str, gpu_ranks)),
            cls.ray_experimental_noset: "1",
        }
        env_vars.update(visible_devices_env_vars)

    @classmethod
    def get_visible_gpus(cls) -> list:
        """
        Return the list of currently visible device IDs.

        Returns:
            list: A list of device ID strings parsed from the visibility environment variable.
        """
        if cls.device_control_env_var is not None:
            return os.environ.get(cls.device_control_env_var, "").split(",")
        return []

    @classmethod
    def get_vllm_worker_class(cls):
        """Return the custom vLLM WorkerWrapper class used in Ray."""
        raise NotImplementedError

    @classmethod
    def get_vllm_run_time_env_vars(cls, gpu_rank: str) -> dict:
        """
        Generate the runtime environment variables required for vLLM execution
        on the specified GPU rank.

        Args:
            gpu_rank (str): The rank or ID of the GPU for which to generate environment variables.

        Returns:
            dict: A dictionary mapping environment variable names to their values.
                The actual keys and values depend on the specific device control
                and runtime requirements for vLLM.

        Raises:
            NotImplementedError: This method must be implemented by subclasses to
                              provide framework-specific environment variables.
        """
        raise NotImplementedError

    @classmethod
    def apply_ulysses_patch(cls) -> None:
        """
        Apply the Ulysses attention runtime patch to the current environment.

        This method sets up or modifies the necessary environment variables, flags,
        or other runtime configurations to enable the Ulysses-optimized attention operations
        in vLLM. It ensures that models using the Ulysses attention implementation
        can run efficiently on the target hardware.

        Returns:
            dict: A dictionary containing the environment variables that were applied
                or modified to enable Ulysses attention support.

        Raises:
            NotImplementedError: This method should be implemented by subclasses to
                                provide framework- and hardware-specific Ulysses patching.
        """
        raise NotImplementedError


def _mem_get_info_used(device_module, device: int) -> int:
    free, total = device_module.mem_get_info(device)
    return int(total - free)
