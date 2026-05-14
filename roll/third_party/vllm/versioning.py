from packaging.version import Version


_VLLM_0_11_MIN = Version("0.11.0")
_VLLM_0_12_MIN = Version("0.12.0")
_VLLM_0_11_V0_RAY_EXECUTOR_MAX = Version("0.11.2")


def uses_vllm_0_11_adapter(version: str) -> bool:
    parsed = Version(version)
    return _VLLM_0_11_MIN <= parsed < _VLLM_0_12_MIN


def supports_vllm_0_11_v0_ray_executor(version: str) -> bool:
    parsed = Version(version)
    return _VLLM_0_11_MIN <= parsed < _VLLM_0_11_V0_RAY_EXECUTOR_MAX


def load_process_weights_after_loading_utils():
    from vllm.model_executor.model_loader.utils import process_weights_after_loading

    try:
        from vllm.model_executor.model_loader.utils import set_default_torch_dtype
    except ImportError:
        from vllm.utils.torch_utils import set_default_torch_dtype

    return process_weights_after_loading, set_default_torch_dtype
