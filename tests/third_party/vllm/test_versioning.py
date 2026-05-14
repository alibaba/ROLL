import vllm

import roll.third_party.vllm as roll_vllm
from roll.third_party.vllm.versioning import (
    load_process_weights_after_loading_utils,
    supports_vllm_0_11_v0_ray_executor,
    uses_vllm_0_11_adapter,
)


def test_uses_vllm_0_11_adapter():
    assert uses_vllm_0_11_adapter("0.11.0")
    assert uses_vllm_0_11_adapter("0.11.1rc1")
    assert uses_vllm_0_11_adapter("0.11.2")
    assert not uses_vllm_0_11_adapter("0.10.2")
    assert not uses_vllm_0_11_adapter("0.12.0")


def test_supports_vllm_0_11_v0_ray_executor():
    assert supports_vllm_0_11_v0_ray_executor("0.11.0")
    assert supports_vllm_0_11_v0_ray_executor("0.11.1rc1")
    assert not supports_vllm_0_11_v0_ray_executor("0.11.2")
    assert not supports_vllm_0_11_v0_ray_executor("0.12.0")


def test_load_process_weights_after_loading_utils():
    if not uses_vllm_0_11_adapter(vllm.__version__):
        return

    process_weights_after_loading, set_default_torch_dtype = load_process_weights_after_loading_utils()
    assert callable(process_weights_after_loading)
    assert callable(set_default_torch_dtype)


def test_roll_vllm_module_exposes_v1_ray_executor():
    if not uses_vllm_0_11_adapter(vllm.__version__):
        return

    assert roll_vllm.ray_executor_class_v1 is not None
