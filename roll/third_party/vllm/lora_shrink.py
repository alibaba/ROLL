"""Stable dense LoRA reductions for the Flash-Next recurrent backbone."""

import importlib
import sys
from functools import wraps


_MODEL_TYPES = {"qwen4_exp", "qwen4_exp_text", "qwen3_8_flash_next", "qwen3_8_flash_next_text"}
_SHRINK_MODULE = "vllm.lora.ops.triton_ops.lora_shrink_op"


def patch_qwen38_lora_shrink(hf_config, lora_config):
    """Disable split-K atomics in this model's dense Triton LoRA shrink.

    FP32 atomic accumulation order varies between identical requests. Flash-Next
    amplifies that small difference through its recurrent layers, changing the
    rollout log probabilities. A single K partition uses a deterministic store.
    vLLM owns one model worker per process. Reinitializing that process for a
    different configuration restores native behavior. This neither enables
    global batch invariance (unsupported by GDN) nor changes the MoE or LoRA
    expand kernels.
    """
    if lora_config is None or getattr(hf_config, "model_type", None) not in _MODEL_TYPES:
        module = sys.modules.get(_SHRINK_MODULE)
        selector = getattr(module, "get_lora_op_configs", None)
        owned = getattr(module, "_roll_qwen38_shrink_selector", None)
        if owned is not None:
            owned._roll_enabled = False
            if selector is owned:
                module.get_lora_op_configs = owned.__wrapped__
        return
    try:
        module = importlib.import_module(_SHRINK_MODULE)
    except ImportError as exc:
        raise RuntimeError("Flash-Next requires the native Triton LoRA shrink API") from exc
    original = getattr(module, "get_lora_op_configs", None)
    if not callable(original):
        raise RuntimeError("Unsupported native LoRA shrink configuration API")
    owned = getattr(module, "_roll_qwen38_shrink_selector", None)
    if original is owned:
        owned._roll_enabled = True
        return
    if owned is not None:
        # Preserve wrappers installed after us; their calls to the old selector
        # become no-ops rather than retaining a stale model configuration.
        owned._roll_enabled = False

    @wraps(original)
    def select(op_type, *args, **kwargs):
        config = original(op_type, *args, **kwargs)
        if not select._roll_enabled or op_type != "shrink":
            return config
        if not isinstance(config, dict) or "split_k" not in config:
            raise RuntimeError("Unsupported native LoRA shrink configuration: missing split_k")
        return {**config, "split_k": 1}

    select._roll_enabled = True
    module._roll_qwen38_shrink_selector = select
    module.get_lora_op_configs = select
