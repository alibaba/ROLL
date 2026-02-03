from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model

from roll.utils.logging import get_logger
from roll.utils.packages import is_transformers_version_greater_than

logger = get_logger()


old_flash_attention_forward = ALL_ATTENTION_FUNCTIONS["flash_attention_2"]
if not is_transformers_version_greater_than("4.53.0"):
    old_update_causal_mask = Qwen2Model._update_causal_mask
else:
    old_update_causal_mask = None


def apply_ulysses_patch():
    from .ulysses_attention import _flash_attention_forward, _update_causal_mask

    if not is_transformers_version_greater_than("4.53.0"):
        ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = _flash_attention_forward
        Qwen2Model._update_causal_mask = _update_causal_mask
        return _flash_attention_forward, _update_causal_mask
    else:
        from .hf_flash_attention_patch import apply_hf_flash_attention_ulysses_patch

        patch_info = apply_hf_flash_attention_ulysses_patch()
        if not patch_info.get("patched", False):
            logger.warning(
                "Failed to apply ulysses_attention patching for transformers>=4.53.0 "
                "(no FlashAttention2 hook patched)."
            )
            return None
        logger.info(f"Applied ulysses_attention patching for transformers>=4.53.0: {patch_info.get('targets')}")
        return patch_info


def unapply_ulysses_patch():
    global old_flash_attention_forward, old_update_causal_mask
    ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = old_flash_attention_forward
    if not is_transformers_version_greater_than("4.53.0"):
        Qwen2Model._update_causal_mask = old_update_causal_mask
    else:
        try:
            from .hf_flash_attention_patch import unapply_hf_flash_attention_ulysses_patch

            unapply_hf_flash_attention_ulysses_patch()
        except Exception:
            pass
