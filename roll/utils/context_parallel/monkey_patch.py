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
        logger.warning("Currently, ulysses_attention patching is not supported for transformers>=4.53.0")
        return None


def unapply_ulysses_patch():
    global old_flash_attention_forward, old_update_causal_mask
    ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = old_flash_attention_forward
    if not is_transformers_version_greater_than("4.53.0"):
        Qwen2Model._update_causal_mask = old_update_causal_mask
