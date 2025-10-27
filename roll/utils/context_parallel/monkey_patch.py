from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model


old_flash_attention_forward = ALL_ATTENTION_FUNCTIONS["flash_attention_2"]
old_update_causal_mask = Qwen2Model._update_causal_mask


def apply_ulysses_patch():
    from .ulysses_attention import _flash_attention_forward, _update_causal_mask

    ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = _flash_attention_forward
    Qwen2Model._update_causal_mask = _update_causal_mask


def unapply_ulysses_patch():
    global old_flash_attention_forward, old_update_causal_mask
    ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = old_flash_attention_forward
    Qwen2Model._update_causal_mask = old_update_causal_mask
