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

# Store original vision forwards for unapply
_original_vision_forwards = {}


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


def _patch_vision_class(cls, key, class_name):
    """Patch a single VisionTransformer class with Vision DP, with idempotency guard."""
    from .vision_dp import create_dp_vision_forward

    if getattr(cls, "_vision_dp_patched", False):
        return
    original = cls.forward
    _original_vision_forwards[key] = original
    cls.forward = create_dp_vision_forward(original)
    cls._vision_dp_patched = True
    logger.info(f"Monkey patch {class_name}.forward for Vision DP")


def apply_vision_dp_patch():
    """Patch VisionTransformer.forward for Vision Data Parallel.

    Distributes whole images across Ulysses SP ranks for parallelized ViT computation.
    Each rank processes 1/sp_size of images, then all-gathers embeddings.

    This reduces ViT peak memory by ~sp_size x (e.g. SP=4 -> ~4x reduction).
    Safe to call multiple times -- each class is only patched once.
    """
    # Patch Qwen2-VL VisionTransformer
    try:
        from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VisionTransformerPretrainedModel

        _patch_vision_class(Qwen2VisionTransformerPretrainedModel, "qwen2_vl", "Qwen2VisionTransformerPretrainedModel")
    except ImportError as e:
        logger.debug(f"Qwen2-VL not available for Vision DP patch: {e}")

    # Patch Qwen2.5-VL VisionTransformer
    try:
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
            Qwen2_5_VisionTransformerPretrainedModel,
        )

        _patch_vision_class(
            Qwen2_5_VisionTransformerPretrainedModel, "qwen2_5_vl", "Qwen2_5_VisionTransformerPretrainedModel"
        )
    except ImportError as e:
        logger.debug(f"Qwen2.5-VL not available for Vision DP patch: {e}")

    # Patch Qwen3-VL VisionModel
    try:
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

        _patch_vision_class(Qwen3VLVisionModel, "qwen3_vl", "Qwen3VLVisionModel")
    except ImportError as e:
        logger.debug(f"Qwen3-VL not available for Vision DP patch: {e}")


def _unapply_vision_class(cls, key):
    """Restore a single VisionTransformer class, clearing the idempotency flag."""
    if key in _original_vision_forwards:
        cls.forward = _original_vision_forwards.pop(key)
        cls._vision_dp_patched = False


def unapply_vision_dp_patch():
    """Restore original VisionTransformer.forward methods."""
    try:
        from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VisionTransformerPretrainedModel

        _unapply_vision_class(Qwen2VisionTransformerPretrainedModel, "qwen2_vl")
    except ImportError:
        pass

    try:
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
            Qwen2_5_VisionTransformerPretrainedModel,
        )

        _unapply_vision_class(Qwen2_5_VisionTransformerPretrainedModel, "qwen2_5_vl")
    except ImportError:
        pass

    try:
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

        _unapply_vision_class(Qwen3VLVisionModel, "qwen3_vl")
    except ImportError:
        pass


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
    unapply_vision_dp_patch()
