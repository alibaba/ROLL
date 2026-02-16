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


def apply_vision_dp_patch():
    """Patch VisionTransformer.forward for Vision Data Parallel.

    Distributes whole images across Ulysses SP ranks for parallelized ViT computation.
    Each rank processes 1/sp_size of images, then all-gathers embeddings.

    This reduces ViT peak memory by ~sp_size x (e.g. SP=4 -> ~4x reduction).
    """
    from .vision_dp import create_dp_vision_forward

    # Patch Qwen2-VL VisionTransformer
    try:
        from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VisionTransformerPretrainedModel

        original = Qwen2VisionTransformerPretrainedModel.forward
        _original_vision_forwards["qwen2_vl"] = original
        Qwen2VisionTransformerPretrainedModel.forward = create_dp_vision_forward(original)
        logger.info("Monkey patch Qwen2VisionTransformerPretrainedModel.forward for Vision DP")
    except ImportError as e:
        logger.debug(f"Qwen2-VL not available for Vision DP patch: {e}")

    # Patch Qwen2.5-VL VisionTransformer
    try:
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
            Qwen2_5_VisionTransformerPretrainedModel,
        )

        original = Qwen2_5_VisionTransformerPretrainedModel.forward
        _original_vision_forwards["qwen2_5_vl"] = original
        Qwen2_5_VisionTransformerPretrainedModel.forward = create_dp_vision_forward(original)
        logger.info("Monkey patch Qwen2_5_VisionTransformerPretrainedModel.forward for Vision DP")
    except ImportError as e:
        logger.debug(f"Qwen2.5-VL not available for Vision DP patch: {e}")

    # Patch Qwen3-VL VisionModel
    try:
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

        original = Qwen3VLVisionModel.forward
        _original_vision_forwards["qwen3_vl"] = original
        Qwen3VLVisionModel.forward = create_dp_vision_forward(original)
        logger.info("Monkey patch Qwen3VLVisionModel.forward for Vision DP")
    except ImportError as e:
        logger.debug(f"Qwen3-VL not available for Vision DP patch: {e}")

    # Patch Qwen3-VL-MoE VisionModel
    try:
        from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeVisionModel

        original = Qwen3VLMoeVisionModel.forward
        _original_vision_forwards["qwen3_vl_moe"] = original
        Qwen3VLMoeVisionModel.forward = create_dp_vision_forward(original)
        logger.info("Monkey patch Qwen3VLMoeVisionModel.forward for Vision DP")
    except ImportError as e:
        logger.debug(f"Qwen3-VL-MoE not available for Vision DP patch: {e}")


def unapply_vision_dp_patch():
    """Restore original VisionTransformer.forward methods."""
    if "qwen2_vl" in _original_vision_forwards:
        try:
            from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VisionTransformerPretrainedModel

            Qwen2VisionTransformerPretrainedModel.forward = _original_vision_forwards.pop("qwen2_vl")
        except ImportError:
            pass

    if "qwen2_5_vl" in _original_vision_forwards:
        try:
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
                Qwen2_5_VisionTransformerPretrainedModel,
            )

            Qwen2_5_VisionTransformerPretrainedModel.forward = _original_vision_forwards.pop("qwen2_5_vl")
        except ImportError:
            pass

    if "qwen3_vl" in _original_vision_forwards:
        try:
            from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

            Qwen3VLVisionModel.forward = _original_vision_forwards.pop("qwen3_vl")
        except ImportError:
            pass

    if "qwen3_vl_moe" in _original_vision_forwards:
        try:
            from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeVisionModel

            Qwen3VLMoeVisionModel.forward = _original_vision_forwards.pop("qwen3_vl_moe")
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
