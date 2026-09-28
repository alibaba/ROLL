"""Qwen-Image diffusion model package.

Exports only dependency-light classes to avoid pulling vllm/vllm_omni into
trainer-side imports. Inference-engine-specific modules must be imported
by their full path:

    from roll.pipeline.diffusion.models.qwen_image.qwen_image_transformer import QwenImageTransformer2DModelFixed
    from roll.pipeline.diffusion.models.qwen_image.vllm_omni_qwen_image_adapter import QwenImagePipelineWithLogProb
"""
from roll.pipeline.diffusion.models.qwen_image.qwen_image_adapter import QwenImageAdapter

__all__ = [
    "QwenImageAdapter",
]
