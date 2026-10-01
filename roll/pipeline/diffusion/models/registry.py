"""Diffusion model adapter registry."""

from roll.pipeline.diffusion.models.base import DiffusionAdapter
from roll.pipeline.diffusion.models.qwen_image import QwenImageAdapter
from roll.pipeline.diffusion.models.wan import WanDiffusionAdapter


DIFFUSION_MODEL_ADAPTERS: dict[str, type[DiffusionAdapter]] = {
    "qwen_image": QwenImageAdapter,
    "wan2_1": WanDiffusionAdapter,
}

SELF_FORCING_MODULES = {
    "wan2_1": "roll.pipeline.diffusion.models.wan.wan_self_forcing",
}


def get_diffusion_model_adapter(variant: str) -> type[DiffusionAdapter]:
    """Return the adapter registered for a diffusion model variant."""
    try:
        return DIFFUSION_MODEL_ADAPTERS[variant]
    except KeyError:
        raise ValueError(
            f"Unknown diffusion_model_variant={variant!r}. Available: {list(DIFFUSION_MODEL_ADAPTERS)}"
        ) from None


def get_self_forcing_module(variant: str) -> str:
    """Return the Self-Forcing implementation registered for a model variant."""
    try:
        return SELF_FORCING_MODULES[variant]
    except KeyError:
        raise ValueError(f"Self-Forcing is not implemented for diffusion_model_variant={variant!r}") from None
