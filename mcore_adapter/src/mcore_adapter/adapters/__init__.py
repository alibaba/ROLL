from ..utils import get_logger, is_peft_available


logger = get_logger(__name__)

if is_peft_available():
    from .lora_layer import apply_megatron_lora
    from .utils import (
        find_all_embedding_modules,
        find_all_linear_modules,
        find_all_router_modules,
        set_linear_is_expert,
    )
else:

    def apply_megatron_lora(*args, **kwargs):
        raise ValueError("PEFT is not available. Please install PEFT to use LoRA adapters.")

    def find_all_linear_modules(model):
        raise ValueError("PEFT is not available. Please install PEFT to use LoRA adapters.")

    def find_all_embedding_modules(model):
        raise ValueError("PEFT is not available. Please install PEFT to use LoRA adapters.")

    def find_all_router_modules(model):
        raise ValueError("PEFT is not available. Please install PEFT to use LoRA adapters.")


__all__ = [
    "apply_megatron_lora",
    "find_all_linear_modules",
    "find_all_embedding_modules",
    "find_all_router_modules",
    "set_linear_is_expert",
]
