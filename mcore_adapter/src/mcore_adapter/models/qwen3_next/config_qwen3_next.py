from dataclasses import dataclass
from typing import List, Optional

from ..auto.config_auto import register_config
from ..model_config import McaModelConfig


@register_config("qwen3_next")
@dataclass
class Qwen3NextConfig(McaModelConfig):
    """Qwen3NextConfig"""
    # Gated Delta Net specific (for linear attention layers)
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32

    layer_types: Optional[List[str]] = None
    full_attention_interval: int = 4

    def __post_init__(self):
        super().__post_init__()
        assert self.tensor_model_parallel_size == 1, "Qwen3Next only supports tensor_model_parallel_size=1"
        assert self.context_parallel_size == 1, "Qwen3Next only supports context_parallel_size=1"

        if self.layer_types is None:
            self.layer_types = [
                "linear_attention"
                if bool((i + 1) % self.full_attention_interval)
                else "full_attention"
                for i in range(self.num_layers)
            ]
