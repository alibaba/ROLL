from dataclasses import dataclass
from typing import Optional

from ..auto.config_auto import register_config
from ..model_config import McaModelConfig


@register_config("qwen3_next")
@dataclass
class Qwen3NextConfig(McaModelConfig):
    """Qwen3NextConfig"""

    # Gated Delta Net specific (for linear attention layers)
    layer_types: Optional[list[str]] = None

    def __post_init__(self):
        super().__post_init__()
        if self.layer_types is None:
            self.layer_types = [
                "linear_attention" if bool((i + 1) % self.linear_attention_freq) else "full_attention"
                for i in range(self.num_layers)
            ]
