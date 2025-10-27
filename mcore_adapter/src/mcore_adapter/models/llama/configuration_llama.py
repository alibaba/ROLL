from dataclasses import dataclass, field
from typing import Optional

from ..auto.config_auto import register_config
from ..model_config import McaModelConfig


@register_config("llama")
@dataclass
class LlamaConfig(McaModelConfig):
    rope_scaling: Optional[dict] = field(
        default=None,
        metadata={"help": "Rope scaling."},
    )

    def __post_init__(self):
        super().__post_init__()
        self.rotary_scaling = self.rope_scaling is not None
