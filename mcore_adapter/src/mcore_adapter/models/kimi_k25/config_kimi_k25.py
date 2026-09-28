from dataclasses import dataclass, field
from typing import Optional

from transformers import PretrainedConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from ..auto.config_auto import register_config
from ..model_config import MLAMcaModelConfig


@register_config("kimi_k25")
@dataclass
class KimiK2_5Config(MLAMcaModelConfig):
    vision_config: Optional[dict] = field(
        default=None,
        metadata={"help": "Vision model config."},
    )
    media_placeholder_token_id: int = 163605

    def __post_init__(self):
        super().__post_init__()
        self.hidden_dropout = 0.0
        self.attention_dropout = 0.0
        hf_vision_config_class = get_class_from_dynamic_module(
            "configuration_kimi_k25.KimiK25VisionConfig",
            self.name_or_path,
        )
        if isinstance(self.vision_config, PretrainedConfig):
            self.vision_config = self.vision_config.to_dict()

        self.hf_vision_config = hf_vision_config_class(**self.vision_config)
        self.merge_kernel_size = tuple(self.hf_vision_config.merge_kernel_size)
        self.vision_token_compress = self.merge_kernel_size[0] * self.merge_kernel_size[1]
        # prefer the value from HF vision config if available
        if hasattr(self.hf_vision_config, "media_placeholder_token_id"):
            self.media_placeholder_token_id = self.hf_vision_config.media_placeholder_token_id
