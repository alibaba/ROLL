from dataclasses import dataclass, field
from typing import Optional

from transformers import PretrainedConfig

from ..auto.config_auto import register_config
from ..model_config import McaModelConfig


@register_config("qwen3_5")
@dataclass
class Qwen3_5Config(McaModelConfig):
    """Qwen3_5Config"""

    # Gated Delta Net specific (for linear attention layers)
    layer_types: Optional[list[str]] = None

    # Vision specific
    vision_start_token_id: int = 248053
    vision_end_token_id: int = 248054
    vision_token_id: int = 248055
    image_token_id: int = 248056
    video_token_id: int = 248057
    vision_config: Optional[dict] = field(
        default=None,
        metadata={"help": "Vision model config."},
    )
    rope_scaling: Optional[dict] = field(
        default=None,
        metadata={"help": "Rope scaling."},
    )

    def __post_init__(self):
        super().__post_init__()
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig

        if isinstance(self.vision_config, PretrainedConfig):
            self.vision_config = self.vision_config.to_dict()
        vision_config_obj = Qwen3_5VisionConfig(**self.vision_config)
        self.merge_size = vision_config_obj.spatial_merge_size
        self.pixel_values_dim = (
            vision_config_obj.patch_size
            * vision_config_obj.patch_size
            * vision_config_obj.in_channels
            * vision_config_obj.temporal_patch_size
        )  # 1176
        self.mrope_section = self.rope_scaling.get("mrope_section")
        self.rotary_base = self.rope_scaling.get("rope_theta")
        self.rotary_percent = self.rope_scaling.get("partial_rotary_factor")

        assert self.hidden_dropout == 0.0, "hidden dropout is Not supported for qwen3_5 yet."
        if self.layer_types is None:
            self.layer_types = [
                "linear_attention" if bool((i + 1) % self.linear_attention_freq) else "full_attention"
                for i in range(self.num_layers)
            ]
