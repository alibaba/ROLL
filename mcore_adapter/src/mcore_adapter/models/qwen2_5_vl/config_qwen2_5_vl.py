from dataclasses import dataclass, field
from typing import Optional

from transformers import PretrainedConfig

from ..auto.config_auto import register_config
from ..model_config import McaModelConfig


@register_config("qwen2_5_vl")
@dataclass
class Qwen2_5_VLConfig(McaModelConfig):
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    vision_token_id: int = 151654
    image_token_id: int = 151655
    video_token_id: int = 151656
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
        from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig

        if isinstance(self.vision_config, PretrainedConfig):
            self.vision_config = self.vision_config.to_dict()
        vision_config_obj = Qwen2_5_VLVisionConfig(**self.vision_config)
        self.merge_size = vision_config_obj.spatial_merge_size
        self.tokens_per_second = vision_config_obj.tokens_per_second
        self.pixel_values_dim = (
            vision_config_obj.patch_size
            * vision_config_obj.patch_size
            * vision_config_obj.in_channels
            * vision_config_obj.temporal_patch_size
        )  # 1176
        self.mrope_section = self.rope_scaling.get("mrope_section")

        assert self.hidden_dropout == 0.0, "hidden dropout is Not supported for qwen2_5_vl yet."
