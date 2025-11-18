from dataclasses import dataclass, field
from typing import Optional

from transformers import PretrainedConfig

from ...utils import get_logger
from ..auto.config_auto import register_config
from ..model_config import McaModelConfig


logger = get_logger(__name__)

@register_config("qwen3_vl")
@dataclass
class Qwen3VLConfig(McaModelConfig):
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    vision_token_id: int = 151654
    image_token_id: int = 151655
    video_token_id: int = 151656
    vision_config: Optional[dict] = field(
        default=None,
        metadata={"help": "Vision model config."},
    )
    text_config: Optional[dict] = field(
        default=None,
        metadata={"help": "Text model config."},
    )
    rope_scaling: Optional[dict] = field(
        default=None,
        metadata={"help": "Rope scaling."},
    )

    def __post_init__(self):
        logger.info(f"{self.text_config}")
        super().__post_init__()
        from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig

        if isinstance(self.vision_config, PretrainedConfig):
            self.vision_config = self.vision_config.to_dict()
        vision_config_obj = Qwen3VLVisionConfig(**self.vision_config)
        self.merge_size = vision_config_obj.spatial_merge_size
        self.pixel_values_dim = (
            vision_config_obj.patch_size
            * vision_config_obj.patch_size
            * vision_config_obj.in_channels
            * vision_config_obj.temporal_patch_size
        )  # 1176
        self.mrope_section = self.rope_scaling.get("mrope_section")
