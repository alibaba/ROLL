from dataclasses import dataclass, field
from typing import Optional

from transformers import PretrainedConfig

from ...utils import get_logger
from ..auto.config_auto import register_config
from ..model_config import McaModelConfig


logger = get_logger(__name__)

@register_config("qwen3_omni_moe")
@dataclass
class Qwen3OmniMoeConfig(McaModelConfig):
    audio_token_id: int = 151675
    image_token_id: int = 151655
    video_token_id: int = 151656
    position_id_per_seconds: int = 13
    audio_start_token_id: int = 151669
    vision_start_token_id: int = 151652
    vision_config: Optional[dict] = field(
        default=None,
        metadata={"help": "Vision model config."},
    )
    audio_config: Optional[dict] = field(
        default=None,
        metadata={"help": "audio model config."},
    )
    # text_config: Optional[dict] = field(
    #     default=None,
    #     metadata={"help": "Text model config."},
    # )
    enable_audio_output: bool = False
    talker_config: Optional[dict] = field(
        default=None,
        metadata={"help": "talker model config."},
    )
    code2wav_config: Optional[dict] = field(
        default=None,
        metadata={"help": "code2wav model config."},
    )
    rope_scaling: Optional[dict] = field(
        default=None,
        metadata={"help": "Rope scaling."},
    )

    def __post_init__(self):
        super().__post_init__()
        from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeVisionEncoderConfig

        if isinstance(self.audio_config, PretrainedConfig):
            self.audio_config = self.audio_config.to_dict()
        if isinstance(self.vision_config, PretrainedConfig):
            self.vision_config = self.vision_config.to_dict()
        if isinstance(self.talker_config, PretrainedConfig):
            self.talker_config = self.talker_config.to_dict()
        if isinstance(self.code2wav_config, PretrainedConfig):
            self.code2wav_config = self.code2wav_config.to_dict()
        vision_config_obj = Qwen3OmniMoeVisionEncoderConfig(**self.vision_config)
        self.merge_size = vision_config_obj.spatial_merge_size
        self.pixel_values_dim = (
            vision_config_obj.patch_size
            * vision_config_obj.patch_size
            * vision_config_obj.in_channels
            * vision_config_obj.temporal_patch_size
        )  # 1536
        assert "mrope_section" in self.rope_scaling, "mrope_section is required"
        self.mrope_section = self.rope_scaling.get("mrope_section")
