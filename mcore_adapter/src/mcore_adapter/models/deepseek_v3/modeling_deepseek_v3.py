from ..auto.modeling_auto import register_model
from ..model_config import MLAMcaModelConfig
from ..model_factory import McaGPTModel


@register_model("deepseek_v3")
class DeepSeekV3Model(McaGPTModel):
    config_class = MLAMcaModelConfig
