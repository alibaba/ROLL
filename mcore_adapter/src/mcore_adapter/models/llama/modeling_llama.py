from ..auto.modeling_auto import register_model
from ..model_factory import McaGPTModel
from .configuration_llama import LlamaConfig


@register_model("llama")
class LlamaModel(McaGPTModel):
    config_class = LlamaConfig
