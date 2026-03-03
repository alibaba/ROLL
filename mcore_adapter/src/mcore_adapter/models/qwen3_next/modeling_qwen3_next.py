from typing import Optional

from ..auto.modeling_auto import register_model
from ..model_factory import McaGPTModel
from .config_qwen3_next import Qwen3NextConfig


@register_model("qwen3_next")
class Qwen3NextModel(McaGPTModel):
    config_class = Qwen3NextConfig

    def _get_transformer_layer_spec(self, config: Optional[Qwen3NextConfig] = None):
        from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
            get_transformer_block_with_experimental_attention_variant_spec,
        )

        config = config or self.config
        assert config.transformer_impl == "transformer_engine", (
            "Qwen3NextModel only supports 'transformer_engine' implementation"
        )
        if config.experimental_attention_variant is not None:
            transformer_block_spec = get_transformer_block_with_experimental_attention_variant_spec(
                config=config, vp_stage=self.vp_stage
            )
        else:
            transformer_block_spec = super()._get_transformer_layer_spec(config)
        return transformer_block_spec
