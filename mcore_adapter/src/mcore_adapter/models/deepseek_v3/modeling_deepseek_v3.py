from typing import Optional

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_mtp_block_spec

from ..auto.modeling_auto import register_model
from ..model_config import MLAMcaModelConfig
from ..model_factory import McaGPTModel


@register_model("deepseek_v3")
class DeepSeekV3Model(McaGPTModel):
    config_class = MLAMcaModelConfig

    def __init__(self, config, **kwargs):
        kwargs["mtp_block_spec"] = self._get_mtp_block_spec(config)
        super().__init__(config, **kwargs)

        if self.mtp_process:
            # MCore-0.12.0 `num_layers_to_build` do not account mtp
            self.decoder.layers = self.decoder.layers[:-1]

    def _get_mtp_block_spec(self, config: Optional["MLAMcaModelConfig"] = None):
        config = config or self.config
        if config.mtp_num_layers and config.mtp_num_layers > 0:
            transformer_layer_spec = self._get_transformer_layer_spec(config)
            use_te = config.transformer_impl == "transformer_engine"
            spec = get_gpt_mtp_block_spec(config, transformer_layer_spec, use_te)
            return spec
        else:
            return None
