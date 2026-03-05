from roll.utils.logging import get_logger

logger = get_logger()

from vllm.transformers_utils.configs.qwen3_5 import Qwen3_5TextConfig
from vllm.transformers_utils.configs.qwen3_5_moe import Qwen3_5MoeTextConfig
from transformers.configuration_utils import PretrainedConfig, layer_type_validation

def Qwen3_5TextConfig_init(
    self,
    vocab_size=248320,
    hidden_size=4096,
    intermediate_size=12288,
    num_hidden_layers=32,
    num_attention_heads=16,
    num_key_value_heads=4,
    hidden_act="silu",
    max_position_embeddings=32768,
    initializer_range=0.02,
    rms_norm_eps=1e-6,
    use_cache=True,
    tie_word_embeddings=False,
    rope_parameters=None,
    attention_bias=False,
    attention_dropout=0.0,
    head_dim=256,
    linear_conv_kernel_dim=4,
    linear_key_head_dim=128,
    linear_value_head_dim=128,
    linear_num_key_heads=16,
    linear_num_value_heads=32,
    layer_types=None,
    pad_token_id=None,
    bos_token_id=None,
    eos_token_id=None,
    **kwargs,
):
    kwargs["ignore_keys_at_rope_validation"] = {
        "mrope_section",
        "mrope_interleaved",
    }
    self.vocab_size = vocab_size
    self.max_position_embeddings = max_position_embeddings
    self.hidden_size = hidden_size
    self.intermediate_size = intermediate_size
    self.num_hidden_layers = num_hidden_layers
    self.num_attention_heads = num_attention_heads
    self.num_key_value_heads = num_key_value_heads
    self.hidden_act = hidden_act
    self.initializer_range = initializer_range
    self.rms_norm_eps = rms_norm_eps
    self.use_cache = use_cache
    self.attention_bias = attention_bias
    self.attention_dropout = attention_dropout
    self.head_dim = head_dim
    self.rope_parameters = rope_parameters
    kwargs.setdefault("partial_rotary_factor", 0.25)

    self.layer_types = layer_types
    if self.layer_types is None:
        interval_pattern = kwargs.get("full_attention_interval", 4)
        self.layer_types = [
            "linear_attention"
            if bool((i + 1) % interval_pattern)
            else "full_attention"
            for i in range(self.num_hidden_layers)
        ]
    layer_type_validation(self.layer_types, self.num_hidden_layers)

    # linear attention part
    self.linear_conv_kernel_dim = linear_conv_kernel_dim
    self.linear_key_head_dim = linear_key_head_dim
    self.linear_value_head_dim = linear_value_head_dim
    self.linear_num_key_heads = linear_num_key_heads
    self.linear_num_value_heads = linear_num_value_heads
    super(Qwen3_5TextConfig, self).__init__(**kwargs)
    # Set these AFTER super().__init__() because transformers v4's
    # PretrainedConfig.__init__ has these as explicit params with different
    # defaults (e.g. tie_word_embeddings=True) that would overwrite our values.
    self.pad_token_id = pad_token_id
    self.bos_token_id = bos_token_id
    self.eos_token_id = eos_token_id
    self.tie_word_embeddings = tie_word_embeddings

def Qwen3_5MoeTextConfig_init(
    self,
    vocab_size=248320,
    hidden_size=2048,
    num_hidden_layers=40,
    num_attention_heads=16,
    num_key_value_heads=2,
    hidden_act="silu",
    max_position_embeddings=32768,
    initializer_range=0.02,
    rms_norm_eps=1e-6,
    use_cache=True,
    tie_word_embeddings=False,
    rope_parameters=None,
    attention_bias=False,
    attention_dropout=0.0,
    head_dim=256,
    linear_conv_kernel_dim=4,
    linear_key_head_dim=128,
    linear_value_head_dim=128,
    linear_num_key_heads=16,
    linear_num_value_heads=32,
    moe_intermediate_size=512,
    shared_expert_intermediate_size=512,
    num_experts_per_tok=8,
    num_experts=256,
    output_router_logits=False,
    router_aux_loss_coef=0.001,
    layer_types=None,
    pad_token_id=None,
    bos_token_id=None,
    eos_token_id=None,
    **kwargs,
):
    kwargs["ignore_keys_at_rope_validation"] = {
        "mrope_section",
        "mrope_interleaved",
    }
    self.vocab_size = vocab_size
    self.max_position_embeddings = max_position_embeddings
    self.hidden_size = hidden_size
    self.num_hidden_layers = num_hidden_layers
    self.num_attention_heads = num_attention_heads
    self.num_key_value_heads = num_key_value_heads
    self.hidden_act = hidden_act
    self.initializer_range = initializer_range
    self.rms_norm_eps = rms_norm_eps
    self.use_cache = use_cache
    self.attention_bias = attention_bias
    self.attention_dropout = attention_dropout
    self.head_dim = head_dim
    self.rope_parameters = rope_parameters
    kwargs.setdefault("partial_rotary_factor", 0.25)

    self.layer_types = layer_types
    if self.layer_types is None:
        interval_pattern = kwargs.get("full_attention_interval", 4)
        self.layer_types = [
            "linear_attention"
            if bool((i + 1) % interval_pattern)
            else "full_attention"
            for i in range(self.num_hidden_layers)
        ]
    layer_type_validation(self.layer_types, self.num_hidden_layers)

    # linear attention part
    self.linear_conv_kernel_dim = linear_conv_kernel_dim
    self.linear_key_head_dim = linear_key_head_dim
    self.linear_value_head_dim = linear_value_head_dim
    self.linear_num_key_heads = linear_num_key_heads
    self.linear_num_value_heads = linear_num_value_heads
    self.moe_intermediate_size = moe_intermediate_size
    self.shared_expert_intermediate_size = shared_expert_intermediate_size
    self.num_experts_per_tok = num_experts_per_tok
    self.num_experts = num_experts
    self.output_router_logits = output_router_logits
    self.router_aux_loss_coef = router_aux_loss_coef
    super(Qwen3_5MoeTextConfig, self).__init__(**kwargs)
    # Set these AFTER super().__init__() because transformers v4's
    # PretrainedConfig.__init__ has these as explicit params with different
    # defaults (e.g. tie_word_embeddings=True) that would overwrite our values.
    self.pad_token_id = pad_token_id
    self.bos_token_id = bos_token_id
    self.eos_token_id = eos_token_id
    self.tie_word_embeddings = tie_word_embeddings

Qwen3_5TextConfig.__init__ = Qwen3_5TextConfig_init
Qwen3_5MoeTextConfig.__init__ = Qwen3_5MoeTextConfig_init
