from ..auto.config_auto import register_config
from ..auto.modeling_auto import register_model
from ..converter.dist_converter import default_dist_config, register_dist_config
from ..converter.template import (
    QKVBiasConverOp,
    QKVConverOp,
    RenameConverOp,
    StackConverOp,
    register_template,
)
from ..model_config import McaModelConfig
from ..model_factory import McaGPTModel


register_config("mistral", McaModelConfig)
register_model("mistral", McaGPTModel)
register_dist_config("mistral", default_dist_config)


register_template(
    "mistral",
    hf_layer_prefix="model.layers.",
    config_hf_to_mca={
        "max_position_embeddings": "max_sequence_length",
        "hidden_size": "hidden_size",
        "num_attention_heads": "num_attention_heads",
        "num_key_value_heads": "num_query_groups",
        "num_hidden_layers": "num_layers",
        "rms_norm_eps": "layernorm_epsilon",
        "vocab_size": "padded_vocab_size",
        "intermediate_size": "ffn_hidden_size",
        "attention_bias": "add_qkv_bias",
        "attention_dropout": "attention_dropout",
        "rope_theta": "rotary_base",
        "tie_word_embeddings": "tie_embeddings_and_output_weights",
    },
    hf_invalid_keys=[".self_attn.rotary_emb.inv_freq"],
    constant_mca_config={
        "swiglu": True,
        "position_embedding_type": "rope",
        "normalization": "RMSNorm",
        "add_bias_linear": False,
        "hidden_dropout": 0.0,
        "rotary_percent": 1.0,
    },
    weight_converters=[
        RenameConverOp(hf_names="lm_head.weight", mca_names="output_layer.weight"),
        RenameConverOp(hf_names="model.embed_tokens.weight", mca_names="embedding.word_embeddings.weight"),
        RenameConverOp(hf_names=".input_layernorm.weight", mca_names=".self_attention.linear_qkv.layer_norm_weight"),
        RenameConverOp(hf_names=".self_attn.o_proj.weight", mca_names=".self_attention.linear_proj.weight"),
        RenameConverOp(hf_names=".post_attention_layernorm.weight", mca_names=".mlp.linear_fc1.layer_norm_weight"),
        RenameConverOp(hf_names=".mlp.down_proj.weight", mca_names=".mlp.linear_fc2.weight"),
        RenameConverOp(hf_names="model.norm.weight", mca_names="decoder.final_layernorm.weight"),
        StackConverOp(
            hf_names=[".mlp.gate_proj.weight", ".mlp.up_proj.weight"], mca_names=".mlp.linear_fc1.weight", dim=0
        ),
        QKVConverOp(
            hf_names=[".self_attn.q_proj.weight", ".self_attn.k_proj.weight", ".self_attn.v_proj.weight"],
            mca_names=".self_attention.linear_qkv.weight",
        ),
    ],
)
