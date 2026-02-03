from dataclasses import dataclass

from ..converter.dist_converter import DistParallelConfig, default_dist_config, register_dist_config
from ..converter.template import (
    QKVBiasConverOp,
    QKVConverOp,
    RenameConverOp,
    StackConverOp,
    Template,
    register_template,
)
from .config_qwen3_vl import Qwen3VLConfig
from .modeling_qwen3_vl import Qwen3VLModel


register_dist_config(
    "qwen3_vl",
    default_dist_config.merge_configs(
        DistParallelConfig(
            pre_process_weights=["vision_model.*"],
            duplicated_weights=["vision_model.*"],
        )
    ),
)


@dataclass
class Qwen3VLTemplate(Template):
    def adjust_config_hf_to_mca(self):
        # NOTE: for `tie_word_embeddings`,
        # in qwen3-vl model like Qwen/Qwen3-VL-4B-Instruct, tie_word_embeddings
        # exists both in inner and outer of text_config, and both are True
        # in qwen3-vl-moe model like Qwen/Qwen3-VL-30B-A3B-Instruct, tie_word_embeddings
        # in outer of text_config is False while it uses the default value True in the
        # inner of text_config
        # currently, both use tie_word_embeddings in the outter of text_config
        non_text_config_keys = set(
            list(filter(lambda k: k.endswith("_token_id"), self.config_hf_to_mca.keys()))
            + ["vision_config", "tie_word_embeddings"]
        )
        new_config_hf_to_mca = {}
        for hf_key, mca_key in self.config_hf_to_mca.items():
            new_hf_key = hf_key
            if hf_key not in non_text_config_keys:
                new_hf_key = "text_config." + new_hf_key
            new_config_hf_to_mca[new_hf_key] = mca_key
        return new_config_hf_to_mca


register_template(
    "qwen3_vl",
    hf_layer_prefix="model.language_model.layers.",
    template_class=Qwen3VLTemplate,
    config_hf_to_mca={
        "max_position_embeddings": "max_sequence_length",
        "hidden_size": "hidden_size",
        "attention_bias": "add_qkv_bias",
        "head_dim": "kv_channels",
        "num_attention_heads": "num_attention_heads",
        "num_key_value_heads": "num_query_groups",
        "num_hidden_layers": "num_layers",
        "rms_norm_eps": "layernorm_epsilon",
        "vocab_size": "padded_vocab_size",
        "attention_dropout": "attention_dropout",
        "rope_theta": "rotary_base",
        "intermediate_size": "ffn_hidden_size",
        "tie_word_embeddings": "tie_embeddings_and_output_weights",

        # vit related
        "vision_start_token_id": "vision_start_token_id",
        "vision_end_token_id": "vision_end_token_id",
        "vision_token_id": "vision_token_id",
        "image_token_id": "image_token_id",
        "video_token_id": "video_token_id",
        "vision_config": "vision_config",
        "rope_scaling": "rope_scaling",
    },
    constant_mca_config={
        "swiglu": True,
        "position_embedding_type": "mrope",
        "normalization": "RMSNorm",
        "add_bias_linear": False,
        "hidden_dropout": 0.0,
        "rotary_percent": 1.0,
        "qk_layernorm": True,
    },
    weight_converters=[
        RenameConverOp(hf_names="lm_head.weight", mca_names="output_layer.weight"),
        RenameConverOp(hf_names="model.language_model.embed_tokens.weight", mca_names="embedding.word_embeddings.weight"),
        RenameConverOp(hf_names=".input_layernorm.weight", mca_names=".self_attention.linear_qkv.layer_norm_weight"),
        RenameConverOp(hf_names=".self_attn.o_proj.weight", mca_names=".self_attention.linear_proj.weight"),
        RenameConverOp(hf_names=".self_attn.q_norm.weight", mca_names=".self_attention.q_layernorm.weight"),
        RenameConverOp(hf_names=".self_attn.k_norm.weight", mca_names=".self_attention.k_layernorm.weight"),
        RenameConverOp(hf_names=".post_attention_layernorm.weight", mca_names=".mlp.linear_fc1.layer_norm_weight"),
        RenameConverOp(hf_names="model.language_model.norm.weight", mca_names="decoder.final_layernorm.weight"),
        StackConverOp(
            hf_names=[".mlp.gate_proj.weight", ".mlp.up_proj.weight"], mca_names=".mlp.linear_fc1.weight", dim=0
        ),
        RenameConverOp(hf_names=".mlp.down_proj.weight", mca_names=".mlp.linear_fc2.weight"),
        QKVConverOp(
            hf_names=[".self_attn.q_proj.weight", ".self_attn.k_proj.weight", ".self_attn.v_proj.weight"],
            mca_names=".self_attention.linear_qkv.weight",
        ),
        QKVBiasConverOp(
            hf_names=[".self_attn.q_proj.bias", ".self_attn.k_proj.bias", ".self_attn.v_proj.bias"],
            mca_names=".self_attention.linear_qkv.bias",
        ),
        RenameConverOp(hf_names="model.visual.{}", mca_names="vision_model.{}"),

    ],
)


__all__ = ["Qwen3VLConfig", "Qwen3VLModel"]
