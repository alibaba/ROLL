from dataclasses import dataclass

from ..converter.dist_converter import (
    DistParallelConfig,
    default_dist_config,
    register_dist_config,
    shared_moe_dist_config,
)
from ..converter.template import (
    QKVBiasConverOp,
    QKVConverOp,
    RenameConverOp,
    StackConverOp,
    Template,
    register_template,
)
from .config_qwen3_omni import Qwen3OmniMoeConfig
from .modeling_qwen3_omni import Qwen3OmniMoeModel


@dataclass
class Qwen3OmniMoeTemplate(Template):
    def adjust_config_hf_to_mca(self):
        non_text_config_keys = set(
            list(filter(lambda k: k.endswith("_token_id"), self.config_hf_to_mca.keys()))
            + ["position_id_per_seconds", "vision_config", "audio_config"]
        )
        audio_output_config_keys = ["enable_audio_output", "talker_config", "code2wav_config"]
        new_config_hf_to_mca = {}
        for hf_key, mca_key in self.config_hf_to_mca.items():
            new_hf_key = hf_key
            if hf_key not in audio_output_config_keys:
                if hf_key not in non_text_config_keys:
                    new_hf_key = "text_config." + new_hf_key
                new_hf_key = "thinker_config." + new_hf_key
            new_config_hf_to_mca[new_hf_key] = mca_key
        return new_config_hf_to_mca


register_dist_config(
    "qwen3_omni_moe",
    default_dist_config.merge_configs(shared_moe_dist_config).merge_configs(
        DistParallelConfig(
            pre_process_weights=["vision_model.*", "audio_model.*"],
            post_process_weights=["talker.*", "code2wav.*"],
            duplicated_weights=["vision_model.*", "audio_model.*", "talker.*", "code2wav.*"],
        )
    ),
)


# NOTE: thinking and instruct both use qwen3_omni_moe as model_type and Qwen3OmniMoeForConditionalGeneration
# as architecture, thus both hf config and weight key has thinker prefix. And it seems the processor cannot
# use list fps thus video should be processed by one by one.
# TODO: Should we use "thinker" for naming template/config/model, would there exist confilicts if we support
# instruct model since thinking and instruct both use qwen3_omni_moe
register_template(
    "qwen3_omni_moe",
    hf_layer_prefix="thinker.model.layers.",
    hf_moe_prefix=".mlp.experts.",
    template_class=Qwen3OmniMoeTemplate,  # Qwen3VLMoeTemplate,
    # hf has hierarchical config for multi-modal models while mca has flat config
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
        "rope_scaling": "rope_scaling",
        "intermediate_size": "ffn_hidden_size",
        "tie_word_embeddings": "tie_embeddings_and_output_weights",
        # MoE related
        "moe_intermediate_size": "moe_ffn_hidden_size",
        "decoder_sparse_step": "moe_layer_freq",
        "num_experts": "num_moe_experts",
        "num_experts_per_tok": "moe_router_topk",
        "router_aux_loss_coef": "moe_aux_loss_coeff",
        # ait ralated, only need for usage in get_rope_index
        "audio_token_id": "audio_token_id",
        "audio_start_token_id": "audio_start_token_id",
        # "audio_end_token_id": "audio_start_token_id",
        # vit related, only need for usage in get_rope_index
        "image_token_id": "image_token_id",
        "video_token_id": "video_token_id",
        "vision_start_token_id": "vision_start_token_id",
        # "vision_end_token_id": "vision_end_token_id",
        "position_id_per_seconds": "position_id_per_seconds",
        "vision_config": "vision_config",
        "audio_config": "audio_config",
        "enable_audio_output": "enable_audio_output",
        "talker_config": "talker_config",
        "code2wav_config": "code2wav_config",
    },
    constant_mca_config={
        "swiglu": True,
        "position_embedding_type": "mrope",  # TM-ROPE
        "normalization": "RMSNorm",
        "add_bias_linear": False,
        "hidden_dropout": 0.0,
        "rotary_percent": 1.0,
        "moe_router_load_balancing_type": "aux_loss",
        "moe_router_pre_softmax": False,
        "qk_layernorm": True,
    },
    weight_converters=[
        RenameConverOp(hf_names="thinker.model.embed_tokens.weight", mca_names="embedding.word_embeddings.weight"),
        RenameConverOp(hf_names="thinker.model.norm.weight", mca_names="decoder.final_layernorm.weight"),
        RenameConverOp(hf_names="thinker.lm_head.weight", mca_names="output_layer.weight"),
        RenameConverOp(hf_names=".input_layernorm.weight", mca_names=".self_attention.linear_qkv.layer_norm_weight"),
        # attention weights
        QKVConverOp(
            hf_names=[".self_attn.q_proj.weight", ".self_attn.k_proj.weight", ".self_attn.v_proj.weight"],
            mca_names=".self_attention.linear_qkv.weight",
        ),
        QKVBiasConverOp(
            hf_names=[".self_attn.q_proj.bias", ".self_attn.k_proj.bias", ".self_attn.v_proj.bias"],
            mca_names=".self_attention.linear_qkv.bias",
        ),  # attention_bias is false actually
        RenameConverOp(hf_names=".self_attn.o_proj.weight", mca_names=".self_attention.linear_proj.weight"),
        RenameConverOp(hf_names=".self_attn.q_norm.weight", mca_names=".self_attention.q_layernorm.weight"),
        RenameConverOp(hf_names=".self_attn.k_norm.weight", mca_names=".self_attention.k_layernorm.weight"),
        RenameConverOp(hf_names=".post_attention_layernorm.weight", mca_names=".pre_mlp_layernorm.weight"),
        # moe weights
        RenameConverOp(hf_names=".mlp.gate.weight", mca_names=".mlp.router.weight"),
        StackConverOp(hf_names=[".gate_proj.weight", ".up_proj.weight"], mca_names=".linear_fc1.weight", dim=0),
        RenameConverOp(hf_names=".down_proj.weight", mca_names=".linear_fc2.weight"),
        RenameConverOp(hf_names="thinker.visual.{}", mca_names="vision_model.{}"),
        # add audio model to make it can be saved and used in hf
        # although the audio_model weights can be put into template.hf_invalid_keys
        RenameConverOp(hf_names="thinker.audio_tower.{}", mca_names="audio_model.{}"),
        RenameConverOp(hf_names="talker.{}", mca_names="talker.{}"),
        RenameConverOp(hf_names="code2wav.{}", mca_names="code2wav.{}"),
    ],
)

__all__ = ["Qwen3OmniMoeConfig", "Qwen3OmniMoeModel"]
