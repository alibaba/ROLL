from megatron.core import mpu

from ..auto.config_auto import register_config
from ..auto.modeling_auto import register_model
from ..converter.convert_utils import (
    get_layer_index,
    get_mca_layer_index,
    get_mca_mtp_layer_index,
    remove_weight_prefix,
)
from ..converter.dist_converter import DistParallelConfig, default_dist_config, mtp_config, register_dist_config
from ..converter.template import (
    QKVBiasConverOp,
    QKVConverOp,
    RenameConverOp,
    StackConverOp,
    Template,
    register_template,
)
from ..model_config import McaModelConfig
from ..model_factory import McaGPTModel


class Glm4MoeTemplate(Template):
    def convert_hf_to_mca_config_kws(self, hf_config, **kw_args):
        partial_rotary_factor = getattr(hf_config, "partial_rotary_factor", None)
        if partial_rotary_factor is not None:
            kw_args["rotary_percent"] = partial_rotary_factor

        if getattr(hf_config, "num_hidden_layers", None) is not None:
            kw_args["num_layers"] = hf_config.num_hidden_layers

        # heads / kv heads
        if getattr(hf_config, "num_attention_heads", None) is not None:
            kw_args["num_attention_heads"] = hf_config.num_attention_heads
        if getattr(hf_config, "num_key_value_heads", None) is not None:
            kw_args["num_query_groups"] = hf_config.num_key_value_heads

        # hidden/intermediate
        if getattr(hf_config, "hidden_size", None) is not None:
            kw_args["hidden_size"] = hf_config.hidden_size
        if getattr(hf_config, "intermediate_size", None) is not None:
            kw_args["ffn_hidden_size"] = hf_config.intermediate_size

        if getattr(hf_config, "routed_scaling_factor", None) is not None:
            kw_args["moe_router_topk_scaling_factor"] = hf_config.routed_scaling_factor

        n_shared_experts = getattr(hf_config, "n_shared_experts", None)
        if n_shared_experts:
            kw_args["moe_shared_expert_intermediate_size"] = (
                hf_config.n_shared_experts * hf_config.moe_intermediate_size
            )
        res = super().convert_hf_to_mca_config_kws(hf_config, **kw_args)

        first_k_dense_replace = getattr(hf_config, "first_k_dense_replace", None)
        if first_k_dense_replace:
            assert first_k_dense_replace < res["num_layers"], "first_k_dense_layers is out of range."
            res["moe_layer_freq"] = [0] * first_k_dense_replace + [1] * (res["num_layers"] - first_k_dense_replace)

        return res

    def convert_mca_to_hf_config(self, mca_config, **kw_args):
        if mca_config.moe_shared_expert_intermediate_size:
            kw_args["n_shared_experts"] = (
                mca_config.moe_shared_expert_intermediate_size // mca_config.moe_ffn_hidden_size
            )
        else:
            kw_args["n_shared_experts"] = 0

        if isinstance(mca_config.moe_layer_freq, list):
            kw_args["first_k_dense_replace"] = mca_config.moe_layer_freq.count(0)
            kw_args["moe_layer_freq"] = 1

        res = super().convert_mca_to_hf_config(mca_config, **kw_args)

        return res

    def add_hf_weight(self, name, weight):
        name2weights = super().add_hf_weight(name, weight)
        if name2weights is None:
            return None
        res = {}
        for name, weight in name2weights.items():
            layer_index = get_mca_layer_index(name)
            if layer_index is not None and layer_index >= self.mca_config.num_layers:
                name = self.convert_mtp_name(name)
            if layer_index is not None and layer_index < self.mca_config.moe_layer_freq.count(0):
                # dense layer use fused `TELayerNormColumnParallelLinear`, change the name
                if "pre_mlp_layernorm" in name:
                    name = name.replace("pre_mlp_layernorm.", "mlp.linear_fc1.layer_norm_")
            res[name] = weight
        return res

    def add_mca_weight(self, name, weight, **kwargs):
        name = self.revert_mtp_name(name)
        layer_index = get_mca_layer_index(name)
        if layer_index is not None and layer_index < self.mca_config.moe_layer_freq.count(0):
            name = name.replace("mlp.linear_fc1.layer_norm_", "pre_mlp_layernorm.")
        name2weights = super().add_mca_weight(name, weight, **kwargs)
        res = {}
        for name, weight in name2weights.items():
            if (
                name == "model.embed_tokens.weight"
                and self.mca_config.pipeline_model_parallel_size > 1
                and mpu.is_pipeline_last_stage()
            ):
                continue
            layer_index = get_layer_index(name, self.hf_layer_prefix)
            if layer_index is not None:
                is_moe_layer = layer_index >= self.mca_config.moe_layer_freq.count(0)
                if not is_moe_layer:
                    name = name.replace("mlp.shared_experts.", "mlp.")
            res[name] = weight
        return res

    def hf_name_to_mca_names(self, hf_name):
        mca_names = super().hf_name_to_mca_names(hf_name)
        if mca_names is None:
            return None
        mtp_mca_names = [self.convert_mtp_name(mca_name) for mca_name in mca_names]
        return mtp_mca_names

    def convert_mtp_name(self, name):
        mca_layer_index = get_mca_layer_index(name)
        if mca_layer_index is None or mca_layer_index < self.mca_config.num_layers:
            return name
        mtp_layer_index = mca_layer_index - self.mca_config.num_layers
        has_transformer_layer = "self_attention" in name or "mlp" in name or "input_layernorm" in name
        name = name.replace("decoder", "mtp")
        pure_name = remove_weight_prefix(name, prefix="mtp.layers.")
        name = (
            "mtp.layers." + str(mtp_layer_index) + (".transformer_layer" if has_transformer_layer else "") + pure_name
        )
        return name

    def revert_mtp_name(self, name):
        if "mtp" in name:
            has_transformer_layer = "self_attention" in name or "mlp" in name or "input_layernorm" in name
            mtp_layer_index = get_mca_mtp_layer_index(name)
            pure_name = remove_weight_prefix(name, prefix="mtp.layers.")
            # only consider padding mtp for now...
            mca_layer_index = mtp_layer_index + self.mca_config.num_layers
            name = (
                "decoder.layers."
                + str(mca_layer_index)
                + (pure_name.replace(".transformer_layer", "") if has_transformer_layer else pure_name)
            )
        return name


register_template(
    "glm4_moe",
    hf_layer_prefix="model.layers.",
    hf_moe_prefix=".mlp.experts.",
    template_class=Glm4MoeTemplate,
    hf_invalid_keys=[
        ".embed_tokens.weight",  # skip layers.x.embed_tokens
        ".shared_head.head.weight",
    ],
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
        # MoE related
        "moe_intermediate_size": "moe_ffn_hidden_size",
        "decoder_sparse_step": "moe_layer_freq",
        "n_routed_experts": "num_moe_experts",  # diff
        "num_experts_per_tok": "moe_router_topk",
        # MTP related
        "num_nextn_predict_layers": "mtp_num_layers",
    },
    constant_mca_config={
        "swiglu": True,
        "position_embedding_type": "rope",
        "normalization": "RMSNorm",
        "add_bias_linear": False,
        "hidden_dropout": 0.0,
        "rotary_percent": 0.5,
        "moe_router_load_balancing_type": "seq_aux_loss",
        "moe_router_pre_softmax": False,
        "qk_layernorm": False,
        "moe_router_enable_expert_bias": True,
        "moe_router_score_function": "sigmoid",
        "mtp_loss_scaling_factor": 0.3,
    },
    weight_converters=[
        RenameConverOp(hf_names="lm_head.weight", mca_names="output_layer.weight"),
        RenameConverOp(hf_names="model.embed_tokens.weight", mca_names="embedding.word_embeddings.weight"),
        RenameConverOp(hf_names=".input_layernorm.weight", mca_names=".self_attention.linear_qkv.layer_norm_weight"),
        RenameConverOp(hf_names=".self_attn.o_proj.weight", mca_names=".self_attention.linear_proj.weight"),
        RenameConverOp(hf_names=".post_attention_layernorm.weight", mca_names=".pre_mlp_layernorm.weight"),
        RenameConverOp(hf_names=".mlp.down_proj.weight", mca_names=".mlp.linear_fc2.weight"),  # first layer
        StackConverOp(
            hf_names=[".mlp.gate_proj.weight", ".mlp.up_proj.weight"], mca_names=".mlp.linear_fc1.weight", dim=0
        ),
        StackConverOp(hf_names=[".gate_proj.weight", ".up_proj.weight"], mca_names=".linear_fc1.weight", dim=0),
        RenameConverOp(hf_names=".down_proj.weight", mca_names=".linear_fc2.weight"),
        RenameConverOp(hf_names="model.norm.weight", mca_names="decoder.final_layernorm.weight"),
        RenameConverOp(hf_names=".mlp.gate.weight", mca_names=".mlp.router.weight"),
        StackConverOp(  # for shared
            hf_names=[".mlp.shared_experts.gate_proj.weight", ".mlp.shared_experts.up_proj.weight"],
            mca_names=".mlp.shared_experts.linear_fc1.weight",
            dim=0,
        ),
        RenameConverOp(
            hf_names=".mlp.shared_experts.down_proj.weight", mca_names=".mlp.shared_experts.linear_fc2.weight"
        ),
        RenameConverOp(hf_names=".mlp.gate.e_score_correction_bias", mca_names=".mlp.router.expert_bias"),
        QKVConverOp(
            hf_names=[".self_attn.q_proj.weight", ".self_attn.k_proj.weight", ".self_attn.v_proj.weight"],
            mca_names=".self_attention.linear_qkv.weight",
        ),
        QKVBiasConverOp(
            hf_names=[".self_attn.q_proj.bias", ".self_attn.k_proj.bias", ".self_attn.v_proj.bias"],
            mca_names=".self_attention.linear_qkv.bias",
        ),
        RenameConverOp(hf_names=".enorm.weight", mca_names=".enorm.weight"),
        RenameConverOp(hf_names=".hnorm.weight", mca_names=".hnorm.weight"),
        RenameConverOp(hf_names=".eh_proj.weight", mca_names=".eh_proj.weight"),
        RenameConverOp(hf_names=".shared_head.norm.weight", mca_names=".final_layernorm.weight"),
    ],
)

register_config("glm4_moe", McaModelConfig)
register_model("glm4_moe", McaGPTModel)
glm_dist_config = default_dist_config.merge_configs(mtp_config).merge_configs(
    DistParallelConfig(
        duplicated_weights=[
            ".mlp.router.expert_bias",
        ],
        grouped_column_map={".linear_fc1.weight": ".mlp.experts.weight1"},
        grouped_row_map={".linear_fc2.weight": ".mlp.experts.weight2"},
        row_parallel_weights=[
            ".self_attention.linear_proj.weight",
            ".mlp.shared_experts.linear_fc2.weight",
            ".linear_fc2.weight",
            ".mlp.linear_fc2.weight",
        ],
        swiglu_weights=[
            ".mlp.shared_experts.linear_fc1.weight",
            ".linear_fc1.weight",
            ".mlp.linear_fc1.weight",
        ],
    )
)
register_dist_config("glm4_moe", glm_dist_config)
