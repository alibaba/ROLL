import torch
from megatron.core import mpu

from ..auto.config_auto import register_config
from ..converter.convert_utils import (
    get_layer_index,
    get_mca_layer_index,
    remove_weight_prefix,
)
from ..converter.dist_converter import mla_dist_config, register_dist_config
from ..converter.template import (
    RenameConverOp,
    StackConverOp,
    Template,
    register_template,
)
from ..model_config import MLAMcaModelConfig
from .modeling_deepseek_v3 import DeepSeekV3Model


class DeepSeekV3Template(Template):
    def convert_hf_to_mca_config_kws(self, hf_config, **kw_args):
        # convert mla related parameters
        rope_scaling = getattr(hf_config, "rope_scaling", None)
        if rope_scaling:
            if rope_scaling.get("original_max_position_embeddings", None):
                kw_args["max_position_embeddings"] = rope_scaling["original_max_position_embeddings"]
            if rope_scaling.get("type", None):
                kw_args["rope_type"] = rope_scaling["type"]
            if rope_scaling.get("factor", None):
                kw_args["rotary_scaling_factor"] = rope_scaling["factor"]
            if rope_scaling.get("mscale_all_dim", None):
                kw_args["mscale_all_dim"] = rope_scaling["mscale_all_dim"]
            if rope_scaling.get("mscale", None):
                kw_args["mscale"] = rope_scaling["mscale"]
            if rope_scaling.get("beta_fast", None):
                kw_args["beta_fast"] = rope_scaling["beta_fast"]
            if rope_scaling.get("beta_slow", None):
                kw_args["beta_slow"] = rope_scaling["beta_slow"]

        # fused backend only support dim <= 128
        torch_dtype = getattr(hf_config, "torch_dtype", None)
        if torch_dtype == torch.bfloat16 or torch_dtype == torch.float16:
            from megatron.core.transformer.enums import AttnBackend

            kw_args["attention_backend"] = AttnBackend.unfused

        # compute moe_shared_expert_intermediate_size
        n_shared_experts = getattr(hf_config, "n_shared_experts", None)
        if n_shared_experts:
            kw_args["moe_shared_expert_intermediate_size"] = (
                hf_config.n_shared_experts * hf_config.moe_intermediate_size
            )

        res = super().convert_hf_to_mca_config_kws(hf_config, **kw_args)

        # set moe_layer_freq for dense + moe hybrid model, suppose all dense layers occur in the first k layers
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

        kw_args["rope_scaling"] = {
            "original_max_position_embeddings": mca_config.max_position_embeddings,
            "type": mca_config.rope_type,
            "factor": mca_config.rotary_scaling_factor,
            "mscale_all_dim": mca_config.mscale_all_dim,
            "mscale": mca_config.mscale,
            "beta_fast": mca_config.beta_fast,
            "beta_slow": mca_config.beta_slow,
        }

        res = super().convert_mca_to_hf_config(mca_config, **kw_args)

        return res

    def _get_mtp_layer_index(self, layer_index):
        if not mpu.is_pipeline_last_stage():
            return None
        if layer_index is None:
            return None

        total_pp_num_layers = self.mca_config.num_layers
        if self.mca_config.account_for_embedding_in_pipeline_split:
            total_pp_num_layers += 1
        if self.mca_config.account_for_loss_in_pipeline_split:
            total_pp_num_layers += 1
        pp_size = mpu.get_pipeline_model_parallel_world_size()
        assert (total_pp_num_layers % pp_size) == 0, (
            "When using mtp, ensure the result layers num can be devideded by pp_size"
        )

        # account for no pipeline parallel
        if pp_size == 1:
            if layer_index < (self.mca_config.num_layers - 1):
                return None
            return layer_index - (self.mca_config.num_layers - 1)

        num_layers_for_pp_rank = total_pp_num_layers // pp_size
        num_layers_in_last_stage = num_layers_for_pp_rank
        if self.mca_config.account_for_loss_in_pipeline_split:
            num_layers_in_last_stage -= 1

        if layer_index < (num_layers_in_last_stage - 1):
            return None

        return layer_index - (num_layers_in_last_stage - 1)

    def add_hf_weight(self, name, weight):
        name2weights = super().add_hf_weight(name, weight)
        if name2weights is None:
            return None
        res = {}
        for name, weight in name2weights.items():
            layer_index = get_mca_layer_index(name)
            if layer_index is not None and layer_index < self.mca_config.moe_layer_freq.count(0):
                # dense layer use fused `TELayerNormColumnParallelLinear`, change the name
                if "pre_mlp_layernorm" in name:
                    name = name.replace("pre_mlp_layernorm.", "mlp.linear_fc1.layer_norm_")
            res[name] = weight
        return res

    def add_mca_weight(self, name, weight):
        layer_index = get_mca_layer_index(name)
        if layer_index is not None and layer_index < self.mca_config.moe_layer_freq.count(0):
            name = name.replace("mlp.linear_fc1.layer_norm_", "pre_mlp_layernorm.")
        name2weights = super().add_mca_weight(name, weight)
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

    def convert_mtp_weights(self, name2weights):
        if name2weights is None:
            return None

        res = {}
        for name, weight in name2weights.items():
            mca_layer_index = get_mca_layer_index(name)
            mtp_layer_index = self._get_mtp_layer_index(mca_layer_index)
            if mtp_layer_index is not None:
                has_transformer_layer = "self_attention" in name or "mlp" in name or "input_layernorm" in name
                name = name.replace("decoder", "mtp")
                pure_name = remove_weight_prefix(name, prefix="mtp.layers.")
                name = (
                    "mtp.layers."
                    + str(mtp_layer_index)
                    + (".transformer_layer" if has_transformer_layer else "")
                    + pure_name
                )
            res[name] = weight
        return res

    def revert_mtp_weights(self, mca_state_dict):
        res = {}
        for name, weight in mca_state_dict.items():
            if "mtp" in name:
                has_transformer_layer = "self_attention" in name or "mlp" in name or "input_layernorm" in name
                mtp_layer_index = get_layer_index(name, prefix="mtp.layers.")
                pure_name = remove_weight_prefix(name, prefix="mtp.layers.")
                # only consider padding mtp for now...
                if self.mca_config.pipeline_model_parallel_size > 1:
                    num_pp_layers = (
                        self.mca_config.num_layers
                        + self.mca_config.account_for_embedding_in_pipeline_split
                        + self.mca_config.account_for_loss_in_pipeline_split
                    )
                    num_layers_in_last_stage = num_pp_layers // self.mca_config.pipeline_model_parallel_size
                    if self.mca_config.account_for_loss_in_pipeline_split:
                        num_layers_in_last_stage -= 1
                    mca_layer_index = mtp_layer_index + (num_layers_in_last_stage - 1)
                else:
                    mca_layer_index = mtp_layer_index + (self.mca_config.num_layers - 1)
                name = (
                    "decoder.layers."
                    + str(mca_layer_index)
                    + (pure_name.replace(".transformer_layer", "") if has_transformer_layer else pure_name)
                )
            res[name] = weight
        return res


register_config("deepseek_v3", MLAMcaModelConfig)
register_dist_config("deepseek_v3", mla_dist_config)


register_template(
    "deepseek_v3",
    template_class=DeepSeekV3Template,
    hf_layer_prefix="model.layers.",
    hf_moe_prefix=".mlp.experts.",
    hf_invalid_keys=[
        ".embed_tokens.weight",  # the mtp is shared, this weight is the same as `model.embed_tokens.weight` in hf,
        ".shared_head.head.weight",
        ".self_attn.rotary_emb.inv_freq",
    ],
    config_hf_to_mca={
        "max_position_embeddings": "max_sequence_length",
        "hidden_size": "hidden_size",
        "num_attention_heads": "num_attention_heads",
        "num_key_value_heads": "num_query_groups",
        "num_hidden_layers": "num_layers",
        "rms_norm_eps": "layernorm_epsilon",
        "vocab_size": "padded_vocab_size",
        "attention_dropout": "attention_dropout",
        "rope_theta": "rotary_base",
        "tie_word_embeddings": "tie_embeddings_and_output_weights",
        "v_head_dim": "v_head_dim",
        "qk_nope_head_dim": "qk_head_dim",
        "qk_rope_head_dim": "qk_pos_emb_head_dim",
        "q_lora_rank": "q_lora_rank",
        "kv_lora_rank": "kv_lora_rank",
        "moe_intermediate_size": "moe_ffn_hidden_size",
        "intermediate_size": "ffn_hidden_size",
        "n_routed_experts": "num_moe_experts",
        "num_experts_per_tok": "moe_router_topk",
        "scoring_func": "moe_router_score_function",
        "n_group": "moe_router_num_groups",
        "topk_group": "moe_router_group_topk",
        "routed_scaling_factor": "moe_router_topk_scaling_factor",
        # MTP related
        "num_nextn_predict_layers": "mtp_num_layers",
    },
    constant_mca_config={
        "swiglu": True,
        "position_embedding_type": "rope",
        "normalization": "RMSNorm",
        "qk_layernorm": True,
        "add_bias_linear": False,
        "add_qkv_bias": False,
        "hidden_dropout": 0.0,
        "rotary_percent": 1.0,
        "moe_router_load_balancing_type": "seq_aux_loss",
        "moe_router_enable_expert_bias": True,
        "moe_router_pre_softmax": True,
        "multi_latent_attention": True,
        "mtp_loss_scaling_factor": 0.3,
    },
    weight_converters=[
        # common weights
        RenameConverOp(hf_names=".input_layernorm.weight", mca_names=".input_layernorm.weight"),
        RenameConverOp(hf_names="lm_head.weight", mca_names="output_layer.weight"),
        RenameConverOp(hf_names="model.embed_tokens.weight", mca_names="embedding.word_embeddings.weight"),
        # attn output
        RenameConverOp(hf_names=".self_attn.o_proj.weight", mca_names=".self_attention.linear_proj.weight"),
        # MLA related weights
        RenameConverOp(hf_names=".self_attn.q_a_proj.weight", mca_names=".self_attention.linear_q_down_proj.weight"),
        RenameConverOp(hf_names=".self_attn.q_proj.weight", mca_names=".self_attention.linear_q_proj.weight"),
        RenameConverOp(
            hf_names=".self_attn.q_a_proj.weight_scale_inv",
            mca_names=".self_attn.q_a_proj.weight_scale_inv._extra_state",
        ),
        RenameConverOp(
            hf_names=".self_attn.q_a_layernorm.weight",
            mca_names=".self_attention.linear_q_up_proj.layer_norm_weight",
        ),
        RenameConverOp(hf_names=".self_attn.q_b_proj.weight", mca_names=".self_attention.linear_q_up_proj.weight"),
        RenameConverOp(
            hf_names=".self_attn.q_b_proj.weight_scale_inv",
            mca_names=".self_attention.q_b_proj.weight_scale_inv._extra_state",
        ),
        RenameConverOp(
            hf_names=".self_attn.kv_a_proj_with_mqa.weight", mca_names=".self_attention.linear_kv_down_proj.weight"
        ),
        RenameConverOp(
            hf_names=".self_attn.kv_a_proj_with_mqa.weight_scale_inv",
            mca_names=".self_attention.kv_a_proj_with_mqa.weight_scale_inv._extra_state",
        ),
        RenameConverOp(
            hf_names=".self_attn.kv_a_layernorm.weight",
            mca_names=".self_attention.linear_kv_up_proj.layer_norm_weight",
        ),
        RenameConverOp(hf_names=".self_attn.kv_b_proj.weight", mca_names=".self_attention.linear_kv_up_proj.weight"),
        RenameConverOp(
            hf_names=".self_attn.kv_b_proj.weight_scale_inv",
            mca_names=".self_attention.kv_b_proj.weight_scale_inv._extra_state",
        ),
        # MoE related weights
        # shared moe
        StackConverOp(
            hf_names=[".mlp.gate_proj.weight", ".mlp.up_proj.weight"], mca_names=[".mlp.linear_fc1.weight"], dim=0
        ),
        RenameConverOp(hf_names=".mlp.down_proj.weight", mca_names=".mlp.linear_fc2.weight"),
        RenameConverOp(hf_names=".mlp.gate_proj.weight_scale_inv", mca_names=".mlp.gate_proj.weight_scale_inv"),
        RenameConverOp(hf_names=".mlp.up_proj.weight_scale_inv", mca_names=".mlp.up_proj.weight_scale_inv"),
        RenameConverOp(hf_names=".mlp.down_proj.weight_scale_inv", mca_names=".mlp.down_proj.weight_scale_inv"),
        # local moe
        # the weight name in deepseek-v3 of shared expert is different......
        StackConverOp(hf_names=[".gate_proj.weight", ".up_proj.weight"], mca_names=[".linear_fc1.weight"], dim=0),
        RenameConverOp(hf_names=".down_proj.weight", mca_names=".linear_fc2.weight"),
        StackConverOp(
            hf_names=[".mlp.shared_experts.gate_proj.weight", ".mlp.shared_experts.up_proj.weight"],
            mca_names=[".mlp.shared_experts.linear_fc1.weight"],
            dim=0,
        ),
        RenameConverOp(
            hf_names=".mlp.shared_experts.down_proj.weight", mca_names=".mlp.shared_experts.linear_fc2.weight"
        ),
        RenameConverOp(hf_names=".mlp.gate.e_score_correction_bias", mca_names=".mlp.router.expert_bias"),
        RenameConverOp(
            hf_names=".mlp.shared_experts.gate_proj.weight_scale_inv",
            mca_names=".mlp.shared_experts.gate_proj.weight_scale_inv",
        ),
        RenameConverOp(
            hf_names=".mlp.shared_experts.up_proj.weight_scale_inv",
            mca_names=".mlp.shared_experts.up_proj.weight_scale_inv",
        ),
        RenameConverOp(
            hf_names=".mlp.shared_experts.down_proj.weight_scale_inv",
            mca_names=".mlp.shared_experts.down_proj.weight_scale_inv",
        ),
        RenameConverOp(hf_names=".down_proj.weight_scale_inv", mca_names=".down_proj.weight_scale_inv"),
        RenameConverOp(hf_names=".up_proj.weight_scale_inv", mca_names=".up_proj.weight_scale_inv"),
        RenameConverOp(hf_names=".gate_proj.weight_scale_inv", mca_names=".gate_proj.weight_scale_inv"),
        # normal transformer weights
        # RenameConverOp(hf_names=".embed_tokens.weight", mca_names=".embed_tokens.weight"),
        RenameConverOp(hf_names=".enorm.weight", mca_names=".enorm.weight"),
        RenameConverOp(hf_names=".hnorm.weight", mca_names=".hnorm.weight"),
        RenameConverOp(hf_names=".eh_proj.weight", mca_names=".eh_proj.weight"),
        RenameConverOp(hf_names=".shared_head.norm.weight", mca_names=".final_layernorm.weight"),
        RenameConverOp(
            hf_names=".self_attn.o_proj.weight_scale_inv", mca_names=".self_attn.o_proj.weight_scale_inv"
        ),
        RenameConverOp(hf_names=".post_attention_layernorm.weight", mca_names=".pre_mlp_layernorm.weight"),
        RenameConverOp(hf_names="model.norm.weight", mca_names="decoder.final_layernorm.weight"),
        RenameConverOp(hf_names=".mlp.gate.weight", mca_names=".mlp.router.weight"),
    ],
)


__all__ = ["DeepSeekV3Model"]
