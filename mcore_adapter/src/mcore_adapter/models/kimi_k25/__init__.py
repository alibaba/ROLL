from dataclasses import dataclass

from ..converter.dist_converter import DistParallelConfig, mla_dist_config, register_dist_config
from ..converter.template import (
    RenameConverOp,
    StackConverOp,
    VisionTemplate,
    register_template,
)
from ..deepseek_v3 import DeepSeekV3Template
from .modeling_kimi_k25 import KimiK2_5Model


register_dist_config(
    "kimi_k25",
    mla_dist_config.merge_configs(
        DistParallelConfig(
            pre_process_weights=["vision_tower.*", "mm_projector.*"],
            duplicated_weights=["vision_tower.*", "mm_projector.*"],
        )
    )
)


@dataclass
class KimiK2_5Template(VisionTemplate, DeepSeekV3Template):

    def convert_hf_to_mca_config_kws(self, hf_config, **kw_args):
        rope_scaling = self.get_hf_config_value(hf_config, "text_config.rope_scaling")
        if rope_scaling:
            if rope_scaling.get("original_max_position_embeddings", None):
                kw_args["original_max_position_embeddings"] = rope_scaling["original_max_position_embeddings"]
            if rope_scaling.get("type", None):
                rope_type = rope_scaling["type"]
                kw_args["rope_type"] = rope_type
                assert rope_type == "yarn", f"only support yarn rope scaling now. but got {rope_type}"
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
        else:
            kw_args["rope_type"] = "rope"
            kw_args["rotary_scaling_factor"] = 1.0

        n_shared_experts = self.get_hf_config_value(hf_config, "text_config.n_shared_experts")
        moe_intermediate_size = self.get_hf_config_value(hf_config, "text_config.moe_intermediate_size")
        if n_shared_experts:
            kw_args["moe_shared_expert_intermediate_size"] = n_shared_experts * moe_intermediate_size

        res = super().convert_hf_to_mca_config_kws(hf_config, **kw_args)

        # parent DeepSeekV3Template reads top-level rope_scaling which is None for VLM
        # (nested under text_config), causing it to overwrite to "rope". Restore here.
        if rope_scaling:
            res["rope_type"] = rope_scaling.get("type", "yarn")
            res["rotary_scaling_factor"] = rope_scaling.get("factor", 1.0)

        first_k_dense_replace = self.get_hf_config_value(hf_config, "text_config.first_k_dense_replace")
        if first_k_dense_replace:
            assert first_k_dense_replace < res["num_layers"], "first_k_dense_layers is out of range."
            res["moe_layer_freq"] = [0] * first_k_dense_replace + [1] * (res["num_layers"] - first_k_dense_replace)

        return res


register_template(
    "kimi_k25",
    hf_layer_prefix="language_model.model.layers.",
    hf_moe_prefix=".mlp.experts.",
    template_class=KimiK2_5Template,
    config_hf_to_mca={
        # text related
        "max_position_embeddings": "max_sequence_length",
        "hidden_size": "hidden_size",
        "num_attention_heads": "num_attention_heads",
        "num_key_value_heads": "num_query_groups",
        "num_hidden_layers": "num_layers",
        "vocab_size": "padded_vocab_size",
        "n_routed_experts": "num_moe_experts",
        "num_experts_per_tok": "moe_router_topk",
        "scoring_func": "moe_router_score_function",
        "rms_norm_eps": "layernorm_epsilon",
        "intermediate_size": "ffn_hidden_size",
        "moe_intermediate_size": "moe_ffn_hidden_size",
        "tie_word_embeddings": "tie_embeddings_and_output_weights",
        "rope_theta": "rotary_base",
        "rope_parameters": "rope_parameters",
        "q_lora_rank": "q_lora_rank",
        "kv_lora_rank": "kv_lora_rank",
        "v_head_dim": "v_head_dim",
        "qk_nope_head_dim": "qk_head_dim",
        "qk_rope_head_dim": "qk_pos_emb_head_dim",
        "n_group": "moe_router_num_groups",
        "topk_group": "moe_router_group_topk",
        "routed_scaling_factor": "moe_router_topk_scaling_factor",
        # vision related
        "vision_config": "vision_config",
    },
    constant_mca_config={
        "swiglu": True,
        "position_embedding_type": "rope",
        "multi_latent_attention": True,
        "qk_layernorm": True,
        "moe_router_enable_expert_bias": True,
        "add_bias_linear": False,
        "add_qkv_bias": False,
        "normalization": "RMSNorm",
        "moe_router_load_balancing_type": "seq_aux_loss",
    },
    weight_converters=[
        # text
        # text -> common
        RenameConverOp(hf_names="language_model.model.embed_tokens.weight", mca_names="embedding.word_embeddings.weight"),
        RenameConverOp(hf_names=".input_layernorm.weight", mca_names=".input_layernorm.weight"),
        # two names for post_attention_layernorm, may change in the template
        RenameConverOp(hf_names=".post_attention_layernorm.weight", mca_names=".pre_mlp_layernorm.weight"),
        RenameConverOp(hf_names="language_model.model.norm.weight", mca_names="decoder.final_layernorm.weight"),
        RenameConverOp(hf_names="language_model.lm_head.weight", mca_names="output_layer.weight"),
        # text -> mla
        RenameConverOp(
            hf_names=".self_attn.q_a_proj.weight", mca_names=".self_attention.linear_q_down_proj.weight"
        ),
        RenameConverOp(
            hf_names=".self_attn.q_a_layernorm.weight", mca_names=".self_attention.linear_q_up_proj.layer_norm_weight",
        ),
        RenameConverOp(
            hf_names=".self_attn.q_b_proj.weight", mca_names=".self_attention.linear_q_up_proj.weight"
        ),
        RenameConverOp(
            hf_names=".self_attn.kv_a_proj_with_mqa.weight", mca_names=".self_attention.linear_kv_down_proj.weight"
        ),
        RenameConverOp(
            hf_names=".self_attn.kv_a_layernorm.weight", mca_names=".self_attention.linear_kv_up_proj.layer_norm_weight",
        ),
        RenameConverOp(
            hf_names=".self_attn.kv_b_proj.weight", mca_names=".self_attention.linear_kv_up_proj.weight"
        ),
        RenameConverOp(
            hf_names=".self_attn.o_proj.weight", mca_names=".self_attention.linear_proj.weight"
        ),
        # text -> mlp
        # text -> dense_mlp
        StackConverOp(
            hf_names=[".mlp.gate_proj.weight", ".mlp.up_proj.weight"], 
            mca_names=[".mlp.linear_fc1.weight"], 
            dim=0
        ),
        RenameConverOp(hf_names=".mlp.down_proj.weight", mca_names=".mlp.linear_fc2.weight"),
        # text -> moe_mlp
        StackConverOp(
            hf_names=[".gate_proj.weight", ".up_proj.weight"], 
            mca_names=[".linear_fc1.weight"], 
            dim=0
        ),
        RenameConverOp(hf_names=".down_proj.weight", mca_names=".linear_fc2.weight"),
        StackConverOp(
            hf_names=[".mlp.shared_experts.gate_proj.weight", ".mlp.shared_experts.up_proj.weight"],
            mca_names=[".mlp.shared_experts.linear_fc1.weight"],
            dim=0,
        ),
        RenameConverOp(
            hf_names=".mlp.shared_experts.down_proj.weight", mca_names=".mlp.shared_experts.linear_fc2.weight"
        ),
        RenameConverOp(hf_names=".mlp.gate.weight", mca_names=".mlp.router.weight"),
        RenameConverOp(hf_names=".mlp.gate.e_score_correction_bias", mca_names=".mlp.router.expert_bias"),
        # vision
        RenameConverOp(hf_names="vision_tower.{}", mca_names="vision_tower.{}"),
        RenameConverOp(hf_names="mm_projector.{}", mca_names="mm_projector.{}"),
    ],
)


__all__ = ["KimiK2_5Model"]
