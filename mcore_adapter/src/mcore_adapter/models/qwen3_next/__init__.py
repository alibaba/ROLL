import re
from dataclasses import dataclass

import torch

from ..converter.dist_converter import (
    default_dist_config,
    gdn_dist_config,
    register_dist_config,
    shared_moe_dist_config,
)
from ..converter.template import (
    ConverOp,
    CopyConverOp,
    GatedQKVConverOp,
    GDNConv1dConverOp,
    RenameConverOp,
    StackConverOp,
    Template,
    register_template,
)
from .config_qwen3_next import Qwen3NextConfig
from .modeling_qwen3_next import Qwen3NextModel


@dataclass
class DropConverOp(ConverOp):
    def _hf_to_mca(self, weights):
        return []

    def _mca_to_hf(self, weights):
        return []


@dataclass
class NextGDNConverOp(ConverOp):
    def __post_init__(self):
        super().__post_init__()
        assert len(self.hf_names) == 2, f"GDNConverOp only support two hf_names {self.hf_names}"
        assert len(self.mca_names) == 1, f"GDNConverOp only support one mca_name {self.mca_names}"

    def _hf_to_mca(self, weights):
        qkvz_weight, ba_weight = weights
        hidden_size = self.mca_config.hidden_size
        qk_head_dim = self.mca_config.linear_key_head_dim
        v_head_dim = self.mca_config.linear_value_head_dim
        num_qk_heads = self.mca_config.linear_num_key_heads
        num_v_heads = self.mca_config.linear_num_value_heads
        qk_dim = qk_head_dim * num_qk_heads
        v_dim = v_head_dim * num_v_heads

        qkvz_reshaped = qkvz_weight.reshape(num_qk_heads, (qk_dim * 2 + v_dim * 2) // num_qk_heads, -1)
        ba_reshaped = ba_weight.reshape(num_qk_heads, 2 * num_v_heads // num_qk_heads, -1)
        q, k, v, z = torch.split(
            qkvz_reshaped,
            [
                qk_head_dim,
                qk_head_dim,
                num_v_heads // num_qk_heads * v_head_dim,
                num_v_heads // num_qk_heads * v_head_dim,
            ],
            dim=1,
        )
        b, a = torch.split(ba_reshaped, [num_v_heads // num_qk_heads, num_v_heads // num_qk_heads], dim=1)
        q, k, v, z, b, a = [weight.reshape(-1, hidden_size) for weight in [q, k, v, z, b, a]]
        in_proj_weight = torch.cat([q, k, v, z, b, a], dim=0).reshape(-1, hidden_size)
        return in_proj_weight

    def _mca_to_hf(self, weights):
        in_proj_weight = weights[0]
        hidden_size = self.mca_config.hidden_size
        qk_head_dim = self.mca_config.linear_key_head_dim
        v_head_dim = self.mca_config.linear_value_head_dim
        num_qk_heads = self.mca_config.linear_num_key_heads
        num_v_heads = self.mca_config.linear_num_value_heads
        qk_dim = qk_head_dim * num_qk_heads
        v_dim = v_head_dim * num_v_heads

        in_proj_weight = in_proj_weight.reshape(-1, hidden_size)
        q, k, v, z, b, a = torch.split(in_proj_weight, [qk_dim, qk_dim, v_dim, v_dim, num_v_heads, num_v_heads], dim=0)
        q, k, v, z, b, a = [weight.reshape(num_qk_heads, -1, hidden_size) for weight in [q, k, v, z, b, a]]
        qkvz_weight = torch.cat([q, k, v, z], dim=1).reshape(-1, hidden_size)
        ba_weight = torch.cat([b, a], dim=1).reshape(-1, hidden_size)
        return [qkvz_weight, ba_weight]


@dataclass
class ZeroCenteredRMSNormConverOp(ConverOp):
    def __post_init__(self):
        super().__post_init__()
        assert len(self.hf_names) == 1, f"ZeroCenteredRMSNormConverOp only support one name {self.hf_names}"
        assert len(self.mca_names) == 1, f"ZeroCenteredRMSNormConverOp only support one name {self.mca_names}"

    def _hf_to_mca(self, weights):
        return weights[0].clone() - 1

    def _mca_to_hf(self, weights):
        return weights[0].clone() + 1


register_dist_config(
    "qwen3_next", default_dist_config.merge_configs(shared_moe_dist_config).merge_configs(gdn_dist_config)
)


@dataclass
class Qwen3NextTemplate(Template):
    def add_hf_weight(self, name, weight):
        pattern = r"^model\.layers\.(\d+)\.input_layernorm\.weight$"
        match = re.match(pattern, name)
        layer_idx = int(match.group(1)) if match else None
        if layer_idx is not None and self.mca_config.layer_types[layer_idx] == "linear_attention":
            return {f"decoder.layers.{layer_idx}.self_attention.in_proj.layer_norm_weight": weight}
        return super().add_hf_weight(name, weight)

    def add_mca_weight(self, name, weight, **kwargs):
        pattern = r"^decoder\.layers\.(\d+)\.self_attention\.in_proj\.layer_norm_weight$"
        match = re.match(pattern, name)
        if not match:
            return super().add_mca_weight(name, weight, **kwargs)
        layer_idx = int(match.group(1)) if match else None
        return {f"model.layers.{layer_idx}.input_layernorm.weight": weight}

    def get_lora_conver_op(self, name, pattern_to_conver_ops: dict[str, ConverOp], lora_rank: int):
        lora_name = name[name.find(".lora") :]
        name = name[: name.find(".lora")] + ".weight"
        op = self.get_conver_op(name, pattern_to_conver_ops)
        if isinstance(op, RenameConverOp):
            op_class = RenameConverOp
            kwargs = {}
        elif "lora_A" in lora_name:
            op_class = CopyConverOp
            kwargs = {}
        elif isinstance(op, StackConverOp):
            op_class = StackConverOp
            kwargs = {"dim": op.dim}
        elif isinstance(op, GatedQKVConverOp):
            op_class = GatedQKVConverOp
            kwargs = {"hidden_size": lora_rank}
        else:
            raise ValueError(f"can not find lora conver op for {name} in {pattern_to_conver_ops}")
        return op_class(
            hf_names=[hf_name.replace(".weight", lora_name) for hf_name in op.hf_names],
            mca_names=[mca_name.replace(".weight", lora_name) for mca_name in op.mca_names],
            _mca_config=op.mca_config,
            **kwargs,
        )


register_template(
    "qwen3_next",
    hf_layer_prefix="model.layers.",
    hf_moe_prefix=".mlp.experts.",
    template_class=Qwen3NextTemplate,
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
        "num_experts": "num_moe_experts",
        "num_experts_per_tok": "moe_router_topk",
        "router_aux_loss_coef": "moe_aux_loss_coeff",
        "shared_expert_intermediate_size": "moe_shared_expert_intermediate_size",
        # Linear attention
        "linear_conv_kernel_dim": "linear_conv_kernel_dim",
        "linear_key_head_dim": "linear_key_head_dim",
        "linear_value_head_dim": "linear_value_head_dim",
        "linear_num_key_heads": "linear_num_key_heads",
        "linear_num_value_heads": "linear_num_value_heads",
        # other special configs
        # "mlp_only_layers": "mlp_only_layers",
        "layer_types": "layer_types",
        "full_attention_interval": "linear_attention_freq",
    },
    constant_mca_config={
        "swiglu": True,
        "position_embedding_type": "rope",
        "normalization": "RMSNorm",
        "add_bias_linear": False,
        "hidden_dropout": 0.0,
        "rotary_percent": 1.0,
        "moe_router_load_balancing_type": "aux_loss",
        "moe_router_pre_softmax": False,
        "qk_layernorm": True,
        "moe_shared_expert_gate": True,
        "layernorm_zero_centered_gamma": True,
        "hetereogenous_dist_checkpoint": True,
        "attention_output_gate": True,
        "experimental_attention_variant": "gated_delta_net",
    },
    weight_converters=[
        RenameConverOp(hf_names="lm_head.weight", mca_names="output_layer.weight"),
        RenameConverOp(hf_names="model.embed_tokens.weight", mca_names="embedding.word_embeddings.weight"),
        RenameConverOp(hf_names=".input_layernorm.weight", mca_names=".self_attention.linear_qkv.layer_norm_weight"),
        RenameConverOp(hf_names=".post_attention_layernorm.weight", mca_names=".pre_mlp_layernorm.weight"),
        RenameConverOp(hf_names="model.norm.weight", mca_names="decoder.final_layernorm.weight"),
        # Experts
        RenameConverOp(hf_names=".down_proj.weight", mca_names=".linear_fc2.weight"),
        StackConverOp(hf_names=[".gate_proj.weight", ".up_proj.weight"], mca_names=".linear_fc1.weight", dim=0),
        RenameConverOp(hf_names=".mlp.gate.weight", mca_names=".mlp.router.weight"),
        RenameConverOp(
            hf_names=".mlp.shared_expert.down_proj.weight", mca_names=".mlp.shared_experts.linear_fc2.weight"
        ),
        RenameConverOp(hf_names=".mlp.shared_expert_gate.weight", mca_names=".mlp.shared_experts.gate_weight"),
        StackConverOp(
            hf_names=[".mlp.shared_expert.gate_proj.weight", ".mlp.shared_expert.up_proj.weight"],
            mca_names=".mlp.shared_experts.linear_fc1.weight",
            dim=0,
        ),
        # Multi-head attention
        GatedQKVConverOp(
            hf_names=[".self_attn.q_proj.weight", ".self_attn.k_proj.weight", ".self_attn.v_proj.weight"],
            mca_names=".self_attention.linear_qkv.weight",
        ),
        RenameConverOp(hf_names=".self_attn.o_proj.weight", mca_names=".self_attention.linear_proj.weight"),
        RenameConverOp(hf_names=".self_attn.q_norm.weight", mca_names=".self_attention.q_layernorm.weight"),
        RenameConverOp(hf_names=".self_attn.k_norm.weight", mca_names=".self_attention.k_layernorm.weight"),
        # Linear attention
        NextGDNConverOp(
            hf_names=[".linear_attn.in_proj_qkvz.weight", ".linear_attn.in_proj_ba.weight"],
            mca_names=".self_attention.in_proj.weight",
        ),
        GDNConv1dConverOp(hf_names=".linear_attn.conv1d.weight", mca_names=".self_attention.conv1d.weight"),
        RenameConverOp(hf_names=".linear_attn.dt_bias", mca_names=".self_attention.dt_bias"),
        RenameConverOp(hf_names=".linear_attn.A_log", mca_names=".self_attention.A_log"),
        ZeroCenteredRMSNormConverOp(hf_names=".linear_attn.norm.weight", mca_names=".self_attention.out_norm.weight"),
        RenameConverOp(hf_names=".linear_attn.out_proj.weight", mca_names=".self_attention.out_proj.weight"),
        # MTP not support
        DropConverOp(hf_names="mtp.*", mca_names=[]),
    ],
)

__all__ = ["Qwen3NextConfig", "Qwen3NextModel"]
