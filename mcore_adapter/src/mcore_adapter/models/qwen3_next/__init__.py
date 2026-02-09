import re
from dataclasses import dataclass

import torch

from ..converter.dist_converter import (
    DistParallelConfig,
    default_dist_config,
    register_dist_config,
    shared_moe_dist_config,
)
from ..converter.template import (
    ConverOp,
    CopyConverOp,
    QKVConverOp,
    RenameConverOp,
    StackConverOp,
    Template,
    register_template,
)
from .config_qwen3_next import Qwen3NextConfig
from .modeling_qwen3_next import Qwen3NextModel


@dataclass
class DropConverOp(ConverOp):
    def __init__(self, hf_names, mca_names):
        super().__init__(hf_names, mca_names)

    def _hf_to_mca(self, weights):
        return []

    def _mca_to_hf(self, weights):
        return []


@dataclass
class NextQKVConverOp(QKVConverOp):
    """query weight used for calculating query_states and gate"""
    def _hf_to_mca(self, weights):
        if self.hidden_size is None:
            self.hidden_size = self.mca_config.hidden_size
        q_weight, k_weight, v_weight = weights
        nh = self.mca_config.num_attention_heads
        ng = self.mca_config.num_query_groups
        dim = self.mca_config.kv_channels
        assert nh % ng == 0
        mca_qkv_weight = torch.cat(
            [
                q_weight.reshape((ng, dim * nh // ng * 2, -1)),
                k_weight.reshape((ng, dim, -1)),
                v_weight.reshape((ng, dim, -1)),
            ],
            dim=1,
        ).reshape((-1, self.hidden_size))
        return mca_qkv_weight

    def _mca_to_hf(self, weights):
        if self.hidden_size is None:
            self.hidden_size = self.mca_config.hidden_size
        qkv_weight = weights[0]
        ng = self.mca_config.num_query_groups
        nh = self.mca_config.num_attention_heads
        dim = self.mca_config.kv_channels
        qkv_weight = qkv_weight.reshape((ng, dim * (nh // ng * 2 + 2), -1))
        qkv_weights = torch.split(qkv_weight, [dim * nh // ng * 2, dim, dim], dim=1)
        q_weight = qkv_weights[0].reshape((-1, self.hidden_size))
        k_weight = qkv_weights[1].reshape((-1, self.hidden_size))
        v_weight = qkv_weights[2].reshape((-1, self.hidden_size))
        return [q_weight, k_weight, v_weight]


linear_attn_dist_config = DistParallelConfig(
    # TODO: support tensor parallel
    duplicated_weights=[
        ".self_attention.in_proj_qkvz.weight",
        ".self_attention.in_proj_ba.weight",
        ".self_attention.conv1d.weight",
        ".self_attention.dt_bias",
        ".self_attention.A_log",
        ".self_attention.norm.weight",
        ".self_attention.out_proj.weight",
        ".input_layernorm.weight",
    ]
)


register_dist_config(
    "qwen3_next", default_dist_config.merge_configs(shared_moe_dist_config).merge_configs(linear_attn_dist_config)
)


@dataclass
class Qwen3NextTemplate(Template):
    def add_hf_weight(self, name, weight):
        pattern = r"^model\.layers\.(\d+)\.input_layernorm\.weight$"
        match = re.match(pattern, name)
        layer_idx = int(match.group(1)) if match else None
        if layer_idx is not None and self.mca_config.layer_types[layer_idx] == "linear_attention":
            return {f"decoder.layers.{layer_idx}.input_layernorm.weight": weight}
        return super().add_hf_weight(name, weight)

    def add_mca_weight(self, name, weight, **kwargs):
        pattern = r"^decoder\.layers\.(\d+)\.input_layernorm\.weight$"
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
        elif isinstance(op, NextQKVConverOp):
            op_class = NextQKVConverOp
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
        "full_attention_interval": "full_attention_interval",
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
        "moe_use_shared_expert_gate": True,
        "layernorm_zero_centered_gamma": True,
        "hetereogenous_dist_checkpoint": True,
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
        NextQKVConverOp(
            hf_names=[".self_attn.q_proj.weight", ".self_attn.k_proj.weight", ".self_attn.v_proj.weight"],
            mca_names=".self_attention.linear_qkv.weight",
        ),
        RenameConverOp(hf_names=".self_attn.o_proj.weight", mca_names=".self_attention.linear_proj.weight"),
        RenameConverOp(hf_names=".self_attn.q_norm.weight", mca_names=".self_attention.q_layernorm.weight"),
        RenameConverOp(hf_names=".self_attn.k_norm.weight", mca_names=".self_attention.k_layernorm.weight"),
        # Linear attention
        RenameConverOp(hf_names=".linear_attn.in_proj_qkvz.weight", mca_names=".self_attention.in_proj_qkvz.weight"),
        RenameConverOp(hf_names=".linear_attn.in_proj_ba.weight", mca_names=".self_attention.in_proj_ba.weight"),
        RenameConverOp(hf_names=".linear_attn.conv1d.weight", mca_names=".self_attention.conv1d.weight"),
        RenameConverOp(hf_names=".linear_attn.dt_bias", mca_names=".self_attention.dt_bias"),
        RenameConverOp(hf_names=".linear_attn.A_log", mca_names=".self_attention.A_log"),
        RenameConverOp(hf_names=".linear_attn.norm.weight", mca_names=".self_attention.norm.weight"),
        RenameConverOp(hf_names=".linear_attn.out_proj.weight", mca_names=".self_attention.out_proj.weight"),
        # MTP not support
        DropConverOp(hf_names="mtp.*", mca_names=[]),
    ],
)


__all__ = ["Qwen3NextConfig", "Qwen3NextModel"]
