import re
from dataclasses import dataclass

import torch

from ..auto.config_auto import register_config
from ..auto.modeling_auto import register_model
from ..converter.dist_converter import (
    DistParallelConfig,
    default_dist_config,
    gdn_dist_config,
    mtp_config,
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
    StackedTensors,
    ZeroCenteredRMSNormConverOp,
    register_template,
)
from ..qwen3_5 import Qwen3_5_GDNConverOp, Qwen3_5Template
from ..qwen3_5.config_qwen3_5 import Qwen3_5Config
from ..qwen3_5.modeling_qwen3_5 import Qwen3_5Model


@dataclass
class SplitConverOp(ConverOp):
    def __post_init__(self):
        super().__post_init__()
        assert len(self.hf_names) == 1, f"SplitConverOp only support one name {self.hf_names}"

    @property
    def mca_config(self) -> "Qwen3_5Config":
        return self._mca_config

    @mca_config.setter
    def mca_config(self, value: "Qwen3_5Config"):
        self._mca_config = value
        if len(self.mca_names) == 1:
            mca_name = self.mca_names[0]
            num_splits = self._mca_config.num_moe_experts
            self.mca_names = [str(i) + mca_name for i in range(num_splits)]

    def _hf_to_mca(self, weights):
        return list(torch.unbind(weights[0], dim=0))

    def _mca_to_hf(self, weights):
        if isinstance(weights[0], StackedTensors):
            return torch.stack([torch.cat(weight.tensors) for weight in weights], dim=0)
        return torch.stack(weights, dim=0)


@dataclass
class SplitStackConverOp(SplitConverOp):
    def _hf_to_mca(self, weights):
        return [StackedTensors(torch.chunk(w, 2, dim=0), dim=0) for w in torch.unbind(weights[0], dim=0)]


register_config("qwen3_5_moe", Qwen3_5Config)
register_model("qwen3_5_moe", Qwen3_5Model)
register_dist_config(
    "qwen3_5_moe",
    default_dist_config.merge_configs(shared_moe_dist_config)
    .merge_configs(gdn_dist_config)
    .merge_configs(mtp_config)
    .merge_configs(
        DistParallelConfig(
            pre_process_weights=["vision_model.*"],
            duplicated_weights=["vision_model.*"],
        )
    ),
)


@dataclass
class Qwen3_5_MoETemplate(Qwen3_5Template):
    def get_lora_conver_op(self, name, pattern_to_conver_ops: dict[str, ConverOp], lora_rank: int):
        lora_name = name[name.find(".lora") :]
        cache_key = f"{name}_{lora_rank}"
        if cache_key in self._lora_op_cache:
            return self._lora_op_cache[cache_key]

        name = name[: name.find(".lora")] + ".weight"
        op = self.get_conver_op(name, pattern_to_conver_ops)
        kwargs = {}
        if isinstance(op, RenameConverOp):
            op_class = RenameConverOp
        elif isinstance(op, (SplitConverOp, SplitStackConverOp)):
            op_class = type(op)
        elif "lora_A" in lora_name:
            op_class = CopyConverOp
        elif isinstance(op, StackConverOp):
            op_class = StackConverOp
            kwargs = {"dim": op.dim}
        elif isinstance(op, GatedQKVConverOp):
            op_class = GatedQKVConverOp
            kwargs = {"hidden_size": lora_rank}
        elif isinstance(op, Qwen3_5_GDNConverOp):
            op_class = type(op)
        else:
            raise ValueError(f"cannot find lora conver op for {name} in {pattern_to_conver_ops}")
        lora_hf_names = [hf_name if hf_name.endswith(".weight") else hf_name + ".weight" for hf_name in op.hf_names]
        lora_hf_names = [hf_name.replace(".weight", lora_name) for hf_name in lora_hf_names]
        lora_mca_names = [mca_name.replace(".weight", lora_name) for mca_name in op.mca_names]
        lora_op = op_class(
            hf_names=lora_hf_names,
            mca_names=lora_mca_names,
            _mca_config=op.mca_config,
            **kwargs,
        )
        self._lora_op_cache[cache_key] = lora_op
        return lora_op


register_template(
    "qwen3_5_moe",
    hf_layer_prefix="model.language_model.layers.",
    hf_moe_prefix=".mlp.experts.",
    template_class=Qwen3_5_MoETemplate,
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
        "intermediate_size": "ffn_hidden_size",
        "tie_word_embeddings": "tie_embeddings_and_output_weights",
        # MoE related
        "moe_intermediate_size": "moe_ffn_hidden_size",
        "decoder_sparse_step": "moe_layer_freq",
        "num_experts": "num_moe_experts",
        "num_experts_per_tok": "moe_router_topk",
        "router_aux_loss_coef": "moe_aux_loss_coeff",
        "shared_expert_intermediate_size": "moe_shared_expert_intermediate_size",
        # vit related
        "vision_start_token_id": "vision_start_token_id",
        "vision_end_token_id": "vision_end_token_id",
        "vision_token_id": "vision_token_id",
        "image_token_id": "image_token_id",
        "video_token_id": "video_token_id",
        "vision_config": "vision_config",
        "rope_parameters": "rope_scaling",
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
        "mtp_num_hidden_layers": "mtp_num_layers",
    },
    constant_mca_config={
        "swiglu": True,
        "position_embedding_type": "mrope",
        "normalization": "RMSNorm",
        "add_bias_linear": False,
        "hidden_dropout": 0.0,
        "moe_router_load_balancing_type": "aux_loss",
        "moe_router_pre_softmax": False,
        "qk_layernorm": True,
        "moe_shared_expert_gate": True,
        "layernorm_zero_centered_gamma": True,
        "hetereogenous_dist_checkpoint": True,
        "attention_output_gate": True,
        "experimental_attention_variant": "gated_delta_net",
        "mtp_loss_scaling_factor": 0.3,
    },
    weight_converters=[
        RenameConverOp(hf_names="lm_head.weight", mca_names="output_layer.weight"),
        RenameConverOp(
            hf_names="model.language_model.embed_tokens.weight", mca_names="embedding.word_embeddings.weight"
        ),
        RenameConverOp(hf_names=".input_layernorm.weight", mca_names=".self_attention.linear_qkv.layer_norm_weight"),
        RenameConverOp(hf_names=".post_attention_layernorm.weight", mca_names=".pre_mlp_layernorm.weight"),
        RenameConverOp(hf_names="model.language_model.norm.weight", mca_names="decoder.final_layernorm.weight"),
        # Stacked experts
        RenameConverOp(hf_names=".mlp.gate.weight", mca_names=".mlp.router.weight"),
        SplitStackConverOp(hf_names="gate_up_proj", mca_names=".linear_fc1.weight"),
        SplitConverOp(hf_names="down_proj", mca_names=".linear_fc2.weight"),
        RenameConverOp(hf_names=".down_proj.weight", mca_names=".linear_fc2.weight"),
        StackConverOp(hf_names=[".gate_proj.weight", ".up_proj.weight"], mca_names=".linear_fc1.weight", dim=0),
        # Shared experts
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
        Qwen3_5_GDNConverOp(
            hf_names=[
                ".linear_attn.in_proj_qkv.weight",
                ".linear_attn.in_proj_z.weight",
                ".linear_attn.in_proj_b.weight",
                ".linear_attn.in_proj_a.weight",
            ],
            mca_names=".self_attention.in_proj.weight",
        ),
        GDNConv1dConverOp(hf_names=".linear_attn.conv1d.weight", mca_names=".self_attention.conv1d.weight"),
        RenameConverOp(hf_names=".linear_attn.dt_bias", mca_names=".self_attention.dt_bias"),
        RenameConverOp(hf_names=".linear_attn.A_log", mca_names=".self_attention.A_log"),
        ZeroCenteredRMSNormConverOp(
            hf_names=".linear_attn.norm.weight", mca_names=".self_attention.out_norm.weight"
        ),
        RenameConverOp(hf_names=".linear_attn.out_proj.weight", mca_names=".self_attention.out_proj.weight"),
        # vit related
        RenameConverOp(hf_names="model.visual.{}", mca_names="vision_model.{}"),
        # mtp related
        RenameConverOp(hf_names=".pre_fc_norm_embedding.weight", mca_names=".enorm.weight"),
        RenameConverOp(hf_names=".pre_fc_norm_hidden.weight", mca_names=".hnorm.weight"),
        RenameConverOp(hf_names=".fc.weight", mca_names=".eh_proj.weight"),
        RenameConverOp(hf_names=".norm.weight", mca_names=".final_layernorm.weight"),
    ],
)
