from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from ..auto.config_auto import register_config
from ..auto.modeling_auto import register_model
from ..converter.convert_utils import (
    convert_to_hf_prefix,
    get_mca_moe_index,
    get_mca_weight_prefix,
    remove_mca_weight_prefix,
)
from ..converter.dist_converter import (
    DistParallelConfig,
    default_dist_config,
    register_dist_config,
    shared_moe_dist_config,
)
from ..converter.template import (
    ConverOp,
    QKVBiasConverOp,
    QKVConverOp,
    RenameConverOp,
    StackedTensors,
    Template,
    register_template,
)
from ..qwen3_vl import Qwen3VLConfig, Qwen3VLModel, Qwen3VLTemplate


if TYPE_CHECKING:
    from megatron.core.transformer import TransformerConfig


@dataclass
class SplitConverOp(ConverOp):
    def __post_init__(self):
        super().__post_init__()
        assert len(self.hf_names) == 1, f"SplitConverOp only support one name {self.hf_names}"

    @property
    def mca_config(self) -> "TransformerConfig":
        return self._mca_config

    @mca_config.setter
    def mca_config(self, value: "TransformerConfig"):
        self._mca_config = value
        if len(self.mca_names) == 1:
            mca_name = self.mca_names[0]
            num_splits = self._mca_config.num_moe_experts
            self.mca_names = [str(i) + mca_name for i in range(num_splits)]

    def _hf_to_mca(self, weights):
        return list(torch.unbind(weights[0].transpose(1, 2).contiguous(), dim=0))

    def _mca_to_hf(self, weights):
        if isinstance(weights[0], StackedTensors):
            return torch.stack([torch.cat(weight.tensors) for weight in weights], dim=0).transpose(1, 2).contiguous()
        return torch.stack(weights, dim=0).transpose(1, 2).contiguous()


@dataclass
class SplitStackConverOp(SplitConverOp):
    def _hf_to_mca(self, weights):
        return [
            StackedTensors(torch.chunk(w, 2, dim=0), dim=0)
            for w in torch.unbind(weights[0].transpose(1, 2).contiguous(), dim=0)
        ]


register_config("qwen3_vl_moe", Qwen3VLConfig)
register_model("qwen3_vl_moe", Qwen3VLModel)
register_dist_config(
    "qwen3_vl_moe",
    default_dist_config.merge_configs(shared_moe_dist_config).merge_configs(
        DistParallelConfig(
            pre_process_weights=["vision_model.*"],
            duplicated_weights=["vision_model.*"],
        )
    ),
)


@dataclass
class Qwen3VLMoeTemplate(Qwen3VLTemplate):
    def add_mca_weight(self, name, weight, **kwargs):
        weight_prefix = get_mca_weight_prefix(name)
        original_name = remove_mca_weight_prefix(name)
        moe_layer_index = get_mca_moe_index(name)
        # Since experts weights are stacked in qwen3_vl_moe,
        # we need to add the moe index to the original name to
        # ensure all experts weights have the same weight_prefix
        if moe_layer_index is not None:
            original_name = str(moe_layer_index) + original_name
            weight_prefix = name[: -len(original_name)]
        if weight_prefix not in self.prefix_name_to_weight:
            self.prefix_name_to_weight[weight_prefix] = {}
        self.prefix_name_to_weight[weight_prefix][original_name] = weight
        prefix_weights = self.prefix_name_to_weight[weight_prefix]
        # However, when looking up the converter, we still use the original name without moe index
        # This is because mca_name_to_converter is built before mca_names reset which happens at
        # model converter init.
        original_name = remove_mca_weight_prefix(name)
        if ".lora_A." in original_name or ".lora_B." in original_name:
            op = self.get_lora_conver_op(original_name, self.mca_name_to_converter, **kwargs)
        else:
            op = self.get_conver_op(original_name, self.mca_name_to_converter)
        name_to_weight = {
            name: prefix_weights.pop(name)
            for name in list(prefix_weights.keys())
            if op.is_required_name(name, mca_name=True)
        }
        conver_res = op(name_to_weight, mca_to_hf=True)
        if conver_res is None:
            # not ready to convert
            self.prefix_name_to_weight[weight_prefix].update(name_to_weight)
            return conver_res
        hf_prefix = convert_to_hf_prefix(weight_prefix, self.hf_layer_prefix, self.hf_moe_prefix)
        return {hf_prefix + name: weight for name, weight in conver_res.items()}


register_template(
    "qwen3_vl_moe",
    hf_layer_prefix="model.language_model.layers.",
    hf_moe_prefix=".mlp.experts.",
    template_class=Qwen3VLMoeTemplate,
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
        RenameConverOp(
            hf_names="model.language_model.embed_tokens.weight", mca_names="embedding.word_embeddings.weight"
        ),
        RenameConverOp(hf_names=".input_layernorm.weight", mca_names=".self_attention.linear_qkv.layer_norm_weight"),
        RenameConverOp(hf_names=".self_attn.o_proj.weight", mca_names=".self_attention.linear_proj.weight"),
        RenameConverOp(hf_names=".self_attn.q_norm.weight", mca_names=".self_attention.q_layernorm.weight"),
        RenameConverOp(hf_names=".self_attn.k_norm.weight", mca_names=".self_attention.k_layernorm.weight"),
        RenameConverOp(hf_names=".post_attention_layernorm.weight", mca_names=".pre_mlp_layernorm.weight"),
        RenameConverOp(hf_names="model.language_model.norm.weight", mca_names="decoder.final_layernorm.weight"),
        SplitStackConverOp(hf_names="gate_up_proj", mca_names=".linear_fc1.weight"),
        SplitConverOp(hf_names="down_proj", mca_names=".linear_fc2.weight"),
        RenameConverOp(hf_names=".mlp.gate.weight", mca_names=".mlp.router.weight"),
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
