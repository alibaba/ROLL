import re
from dataclasses import dataclass

import torch

from ..converter.convert_utils import StackedTensors
from ..converter.dist_converter import (
    DistParallelConfig,
    default_dist_config,
    gdn_dist_config,
    register_dist_config,
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
from .config_qwen3_5 import Qwen3_5Config
from .modeling_qwen3_5 import Qwen3_5Model


@dataclass
class DropConverOp(ConverOp):
    def _hf_to_mca(self, weights):
        return []

    def _mca_to_hf(self, weights):
        return []


@dataclass
class Qwen3_5_GDNConverOp(ConverOp):
    def __post_init__(self):
        super().__post_init__()
        assert len(self.hf_names) == 4, f"GDNConverOp only support four hf_names {self.hf_names}"
        assert len(self.mca_names) == 1, f"GDNConverOp only support one mca_name {self.mca_names}"

    def _hf_to_mca(self, weights):
        qkv_weight, z_weight, b_weight, a_weight = weights
        qk_head_dim = self.mca_config.linear_key_head_dim
        v_head_dim = self.mca_config.linear_value_head_dim
        num_qk_heads = self.mca_config.linear_num_key_heads
        num_v_heads = self.mca_config.linear_num_value_heads
        qk_dim = qk_head_dim * num_qk_heads
        v_dim = v_head_dim * num_v_heads

        q, k, v = torch.split(
            qkv_weight,
            [
                qk_dim,
                qk_dim,
                v_dim,
            ],
            dim=0,
        )
        z = z_weight.reshape(v_dim, -1)
        b = b_weight.reshape(num_v_heads, -1)
        a = a_weight.reshape(num_v_heads, -1)
        return StackedTensors(tensors=[q, k, v, z, b, a], dim=0)

    def _mca_to_hf(self, weights):
        if len(weights) == 1:
            assert isinstance(weights[0], StackedTensors)
            q, k, v, z, b, a = weights[0].tensors
            qkv = torch.cat([q, k, v], dim=0)
            return [qkv, z, b, a]


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
    "qwen3_5",
    default_dist_config.merge_configs(gdn_dist_config).merge_configs(
        DistParallelConfig(
            pre_process_weights=["vision_model.*"],
            duplicated_weights=["vision_model.*"],
        )
    ),
)


@dataclass
class Qwen3_5Template(Template):
    def adjust_config_hf_to_mca(self):
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

    def add_hf_weight(self, name, weight):
        pattern = r"^model\.language_model\.layers\.(\d+)\.input_layernorm\.weight$"
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
        return {f"model.language_model.layers.{layer_idx}.input_layernorm.weight": weight}

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
            raise ValueError(f"cannot find lora conver op for {name} in {pattern_to_conver_ops}")
        return op_class(
            hf_names=[hf_name.replace(".weight", lora_name) for hf_name in op.hf_names],
            mca_names=[mca_name.replace(".weight", lora_name) for mca_name in op.mca_names],
            _mca_config=op.mca_config,
            **kwargs,
        )


register_template(
    "qwen3_5",
    hf_layer_prefix="model.language_model.layers.",
    template_class=Qwen3_5Template,
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
    },
    constant_mca_config={
        "swiglu": True,
        "position_embedding_type": "mrope",
        "normalization": "RMSNorm",
        "add_bias_linear": False,
        "hidden_dropout": 0.0,
        "qk_layernorm": True,
        "layernorm_zero_centered_gamma": True,
        "hetereogenous_dist_checkpoint": True,
        "attention_output_gate": True,
        "linear_attention_type": "gated_delta_net",
    },
    weight_converters=[
        RenameConverOp(hf_names="lm_head.weight", mca_names="output_layer.weight"),
        RenameConverOp(
            hf_names="model.language_model.embed_tokens.weight", mca_names="embedding.word_embeddings.weight"
        ),
        RenameConverOp(hf_names=".input_layernorm.weight", mca_names=".self_attention.linear_qkv.layer_norm_weight"),
        RenameConverOp(hf_names=".post_attention_layernorm.weight", mca_names=".mlp.linear_fc1.layer_norm_weight"),
        RenameConverOp(hf_names="model.language_model.norm.weight", mca_names="decoder.final_layernorm.weight"),
        RenameConverOp(hf_names=".mlp.down_proj.weight", mca_names=".mlp.linear_fc2.weight"),
        StackConverOp(
            hf_names=[".mlp.gate_proj.weight", ".mlp.up_proj.weight"], mca_names=".mlp.linear_fc1.weight", dim=0
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
        DropConverOp(hf_names="mtp.*", mca_names=[]),
    ],
)

__all__ = ["Qwen3_5Config", "Qwen3_5Model"]
