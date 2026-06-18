import re
from dataclasses import dataclass
from typing import Optional

import torch

from ..converter.convert_utils import (
    MCA_MTP_MOE_PREFIX,
    MCA_MTP_PREFIX,
    StackedTensors,
    convert_to_mca_prefix,
    get_weight_prefix,
    remove_mca_mtp_weight_prefix,
    remove_weight_prefix,
)
from ..converter.dist_converter import (
    DistParallelConfig,
    default_dist_config,
    gdn_dist_config,
    mtp_config,
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
    ZeroCenteredRMSNormConverOp,
    register_template,
)
from .config_qwen3_5 import Qwen3_5Config
from .modeling_qwen3_5 import Qwen3_5Model


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
        assert len(weights) == 1
        assert isinstance(weights[0], StackedTensors)
        q, k, v, z, b, a = weights[0].tensors
        qkv = torch.cat([q, k, v], dim=0)
        return [qkv, z, b, a]


register_dist_config(
    "qwen3_5",
    default_dist_config.merge_configs(gdn_dist_config)
    .merge_configs(mtp_config)
    .merge_configs(
        DistParallelConfig(
            pre_process_weights=["vision_model.*"],
            duplicated_weights=["vision_model.*"],
        )
    ),
)


@dataclass
class Qwen3_5Template(Template):
    def __post_init__(self):
        super().__post_init__()
        self.hf_ln_pattern = re.compile(r"^model\.language_model\.layers\.(\d+)\.input_layernorm\.weight$")
        self.mca_ln_pattern = re.compile(r"^decoder\.layers\.(\d+)\.self_attention\.in_proj\.layer_norm_weight$")

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
        match = re.match(self.hf_ln_pattern, name)
        layer_idx = int(match.group(1)) if match else None
        if layer_idx is not None and self.mca_config.layer_types[layer_idx] == "linear_attention":
            return {f"decoder.layers.{layer_idx}.self_attention.in_proj.layer_norm_weight": weight}
        if not name.startswith("mtp"):
            return super().add_hf_weight(name, weight)
        weight_prefix = "mtp.layers.0" if name.startswith("mtp.layers.0") else "mtp"
        if self.hf_moe_prefix is not None and self.hf_moe_prefix in name:
            weight_prefix = get_weight_prefix(name, "mtp.layers.", moe_prefix=self.hf_moe_prefix)
        original_name = name.removeprefix(weight_prefix)
        if weight_prefix not in self.prefix_name_to_weight:
            self.prefix_name_to_weight[weight_prefix] = {}
        self.prefix_name_to_weight[weight_prefix][original_name] = weight
        # weights in the same layer
        prefix_weights = self.prefix_name_to_weight[weight_prefix]
        op = self.get_conver_op(original_name, self.hf_name_to_converter)
        name_to_weight = {
            name: prefix_weights.pop(name)
            for name in list(prefix_weights.keys())
            if op.is_required_name(name, mca_name=False)
        }
        conver_res = op(name_to_weight, mca_to_hf=False)
        if conver_res is None:
            # not ready to convert
            self.prefix_name_to_weight[weight_prefix].update(name_to_weight)
            return conver_res
        has_transformer_layer = "self_attention" in name or "mlp" in name or "input_layernorm" in name
        mca_prefix = "mtp.layers.0" + (".mtp_model_layer" if has_transformer_layer else "")
        if self.hf_moe_prefix is not None and self.hf_moe_prefix in name:
            mca_prefix = convert_to_mca_prefix(weight_prefix, self.hf_layer_prefix, self.hf_moe_prefix)
            mca_prefix = mca_prefix.replace("mtp.layers.0", "mtp.layers.0.mtp_model_layer", 1)
        return {mca_prefix + name: weight for name, weight in conver_res.items()}

    def add_mca_weight(self, name, weight, **kwargs):
        match = re.match(self.mca_ln_pattern, name)
        if match:
            layer_idx = int(match.group(1))
            return {f"model.language_model.layers.{layer_idx}.input_layernorm.weight": weight}
        if not name.startswith("mtp"):
            return super().add_mca_weight(name, weight, **kwargs)
        if MCA_MTP_MOE_PREFIX in name:
            # MTP MoE weight: include expert index in prefix
            weight_prefix = get_weight_prefix(name, MCA_MTP_PREFIX, MCA_MTP_MOE_PREFIX)
        else:
            weight_prefix = (
                "mtp.layers.0.mtp_model_layer" if name.startswith("mtp.layers.0.mtp_model_layer") else "mtp.layers.0"
            )
        original_name = remove_mca_mtp_weight_prefix(name)
        if weight_prefix not in self.prefix_name_to_weight:
            self.prefix_name_to_weight[weight_prefix] = {}
        self.prefix_name_to_weight[weight_prefix][original_name] = weight
        prefix_weights = self.prefix_name_to_weight[weight_prefix]
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
        if MCA_MTP_MOE_PREFIX in name:
            # Convert MTP MoE prefix: remove .mtp_model_layer and .local_experts.
            hf_prefix = weight_prefix.replace(".mtp_model_layer", "").replace(".local_experts.", ".")
        else:
            hf_prefix = "mtp.layers.0" if name.startswith("mtp.layers.0.mtp_model_layer") else "mtp"
        result = {hf_prefix + name: weight for name, weight in conver_res.items()}
        return result

    def hf_name_to_mca_names(self, hf_name) -> Optional[list[str]]:
        if not hf_name.startswith("mtp"):
            return super().hf_name_to_mca_names(hf_name)
        weight_prefix = "mtp.layers.0" if hf_name.startswith("mtp.layers.0") else "mtp"
        original_name = hf_name.removeprefix(weight_prefix)
        if self.hf_moe_prefix is not None:
            original_name = remove_weight_prefix(original_name, self.hf_moe_prefix)
        if original_name in self.hf_invalid_keys:
            return None
        op = self.get_conver_op(original_name, self.hf_name_to_converter)
        has_transformer_layer = "self_attention" in hf_name or "mlp" in hf_name or "input_layernorm" in hf_name
        mca_prefix = "mtp.layers.0" + (".mtp_model_layer" if has_transformer_layer else "")
        return [mca_prefix + name for name in op.mca_names]

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
        elif "lora_A" in lora_name:
            op_class = CopyConverOp
        elif isinstance(op, StackConverOp):
            op_class = StackConverOp
            kwargs = {"dim": op.dim}
        elif isinstance(op, GatedQKVConverOp):
            op_class = GatedQKVConverOp
            kwargs = {"hidden_size": lora_rank}
        elif isinstance(op, Qwen3_5_GDNConverOp):
            op_class = Qwen3_5_GDNConverOp
        else:
            raise ValueError(f"cannot find lora conver op for {name} in {pattern_to_conver_ops}")
        lora_op = op_class(
            hf_names=[hf_name.replace(".weight", lora_name) for hf_name in op.hf_names],
            mca_names=[mca_name.replace(".weight", lora_name) for mca_name in op.mca_names],
            _mca_config=op.mca_config,
            **kwargs,
        )
        self._lora_op_cache[cache_key] = lora_op
        return lora_op


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
        "mtp_num_hidden_layers": "mtp_num_layers",
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
        "experimental_attention_variant": "gated_delta_net",
        "mtp_loss_scaling_factor": 0.3,
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
        RenameConverOp(hf_names=".pre_fc_norm_embedding.weight", mca_names=".enorm.weight"),
        RenameConverOp(hf_names=".pre_fc_norm_hidden.weight", mca_names=".hnorm.weight"),
        RenameConverOp(hf_names=".fc.weight", mca_names=".eh_proj.weight"),
        RenameConverOp(hf_names=".norm.weight", mca_names=".final_layernorm.weight"),
    ],
)

__all__ = ["Qwen3_5Config", "Qwen3_5Model"]
