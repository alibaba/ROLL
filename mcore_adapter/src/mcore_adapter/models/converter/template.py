import json
import re
from abc import ABC
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import torch
from megatron.core import mpu
from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from ...utils import get_logger
from .convert_utils import (
    StackedTensors,
    convert_to_hf_prefix,
    convert_to_mca_prefix,
    get_layer_index,
    get_mca_layer_index,
    get_mca_weight_prefix,
    get_weight_prefix,
    remove_mca_weight_prefix,
    remove_weight_prefix,
)


if TYPE_CHECKING:
    from megatron.core.transformer import TransformerConfig
    from transformers import PretrainedConfig

logger = get_logger(__name__)


@dataclass
class ConverOp(ABC):
    """
    all names in ConverOp should not have layer prefix
    """

    hf_names: Union[str, list]
    mca_names: Union[str, list]
    mca_config: "TransformerConfig" = None

    def __post_init__(self):
        if isinstance(self.hf_names, str):
            self.hf_names = [self.hf_names]
        if isinstance(self.mca_names, str):
            self.mca_names = [self.mca_names]

    def __call__(self, name_to_weight: Dict[str, torch.Tensor], mca_to_hf: bool = False) -> Any:
        weight_len = len(self.mca_names if mca_to_hf else self.hf_names)
        if weight_len > len(name_to_weight):
            # not enough to convert
            return None
        if mca_to_hf:
            return self.mca_to_hf(name_to_weight)
        else:
            return self.hf_to_mca(name_to_weight)

    @staticmethod
    def _name_to_pattern(name: str):
        return name.replace(".", "\.").replace("{}", "(.*)")

    def is_required_name(self, name, mca_name: bool):
        required_names = self.mca_names if mca_name else self.hf_names
        if name in required_names:
            return True
        for pattern in required_names:
            re_pattern = self._name_to_pattern(pattern)
            if re.match(re_pattern, name):
                return True
        return False

    def _to_names_and_weights(
        self, from_names: List[str], to_names: List[str], name_to_weight: Dict[str, torch.Tensor]
    ) -> Tuple[List[str], List[torch.Tensor]]:
        weights = []
        match = None
        for from_name in from_names:
            if from_name in name_to_weight:
                weight = name_to_weight[from_name]
            elif "{}" in from_name:
                re_pattern = self._name_to_pattern(from_name)
                for name in name_to_weight:
                    match = re.findall(re_pattern, name)
                    if match:
                        weight = name_to_weight[name]
                        break
                if not match:
                    raise ValueError(f"Cannot find match {from_name} in {name_to_weight.keys()}")
            else:
                raise ValueError(f"Cannot find {from_name} in {name_to_weight.keys()}")
            weights.append(weight)

        if match:
            to_names = [to_name.format(*match) for to_name in to_names]
        return to_names, weights

    def hf_to_mca(self, name_to_weight: Dict[str, torch.Tensor]):
        names, weights = self._to_names_and_weights(self.hf_names, self.mca_names, name_to_weight)
        mca_weights = self._hf_to_mca(weights)
        if isinstance(mca_weights, (torch.Tensor, StackedTensors)):
            mca_weights = [mca_weights]
        assert len(names) == len(mca_weights), f"names: {names}, weights: {mca_weights}"
        return {names[i]: mca_weights[i] for i in range(len(names))}

    def mca_to_hf(self, name_to_weight: Dict[str, torch.Tensor]):
        names, weights = self._to_names_and_weights(self.mca_names, self.hf_names, name_to_weight)
        hf_weights = self._mca_to_hf(weights)
        if isinstance(hf_weights, (torch.Tensor, StackedTensors)):
            hf_weights = [hf_weights]
        assert len(names) == len(hf_weights), f"names: {names}, weights: {hf_weights}"
        return {names[i]: hf_weights[i] for i in range(len(names))}

    def _hf_to_mca(self, weights: List[torch.Tensor]) -> List[torch.Tensor]:
        raise NotImplementedError()

    def _mca_to_hf(self, weights: List[torch.Tensor]) -> List[torch.Tensor]:
        raise NotImplementedError()


@dataclass
class RenameConverOp(ConverOp):
    def __post_init__(self):
        super().__post_init__()
        assert len(self.hf_names) == 1, f"RenameConverOp only support one name {self.hf_names}"
        assert len(self.mca_names) == 1, f"RenameConverOp only support one name {self.mca_names}"

    def _hf_to_mca(self, weights):
        return weights

    def _mca_to_hf(self, weights):
        return weights


@dataclass
class ConcatConverOp(ConverOp):
    dim: int = 0

    def __post_init__(self):
        super().__post_init__()
        assert (len(self.hf_names) == 1) != (
            len(self.mca_names) == 1
        ), f"ConcatConverOp only supports one name as target {self.hf_names} {self.mca_names}"

    def _hf_to_mca(self, weights):
        if len(weights) == 1:
            return torch.chunk(weights[0], len(self.mca_names), dim=self.dim)
        return torch.cat(weights, dim=self.dim)

    def _mca_to_hf(self, weights):
        if len(weights) == 1:
            return torch.chunk(weights[0], len(self.hf_names), dim=self.dim)
        return torch.cat(weights, dim=self.dim)


@dataclass
class StackConverOp(ConverOp):
    dim: int = 0

    def __post_init__(self):
        super().__post_init__()
        assert (len(self.hf_names) == 1) != (
            len(self.mca_names) == 1
        ), f"StackConverOp only supports one name as target {self.hf_names} {self.mca_names}"

    def _hf_to_mca(self, weights):
        if len(weights) == 1:
            assert isinstance(weights[0], StackedTensors)
            return weights[0].tensors
        return StackedTensors(tensors=weights, dim=self.dim)

    def _mca_to_hf(self, weights):
        if len(weights) == 1:
            assert isinstance(weights[0], StackedTensors)
            return weights[0].tensors
        return StackedTensors(tensors=weights, dim=self.dim)


class QKVConverOp(ConverOp):
    def __post_init__(self):
        super().__post_init__()
        assert len(self.hf_names) == 3, f"QKVConverOp only support three hf_names {self.hf_names}"
        assert len(self.mca_names) == 1, f"QKVConverOp only support one mca_name {self.mca_names}"

    def _hf_to_mca(self, weights):
        q_weight, k_weight, v_weight = weights
        nh = self.mca_config.num_attention_heads
        ng = self.mca_config.num_query_groups
        dim = self.mca_config.kv_channels
        assert nh % ng == 0
        mca_qkv_weight = torch.cat(
            [
                q_weight.reshape((ng, dim * nh // ng, -1)),
                k_weight.reshape((ng, dim, -1)),
                v_weight.reshape((ng, dim, -1)),
            ],
            dim=1,
        ).reshape((-1, self.mca_config.hidden_size))
        return mca_qkv_weight

    def _mca_to_hf(self, weights):
        qkv_weight = weights[0]
        ng = self.mca_config.num_query_groups
        nh = self.mca_config.num_attention_heads
        dim = self.mca_config.kv_channels
        qkv_weight = qkv_weight.reshape((ng, dim * (nh // ng + 2), -1))
        qkv_weights = torch.split(qkv_weight, [dim * nh // ng, dim, dim], dim=1)
        q_weight = qkv_weights[0].reshape((-1, self.mca_config.hidden_size))
        k_weight = qkv_weights[1].reshape((-1, self.mca_config.hidden_size))
        v_weight = qkv_weights[2].reshape((-1, self.mca_config.hidden_size))
        return [q_weight, k_weight, v_weight]


class QKVBiasConverOp(ConverOp):
    def __post_init__(self):
        super().__post_init__()
        assert len(self.hf_names) == 3, f"QKVBiasConverOp only support three hf_names {self.hf_names}"
        assert len(self.mca_names) == 1, f"QKVBiasConverOp only support one mca_name {self.mca_names}"

    def _hf_to_mca(self, weights):
        q_weight, k_weight, v_weight = weights
        nh = self.mca_config.num_attention_heads
        ng = self.mca_config.num_query_groups
        dim = self.mca_config.kv_channels
        assert nh % ng == 0
        mca_qkv_weight = torch.cat(
            [
                q_weight.reshape((ng, dim * nh // ng)),
                k_weight.reshape((ng, dim)),
                v_weight.reshape((ng, dim)),
            ],
            dim=1,
        ).reshape((-1))
        return mca_qkv_weight

    def _mca_to_hf(self, weights):
        qkv_weight = weights[0]
        ng = self.mca_config.num_query_groups
        nh = self.mca_config.num_attention_heads
        dim = self.mca_config.kv_channels
        qkv_weight = qkv_weight.reshape((ng, dim * (nh // ng + 2), -1))
        qkv_weights = torch.split(qkv_weight, [dim * nh // ng, dim, dim], dim=1)
        q_weight = qkv_weights[0].reshape((-1))
        k_weight = qkv_weights[1].reshape((-1))
        v_weight = qkv_weights[2].reshape((-1))
        return [q_weight, k_weight, v_weight]


@dataclass
class Template:
    hf_model_type: str
    hf_layer_prefix: str
    config_hf_to_mca: Dict[str, str]
    weight_converters: List[ConverOp]
    constant_mca_config: Dict[str, Any]
    constant_hf_config: Dict[str, Any] = field(default_factory=dict)
    hf_moe_prefix: Optional[str] = None
    hf_invalid_keys: List[str] = field(default_factory=list)
    config_mca_to_hf: Optional[Dict[str, str]] = None
    hf_name_to_converter: Dict[str, ConverOp] = field(default_factory=dict)
    mca_name_to_converter: Dict[str, ConverOp] = field(default_factory=dict)
    prefix_name_to_weight: Dict[str, Dict[str, torch.Tensor]] = field(default_factory=dict)

    def __post_init__(self):
        if self.config_mca_to_hf is None:
            self.config_mca_to_hf = {v: k for k, v in self.config_hf_to_mca.items()}
        self.hf_name_to_converter = {}
        self.mca_name_to_converter = {}
        for converter in self.weight_converters:
            for hf_name in converter.hf_names:
                self.hf_name_to_converter[hf_name] = converter
            for mca_name in converter.mca_names:
                self.mca_name_to_converter[mca_name] = converter
        self.release()

    def release(self):
        weights_not_converted = [
            (prefix, name, weight.size())
            for prefix, name2weight in self.prefix_name_to_weight.items()
            for name, weight in name2weight.items()
        ]
        if len(weights_not_converted) > 0:
            logger.warning(f"weights not converted {len(weights_not_converted)} {weights_not_converted}")
        self.prefix_name_to_weight = {}

    def convert_hf_to_mca_config(self, hf_config, **kw_args):
        from mcore_adapter.models import AutoConfig as AutoMcaModelConfig

        kw_args = self.convert_hf_to_mca_config_kws(hf_config, **kw_args)
        return AutoMcaModelConfig.for_model(self.hf_model_type, **kw_args)

    def convert_hf_to_mca_config_kws(self, hf_config: "PretrainedConfig", **kw_args):
        for k, v in self.config_hf_to_mca.items():
            if hasattr(hf_config, k):
                kw_args[v] = getattr(hf_config, k)
        kw_args["hf_model_type"] = self.hf_model_type
        kw_args["name_or_path"] = hf_config.name_or_path
        kw_args["hf_config_json"] = hf_config.to_json_string()
        return {**kw_args, **self.constant_mca_config}

    def convert_mca_to_hf_config(self, mca_config, **kw_args):
        for k, v in self.config_mca_to_hf.items():
            if hasattr(mca_config, k):
                kw_args[v] = getattr(mca_config, k)
        kw_args.update(self.constant_hf_config)
        kw_args["name_or_path"] = mca_config.name_or_path
        config_dict = json.loads(mca_config.hf_config_json)
        kw_args = {**config_dict, **kw_args}
        kw_args["model_type"] = self.hf_model_type
        has_remote_code = "auto_map" in config_dict and "AutoConfig" in config_dict["auto_map"]
        if has_remote_code:
            class_ref = config_dict["auto_map"]["AutoConfig"]
            config_class = get_class_from_dynamic_module(class_ref, mca_config.name_or_path)
            config_class.register_for_auto_class()
            return config_class.from_dict(kw_args)
        return AutoConfig.for_model(**kw_args)

    def set_mca_config_for_ops(self, mca_config):
        self.mca_config = mca_config
        for converter in self.weight_converters:
            converter.mca_config = mca_config

    def add_hf_weight(self, name, weight):
        weight_prefix = get_weight_prefix(name, self.hf_layer_prefix, moe_prefix=self.hf_moe_prefix)
        original_name = remove_weight_prefix(name, self.hf_layer_prefix, moe_prefix=self.hf_moe_prefix)
        if original_name in self.hf_invalid_keys:
            return None
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
        mca_prefix = convert_to_mca_prefix(weight_prefix, self.hf_layer_prefix, self.hf_moe_prefix)
        return {mca_prefix + name: weight for name, weight in conver_res.items()}

    def add_mca_weight(self, name, weight):
        weight_prefix = get_mca_weight_prefix(name)
        original_name = remove_mca_weight_prefix(name)
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
        hf_prefix = convert_to_hf_prefix(weight_prefix, self.hf_layer_prefix, self.hf_moe_prefix)
        return {hf_prefix + name: weight for name, weight in conver_res.items()}

    def get_conver_op(self, name, pattern_to_conver_ops: Dict[str, ConverOp]):
        if name in pattern_to_conver_ops:
            return pattern_to_conver_ops[name]
        for pattern in sorted(pattern_to_conver_ops, key=lambda x: len(x), reverse=True):
            re_pattern = pattern.replace("{}", "(.*?)")
            if re.match(re_pattern, name):
                return pattern_to_conver_ops[pattern]
        raise ValueError(f"can not find conver op for {name} in {pattern_to_conver_ops}")

    def hf_name_to_mca_names(self, hf_name) -> Optional[List[str]]:
        weight_prefix = get_weight_prefix(hf_name, self.hf_layer_prefix, moe_prefix=self.hf_moe_prefix)
        original_name = remove_weight_prefix(hf_name, self.hf_layer_prefix, moe_prefix=self.hf_moe_prefix)
        if original_name in self.hf_invalid_keys:
            return None
        op = self.get_conver_op(original_name, self.hf_name_to_converter)
        mca_prefix = convert_to_mca_prefix(weight_prefix, self.hf_layer_prefix, self.hf_moe_prefix)
        return [mca_prefix + name for name in op.mca_names]


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

        if res.get("mtp_num_layers"):
            res["num_layers"] += 1

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

        if mca_config.mtp_num_layers:
            res.num_hidden_layers = mca_config.num_layers - 1

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


templates: Dict[str, Template] = {}


def register_template(
    hf_model_type,
    config_hf_to_mca,
    weight_converters,
    hf_layer_prefix,
    hf_invalid_keys=[],
    template_class: Template = Template,
    constant_mca_config={},
    constant_hf_config={},
    **kwargs,
):
    templates[hf_model_type] = template_class(
        hf_model_type=hf_model_type,
        hf_layer_prefix=hf_layer_prefix,
        hf_invalid_keys=hf_invalid_keys,
        config_hf_to_mca=config_hf_to_mca,
        constant_mca_config=constant_mca_config,
        constant_hf_config=constant_hf_config,
        weight_converters=weight_converters,
        **kwargs,
    )


def get_template(name) -> Template:
    return templates[name]


register_template(
    "llama",
    hf_layer_prefix="model.layers.",
    config_hf_to_mca={
        "max_position_embeddings": "max_sequence_length",
        "hidden_size": "hidden_size",
        "num_attention_heads": "num_attention_heads",
        "num_key_value_heads": "num_query_groups",
        "num_hidden_layers": "num_layers",
        "rms_norm_eps": "layernorm_epsilon",
        "vocab_size": "padded_vocab_size",
        "intermediate_size": "ffn_hidden_size",
        "attention_bias": "add_qkv_bias",
        "attention_dropout": "attention_dropout",
        "rope_theta": "rotary_base",
        "tie_word_embeddings": "tie_embeddings_and_output_weights",
    },
    hf_invalid_keys=[".self_attn.rotary_emb.inv_freq"],
    constant_mca_config={
        "swiglu": True,
        "position_embedding_type": "rope",
        "normalization": "RMSNorm",
        "add_bias_linear": False,
        "hidden_dropout": 0.0,
        "rotary_percent": 1.0,
    },
    weight_converters=[
        RenameConverOp(hf_names="lm_head.weight", mca_names="output_layer.weight"),
        RenameConverOp(hf_names="model.embed_tokens.weight", mca_names="embedding.word_embeddings.weight"),
        RenameConverOp(hf_names=".input_layernorm.weight", mca_names=".self_attention.linear_qkv.layer_norm_weight"),
        RenameConverOp(hf_names=".self_attn.o_proj.weight", mca_names=".self_attention.linear_proj.weight"),
        RenameConverOp(hf_names=".post_attention_layernorm.weight", mca_names=".mlp.linear_fc1.layer_norm_weight"),
        RenameConverOp(hf_names=".mlp.down_proj.weight", mca_names=".mlp.linear_fc2.weight"),
        RenameConverOp(hf_names="model.norm.weight", mca_names="decoder.final_layernorm.weight"),
        StackConverOp(
            hf_names=[".mlp.gate_proj.weight", ".mlp.up_proj.weight"], mca_names=".mlp.linear_fc1.weight", dim=0
        ),
        QKVConverOp(
            hf_names=[".self_attn.q_proj.weight", ".self_attn.k_proj.weight", ".self_attn.v_proj.weight"],
            mca_names=".self_attention.linear_qkv.weight",
        ),
    ],
)


register_template(
    "qwen2",
    hf_layer_prefix="model.layers.",
    config_hf_to_mca={
        "max_position_embeddings": "max_sequence_length",
        "hidden_size": "hidden_size",
        "num_attention_heads": "num_attention_heads",
        "num_key_value_heads": "num_query_groups",
        "num_hidden_layers": "num_layers",
        "rms_norm_eps": "layernorm_epsilon",
        "vocab_size": "padded_vocab_size",
        "intermediate_size": "ffn_hidden_size",
        "attention_dropout": "attention_dropout",
        "rope_theta": "rotary_base",
        "tie_word_embeddings": "tie_embeddings_and_output_weights",
    },
    constant_mca_config={
        "swiglu": True,
        "position_embedding_type": "rope",
        "normalization": "RMSNorm",
        "add_bias_linear": False,
        "add_qkv_bias": True,
        "hidden_dropout": 0.0,
        "rotary_percent": 1.0,
    },
    weight_converters=[
        RenameConverOp(hf_names="lm_head.weight", mca_names="output_layer.weight"),
        RenameConverOp(hf_names="model.embed_tokens.weight", mca_names="embedding.word_embeddings.weight"),
        RenameConverOp(hf_names=".input_layernorm.weight", mca_names=".self_attention.linear_qkv.layer_norm_weight"),
        RenameConverOp(hf_names=".self_attn.o_proj.weight", mca_names=".self_attention.linear_proj.weight"),
        RenameConverOp(hf_names=".post_attention_layernorm.weight", mca_names=".mlp.linear_fc1.layer_norm_weight"),
        RenameConverOp(hf_names=".mlp.down_proj.weight", mca_names=".mlp.linear_fc2.weight"),
        RenameConverOp(hf_names="model.norm.weight", mca_names="decoder.final_layernorm.weight"),
        StackConverOp(
            hf_names=[".mlp.gate_proj.weight", ".mlp.up_proj.weight"], mca_names=".mlp.linear_fc1.weight", dim=0
        ),
        QKVConverOp(
            hf_names=[".self_attn.q_proj.weight", ".self_attn.k_proj.weight", ".self_attn.v_proj.weight"],
            mca_names=".self_attention.linear_qkv.weight",
        ),
        QKVBiasConverOp(
            hf_names=[".self_attn.q_proj.bias", ".self_attn.k_proj.bias", ".self_attn.v_proj.bias"],
            mca_names=".self_attention.linear_qkv.bias",
        ),
    ],
)


register_template(
    "qwen2_moe",
    hf_layer_prefix="model.layers.",
    hf_moe_prefix=".mlp.experts.",
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
        "moe_intermediate_size": "ffn_hidden_size",
        "tie_word_embeddings": "tie_embeddings_and_output_weights",
        # MoE related
        "decoder_sparse_step": "moe_layer_freq",
        "num_experts": "num_moe_experts",
        "num_experts_per_tok": "moe_router_topk",
        "router_aux_loss_coef": "moe_aux_loss_coeff",
        "shared_expert_intermediate_size": "moe_shared_expert_intermediate_size",
    },
    constant_mca_config={
        "swiglu": True,
        "position_embedding_type": "rope",
        "normalization": "RMSNorm",
        "add_bias_linear": False,
        "add_qkv_bias": True,
        "hidden_dropout": 0.0,
        "rotary_percent": 1.0,
        "moe_router_load_balancing_type": "aux_loss",
        "moe_router_pre_softmax": True,
        "moe_use_shared_expert_gate": True,
    },
    weight_converters=[
        RenameConverOp(hf_names="lm_head.weight", mca_names="output_layer.weight"),
        RenameConverOp(hf_names="model.embed_tokens.weight", mca_names="embedding.word_embeddings.weight"),
        RenameConverOp(hf_names=".input_layernorm.weight", mca_names=".self_attention.linear_qkv.layer_norm_weight"),
        RenameConverOp(hf_names=".self_attn.o_proj.weight", mca_names=".self_attention.linear_proj.weight"),
        RenameConverOp(hf_names=".post_attention_layernorm.weight", mca_names=".pre_mlp_layernorm.weight"),
        RenameConverOp(hf_names=".down_proj.weight", mca_names=".linear_fc2.weight"),
        RenameConverOp(hf_names="model.norm.weight", mca_names="decoder.final_layernorm.weight"),
        StackConverOp(hf_names=[".gate_proj.weight", ".up_proj.weight"], mca_names=".linear_fc1.weight", dim=0),
        StackConverOp(
            hf_names=[".mlp.shared_expert.gate_proj.weight", ".mlp.shared_expert.up_proj.weight"],
            mca_names=".mlp.shared_experts.linear_fc1.weight",
            dim=0,
        ),
        RenameConverOp(hf_names=".mlp.gate.weight", mca_names=".mlp.router.weight"),
        RenameConverOp(
            hf_names=".mlp.shared_expert.down_proj.weight", mca_names=".mlp.shared_experts.linear_fc2.weight"
        ),
        RenameConverOp(hf_names=".mlp.shared_expert_gate.weight", mca_names=".mlp.shared_experts.gate_weight"),
        QKVConverOp(
            hf_names=[".self_attn.q_proj.weight", ".self_attn.k_proj.weight", ".self_attn.v_proj.weight"],
            mca_names=".self_attention.linear_qkv.weight",
        ),
        QKVBiasConverOp(
            hf_names=[".self_attn.q_proj.bias", ".self_attn.k_proj.bias", ".self_attn.v_proj.bias"],
            mca_names=".self_attention.linear_qkv.bias",
        ),
    ],
)


register_template(
    "qwen3",
    hf_layer_prefix="model.layers.",
    hf_moe_prefix=".mlp.experts.",
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
    },
    constant_mca_config={
        "swiglu": True,
        "position_embedding_type": "rope",
        "normalization": "RMSNorm",
        "add_bias_linear": False,
        "hidden_dropout": 0.0,
        "rotary_percent": 1.0,
        "qk_layernorm": True,
    },
    weight_converters=[
        RenameConverOp(hf_names="lm_head.weight", mca_names="output_layer.weight"),
        RenameConverOp(hf_names="model.embed_tokens.weight", mca_names="embedding.word_embeddings.weight"),
        RenameConverOp(hf_names=".input_layernorm.weight", mca_names=".self_attention.linear_qkv.layer_norm_weight"),
        RenameConverOp(hf_names=".self_attn.o_proj.weight", mca_names=".self_attention.linear_proj.weight"),
        RenameConverOp(hf_names=".self_attn.q_norm.weight", mca_names=".self_attention.q_layernorm.weight"),
        RenameConverOp(hf_names=".self_attn.k_norm.weight", mca_names=".self_attention.k_layernorm.weight"),
        RenameConverOp(hf_names=".post_attention_layernorm.weight", mca_names=".mlp.linear_fc1.layer_norm_weight"),
        RenameConverOp(hf_names=".down_proj.weight", mca_names=".linear_fc2.weight"),
        RenameConverOp(hf_names="model.norm.weight", mca_names="decoder.final_layernorm.weight"),
        StackConverOp(
            hf_names=[".mlp.gate_proj.weight", ".mlp.up_proj.weight"], mca_names=".mlp.linear_fc1.weight", dim=0
        ),
        RenameConverOp(hf_names=".mlp.down_proj.weight", mca_names=".mlp.linear_fc2.weight"),
        QKVConverOp(
            hf_names=[".self_attn.q_proj.weight", ".self_attn.k_proj.weight", ".self_attn.v_proj.weight"],
            mca_names=".self_attention.linear_qkv.weight",
        ),
        QKVBiasConverOp(
            hf_names=[".self_attn.q_proj.bias", ".self_attn.k_proj.bias", ".self_attn.v_proj.bias"],
            mca_names=".self_attention.linear_qkv.bias",
        ),
    ],
)


register_template(
    "qwen3_moe",
    hf_layer_prefix="model.layers.",
    hf_moe_prefix=".mlp.experts.",
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
    },
    weight_converters=[
        RenameConverOp(hf_names="lm_head.weight", mca_names="output_layer.weight"),
        RenameConverOp(hf_names="model.embed_tokens.weight", mca_names="embedding.word_embeddings.weight"),
        RenameConverOp(hf_names=".input_layernorm.weight", mca_names=".self_attention.linear_qkv.layer_norm_weight"),
        RenameConverOp(hf_names=".self_attn.o_proj.weight", mca_names=".self_attention.linear_proj.weight"),
        RenameConverOp(hf_names=".self_attn.q_norm.weight", mca_names=".self_attention.q_layernorm.weight"),
        RenameConverOp(hf_names=".self_attn.k_norm.weight", mca_names=".self_attention.k_layernorm.weight"),
        RenameConverOp(hf_names=".post_attention_layernorm.weight", mca_names=".pre_mlp_layernorm.weight"),
        RenameConverOp(hf_names=".down_proj.weight", mca_names=".linear_fc2.weight"),
        RenameConverOp(hf_names="model.norm.weight", mca_names="decoder.final_layernorm.weight"),
        StackConverOp(hf_names=[".gate_proj.weight", ".up_proj.weight"], mca_names=".linear_fc1.weight", dim=0),
        RenameConverOp(hf_names=".mlp.gate.weight", mca_names=".mlp.router.weight"),
        QKVConverOp(
            hf_names=[".self_attn.q_proj.weight", ".self_attn.k_proj.weight", ".self_attn.v_proj.weight"],
            mca_names=".self_attention.linear_qkv.weight",
        ),
        QKVBiasConverOp(
            hf_names=[".self_attn.q_proj.bias", ".self_attn.k_proj.bias", ".self_attn.v_proj.bias"],
            mca_names=".self_attention.linear_qkv.bias",
        ),
    ],
)


register_template(
    "mistral",
    hf_layer_prefix="model.layers.",
    config_hf_to_mca={
        "max_position_embeddings": "max_sequence_length",
        "hidden_size": "hidden_size",
        "num_attention_heads": "num_attention_heads",
        "num_key_value_heads": "num_query_groups",
        "num_hidden_layers": "num_layers",
        "rms_norm_eps": "layernorm_epsilon",
        "vocab_size": "padded_vocab_size",
        "intermediate_size": "ffn_hidden_size",
        "attention_bias": "add_qkv_bias",
        "attention_dropout": "attention_dropout",
        "rope_theta": "rotary_base",
        "tie_word_embeddings": "tie_embeddings_and_output_weights",
    },
    hf_invalid_keys=[".self_attn.rotary_emb.inv_freq"],
    constant_mca_config={
        "swiglu": True,
        "position_embedding_type": "rope",
        "normalization": "RMSNorm",
        "add_bias_linear": False,
        "hidden_dropout": 0.0,
        "rotary_percent": 1.0,
    },
    weight_converters=[
        RenameConverOp(hf_names="lm_head.weight", mca_names="output_layer.weight"),
        RenameConverOp(hf_names="model.embed_tokens.weight", mca_names="embedding.word_embeddings.weight"),
        RenameConverOp(hf_names=".input_layernorm.weight", mca_names=".self_attention.linear_qkv.layer_norm_weight"),
        RenameConverOp(hf_names=".self_attn.o_proj.weight", mca_names=".self_attention.linear_proj.weight"),
        RenameConverOp(hf_names=".post_attention_layernorm.weight", mca_names=".mlp.linear_fc1.layer_norm_weight"),
        RenameConverOp(hf_names=".mlp.down_proj.weight", mca_names=".mlp.linear_fc2.weight"),
        RenameConverOp(hf_names="model.norm.weight", mca_names="decoder.final_layernorm.weight"),
        StackConverOp(
            hf_names=[".mlp.gate_proj.weight", ".mlp.up_proj.weight"], mca_names=".mlp.linear_fc1.weight", dim=0
        ),
        QKVConverOp(
            hf_names=[".self_attn.q_proj.weight", ".self_attn.k_proj.weight", ".self_attn.v_proj.weight"],
            mca_names=".self_attention.linear_qkv.weight",
        ),
    ],
)


register_template(
    "mixtral",
    hf_layer_prefix="model.layers.",
    hf_moe_prefix=".block_sparse_moe.experts.",
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
        "num_local_experts": "num_moe_experts",
        "num_experts_per_tok": "moe_router_topk",
        "router_aux_loss_coef": "moe_aux_loss_coeff",
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
    },
    weight_converters=[
        RenameConverOp(hf_names="lm_head.weight", mca_names="output_layer.weight"),
        RenameConverOp(hf_names="model.embed_tokens.weight", mca_names="embedding.word_embeddings.weight"),
        RenameConverOp(hf_names=".input_layernorm.weight", mca_names=".self_attention.linear_qkv.layer_norm_weight"),
        RenameConverOp(hf_names=".self_attn.o_proj.weight", mca_names=".self_attention.linear_proj.weight"),
        RenameConverOp(hf_names=".self_attn.q_norm.weight", mca_names=".self_attention.q_layernorm.weight"),
        RenameConverOp(hf_names=".self_attn.k_norm.weight", mca_names=".self_attention.k_layernorm.weight"),
        RenameConverOp(hf_names=".post_attention_layernorm.weight", mca_names=".pre_mlp_layernorm.weight"),
        RenameConverOp(hf_names=".w2.weight", mca_names=".linear_fc2.weight"),
        RenameConverOp(hf_names="model.norm.weight", mca_names="decoder.final_layernorm.weight"),
        StackConverOp(hf_names=[".w1.weight", ".w3.weight"], mca_names=".linear_fc1.weight", dim=0),
        RenameConverOp(hf_names=".block_sparse_moe.gate.weight", mca_names=".mlp.router.weight"),
        QKVConverOp(
            hf_names=[".self_attn.q_proj.weight", ".self_attn.k_proj.weight", ".self_attn.v_proj.weight"],
            mca_names=".self_attention.linear_qkv.weight",
        ),
        QKVBiasConverOp(
            hf_names=[".self_attn.q_proj.bias", ".self_attn.k_proj.bias", ".self_attn.v_proj.bias"],
            mca_names=".self_attention.linear_qkv.bias",
        ),
    ],
)


register_template(
    "qwen2_vl",
    hf_layer_prefix="model.layers.",
    config_hf_to_mca={
        "max_position_embeddings": "max_sequence_length",
        "hidden_size": "hidden_size",
        "num_attention_heads": "num_attention_heads",
        "num_key_value_heads": "num_query_groups",
        "num_hidden_layers": "num_layers",
        "rms_norm_eps": "layernorm_epsilon",
        "vocab_size": "padded_vocab_size",
        "intermediate_size": "ffn_hidden_size",
        "attention_dropout": "attention_dropout",
        "rope_theta": "rotary_base",
        "tie_word_embeddings": "tie_embeddings_and_output_weights",
        # qwen2_vl related
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
        "position_embedding_type": "rope",
        "normalization": "RMSNorm",
        "add_bias_linear": False,
        "add_qkv_bias": True,
        "hidden_dropout": 0.0,
        "rotary_percent": 1.0,
    },
    weight_converters=[
        RenameConverOp(hf_names="lm_head.weight", mca_names="output_layer.weight"),
        RenameConverOp(hf_names="model.embed_tokens.weight", mca_names="embedding.word_embeddings.weight"),
        RenameConverOp(hf_names=".input_layernorm.weight", mca_names=".self_attention.linear_qkv.layer_norm_weight"),
        RenameConverOp(hf_names=".self_attn.o_proj.weight", mca_names=".self_attention.linear_proj.weight"),
        RenameConverOp(hf_names=".post_attention_layernorm.weight", mca_names=".mlp.linear_fc1.layer_norm_weight"),
        RenameConverOp(hf_names=".mlp.down_proj.weight", mca_names=".mlp.linear_fc2.weight"),
        RenameConverOp(hf_names="model.norm.weight", mca_names="decoder.final_layernorm.weight"),
        StackConverOp(
            hf_names=[".mlp.gate_proj.weight", ".mlp.up_proj.weight"], mca_names=".mlp.linear_fc1.weight", dim=0
        ),
        QKVConverOp(
            hf_names=[".self_attn.q_proj.weight", ".self_attn.k_proj.weight", ".self_attn.v_proj.weight"],
            mca_names=".self_attention.linear_qkv.weight",
        ),
        QKVBiasConverOp(
            hf_names=[".self_attn.q_proj.bias", ".self_attn.k_proj.bias", ".self_attn.v_proj.bias"],
            mca_names=".self_attention.linear_qkv.bias",
        ),
        RenameConverOp(hf_names="visual.{}", mca_names="vision_model.{}"),
    ],
)

register_template(
    "qwen2_5_vl",
    hf_layer_prefix="model.layers.",
    config_hf_to_mca={
        "max_position_embeddings": "max_sequence_length",
        "hidden_size": "hidden_size",
        "num_attention_heads": "num_attention_heads",
        "num_key_value_heads": "num_query_groups",
        "num_hidden_layers": "num_layers",
        "rms_norm_eps": "layernorm_epsilon",
        "vocab_size": "padded_vocab_size",
        "intermediate_size": "ffn_hidden_size",
        "attention_dropout": "attention_dropout",
        "rope_theta": "rotary_base",
        "tie_word_embeddings": "tie_embeddings_and_output_weights",
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
        "position_embedding_type": "rope",
        "normalization": "RMSNorm",
        "add_bias_linear": False,
        "add_qkv_bias": True,
        "hidden_dropout": 0.0,
        "rotary_percent": 1.0,
    },
    weight_converters=[
        RenameConverOp(hf_names="lm_head.weight", mca_names="output_layer.weight"),
        RenameConverOp(hf_names="model.embed_tokens.weight", mca_names="embedding.word_embeddings.weight"),
        RenameConverOp(hf_names=".input_layernorm.weight", mca_names=".self_attention.linear_qkv.layer_norm_weight"),
        RenameConverOp(hf_names=".self_attn.o_proj.weight", mca_names=".self_attention.linear_proj.weight"),
        RenameConverOp(hf_names=".post_attention_layernorm.weight", mca_names=".mlp.linear_fc1.layer_norm_weight"),
        RenameConverOp(hf_names=".mlp.down_proj.weight", mca_names=".mlp.linear_fc2.weight"),
        RenameConverOp(hf_names="model.norm.weight", mca_names="decoder.final_layernorm.weight"),
        StackConverOp(
            hf_names=[".mlp.gate_proj.weight", ".mlp.up_proj.weight"], mca_names=".mlp.linear_fc1.weight", dim=0
        ),
        QKVConverOp(
            hf_names=[".self_attn.q_proj.weight", ".self_attn.k_proj.weight", ".self_attn.v_proj.weight"],
            mca_names=".self_attention.linear_qkv.weight",
        ),
        QKVBiasConverOp(
            hf_names=[".self_attn.q_proj.bias", ".self_attn.k_proj.bias", ".self_attn.v_proj.bias"],
            mca_names=".self_attention.linear_qkv.bias",
        ),
        RenameConverOp(hf_names="visual.{}", mca_names="vision_model.{}"),
    ],
)


register_template(
    "deepseek_v3",
    template_class=DeepSeekV3Template,
    hf_layer_prefix="model.layers.",
    hf_moe_prefix=".mlp.experts.",
    hf_invalid_keys=[
        # ".mlp.gate.e_score_correction_bias", # support in the future
        ".embed_tokens.weight",  # the mtp is shared, this weight is the same as `model.embed_tokens.weight` in hf,
        ".shared_head.head.weight",
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
        RenameConverOp(
            hf_names=".self_attn.q_a_proj.weight_scale_inv",
            mca_names=".self_attn.q_a_proj.weight_scale_inv._extra_state",
        ),
        RenameConverOp(
            hf_names=".self_attn.q_a_layernorm.weight", mca_names=".self_attention.linear_q_up_proj.layer_norm_weight"
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
        # RenameConverOp(hf_names=".shared_head.head.weight", mca_names=".shared_head.head.weight"),
        RenameConverOp(hf_names=".self_attn.o_proj.weight_scale_inv", mca_names=".self_attn.o_proj.weight_scale_inv"),
        RenameConverOp(hf_names=".post_attention_layernorm.weight", mca_names=".pre_mlp_layernorm.weight"),
        RenameConverOp(hf_names="model.norm.weight", mca_names="decoder.final_layernorm.weight"),
        RenameConverOp(hf_names=".mlp.gate.weight", mca_names=".mlp.router.weight"),
    ],
)
