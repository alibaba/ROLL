import json
import os
import re
from abc import ABC
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import torch
from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from ...utils import get_logger
from .convert_utils import (
    StackedTensors,
    convert_to_hf_prefix,
    convert_to_mca_prefix,
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
    _mca_config: "TransformerConfig" = field(default=None, repr=False)

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

    @property
    def mca_config(self) -> "TransformerConfig":
        return self._mca_config

    @mca_config.setter
    def mca_config(self, value: "TransformerConfig"):
        self._mca_config = value

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
class CopyConverOp(ConverOp):
    def __post_init__(self):
        super().__post_init__()
        assert (len(self.hf_names) == 1) != (len(self.mca_names) == 1), (
            f"CopyConverOp only supports one name as target {self.hf_names} {self.mca_names}"
        )

    def _hf_to_mca(self, weights):
        return weights * len(self.mca_names)

    def _mca_to_hf(self, weights):
        return weights * len(self.hf_names)


@dataclass
class ConcatConverOp(ConverOp):
    dim: int = 0

    def __post_init__(self):
        super().__post_init__()
        assert (len(self.hf_names) == 1) != (len(self.mca_names) == 1), (
            f"ConcatConverOp only supports one name as target {self.hf_names} {self.mca_names}"
        )

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
        assert (len(self.hf_names) == 1) != (len(self.mca_names) == 1), (
            f"StackConverOp only supports one name as target {self.hf_names} {self.mca_names}"
        )

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


@dataclass
class QKVConverOp(ConverOp):
    hidden_size: Optional[int] = None

    def __post_init__(self):
        super().__post_init__()
        assert len(self.hf_names) == 3, f"QKVConverOp only support three hf_names {self.hf_names}"
        assert len(self.mca_names) == 1, f"QKVConverOp only support one mca_name {self.mca_names}"

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
                q_weight.reshape((ng, dim * nh // ng, -1)),
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
        qkv_weight = qkv_weight.reshape((ng, dim * (nh // ng + 2), -1))
        qkv_weights = torch.split(qkv_weight, [dim * nh // ng, dim, dim], dim=1)
        q_weight = qkv_weights[0].reshape((-1, self.hidden_size))
        k_weight = qkv_weights[1].reshape((-1, self.hidden_size))
        v_weight = qkv_weights[2].reshape((-1, self.hidden_size))
        return [q_weight, k_weight, v_weight]


@dataclass
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
        self.config_hf_to_mca = self.adjust_config_hf_to_mca()
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

    def adjust_config_hf_to_mca(self):
        return self.config_hf_to_mca

    def get_hf_config_value(self, hf_config, key, cfg_errs: List[str] = []):
        for name in key.split("."):
            if not hasattr(hf_config, name):
                # warn instead of assert to be backward compatible
                # some cfg not exist in hf_config, such as vision_token_id
                logger.warning(f"{key=} not exists in hf_config for get_hf_config_value")
                cfg_errs.append(key)
                return
            hf_config = getattr(hf_config, name)
        return hf_config

    def set_hf_config_value(self, hf_config, key, value):
        # hf_config is a dict from config.to_dict() by `to_json_string(use_diff=True)`,
        # sub-configs with PretrainedConfig type would be convert to dict
        # use_diff makes hf_config only contain items whose value is different from default
        raw_hf_config = hf_config
        names = key.split(".")
        for i, name in enumerate(names):
            if isinstance(hf_config, dict):
                if name not in hf_config:
                    # to be backward compatible
                    # always put mca config value into hf config kw_args
                    logger.warning(
                        f"{key=} not exists in hf_config for set_hf_config_value, "
                        f"ignore this if no warning in get_hf_config_value"
                    )
                    raw_hf_config[key] = value
                if i == len(names) - 1:
                    hf_config[name] = value
                else:
                    hf_config = hf_config[name]
            else:
                if not hasattr(hf_config, name):
                    # to be backward compatible
                    # always put mca config value into hf config kw_args
                    logger.warning(
                        f"{key=} not exists in hf_config for set_hf_config_value, "
                        f"ignore this if no warning in get_hf_config_value"
                    )
                    raw_hf_config[key] = value
                if i == len(names) - 1:
                    setattr(hf_config, name, value)
                else:
                    hf_config = getattr(hf_config, name)

    def convert_hf_to_mca_config(self, hf_config, **kw_args):
        from ...models.auto.config_auto import AutoConfig as AutoMcaModelConfig

        kw_args = self.convert_hf_to_mca_config_kws(hf_config, **kw_args)
        return AutoMcaModelConfig.for_model(self.hf_model_type, **kw_args)

    def convert_hf_to_mca_config_kws(self, hf_config: "PretrainedConfig", **kw_args):
        for k, v in self.config_hf_to_mca.items():
            cfg_errs = []
            cfg_value = self.get_hf_config_value(hf_config, k, cfg_errs)
            if not cfg_errs:  # cfg_value can be any, use cfg_errs to check
                kw_args[v] = cfg_value
        kw_args["hf_model_type"] = self.hf_model_type
        kw_args["name_or_path"] = hf_config.name_or_path
        kw_args["hf_config_json"] = hf_config.to_json_string()
        return {**kw_args, **self.constant_mca_config}

    def convert_mca_to_hf_config(self, mca_config, **kw_args):
        config_dict = json.loads(mca_config.hf_config_json)
        for k, v in self.config_mca_to_hf.items():
            if hasattr(mca_config, k):
                self.set_hf_config_value(config_dict, v, getattr(mca_config, k))
        kw_args.update(self.constant_hf_config)
        kw_args["name_or_path"] = mca_config.name_or_path
        kw_args = {**config_dict, **kw_args}
        kw_args["model_type"] = self.hf_model_type
        has_remote_code = "auto_map" in config_dict and "AutoConfig" in config_dict["auto_map"]
        if has_remote_code:
            class_ref = config_dict["auto_map"]["AutoConfig"]
            pretrained_model_name_or_path = mca_config.name_or_path
            automap_cache_path = mca_config.get_automap_cache()
            read_cache = os.path.isdir(automap_cache_path) and any(f.endswith('.py') for f in os.listdir(automap_cache_path))
            if read_cache:
                pretrained_model_name_or_path = automap_cache_path
            config_class = get_class_from_dynamic_module(class_ref, pretrained_model_name_or_path)
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

    def add_mca_weight(self, name, weight, **kwargs):
        weight_prefix = get_mca_weight_prefix(name)
        original_name = remove_mca_weight_prefix(name)
        if weight_prefix not in self.prefix_name_to_weight:
            self.prefix_name_to_weight[weight_prefix] = {}
        self.prefix_name_to_weight[weight_prefix][original_name] = weight
        prefix_weights = self.prefix_name_to_weight[weight_prefix]
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

    def get_conver_op(self, name, pattern_to_conver_ops: Dict[str, ConverOp]):
        if name in pattern_to_conver_ops:
            return pattern_to_conver_ops[name]
        for pattern in sorted(pattern_to_conver_ops, key=lambda x: len(x), reverse=True):
            re_pattern = pattern.replace("{}", "(.*?)")
            if re.match(re_pattern, name):
                return pattern_to_conver_ops[pattern]
        raise ValueError(f"can not find conver op for {name} in {pattern_to_conver_ops}")

    def get_lora_conver_op(self, name, pattern_to_conver_ops: Dict[str, ConverOp], lora_rank: int):
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
        elif isinstance(op, QKVConverOp):
            op_class = QKVConverOp
            kwargs = {"hidden_size": lora_rank}
        else:
            raise ValueError(f"can not find lora conver op for {name} in {pattern_to_conver_ops}")
        return op_class(
            hf_names=[hf_name.replace(".weight", lora_name) for hf_name in op.hf_names],
            mca_names=[mca_name.replace(".weight", lora_name) for mca_name in op.mca_names],
            _mca_config=op.mca_config,
            **kwargs,
        )

    def hf_name_to_mca_names(self, hf_name) -> Optional[List[str]]:
        weight_prefix = get_weight_prefix(hf_name, self.hf_layer_prefix, moe_prefix=self.hf_moe_prefix)
        original_name = remove_weight_prefix(hf_name, self.hf_layer_prefix, moe_prefix=self.hf_moe_prefix)
        if original_name in self.hf_invalid_keys:
            return None
        op = self.get_conver_op(original_name, self.hf_name_to_converter)
        mca_prefix = convert_to_mca_prefix(weight_prefix, self.hf_layer_prefix, self.hf_moe_prefix)
        return [mca_prefix + name for name in op.mca_names]


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
