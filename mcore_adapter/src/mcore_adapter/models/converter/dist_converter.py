import fnmatch
import os
from dataclasses import dataclass, field
from itertools import product
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import torch

from ...utils import get_logger
from .convert_utils import (
    StackedTensors,
    add_mca_layer_prefix,
    extract_suffix_number,
    get_mca_layer_index,
    get_mca_moe_index,
    remove_mca_weight_prefix,
    te_grouped_moe_available,
)


if TYPE_CHECKING:
    from torch import Tensor

    from mcore_adapter.models import McaModelConfig


logger = get_logger(__name__)


ASSERT_SP_CONSISTENCY = os.getenv("ASSERT_SP_CONSISTENCY", "1") == "1"


@dataclass
class DistParallelConfig:
    """
    Dataclass for mapping weights to their respective parallelism strategies.
    """

    module_prefix: Optional[str] = None  # the prefix of the parallel module for this config
    pre_process_weights: List[str] = field(default_factory=list)
    post_process_weights: List[str] = field(default_factory=list)
    # tensor parallel
    duplicated_weights: List[str] = field(default_factory=list)
    column_parallel_weights: List[str] = field(default_factory=list)
    row_parallel_weights: List[str] = field(default_factory=list)
    swiglu_weights: List[str] = field(default_factory=list)

    # ungrouped TE name to grouped
    grouped_column_map: Dict[str, str] = field(default_factory=dict)
    grouped_row_map: Dict[str, str] = field(default_factory=dict)

    te_to_local_key_map: dict = field(default_factory=dict)

    def __post_init__(self):
        self.local_to_te_key_map = {v: k for k, v in self.te_to_local_key_map.items()}
        self.grouped_column_weights = list(self.grouped_column_map.keys()) + list(self.grouped_column_map.values())
        self.grouped_row_weights = list(self.grouped_row_map.keys()) + list(self.grouped_row_map.values())
        self.grouped_map = {**self.grouped_column_map, **self.grouped_row_map}
        self.grouped_reverse_map = {v: k for k, v in self.grouped_map.items()}

    def merge_configs(self, other: "DistParallelConfig") -> "DistParallelConfig":
        """
        Merges another ParallelWeightConfig into this one and returns a new object
        with the combined configuration.
        """
        if other is None:
            return self
        return DistParallelConfig(
            module_prefix=other.module_prefix or self.module_prefix,
            pre_process_weights=self.pre_process_weights + other.pre_process_weights,
            post_process_weights=self.post_process_weights + other.post_process_weights,
            duplicated_weights=self.duplicated_weights + other.duplicated_weights,
            column_parallel_weights=self.column_parallel_weights + other.column_parallel_weights,
            row_parallel_weights=self.row_parallel_weights + other.row_parallel_weights,
            swiglu_weights=self.swiglu_weights + other.swiglu_weights,
            grouped_column_map={**self.grouped_column_map, **other.grouped_column_map},
            grouped_row_map={**self.grouped_row_map, **other.grouped_row_map},
            te_to_local_key_map={**self.te_to_local_key_map, **other.te_to_local_key_map},
        )


default_dist_config = DistParallelConfig(
    pre_process_weights=["embedding.word_embeddings.weight"],
    post_process_weights=["output_layer.weight", "decoder.final_layernorm.weight"],
    duplicated_weights=[
        ".self_attention.linear_qkv.layer_norm_weight",
        ".mlp.linear_fc1.layer_norm_weight",
        "decoder.final_layernorm.weight",
        ".mlp.router.weight",
        ".pre_mlp_layernorm.weight",
        ".self_attention.q_layernorm.weight",
        ".self_attention.k_layernorm.weight",
    ],
    column_parallel_weights=[
        "embedding.word_embeddings.weight",
        "output_layer.weight",
        ".self_attention.linear_qkv.weight",
        ".mlp.linear_fc1.weight",
        ".linear_fc1.weight",
    ],
    row_parallel_weights=[".self_attention.linear_proj.weight", ".mlp.linear_fc2.weight", ".linear_fc2.weight"],
    swiglu_weights=[".mlp.linear_fc1.weight", ".linear_fc1.weight"],
    grouped_column_map={".linear_fc1.weight": ".mlp.experts.weight1"},
    grouped_row_map={".linear_fc2.weight": ".mlp.experts.weight2"},
    te_to_local_key_map={
        ".self_attention.linear_qkv.layer_norm_weight": ".input_layernorm.weight",
        ".mlp.linear_fc1.layer_norm_weight": ".pre_mlp_layernorm.weight",
    },
)


te_moe_config = DistParallelConfig(
    grouped_column_map={".linear_fc1.weight": ".mlp.experts.linear_fc1.weight"},
    grouped_row_map={".linear_fc2.weight": ".mlp.experts.linear_fc2.weight"},
)


dist_configs: Dict[str, List[DistParallelConfig]] = {}


def register_dist_config(names: Union[str, List[str]], configs: Union[List[DistParallelConfig], DistParallelConfig]):
    if not isinstance(configs, list):
        configs = [configs]
    if not isinstance(names, list):
        names = [names]
    assert len(configs) == len({c.module_prefix for c in configs}), (
        f"mca_prefix must be different in configs for {names}"
    )
    for name in names:
        assert name not in dist_configs, f"{name} already registered"
        dist_configs[name] = configs


def get_dist_config(name):
    dist_config = dist_configs.get(name, [default_dist_config])
    return dist_config


shared_moe_dist_config = DistParallelConfig(
    duplicated_weights=[".mlp.shared_experts.gate_weight"],
    row_parallel_weights=[".mlp.shared_experts.linear_fc2.weight"],
    swiglu_weights=[".mlp.shared_experts.linear_fc1.weight"],
    te_to_local_key_map={".pre_mlp_layernorm.weight": ".pre_mlp_layernorm.weight"},
)


register_dist_config(
    ["qwen2_moe", "qwen3_moe"],
    default_dist_config.merge_configs(shared_moe_dist_config),
)


register_dist_config(
    ["qwen2_vl", "qwen2_5_vl"],
    [
        default_dist_config,
        DistParallelConfig(module_prefix="vision_model.", pre_process_weights=["*"], duplicated_weights=["*"]),
    ],
)


class DistModuleConverter:
    """
    convert parted of the model weight to model parallel
    """

    def __init__(
        self,
        dist_config: "DistParallelConfig",
        mca_config: "McaModelConfig",
        tensor_model_parallel_rank: int = 0,
        pipeline_model_parallel_rank: int = 0,
        expert_model_parallel_rank: int = 0,
        virtual_pipeline_model_parallel_rank: int = 0,
        revert: bool = False,
    ):
        self.mca_config = mca_config
        self.num_experts = mca_config.num_moe_experts
        self.tensor_model_parallel_rank = tensor_model_parallel_rank or 0
        self.pipeline_model_parallel_rank = pipeline_model_parallel_rank or 0
        self.expert_model_parallel_rank = expert_model_parallel_rank or 0
        self.virtual_pipeline_model_parallel_rank = virtual_pipeline_model_parallel_rank or 0
        self.swiglu = mca_config.swiglu
        self.revert = revert

        self.use_te_grouped_moe = (
            mca_config.moe_grouped_gemm
            and not getattr(mca_config, "moe_use_legacy_grouped_gemm", False)
            and mca_config.transformer_impl == "transformer_engine"
            and te_grouped_moe_available()
        )
        if self.use_te_grouped_moe:
            dist_config = dist_config.merge_configs(te_moe_config)
        self.config = dist_config

        self.num_layers_per_virtual_rank = self._get_num_layers_per_virtual_rank()
        self.num_layers_for_expert = None
        if self.num_experts is not None:
            assert self.num_experts % self.mca_config.expert_model_parallel_size == 0
            self.num_layers_for_expert = self.num_experts // self.mca_config.expert_model_parallel_size

        self.weights_waiting_for_convert: Dict[str, Dict[Union[int, str], "Tensor"]] = {}

    def _get_num_layers_per_virtual_rank(self):
        num_layers = self.mca_config.num_layers
        pipeline_size = self.mca_config.pipeline_model_parallel_size or 1
        virtual_pipeline_size = self.mca_config.virtual_pipeline_model_parallel_size or 1
        if self.mca_config.account_for_embedding_in_pipeline_split:
            num_layers += 1
        if self.mca_config.account_for_loss_in_pipeline_split:
            num_layers += 1
        assert num_layers % (pipeline_size * virtual_pipeline_size) == 0
        return num_layers // (pipeline_size * virtual_pipeline_size)

    def is_on_this_rank(self, weight_name: str):
        if self.revert:
            return True

        def on_this_pipeline():
            if self.pipeline_model_parallel_rank is None:
                return True
            if self.name_match(weight_name, self.config.pre_process_weights):
                return self.pipeline_model_parallel_rank == 0 and self.virtual_pipeline_model_parallel_rank == 0
            if self.name_match(weight_name, self.config.post_process_weights):
                return self.pipeline_model_parallel_rank == (
                    self.mca_config.pipeline_model_parallel_size - 1
                ) and self.virtual_pipeline_model_parallel_rank == (
                    (self.mca_config.virtual_pipeline_model_parallel_size or 1) - 1
                )
            index = get_mca_layer_index(weight_name)
            if index is None:
                return True
            index_pp_rank, index_vp_rank = self._get_layer_info(index)[1:]
            return (
                index_pp_rank == self.pipeline_model_parallel_rank
                and index_vp_rank == self.virtual_pipeline_model_parallel_rank
            )

        def on_this_experts():
            if self.expert_model_parallel_rank is None or self.num_experts is None:
                return True
            moe_index = self.get_local_moe_index(weight_name)
            if moe_index is None:
                return True
            assert isinstance(moe_index, int), f"moe_index: {moe_index}"
            return (moe_index // self.num_layers_for_expert) == self.expert_model_parallel_rank

        return on_this_experts() and on_this_pipeline()

    def _convert_column_parallel(self, weight: "Tensor"):
        return torch.chunk(weight, self.mca_config.tensor_model_parallel_size, dim=0)[
            self.tensor_model_parallel_rank
        ]

    def _revert_column_parallel(self, weights: List["Tensor"]):
        assert len(weights) == self.mca_config.tensor_model_parallel_size
        return torch.cat(weights, dim=0)

    def handle_column_parallel(self, name: str, weights: Union["Tensor", List["Tensor"]]) -> Dict[str, "Tensor"]:
        if self.revert:
            weight = self._revert_column_parallel(weights)
        else:
            weight = self._convert_column_parallel(weights)
        name = self.name_relocate(name)
        return {name: weight}

    def _convert_row_parallel(self, weight: "Tensor"):
        return torch.chunk(weight, self.mca_config.tensor_model_parallel_size, dim=1)[
            self.tensor_model_parallel_rank
        ]

    def _revert_row_parallel(self, weights: List["Tensor"]):
        assert len(weights) == self.mca_config.tensor_model_parallel_size
        return torch.cat(weights, dim=1)

    def handle_row_parallel(self, name: str, weights: Union["Tensor", List["Tensor"]]) -> Dict[str, "Tensor"]:
        if self.revert:
            weight = self._revert_row_parallel(weights)
        else:
            weight = self._convert_row_parallel(weights)
        name = self.name_relocate(name)
        return {name: weight}

    def _convert_swiglu(self, weight: "Tensor"):
        assert self.swiglu and isinstance(weight, StackedTensors) and len(weight.tensors) == 2 and weight.dim == 0, (
            f"weight: {weight} swiglu: {self.swiglu}"
        )
        weight_w = self._convert_column_parallel(weight.tensors[0])
        weight_v = self._convert_column_parallel(weight.tensors[1])
        return torch.cat([weight_w, weight_v], dim=0)

    def _revert_swiglu(self, weights: List["Tensor"]):
        weights = [torch.chunk(weight, 2, dim=0) for weight in weights]
        weights_w = [weight_w[0] for weight_w in weights]
        weights_v = [weight_v[1] for weight_v in weights]
        weight_w = self._revert_column_parallel(weights_w)
        weight_v = self._revert_column_parallel(weights_v)
        return StackedTensors([weight_w, weight_v], dim=0)

    def handle_swiglu(self, name: str, weights: Union["Tensor", List["Tensor"]]) -> Dict[str, "Tensor"]:
        if self.revert:
            weight = self._revert_swiglu(weights)
        else:
            weight = self._convert_swiglu(weights)
        name = self.name_relocate(name)
        return {name: weight}

    def get_pure_name(self, name: str):
        # pure name is the te name without the prefix used to identify parallel strategy
        if self.config.module_prefix:
            name = name.replace(self.config.module_prefix, "")
        pure_name = remove_mca_weight_prefix(name)
        if self.use_te_grouped_moe:
            suffix_num = extract_suffix_number(pure_name)
            if suffix_num is not None and pure_name[: -len(suffix_num)] in self.config.grouped_reverse_map:
                pure_name = pure_name[: -len(suffix_num)]
        if self.mca_config.transformer_impl == "local":
            if self.revert and pure_name in self.config.local_to_te_key_map:
                pure_name = self.config.local_to_te_key_map[pure_name]
        return pure_name

    def name_relocate(self, name: str, moe_index: Optional[int] = None):
        relocated_name = self._name_relocate(name, moe_index)
        if self.config.module_prefix:
            relocated_name = self.config.module_prefix + relocated_name
        return relocated_name

    def _name_relocate(self, name: str, moe_index: Optional[int] = None):
        pure_name = self.get_pure_name(name)
        if self.mca_config.transformer_impl == "local":
            if self.revert:  # when revert to hf, convert to te name
                pure_name = self.config.local_to_te_key_map.get(pure_name, pure_name)
            else:
                pure_name = self.config.te_to_local_key_map.get(pure_name, pure_name)
        layer_index = get_mca_layer_index(name)
        moe_index = get_mca_moe_index(name) if moe_index is None else moe_index
        if layer_index is None:
            return pure_name
        if self.revert:
            layer_index = self.get_global_layer_index(layer_index)
        else:
            layer_index = self.get_local_layer_index(layer_index)
        if moe_index is not None:
            if self.revert:
                if self.mca_config.moe_grouped_gemm:
                    pure_name = self.config.grouped_reverse_map[pure_name]
                moe_index = self.num_layers_for_expert * self.expert_model_parallel_rank + moe_index
            else:
                if self.mca_config.moe_grouped_gemm:
                    moe_index = None
                    pure_name = self.config.grouped_map[pure_name]
                else:
                    moe_index = moe_index % self.num_layers_for_expert
        return add_mca_layer_prefix(pure_name, layer_index, moe_index)

    def _get_layer_info(self, global_layer_index: int):
        offset = 1 if self.mca_config.account_for_embedding_in_pipeline_split else 0
        local_index = (global_layer_index + offset) % self.num_layers_per_virtual_rank
        chunk_index = (global_layer_index + offset) // self.num_layers_per_virtual_rank
        pp_rank = chunk_index % self.mca_config.pipeline_model_parallel_size
        vp_rank = chunk_index // self.mca_config.pipeline_model_parallel_size
        if pp_rank == 0 and vp_rank == 0 and self.mca_config.account_for_embedding_in_pipeline_split:
            local_index -= 1
        return local_index, pp_rank, vp_rank

    def get_local_layer_index(self, global_layer_index: int):
        return self._get_layer_info(global_layer_index)[0]

    def get_global_layer_index(self, local_layer_index: int):
        chunk_index = (
            self.pipeline_model_parallel_rank
            + self.virtual_pipeline_model_parallel_rank * self.mca_config.pipeline_model_parallel_size
        )
        global_layer_index = local_layer_index + chunk_index * self.num_layers_per_virtual_rank
        if self.mca_config.account_for_embedding_in_pipeline_split and chunk_index > 0:
            global_layer_index -= 1
        return global_layer_index

    def handle_duplicated(self, name: str, weights: Union["Tensor", List["Tensor"]]) -> Dict[str, "Tensor"]:
        if self.revert:
            weight = weights[0]
            for w in weights[1:]:
                if w.equal(weight):
                    continue
                message = f"{name} weights are not equal diff sum: {torch.sum(torch.abs(w - weight))}"
                if ASSERT_SP_CONSISTENCY:
                    raise ValueError(message)
                else:
                    logger.warning(message)
                break
        else:
            weight = weights
        name = self.name_relocate(name)
        return {name: weight}

    def _convert_te_grouped_column(self, name: str, weights: "Tensor"):
        if self.swiglu:
            weights = self._convert_swiglu(weights)
        else:
            weights = self._convert_column_parallel(weights)
        # weights = weights.transpose(0, 1)
        moe_index = get_mca_moe_index(name) % self.num_layers_for_expert
        relocated_name = self.name_relocate(name) + str(moe_index)
        return {relocated_name: weights}

    def _revert_te_grouped_column(self, name: str, weights: List["Tensor"]):
        if self.swiglu:
            weight = self._revert_swiglu(weights)
        else:
            weight = self._revert_column_parallel(weights)
        moe_index = int(extract_suffix_number(name))
        return {self.name_relocate(name, moe_index=moe_index): weight}

    def _convert_grouped_column(self, name: str, weights: "Tensor"):
        if self.swiglu:
            weights = self._convert_swiglu(weights)
        else:
            weights = self._convert_column_parallel(weights)
        weights = weights.transpose(0, 1)
        relocated_name = self.name_relocate(name)
        moe_index = get_mca_moe_index(name) % self.num_layers_for_expert
        if relocated_name not in self.weights_waiting_for_convert:
            self.weights_waiting_for_convert[relocated_name] = {}
        self.weights_waiting_for_convert[relocated_name][moe_index] = weights
        if len(self.weights_waiting_for_convert[relocated_name]) < self.num_layers_for_expert:
            return None  # not ready to convert
        weights = sorted(self.weights_waiting_for_convert[relocated_name].items(), key=lambda x: x[0])
        weights = [weight[1] for weight in weights]
        return {relocated_name: torch.stack(weights, dim=0).view(self.mca_config.hidden_size, -1)}

    def _revert_grouped_column(self, name: str, weights: List["Tensor"]):
        def _revert_grouped(weight: "Tensor"):
            weight = weight.view(self.num_layers_for_expert, self.mca_config.hidden_size, -1)
            expert_weights = torch.unbind(weight, dim=0)
            return [weight.transpose(0, 1) for weight in expert_weights]

        # [tp, expert_num_per_rank]
        ungrouped_weights = [_revert_grouped(weight) for weight in weights]
        # [expert_num_per_rank, tp]
        ungrouped_weights = [[weights[i] for weights in ungrouped_weights] for i in range(self.num_layers_for_expert)]

        def _revert_column(weights: List["Tensor"]):
            if self.swiglu:
                return self._revert_swiglu(weights)
            else:
                return self._revert_column_parallel(weights)

        ungrouped_weights = [_revert_column(weights) for weights in ungrouped_weights]
        return {
            self.name_relocate(name, moe_index=moe_index): weight for moe_index, weight in enumerate(ungrouped_weights)
        }

    def handle_grouped_column(self, name: str, weights: Union["Tensor", List["Tensor"]]) -> Dict[str, "Tensor"]:
        if self.revert:
            if self.use_te_grouped_moe:
                return self._revert_te_grouped_column(name, weights)
            return self._revert_grouped_column(name, weights)
        else:
            if self.use_te_grouped_moe:
                return self._convert_te_grouped_column(name, weights)
            return self._convert_grouped_column(name, weights)

    def _convert_te_grouped_row(self, name: str, weights: "Tensor"):
        weights = self._convert_row_parallel(weights)
        # weights = weights.transpose(0, 1)
        moe_index = get_mca_moe_index(name) % self.num_layers_for_expert
        relocated_name = self.name_relocate(name) + str(moe_index)
        return {relocated_name: weights}

    def _revert_te_grouped_row(self, name: str, weights: List["Tensor"]):
        # weights = [weight.transpose(0, 1) for weight in weights]
        weights = self._revert_row_parallel(weights)
        moe_index = int(extract_suffix_number(name))
        return {self.name_relocate(name, moe_index=moe_index): weights}

    def _convert_grouped_row(self, name: str, weights: "Tensor"):
        weights = self._convert_row_parallel(weights)
        weights = weights.transpose(0, 1)
        relocated_name = self.name_relocate(name)
        moe_index = get_mca_moe_index(name) % self.num_layers_for_expert
        if relocated_name not in self.weights_waiting_for_convert:
            self.weights_waiting_for_convert[relocated_name] = {}
        self.weights_waiting_for_convert[relocated_name][moe_index] = weights
        if len(self.weights_waiting_for_convert[relocated_name]) < self.num_layers_for_expert:
            return None  # not ready to convert
        weights = sorted(self.weights_waiting_for_convert[relocated_name].items(), key=lambda x: x[0])
        weights = [weight[1] for weight in weights]
        return {relocated_name: torch.stack(weights, dim=0).view(-1, self.mca_config.hidden_size)}

    def _revert_grouped_row(self, name, weights: List["Tensor"]):
        def _revert_grouped(weight: "Tensor"):
            weight = weight.view(self.num_layers_for_expert, -1, self.mca_config.hidden_size)
            expert_weights = torch.unbind(weight, dim=0)
            return [weight.transpose(0, 1) for weight in expert_weights]

        # [tp, expert_num_per_rank]
        ungrouped_weights = [_revert_grouped(weight) for weight in weights]
        # [expert_num_per_rank, tp]
        ungrouped_weights = [[weights[i] for weights in ungrouped_weights] for i in range(self.num_layers_for_expert)]
        ungrouped_weights = [self._revert_row_parallel(weights) for weights in ungrouped_weights]
        return {
            self.name_relocate(name, moe_index=moe_index): weight for moe_index, weight in enumerate(ungrouped_weights)
        }

    def handle_grouped_row(self, name: str, weights: Union["Tensor", List["Tensor"]]) -> Dict[str, "Tensor"]:
        if self.revert:
            if self.use_te_grouped_moe:
                return self._revert_te_grouped_row(name, weights)
            return self._revert_grouped_row(name, weights)
        else:
            if self.use_te_grouped_moe:
                return self._convert_te_grouped_row(name, weights)
            return self._convert_grouped_row(name, weights)

    def name_match(self, pure_name: str, patterns: List[str]):
        if pure_name in patterns:
            return True
        for pattern in patterns:
            if fnmatch.fnmatch(pure_name, pattern):
                return True
        return False

    def get_local_moe_index(self, name: str) -> Optional[Union[int, List[int]]]:
        if self.config.module_prefix:
            name = name.replace(self.config.module_prefix, "")
        pure_name = remove_mca_weight_prefix(name)
        if self.use_te_grouped_moe:
            suffix_num = extract_suffix_number(pure_name)
            if suffix_num is not None and pure_name[: -len(suffix_num)] in self.config.grouped_reverse_map:
                return int(suffix_num)
        if self.mca_config.moe_grouped_gemm:
            if pure_name in self.config.grouped_reverse_map:
                return list(range(self.num_layers_for_expert))
        return get_mca_moe_index(name)

    def get_global_moe_index(self, name: str) -> Optional[Union[int, List[int]]]:
        local_moe_index = self.get_local_moe_index(name)
        if local_moe_index is None:
            return None
        local_to_global = lambda i: i + self.num_layers_for_expert * self.expert_model_parallel_rank
        if isinstance(local_moe_index, int):
            return local_to_global(local_moe_index)
        else:
            return [local_to_global(i) for i in local_moe_index]

    def dist_convert(self, name: str, weights: Union["Tensor", List["Tensor"]]) -> Dict[str, "Tensor"]:
        if not self.is_on_this_rank(name):
            return None
        pure_name = self.get_pure_name(name)
        if pure_name.endswith(".bias"):
            pure_name = pure_name.replace(".bias", ".weight")
        if self.mca_config.moe_grouped_gemm and self.name_match(pure_name, self.config.grouped_column_weights):
            return self.handle_grouped_column(name, weights)
        if self.mca_config.moe_grouped_gemm and self.name_match(pure_name, self.config.grouped_row_weights):
            return self.handle_grouped_row(name, weights)
        if self.swiglu and self.name_match(pure_name, self.config.swiglu_weights):
            return self.handle_swiglu(name, weights)
        if self.name_match(pure_name, self.config.duplicated_weights):
            return self.handle_duplicated(name, weights)
        if self.name_match(pure_name, self.config.column_parallel_weights):
            return self.handle_column_parallel(name, weights)
        if self.name_match(pure_name, self.config.row_parallel_weights):
            return self.handle_row_parallel(name, weights)
        raise ValueError(f"name: {name}, pure_name: {pure_name}, config {self.config} swiglu: {self.swiglu}")


class DistConverter:
    def __init__(
        self,
        mca_config: "McaModelConfig",
        tensor_model_parallel_rank: int = 0,
        pipeline_model_parallel_rank: int = 0,
        expert_model_parallel_rank: int = 0,
        virtual_pipeline_model_parallel_rank: int = 0,
        **kwargs,
    ):
        dist_configs = get_dist_config(mca_config.hf_model_type)
        self.dist_configs = dist_configs
        self.mca_config = mca_config
        self.tensor_model_parallel_rank = tensor_model_parallel_rank
        self.pipeline_model_parallel_rank = pipeline_model_parallel_rank
        self.expert_model_parallel_rank = expert_model_parallel_rank
        self.virtual_pipeline_model_parallel_rank = virtual_pipeline_model_parallel_rank
        module_kwargs = {
            "mca_config": mca_config,
            "tensor_model_parallel_rank": tensor_model_parallel_rank,
            "pipeline_model_parallel_rank": pipeline_model_parallel_rank,
            "expert_model_parallel_rank": expert_model_parallel_rank,
            "virtual_pipeline_model_parallel_rank": virtual_pipeline_model_parallel_rank,
            **kwargs,
        }
        self.module_converters = {
            config.module_prefix or "": DistModuleConverter(config, **module_kwargs) for config in dist_configs
        }
        self.sorted_prefixes = sorted(self.module_converters.keys(), key=lambda x: len(x), reverse=True)

    def __call__(self, name: str, weights: Union["Tensor", List["Tensor"]]):
        return self.dist_convert(name=name, weights=weights)

    def get_module_converter(self, name: str):
        for prefix in self.sorted_prefixes:
            if name.startswith(prefix):
                return self.module_converters[prefix]
        raise ValueError(f"Didn't find prefix for name: {name} prefixes: {self.sorted_prefixes}")

    def dist_convert(self, name: str, weights: Union["Tensor", List["Tensor"]]) -> Dict[str, "Tensor"]:
        return self.get_module_converter(name).dist_convert(name, weights)

    def is_on_this_rank(self, name: str):
        return self.get_module_converter(name).is_on_this_rank(name)

    def get_local_moe_index(self, name: str):
        return self.get_module_converter(name).get_local_moe_index(name)

    def get_global_moe_index(self, name: str):
        return self.get_module_converter(name).get_global_moe_index(name)

    @staticmethod
    def dist_converter_iter(mca_config: "McaModelConfig", **kwargs):
        for tp_rank, pp_rank, ep_rank in product(
            range(mca_config.tensor_model_parallel_size),
            range(mca_config.pipeline_model_parallel_size),
            range(mca_config.expert_model_parallel_size),
        ):
            yield DistConverter(
                mca_config,
                tensor_model_parallel_rank=tp_rank,
                pipeline_model_parallel_rank=pp_rank,
                expert_model_parallel_rank=ep_rank,
                **kwargs,
            )
