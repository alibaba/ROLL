import fnmatch
import os
import warnings
from dataclasses import dataclass, field
from itertools import product
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import torch
from megatron.core.transformer.pipeline_parallel_layer_layout import LayerType, PipelineParallelLayerLayout

from ...utils import get_logger, is_megatron_llama
from .convert_utils import (
    StackedTensors,
    add_mca_layer_prefix,
    add_mca_mtp_layer_prefix,
    extract_suffix_number,
    get_mca_layer_index,
    get_mca_moe_index,
    remove_mca_weight_prefix,
    te_grouped_moe_available,
)


if TYPE_CHECKING:
    from torch import Tensor

    from ..model_config import McaModelConfig


logger = get_logger(__name__)


ASSERT_SP_CONSISTENCY = os.getenv("ASSERT_SP_CONSISTENCY", "1") == "1"

MCORE_WORD_EMBEDDING = "embedding.word_embeddings.weight"
MCORE_LM_HEAD = "output_layer.weight"


@dataclass
class DistParallelConfig:
    """
    Dataclass for mapping weights to their respective parallelism strategies.
    """

    pre_process_weights: List[str] = field(default_factory=list)
    post_process_weights: List[str] = field(default_factory=list)
    # tensor parallel
    duplicated_weights: List[str] = field(default_factory=list)
    column_parallel_weights: List[str] = field(default_factory=list)
    row_parallel_weights: List[str] = field(default_factory=list)
    swiglu_weights: List[str] = field(default_factory=list)

    # ungrouped TE name to grouped
    grouped_duplicated_map: Dict[str, str] = field(default_factory=dict)
    grouped_column_map: Dict[str, str] = field(default_factory=dict)
    grouped_row_map: Dict[str, str] = field(default_factory=dict)

    te_to_local_key_map: dict = field(default_factory=dict)

    def __post_init__(self):
        self.local_to_te_key_map = {v: k for k, v in self.te_to_local_key_map.items()}
        self.grouped_duplicated_weights = list(self.grouped_duplicated_map.keys()) + list(
            self.grouped_duplicated_map.values()
        )
        self.grouped_column_weights = list(self.grouped_column_map.keys()) + list(self.grouped_column_map.values())
        self.grouped_row_weights = list(self.grouped_row_map.keys()) + list(self.grouped_row_map.values())
        self.grouped_map = {**self.grouped_duplicated_map, **self.grouped_column_map, **self.grouped_row_map}
        self.grouped_reverse_map = {v: k for k, v in self.grouped_map.items()}

    def merge_configs(self, other: "DistParallelConfig") -> "DistParallelConfig":
        """
        Merges another ParallelWeightConfig into this one and returns a new object
        with the combined configuration.
        """
        if other is None:
            return self
        return DistParallelConfig(
            pre_process_weights=self.pre_process_weights + other.pre_process_weights,
            post_process_weights=self.post_process_weights + other.post_process_weights,
            duplicated_weights=self.duplicated_weights + other.duplicated_weights,
            column_parallel_weights=self.column_parallel_weights + other.column_parallel_weights,
            row_parallel_weights=self.row_parallel_weights + other.row_parallel_weights,
            swiglu_weights=self.swiglu_weights + other.swiglu_weights,
            grouped_duplicated_map={**self.grouped_duplicated_map, **other.grouped_duplicated_map},
            grouped_column_map={**self.grouped_column_map, **other.grouped_column_map},
            grouped_row_map={**self.grouped_row_map, **other.grouped_row_map},
            te_to_local_key_map={**self.te_to_local_key_map, **other.te_to_local_key_map},
        )


lora_config = DistParallelConfig(
    duplicated_weights=[
        ".self_attention.linear_proj.lora_B.weight",
        ".self_attention.linear_qkv.lora_A.weight",
        ".mlp.linear_fc1.lora_A.weight",
        ".linear_fc1.lora_A.weight",
        ".mlp.linear_fc2.lora_B.weight",
        ".linear_fc2.lora_B.weight",
    ],
    column_parallel_weights=[
        ".self_attention.linear_qkv.lora_B.weight",
        ".mlp.linear_fc1.lora_B.weight",
        ".linear_fc1.lora_B.weight",
    ],
    row_parallel_weights=[
        ".self_attention.linear_proj.lora_A.weight",
        ".mlp.linear_fc2.lora_A.weight",
        ".linear_fc2.lora_A.weight",
    ],
    swiglu_weights=[".mlp.linear_fc1.lora_B.weight", ".linear_fc1.lora_B.weight"],
)


default_dist_config = DistParallelConfig(
    pre_process_weights=[MCORE_WORD_EMBEDDING],
    post_process_weights=[MCORE_LM_HEAD, "decoder.final_layernorm.weight"],
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
        MCORE_WORD_EMBEDDING,
        MCORE_LM_HEAD,
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
).merge_configs(lora_config)


lora_te_moe_config = DistParallelConfig(
    grouped_duplicated_map={
        ".linear_fc1.lora_A.weight": ".mlp.experts.linear_fc1.lora_A.weight",
        ".linear_fc2.lora_B.weight": ".mlp.experts.linear_fc2.lora_B.weight",
    },
    grouped_column_map={".linear_fc1.lora_B.weight": ".mlp.experts.linear_fc1.lora_B.weight"},
    grouped_row_map={".linear_fc2.lora_A.weight": ".mlp.experts.linear_fc2.lora_A.weight"},
)


te_moe_config = DistParallelConfig(
    grouped_column_map={".linear_fc1.weight": ".mlp.experts.linear_fc1.weight"},
    grouped_row_map={".linear_fc2.weight": ".mlp.experts.linear_fc2.weight"},
).merge_configs(lora_te_moe_config)


mtp_config = DistParallelConfig(
    duplicated_weights=[
        ".enorm.weight",
        ".hnorm.weight",
        ".final_layernorm.weight",
    ],
    column_parallel_weights=[
        ".eh_proj.weight",
    ],
)


mla_dist_config = DistParallelConfig(
    pre_process_weights=[MCORE_WORD_EMBEDDING],
    post_process_weights=[MCORE_LM_HEAD, "decoder.final_layernorm.weight"],
    duplicated_weights=[
        ".self_attention.q_layernorm.weight",
        ".input_layernorm.weight",
        "decoder.final_layernorm.weight",
        ".pre_mlp_layernorm.weight",
        ".self_attention.kv_layernorm.weight",
        ".mlp.router.weight",
        ".mlp.router.expert_bias",
        ".mlp.linear_fc1.layer_norm_weight",
        ".self_attention.linear_q_up_proj.layer_norm_weight",
        ".self_attention.linear_kv_up_proj.layer_norm_weight",
    ],
    column_parallel_weights=[
        MCORE_WORD_EMBEDDING,
        MCORE_LM_HEAD,
        ".self_attention.linear_q_down_proj.weight",
        ".self_attention.linear_q_up_proj.weight",
        ".self_attention.linear_q_proj.weight",
        ".self_attention.linear_kv_down_proj.weight",
        ".self_attention.linear_kv_up_proj.weight",
    ],
    grouped_column_map={".linear_fc1.weight": ".mlp.experts.weight1"},
    grouped_row_map={".linear_fc2.weight": ".mlp.experts.weight2"},
    row_parallel_weights=[
        ".self_attention.linear_proj.weight",
        ".mlp.shared_experts.linear_fc2.weight",
        ".linear_fc2.weight",
        ".mlp.linear_fc2.weight",
    ],
    swiglu_weights=[
        ".mlp.shared_experts.linear_fc1.weight",
        ".linear_fc1.weight",
        ".mlp.linear_fc1.weight",
    ],
).merge_configs(mtp_config)


megatron_llama_config = DistParallelConfig(
    duplicated_weights=[".input_layernorm.weight"],
    grouped_column_map={".linear_fc1.weight": ".mlp.weight1"},
    grouped_row_map={".linear_fc2.weight": ".mlp.weight2"},
)


dist_configs: Dict[str, List[DistParallelConfig]] = {}


def register_dist_config(names: Union[str, List[str]], config: DistParallelConfig):
    if not isinstance(names, list):
        names = [names]
    for name in names:
        assert name not in dist_configs, f"{name} already registered"
        dist_configs[name] = config


def get_dist_config(name) -> DistParallelConfig:
    dist_config = dist_configs.get(name, [default_dist_config])
    return dist_config


lora_shared_moe_dist_config = DistParallelConfig(
    duplicated_weights=[
        ".mlp.shared_experts.linear_fc1.lora_A.weight",
        ".mlp.shared_experts.linear_fc2.lora_B.weight",
    ],
    column_parallel_weights=[
        ".mlp.shared_experts.linear_fc1.lora_B.weight",
    ],
    row_parallel_weights=[
        ".mlp.shared_experts.linear_fc2.lora_A.weight",
    ],
    swiglu_weights=[".mlp.shared_experts.linear_fc1.lora_B.weight"],
)


shared_moe_dist_config = DistParallelConfig(
    duplicated_weights=[".mlp.shared_experts.gate_weight"],
    row_parallel_weights=[".mlp.shared_experts.linear_fc2.weight"],
    swiglu_weights=[".mlp.shared_experts.linear_fc1.weight"],
    te_to_local_key_map={".pre_mlp_layernorm.weight": ".pre_mlp_layernorm.weight"},
).merge_configs(lora_shared_moe_dist_config)


class DistConverter:
    """
    convert parted of the model weight to model parallel
    """

    def __init__(
        self,
        mca_config: "McaModelConfig",
        tensor_model_parallel_rank: int = 0,
        pipeline_model_parallel_rank: int = 0,
        expert_model_parallel_rank: int = 0,
        virtual_pipeline_model_parallel_rank: int = 0,
        revert: bool = False,
        efficient_mode: bool = False,  # not check the consistency of weights
    ):
        self.mca_config = mca_config
        self.num_experts = mca_config.num_moe_experts
        self.tensor_model_parallel_rank = tensor_model_parallel_rank or 0
        self.pipeline_model_parallel_rank = pipeline_model_parallel_rank or 0
        self.expert_model_parallel_rank = expert_model_parallel_rank or 0
        self.virtual_pipeline_model_parallel_rank = virtual_pipeline_model_parallel_rank or 0
        self.swiglu = mca_config.swiglu
        self.revert = revert
        self.efficient_mode = efficient_mode

        self.use_te_grouped_moe = (
            mca_config.moe_grouped_gemm
            and not getattr(mca_config, "moe_use_legacy_grouped_gemm", False)
            and mca_config.transformer_impl == "transformer_engine"
            and te_grouped_moe_available()
        )
        dist_config = get_dist_config(mca_config.hf_model_type)
        if self.use_te_grouped_moe:
            dist_config = dist_config.merge_configs(te_moe_config)
        if is_megatron_llama():
            dist_config = dist_config.merge_configs(megatron_llama_config)
        self.config = dist_config
        self.layout: PipelineParallelLayerLayout = self.mca_config.pipeline_model_parallel_layout

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
        if self.layout is not None:
            return None  # not need while using layout

        if self.mca_config.account_for_embedding_in_pipeline_split:
            num_layers += 1
        if self.mca_config.account_for_loss_in_pipeline_split:
            num_layers += 1
        assert num_layers % (pipeline_size * virtual_pipeline_size) == 0
        return num_layers // (pipeline_size * virtual_pipeline_size)

    def is_on_this_rank(self, weight_name: str, vp_stage: Optional[int] = None):
        if vp_stage is None:
            vp_stage = self.virtual_pipeline_model_parallel_rank
        if self.revert:
            return True

        def on_this_pipeline():
            if self.pipeline_model_parallel_rank is None:
                return True
            if weight_name.startswith("mtp.layers."):
                return self.mca_config.mtp_num_layers and self.is_pipeline_last_stage(vp_stage=vp_stage)

            if self.name_match(weight_name, self.config.pre_process_weights):
                # mtp and tie_embeddings_and_output_weights use embedding weights in last stage
                if weight_name == MCORE_WORD_EMBEDDING and (
                    self.mca_config.mtp_num_layers or self.mca_config.tie_embeddings_and_output_weights
                ):
                    if self.is_pipeline_last_stage(vp_stage=vp_stage):
                        return True
                return self.is_pipeline_first_stage(vp_stage=vp_stage)
            if self.name_match(weight_name, self.config.post_process_weights):
                return self.is_pipeline_last_stage(vp_stage=vp_stage)
            index = get_mca_layer_index(weight_name)
            if index is None:
                return True
            index_pp_rank, index_vp_rank = self._get_layer_info(index)[1:]
            return index_pp_rank == self.pipeline_model_parallel_rank and index_vp_rank == vp_stage

        def on_this_experts():
            if self.expert_model_parallel_rank is None or self.num_experts is None:
                return True
            moe_index = self.get_local_moe_index(weight_name)
            if moe_index is None:
                return True
            assert isinstance(moe_index, int), f"moe_index: {moe_index}"
            return (moe_index // self.num_layers_for_expert) == self.expert_model_parallel_rank

        return on_this_experts() and on_this_pipeline()

    def is_pipeline_last_stage(self, vp_stage: int):
        return self.pipeline_model_parallel_rank == (
            self.mca_config.pipeline_model_parallel_size - 1
        ) and vp_stage == ((self.mca_config.virtual_pipeline_model_parallel_size or 1) - 1)

    def is_pipeline_first_stage(self, vp_stage: int):
        return self.pipeline_model_parallel_rank == 0 and vp_stage == 0

    def _convert_column_parallel(self, weight: "Tensor"):
        return torch.chunk(weight, self.mca_config.tensor_model_parallel_size, dim=0)[
            self.tensor_model_parallel_rank
        ]

    def _revert_column_parallel(self, weights: List["Tensor"]):
        assert len(weights) == self.mca_config.tensor_model_parallel_size
        if len(weights) == 1:
            return weights[0]
        return torch.cat(weights, dim=0)

    def handle_column_parallel(self, name: str, weights: Union["Tensor", List["Tensor"]]) -> Dict[str, "Tensor"]:
        if self.revert:
            weight = self._revert_column_parallel(weights)
        else:
            weight = self._convert_column_parallel(weights)
        name = self._name_relocate(name)
        return {name: weight}

    def _convert_row_parallel(self, weight: "Tensor"):
        return torch.chunk(weight, self.mca_config.tensor_model_parallel_size, dim=1)[
            self.tensor_model_parallel_rank
        ]

    def _revert_row_parallel(self, weights: List["Tensor"]):
        assert len(weights) == self.mca_config.tensor_model_parallel_size
        if len(weights) == 1:
            return weights[0]
        return torch.cat(weights, dim=1)

    def handle_row_parallel(self, name: str, weights: Union["Tensor", List["Tensor"]]) -> Dict[str, "Tensor"]:
        if self.revert:
            weight = self._revert_row_parallel(weights)
        else:
            weight = self._convert_row_parallel(weights)
        name = self._name_relocate(name)
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
        name = self._name_relocate(name)
        return {name: weight}

    def get_pure_name(self, name: str):
        # pure name is the te name without the prefix used to identify parallel strategy
        pure_name = remove_mca_weight_prefix(name)
        if self.use_te_grouped_moe:
            suffix_num = extract_suffix_number(pure_name)
            if suffix_num is not None and self.name_match(
                pure_name[: -len(suffix_num)], self.config.grouped_reverse_map
            ):
                pure_name = pure_name[: -len(suffix_num)]
        if self.mca_config.transformer_impl == "local":
            if self.revert and pure_name in self.config.local_to_te_key_map:
                pure_name = self.config.local_to_te_key_map[pure_name]
        return pure_name

    def _name_relocate(self, name: str, moe_index: Optional[int] = None, moe_index_preprocessed: bool = False):
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

        if moe_index is not None:
            if self.revert:
                if self.mca_config.moe_grouped_gemm:
                    pure_name = self.get_matched_name(pure_name, self.config.grouped_reverse_map)
                if not moe_index_preprocessed:
                    moe_index = self.num_layers_for_expert * self.expert_model_parallel_rank + moe_index
            else:
                if self.mca_config.moe_grouped_gemm:
                    moe_index = None
                    pure_name = self.config.grouped_map[pure_name]
                else:
                    moe_index = moe_index % self.num_layers_for_expert
        if name.startswith("mtp.layers."):
            return add_mca_mtp_layer_prefix(pure_name, layer_index, moe_index)
        return add_mca_layer_prefix(pure_name, layer_index, moe_index)

    def _get_layer_info(self, global_layer_index: int):
        if self.layout is not None:
            offset = 0
            vp_size = self.mca_config.virtual_pipeline_model_parallel_size or 1
            for vpp_rank in range(vp_size):
                for pp_rank in range(self.mca_config.pipeline_model_parallel_size):
                    new_offset = offset + self.layout.layout[pp_rank][vpp_rank].count(LayerType.decoder)
                    if new_offset > global_layer_index:
                        return global_layer_index - offset, pp_rank, vpp_rank
                    offset = new_offset
            raise ValueError(f"{global_layer_index=} not in {self.layout=}")

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

    def get_global_layer_index(self, local_layer_index: int, vp_stage: int):
        if self.layout is not None:
            return self.layout.get_layer_offset(vp_stage=vp_stage) + local_layer_index

        chunk_index = self.pipeline_model_parallel_rank + vp_stage * self.mca_config.pipeline_model_parallel_size
        global_layer_index = local_layer_index + chunk_index * self.num_layers_per_virtual_rank
        if self.mca_config.account_for_embedding_in_pipeline_split and chunk_index > 0:
            global_layer_index -= 1
        return global_layer_index

    def handle_duplicated(self, name: str, weights: Union["Tensor", List["Tensor"]]) -> Dict[str, "Tensor"]:
        if self.revert:
            weight = weights[0]
            if not self.efficient_mode:
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
        name = self._name_relocate(name)
        return {name: weight}

    def handle_grouped_duplicated(self, name: str, weights: Union["Tensor", List["Tensor"]]) -> Dict[str, "Tensor"]:
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
            raise NotImplementedError()
        moe_index = int(extract_suffix_number(name))
        return {self._name_relocate(name, moe_index=moe_index): weight}

    def _convert_te_grouped_column(self, name: str, weights: "Tensor"):
        if self.swiglu:
            weights = self._convert_swiglu(weights)
        else:
            weights = self._convert_column_parallel(weights)
        # weights = weights.transpose(0, 1)
        moe_index = get_mca_moe_index(name) % self.num_layers_for_expert
        relocated_name = self._name_relocate(name) + str(moe_index)
        return {relocated_name: weights}

    def _revert_te_grouped_column(self, name: str, weights: List["Tensor"], moe_index_preprocessed: bool = False):
        if self.swiglu:
            weight = self._revert_swiglu(weights)
        else:
            weight = self._revert_column_parallel(weights)
        moe_index = int(extract_suffix_number(name))
        return {self._name_relocate(name, moe_index=moe_index, moe_index_preprocessed=moe_index_preprocessed): weight}

    def _convert_grouped_column(self, name: str, weights: "Tensor"):
        if self.swiglu:
            weights = self._convert_swiglu(weights)
        else:
            weights = self._convert_column_parallel(weights)
        weights = weights.transpose(0, 1)
        relocated_name = self._name_relocate(name)
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
            self._name_relocate(name, moe_index=moe_index): weight
            for moe_index, weight in enumerate(ungrouped_weights)
        }

    def handle_grouped_column(
        self, name: str, weights: Union["Tensor", List["Tensor"]], moe_index_preprocessed: bool = False
    ) -> Dict[str, "Tensor"]:
        if self.revert:
            if self.use_te_grouped_moe:
                return self._revert_te_grouped_column(name, weights, moe_index_preprocessed=moe_index_preprocessed)
            return self._revert_grouped_column(name, weights)
        else:
            if self.use_te_grouped_moe:
                return self._convert_te_grouped_column(name, weights)
            return self._convert_grouped_column(name, weights)

    def _convert_te_grouped_row(self, name: str, weights: "Tensor"):
        weights = self._convert_row_parallel(weights)
        moe_index = get_mca_moe_index(name) % self.num_layers_for_expert
        relocated_name = self._name_relocate(name) + str(moe_index)
        return {relocated_name: weights}

    def _revert_te_grouped_row(self, name: str, weights: List["Tensor"], moe_index_preprocessed: bool = False):
        weights = self._revert_row_parallel(weights)
        moe_index = int(extract_suffix_number(name))
        return {self._name_relocate(name, moe_index=moe_index, moe_index_preprocessed=moe_index_preprocessed): weights}

    def _convert_grouped_row(self, name: str, weights: "Tensor"):
        weights = self._convert_row_parallel(weights)
        weights = weights.transpose(0, 1)
        relocated_name = self._name_relocate(name)
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
            self._name_relocate(name, moe_index=moe_index): weight
            for moe_index, weight in enumerate(ungrouped_weights)
        }

    def handle_grouped_row(
        self, name: str, weights: Union["Tensor", List["Tensor"]], moe_index_preprocessed: bool = False
    ) -> Dict[str, "Tensor"]:
        if self.revert:
            if self.use_te_grouped_moe:
                return self._revert_te_grouped_row(name, weights, moe_index_preprocessed=moe_index_preprocessed)
            return self._revert_grouped_row(name, weights)
        else:
            if self.use_te_grouped_moe:
                return self._convert_te_grouped_row(name, weights)
            return self._convert_grouped_row(name, weights)

    def name_match(self, pure_name: str, patterns: list[str] | dict[str, Any]):
        if pure_name in patterns:
            return True
        for pattern in patterns:
            if fnmatch.fnmatch(pure_name, pattern):
                return True
        return False

    def get_matched_name(self, name: str, weight_map: dict[str, Any]) -> Optional[str]:
        if name in weight_map:
            return weight_map[name]
        for key in weight_map:
            if fnmatch.fnmatch(name, key):
                name_pattern = weight_map[key]
                return name_pattern[: name_pattern.find(".lora")] + name[name.find(".lora") :]

    def get_local_moe_index(self, name: str) -> Optional[Union[int, List[int]]]:
        pure_name = remove_mca_weight_prefix(name)
        if self.use_te_grouped_moe:
            suffix_num = extract_suffix_number(pure_name)
            if suffix_num is not None and self.name_match(
                pure_name[: -len(suffix_num)], self.config.grouped_reverse_map
            ):
                return int(suffix_num)
        if self.mca_config.moe_grouped_gemm:
            if self.name_match(pure_name, self.config.grouped_reverse_map):
                return list(range(self.num_layers_for_expert))
        return get_mca_moe_index(name)

    def get_global_moe_index(self, name: str) -> Optional[Union[int, List[int]]]:
        local_moe_index = self.get_local_moe_index(name)
        if local_moe_index is None:
            return None

        def local_to_global(i):
            return i + self.num_layers_for_expert * self.expert_model_parallel_rank

        if isinstance(local_moe_index, int):
            return local_to_global(local_moe_index)
        else:
            return [local_to_global(i) for i in local_moe_index]

    def preprocess_layer_index(self, name: str, vp_stage: int) -> str:
        """
        Preprocess layer index for pipeline parallelism.
        Converts between global and local layer indices before calling name_relocate.
        """
        layer_index = get_mca_layer_index(name)
        if layer_index is None:
            return name
        moe_index = get_mca_moe_index(name)

        if self.revert:
            layer_index = self.get_global_layer_index(layer_index, vp_stage=vp_stage)
        else:
            layer_index = self.get_local_layer_index(layer_index)

        if name.startswith("mtp.layers."):
            return add_mca_mtp_layer_prefix(remove_mca_weight_prefix(name), layer_index, moe_index)
        return add_mca_layer_prefix(remove_mca_weight_prefix(name), layer_index, moe_index)

    def dist_convert(
        self,
        name: str,
        weights: Union["Tensor", List["Tensor"]],
        vp_stage: Optional[int] = None,
        layer_index_preprocessed: bool = False,
        moe_index_preprocessed: bool = False,
    ) -> Dict[str, "Tensor"]:
        """
        Convert weights for distributed parallelism.

        Args:
            name: Weight name
            weights: Weight tensor(s)
            vp_stage: Virtual pipeline stage
            layer_index_preprocessed: If True, the name's layer index has already been preprocessed
                for pipeline parallelism by the caller. If False (default), DistConverter will
                handle the layer index conversion between global and local indices.
            moe_index_preprocessed: If True, the name's moe index has already been preprocessed
                for expert parallelism by the caller. If False (default), DistConverter will
                handle the moe index conversion between global and local indices.
        """
        if vp_stage is None:
            vp_stage = self.virtual_pipeline_model_parallel_rank
        if (
            self.mca_config.tie_embeddings_and_output_weights
            and self.mca_config.pipeline_model_parallel_size > 1
            and self.is_pipeline_last_stage(vp_stage=vp_stage)
        ):
            if self.revert and name == MCORE_LM_HEAD:
                return None  # don't need a duplicate lm head
            elif not self.revert and name == MCORE_WORD_EMBEDDING:
                name = MCORE_LM_HEAD  # load word embedding weight to lm head

        if not self.is_on_this_rank(name, vp_stage=vp_stage):
            return None

        if not layer_index_preprocessed:
            name = self.preprocess_layer_index(name, vp_stage)

        pure_name = self.get_pure_name(name)
        if pure_name.endswith(".bias"):
            pure_name = pure_name.replace(".bias", ".weight")
        if self.mca_config.moe_grouped_gemm and self.name_match(pure_name, self.config.grouped_duplicated_weights):
            return self.handle_grouped_duplicated(name, weights)
        if self.mca_config.moe_grouped_gemm and self.name_match(pure_name, self.config.grouped_column_weights):
            return self.handle_grouped_column(name, weights, moe_index_preprocessed=moe_index_preprocessed)
        if self.mca_config.moe_grouped_gemm and self.name_match(pure_name, self.config.grouped_row_weights):
            return self.handle_grouped_row(name, weights, moe_index_preprocessed=moe_index_preprocessed)
        if self.swiglu and self.name_match(pure_name, self.config.swiglu_weights):
            return self.handle_swiglu(name, weights)
        if self.name_match(pure_name, self.config.duplicated_weights):
            return self.handle_duplicated(name, weights)
        if self.name_match(pure_name, self.config.column_parallel_weights):
            return self.handle_column_parallel(name, weights)
        if self.name_match(pure_name, self.config.row_parallel_weights):
            return self.handle_row_parallel(name, weights)
        raise ValueError(f"name: {name}, pure_name: {pure_name}, config {self.config} swiglu: {self.swiglu}")

    def is_tensor_parallel_dup_weight(self, name: str) -> bool:
        pure_name = self.get_pure_name(name)
        return self.name_match(pure_name, self.config.duplicated_weights)

    def is_expert_parallel_weight(self, name: str) -> bool:
        return self.get_local_moe_index(name) is not None

    def __call__(self, name: str, weights: Union["Tensor", List["Tensor"]], vp_stage: Optional[int] = None):
        return self.dist_convert(name=name, weights=weights, vp_stage=vp_stage)

    @staticmethod
    def dist_converter_iter(mca_config: "McaModelConfig", **kwargs):
        warnings.warn("dist_converter_iter is deprecated", DeprecationWarning)
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
