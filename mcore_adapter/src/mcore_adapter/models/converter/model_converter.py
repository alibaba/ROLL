import gc
import json
import os
from typing import TYPE_CHECKING, Dict, Optional, Union

import torch
import torch.distributed as dist
from megatron.core import mpu
from transformers.modeling_utils import (
    get_checkpoint_shard_files,
    load_state_dict,
)
from transformers.utils import (
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    is_peft_available,
    is_safetensors_available,
)

from ...utils import get_logger
from .convert_utils import (
    MAX_SHARD_SIZE,
    SendBucketManager,
    StateDictSplitState,
    all_gather_tensors,
    allgather_parallel_objs,
    gather_tensor_parallel,
    get_tensor_size,
    parse_size_to_int,
    resize_embedding_layer,
)
from .dist_converter import MCORE_WORD_EMBEDDING, MCORE_LM_HEAD, DistConverter
from .template import get_template


if is_peft_available():
    from peft import PeftModel, get_peft_model_state_dict

if is_safetensors_available():
    from safetensors.torch import save_file as safe_save_file


if TYPE_CHECKING:
    from torch import Tensor

    from ..model_config import McaModelConfig

logger = get_logger(__name__)


class ModelConverter:
    def __init__(
        self,
        mca_config: "McaModelConfig",
        tensor_model_parallel_rank: Optional[int] = None,
        pipeline_model_parallel_rank: Optional[int] = None,
        expert_model_parallel_rank: Optional[int] = None,
        to_hf: bool = False,
        verbose=False,
        efficient_mode: bool = False,
        resized_vocab_size: int = None,
    ):
        self.mca_config = mca_config
        self.verbose = verbose
        self.template = get_template(mca_config.hf_model_type)
        self.template.set_mca_config_for_ops(self.mca_config)
        if tensor_model_parallel_rank is None:
            tensor_model_parallel_rank = mpu.get_tensor_model_parallel_rank()
        if pipeline_model_parallel_rank is None:
            pipeline_model_parallel_rank = mpu.get_pipeline_model_parallel_rank()
        if expert_model_parallel_rank is None:
            expert_model_parallel_rank = mpu.get_expert_model_parallel_rank()
        self.dist_converter = DistConverter(
            self.mca_config,
            tensor_model_parallel_rank=tensor_model_parallel_rank,
            pipeline_model_parallel_rank=pipeline_model_parallel_rank,
            expert_model_parallel_rank=expert_model_parallel_rank,
            revert=to_hf,
            efficient_mode=efficient_mode,
        )
        self.resized_vocab_size = resized_vocab_size

    def log(self, msg):
        if self.verbose:
            logger.info(msg)

    def load_mca_state_dict_from_hf(self, model_path: str, vp_stage: int):
        state_dict_iter = self.hf_state_dict_iter(model_path, vp_stage=vp_stage)
        mca_state_dict = self.get_mca_state_dict(state_dict_iter, vp_stage=vp_stage)
        return mca_state_dict

    def get_needed_hf_files(self, path, vp_stage: int):
        files = []
        metadata = None
        hf_weight_index_path = os.path.join(path, SAFE_WEIGHTS_INDEX_NAME)
        if os.path.exists(hf_weight_index_path) and len(files) == 0:
            files, metadata = get_checkpoint_shard_files(path, hf_weight_index_path)

        pt_weight_index_path = os.path.join(path, WEIGHTS_INDEX_NAME)
        if os.path.exists(pt_weight_index_path) and len(files) == 0:
            files, metadata = get_checkpoint_shard_files(path, pt_weight_index_path)

        hf_weight_path = os.path.join(path, SAFE_WEIGHTS_NAME)
        if os.path.exists(hf_weight_path) and len(files) == 0:
            files.append(hf_weight_path)

        pt_weight_path = os.path.join(path, WEIGHTS_NAME)
        if os.path.exists(pt_weight_path) and len(files) == 0:
            files.append(pt_weight_path)

        if metadata is None:
            return set(files)

        needed_files = set()
        for weight_name in metadata["all_checkpoint_keys"]:
            if self.is_needed_hf_name(weight_name, vp_stage=vp_stage):
                needed_files.add(metadata["weight_map"][weight_name])
        return {os.path.join(path, file) for file in needed_files}

    def is_needed_hf_name(self, name, vp_stage: int):
        mca_names = self.template.hf_name_to_mca_names(name)
        if mca_names is None:
            return False
        return any(self.dist_converter.is_on_this_rank(name, vp_stage=vp_stage) for name in mca_names)

    def hf_state_dict_iter(self, path, vp_stage: int):
        files = self.get_needed_hf_files(path, vp_stage=vp_stage)
        for file in files:
            state_dict = load_state_dict(file)
            for k, v in state_dict.items():
                if not self.is_needed_hf_name(k, vp_stage=vp_stage):
                    continue
                yield k, v

    def get_mca_state_dict(self, state_dict_iter, vp_stage: int):
        mca_state_dict = {}

        for name, weight in state_dict_iter:
            converted_state_dict = self.template.add_hf_weight(name, weight)
            if converted_state_dict is not None:
                for mca_name, mca_weight in converted_state_dict.items():
                    # resize before tensor parallel conversion
                    if self.resized_vocab_size and (
                        (mca_name == MCORE_WORD_EMBEDDING) or 
                        (mca_name == MCORE_LM_HEAD and not self.mca_config.tie_embeddings_and_output_weights)
                    ):
                        mca_weight = resize_embedding_layer(mca_weight, self.resized_vocab_size)
                    named_weights = self.dist_converter.dist_convert(mca_name, mca_weight, vp_stage=vp_stage)
                    if named_weights is not None:
                        mca_state_dict.update(named_weights)
                        self.log(f"hf_name: {name} -> mca_name: {list(named_weights.keys())}")
                    else:
                        self.log(
                            f"hf_name: {name} not on this rank: pp {self.dist_converter.pipeline_model_parallel_rank}"
                            f" ep: {self.dist_converter.expert_model_parallel_rank} or not ready to convert"
                        )
            else:
                self.log(f"hf_name: {name} added but not converted")
        self.template.release()
        return mca_state_dict

    def _mca_named_params_with_vp_stage(self, models):
        for vp_stage, model in enumerate(models):
            if is_peft_available() and isinstance(model, PeftModel):
                for adapter_name in model.peft_config.keys():
                    mca_state_dict = get_peft_model_state_dict(model, model.state_dict_for_save_checkpoint(), adapter_name)
                    mca_state_dict = {k: v for k, v in mca_state_dict.items() if not k.endswith("._extra_state")}
                    for mca_name, weight in sorted(mca_state_dict.items()):
                        yield adapter_name, vp_stage, mca_name, weight
            else:
                mca_state_dict = model.state_dict_for_save_checkpoint()
                mca_state_dict = {k: v for k, v in mca_state_dict.items() if not k.endswith("._extra_state")}
                for mca_name, weight in sorted(mca_state_dict.items()):
                    yield None, vp_stage, mca_name, weight

    def convert_to_hf(
        self,
        mca_state_dict: Dict[str, list["Tensor"]],
        vp_stage: Optional[int] = None,
        layer_index_preprocessed: bool = False,
        moe_index_preprocessed: bool = False,
        **kwargs,
    ) -> Dict[str, "Tensor"]:
        """
        Convert Mca state dict to HuggingFace format.

        Args:
            mca_state_dict: Dictionary of mca weight names to tensor lists
            vp_stage: Virtual pipeline stage
            layer_index_preprocessed: If True, the weight names' layer indices have already been
                preprocessed for pipeline parallelism by the caller. If False (default),
                DistConverter will handle the layer index conversion between global and local indices.
            moe_index_preprocessed: If True, the weight names' moe indices have already been
                preprocessed for expert parallelism by the caller. If False (default),
                DistConverter will handle the moe index conversion between global and local indices.
        """
        if vp_stage is None:
            vp_stage = mpu.get_virtual_pipeline_model_parallel_rank()

        hf_state_dict = {}
        for mca_name, weights in mca_state_dict.items():
            merged_named_weights = self.dist_converter.dist_convert(
                mca_name,
                weights,
                vp_stage=vp_stage,
                layer_index_preprocessed=layer_index_preprocessed,
                moe_index_preprocessed=moe_index_preprocessed,
            )
            if merged_named_weights is None:
                continue
            converted = {}
            for merged_name, merged_weight in merged_named_weights.items():
                converted_state_dict = self.template.add_mca_weight(merged_name, merged_weight, **kwargs)
                if converted_state_dict is not None:
                    converted.update(converted_state_dict)
                else:
                    self.log(f"mca_name: {merged_name} added but not converted")
            hf_state_dict.update(converted or {})
        return hf_state_dict

    def save_model_as_hf_inflight(
        self,
        models,
        save_directory: str,
        save_safetensors: bool = True,
        max_shard_size: Union[int, str] = MAX_SHARD_SIZE,
        move_to_cpu: bool = False,
    ):
        assert self.dist_converter.revert, "save_model_as_hf_inflight only support to_hf ModelConverter"
        if not mpu.model_parallel_is_initialized():
            raise RuntimeError("Model parallelism must be initialized before save as hf inflight.")
        if not mpu.get_expert_data_parallel_rank() == 0:
            return

        self.save_hf_config(save_directory, mca_config=models[0].config)

        shard_state = StateDictSplitState(max_shard_size=max_shard_size)
        if isinstance(max_shard_size, str):
            max_shard_size = parse_size_to_int(max_shard_size)

        expert_parallel = self.mca_config.expert_model_parallel_size > 1
        only_need_expert = expert_parallel and mpu.get_expert_model_parallel_rank() > 0
        last_adapter_name = None
        for adapter_name, vp_stage, mca_name, weight in self._mca_named_params_with_vp_stage(models):
            if only_need_expert and not self.dist_converter.is_expert_parallel_weight(mca_name):
                continue
            weights = gather_tensor_parallel(weight, async_op=False)
            if weights is None:  # only tp_rank0 need to convert and save
                continue
            if move_to_cpu and isinstance(weights, list):
                weights = [w.cpu() for w in weights]
            converted_state_dict = self.convert_to_hf(mca_state_dict={mca_name: weights}, vp_stage=vp_stage)
            self.save_hf_shard_state_dict(
                shard_state,
                os.path.join(save_directory, adapter_name) if adapter_name is not None else save_directory,
                converted_state_dict,
                save_safetensors,
            )

            if (
                adapter_name is not None
                and adapter_name != last_adapter_name
                and mpu.get_tensor_model_parallel_rank() == 0
            ):
                self.save_shard_state_meta(shard_state, save_directory, save_safetensors)

            if adapter_name is not None:
                last_adapter_name = adapter_name

        if mpu.get_tensor_model_parallel_rank() == 0:
            self.save_shard_state_meta(shard_state, save_directory, save_safetensors)

    def _auto_bucket_size(self):
        # TODO: optimize this by max weight size
        group_size = max(
            self.mca_config.expert_model_parallel_size,
            self.mca_config.pipeline_model_parallel_size,
            self.mca_config.tensor_model_parallel_size,
        )
        bucket_size = max(512 * 1024 * 1024, 128 * 1024 * 1024 * group_size)
        return bucket_size

    def save_hf_config(self, save_directory: str, mca_config: Optional["McaModelConfig"] = None):
        if not dist.get_rank() == 0:
            return
        mca_config = mca_config or self.mca_config
        hf_config = self.template.convert_mca_to_hf_config(mca_config)
        hf_config.save_pretrained(save_directory)
        mca_config.save_hf_auto_map_files(save_directory)

    def save_hf_shard_state_dict(
        self,
        shard_state: StateDictSplitState,
        save_directory: str,
        hf_state_dict: Optional[Dict[str, "Tensor"]] = None,
        save_safetensors: bool = True,
    ):
        for name, weight in hf_state_dict.items():
            weight_size = get_tensor_size(weight)
            if weight_size + shard_state.current_shard_size > shard_state.max_shard_size:
                self._save_shard_state_dict(shard_state, save_directory, save_safetensors)
            shard_state.current_shard_size += weight_size
            shard_state.current_shard[name] = weight

        if shard_state.current_shard_size >= shard_state.max_shard_size:
            self._save_shard_state_dict(shard_state, save_directory, save_safetensors)

    def save_shard_state_meta(
        self, shard_state: StateDictSplitState, save_directory: str, save_safetensors: bool = True
    ):
        # make sure all weights are saved
        self._save_shard_state_dict(shard_state, save_directory, save_safetensors)

        # 1. collect the expert parallel state
        if self.mca_config.expert_model_parallel_size > 1:
            shard_state = self._merge_shard_state(shard_state, group=mpu.get_expert_model_parallel_group())
        # 2. collect the pipeline parallel state
        if self.mca_config.pipeline_model_parallel_size > 1:
            shard_state = self._merge_shard_state(shard_state, group=mpu.get_pipeline_model_parallel_group())
        # 3. save only on rank0
        if dist.get_rank() == 0:
            index = {
                "metadata": {"total_size": shard_state.total_size},
                "weight_map": shard_state.tensor_to_filename,
            }
            save_index_file = SAFE_WEIGHTS_INDEX_NAME if save_safetensors else WEIGHTS_INDEX_NAME
            save_index_file = os.path.join(save_directory, save_index_file)
            with open(save_index_file, "w", encoding="utf-8") as f:
                content = json.dumps(index, indent=2, sort_keys=True) + "\n"
                f.write(content)
            logger.info(
                f"The model has been saved in {len(shard_state.filename_to_tensors)} checkpoint shards. "
                f"You can find where each parameters has been saved in the index located at {save_index_file}."
            )

    def _merge_shard_state(self, shard_state: StateDictSplitState, group):
        gathered_shard_states = allgather_parallel_objs(shard_state, group=group)
        return StateDictSplitState.merge_states(gathered_shard_states)

    def _save_shard_state_dict(
        self, shard_state: StateDictSplitState, save_directory: str, save_safetensors: bool = True
    ):
        if len(shard_state.current_shard) == 0:
            return
        rank = dist.get_rank()
        file_index = len(shard_state.filename_to_tensors)
        weights_name = SAFE_WEIGHTS_NAME if save_safetensors else WEIGHTS_NAME
        # TODO: this file name format is not same as hf, but it works for now
        suffix = f"{rank}_{file_index}"
        shard_file_name = weights_name.replace(".bin", f"{suffix}.bin").replace(
            ".safetensors", f"{suffix}.safetensors"
        )
        shard_state.filename_to_tensors[shard_file_name] = list(shard_state.current_shard.keys())
        shard_state.tensor_to_filename.update({k: shard_file_name for k in shard_state.current_shard.keys()})

        save_file_name = os.path.join(save_directory, shard_file_name)
        if save_safetensors:
            safe_save_file(shard_state.current_shard, save_file_name, metadata={"format": "pt"})
        else:
            torch.save(shard_state.current_shard, save_file_name)

        shard_state.current_shard = {}
        shard_state.current_shard_size = 0
        gc.collect()
