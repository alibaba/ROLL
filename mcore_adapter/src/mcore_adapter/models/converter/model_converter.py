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
)
from .dist_converter import DistConverter
from .template import get_template


if is_safetensors_available():
    from safetensors.torch import save_file as safe_save_file


if TYPE_CHECKING:
    from torch import Tensor

    from ..model_config import McaModelConfig

logger = get_logger(__name__)


class ModelConverter:
    def __init__(self, mca_config: "McaModelConfig", model_name_or_path: str = None, verbose=False):
        self.model_name_or_path = model_name_or_path
        self.mca_config = mca_config
        self.verbose = verbose
        self.template = get_template(mca_config.hf_model_type)
        self.template.set_mca_config_for_ops(self.mca_config)

    def log(self, msg):
        if self.verbose:
            logger.info(msg)

    def load_mca_state_dict_from_hf(
        self,
        tensor_model_parallel_rank: Optional[int] = None,
        pipeline_model_parallel_rank: Optional[int] = None,
        expert_model_parallel_rank: Optional[int] = None,
        virtual_pipeline_model_parallel_rank: Optional[int] = None,
    ):
        tensor_model_parallel_rank = tensor_model_parallel_rank or mpu.get_tensor_model_parallel_rank()
        pipeline_model_parallel_rank = pipeline_model_parallel_rank or mpu.get_pipeline_model_parallel_rank()
        expert_model_parallel_rank = expert_model_parallel_rank or mpu.get_expert_model_parallel_rank()
        virtual_pipeline_model_parallel_rank = (
            virtual_pipeline_model_parallel_rank or mpu.get_virtual_pipeline_model_parallel_rank()
        )
        dist_converter = DistConverter(
            self.mca_config,
            tensor_model_parallel_rank=tensor_model_parallel_rank,
            pipeline_model_parallel_rank=pipeline_model_parallel_rank,
            expert_model_parallel_rank=expert_model_parallel_rank,
            virtual_pipeline_model_parallel_rank=virtual_pipeline_model_parallel_rank,
            revert=False,
        )
        state_dict_iter = self.hf_state_dict_iter(self.model_name_or_path, dist_converter)
        mca_state_dict = self.get_mca_state_dict(dist_converter, state_dict_iter)
        return mca_state_dict

    def get_needed_hf_files(self, path, dist_converter: "DistConverter"):
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
            if self.is_needed_hf_name(weight_name, dist_converter):
                needed_files.add(metadata["weight_map"][weight_name])
        return {os.path.join(path, file) for file in needed_files}

    def is_needed_hf_name(self, name, dist_converter: "DistConverter"):
        mca_names = self.template.hf_name_to_mca_names(name)
        if mca_names is None:
            return False
        return any(dist_converter.is_on_this_rank(name) for name in mca_names)

    def hf_state_dict_iter(self, path, dist_converter: "DistConverter"):
        files = self.get_needed_hf_files(path, dist_converter)
        for file in files:
            state_dict = load_state_dict(file)
            for k, v in state_dict.items():
                if not self.is_needed_hf_name(k, dist_converter):
                    continue
                yield k, v

    def get_mca_state_dict(self, dist_converter: "DistConverter", state_dict_iter):
        mca_state_dict = {}

        for name, weight in state_dict_iter:
            converted_state_dict = self.template.add_hf_weight(name, weight)
            if converted_state_dict is not None:
                for mca_name, mca_weight in converted_state_dict.items():
                    named_weights = dist_converter(mca_name, mca_weight)
                    if self.mca_config.mtp_num_layers:
                        named_weights = self.template.convert_mtp_weights(named_weights)
                    if named_weights is not None:
                        mca_state_dict.update(named_weights)
                        self.log(f"hf_name: {name} -> mca_name: {list(named_weights.keys())}")
                    else:
                        self.log(
                            f"hf_name: {name} not on this rank: pp {dist_converter.pipeline_model_parallel_rank}"
                            f" ep: {dist_converter.expert_model_parallel_rank} or not ready to convert"
                        )
            else:
                self.log(f"hf_name: {name} added but not converted")
        self.template.release()
        return mca_state_dict

    def _mca_named_params_with_reverter(self, models):
        expert_parallel = self.mca_config.expert_model_parallel_size > 1
        for vp, model in enumerate(models):
            dist_reverter = DistConverter(
                mca_config=self.mca_config,
                tensor_model_parallel_rank=mpu.get_tensor_model_parallel_rank(),
                pipeline_model_parallel_rank=mpu.get_pipeline_model_parallel_rank(),
                expert_model_parallel_rank=mpu.get_expert_model_parallel_rank() if expert_parallel else 0,
                virtual_pipeline_model_parallel_rank=vp,
                revert=True,
            )
            mca_state_dict = model.state_dict_for_save_checkpoint()
            mca_state_dict = {k: v for k, v in mca_state_dict.items() if not k.endswith("._extra_state")}
            if self.mca_config.mtp_num_layers:
                mca_state_dict = self.template.revert_mtp_weights(mca_state_dict)
            for mca_name, weight in sorted(mca_state_dict.items()):
                yield dist_reverter, mca_name, weight

    def save_model_as_hf_inflight(
        self,
        models,
        save_directory: str,
        save_safetensors: bool = True,
        max_shard_size: Union[int, str] = MAX_SHARD_SIZE,
    ):
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
        for dist_reverter, mca_name, weight in self._mca_named_params_with_reverter(models):
            if only_need_expert and dist_reverter.get_local_moe_index(mca_name) is None:
                continue
            weights = gather_tensor_parallel(weight, async_op=False)
            if weights is None:  # only tp_rank0 need to convert and save
                continue
            mca_named_weights = dist_reverter(mca_name, weights)
            if mca_named_weights is None:
                continue

            converted_state_dict = {}
            for mca_name, mca_weight in mca_named_weights.items():
                converted = self.template.add_mca_weight(mca_name, mca_weight)
                assert len(set(converted_state_dict.keys()).intersection(converted.keys())) == 0, (
                    f"converted_state_dict: {converted_state_dict.keys()} converted: {converted.keys()}"
                )
                if converted:
                    converted_state_dict.update(converted)
            self.save_hf_shard_state_dict(shard_state, save_directory, converted_state_dict, save_safetensors)

        if mpu.get_tensor_model_parallel_rank() == 0:
            self.save_shard_state_meta(shard_state, save_directory, save_safetensors)

    def all_gather_weights_as_hf_inflight(self, models):
        expert_parallel = self.mca_config.expert_model_parallel_size > 1
        for dist_reverter, mca_name, weight in self._mca_named_params_with_reverter(models):
            moe_index = dist_reverter.get_local_moe_index(mca_name)
            group = (
                mpu.get_tensor_model_parallel_group() if moe_index is None else mpu.get_expert_tensor_parallel_group()
            )
            if dist.get_world_size(group) == 1:
                weights = [weight]
            else:
                weights = all_gather_tensors(weight, async_op=False, group=group)
            mca_named_weights = dist_reverter(mca_name, weights)
            if mca_named_weights is None:
                continue
            for mca_name, mca_weight in mca_named_weights.items():
                converted = self.template.add_mca_weight(mca_name, mca_weight)
                converted = converted or {}
                for name, weight in converted.items():
                    if expert_parallel and moe_index is not None:
                        names = allgather_parallel_objs(name, group=mpu.get_expert_model_parallel_group())
                        weights = all_gather_tensors(
                            weight, async_op=False, group=mpu.get_expert_model_parallel_group()
                        )
                        for name, weight in zip(names, weights):
                            yield name, weight
                    else:
                        yield name, weight

    def all_gather_weights_as_hf_bucket(self, models, bucket_size: int = None):
        bucket_manager = SendBucketManager(bucket_size or self._auto_bucket_size())
        for name, weight in self.all_gather_weights_as_hf_inflight(models):
            yield from bucket_manager.push_tensor(weight, name=name)
        last_meta, last_buffer = bucket_manager.pop_last_bucket()
        if last_meta is not None:
            yield last_meta, last_buffer

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
