import gc
import json
import os
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING, Optional

import torch
from megatron.core import dist_checkpointing, mpu 
from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed
from safetensors.torch import save_file
from tqdm import tqdm
from transformers import (
    AutoConfig as HfAutoConfig,
)
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForTextToWaveform,
    AutoProcessor,
    AutoTokenizer,
)
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.models.auto.auto_factory import _get_model_class
from transformers.utils import is_peft_available

from ...checkpointing import find_dist_ckpt, get_checkpoint_name, save_config_and_state_dict
from ...constants import ADAPTER_CONFIG_NAME
from ...training_args import DistributingParallelArguments
from ...utils import get_logger
from ..auto.config_auto import AutoConfig
from ..auto.modeling_auto import get_model_cls
from .convert_utils import MAX_SHARD_SIZE
from .model_converter import ModelConverter
from .template import get_template


if is_peft_available():
    from peft import LoraConfig, PeftConfig, get_peft_model, set_peft_model_state_dict

if TYPE_CHECKING:
    from transformers import PretrainedConfig
    from ...training_args import DistributingParallelArguments
    from ..model_config import McaModelConfig
    from .template import Template

logger = get_logger(__name__)


class BaseHFConverter(ABC):
    """
    Abstract base class for converting Mca checkpoints to Hugging Face format.
    Encapsulates common logic for loading configs, streaming weights, and saving artifacts.
    """

    def __init__(
        self,
        checkpoint_path: str,
        save_directory: str,
        torch_dtype: Optional[torch.dtype],
        verbose: bool,
    ):
        self.checkpoint_path = checkpoint_path
        self.save_directory = save_directory
        self.verbose = verbose

        self.mca_config: "McaModelConfig"
        self.hf_config: "PretrainedConfig"
        self.template: "Template"
        self._setup()

        self.torch_dtype = torch_dtype if torch_dtype is not None else self.mca_config.params_dtype

    def _setup(self):
        """Loads Mca config, converts it to HF config"""
        # load mca_config
        self.mca_config = AutoConfig.from_pretrained(self.checkpoint_path)
        if self.mca_config is None:
            raise ValueError("No mca config found in checkpoint")
        if self.mca_config.hf_model_type is None:
            raise ValueError("No hf model type found in mca config")

        self.template = get_template(self.mca_config.hf_model_type)
        self.hf_config = self.template.convert_mca_to_hf_config(self.mca_config)
        self.template.set_mca_config_for_ops(self.mca_config)

        mpu.set_expert_model_parallel_world_size(self.mca_config.expert_model_parallel_size)
        mpu.set_pipeline_model_parallel_world_size(self.mca_config.pipeline_model_parallel_size)
        mpu.set_tensor_model_parallel_world_size(self.mca_config.tensor_model_parallel_size)
        if self.mca_config.virtual_pipeline_model_parallel_size is not None:
            mpu.set_virtual_pipeline_model_parallel_world_size(self.mca_config.virtual_pipeline_model_parallel_size)

    def _stream_hf_weights(self, checkpoint_path: str, use_mmap: bool = False, **kwargs):
        """A generator that loads, converts, and yields HF weights from a given checkpoint path."""

        def log(msg):
            if self.verbose:
                logger.info(msg)

        for pp_rank, ep_rank in product(
            range(self.mca_config.pipeline_model_parallel_size), range(self.mca_config.expert_model_parallel_size)
        ):
            state_dicts = [
                torch.load(
                    get_checkpoint_name(
                        checkpoint_path,
                        tensor_rank=tp_rank,
                        pipeline_rank=pp_rank,
                        pipeline_parallel=self.mca_config.pipeline_model_parallel_size > 1,
                        expert_rank=ep_rank,
                        expert_parallel=self.mca_config.expert_model_parallel_size > 1,
                    ),
                    map_location="cpu",
                    mmap=use_mmap,
                )
                for tp_rank in range(self.mca_config.tensor_model_parallel_size)
            ]

            mpu.set_pipeline_model_parallel_rank(pp_rank)
            mpu.set_expert_model_parallel_rank(ep_rank)
            mpu.set_tensor_model_parallel_rank(0)
            converter = ModelConverter(
                mca_config=self.mca_config,
                pipeline_model_parallel_rank=pp_rank,
                expert_model_parallel_rank=ep_rank,
                tensor_model_parallel_rank=0,
                verbose=self.verbose,
                to_hf=True,
            )

            vp_on = (self.mca_config.virtual_pipeline_model_parallel_size or 1) > 1
            for i in range(self.mca_config.virtual_pipeline_model_parallel_size or 1):
                if vp_on:
                    mpu.set_virtual_pipeline_model_parallel_rank(i)

                v_state_dicts = [sd.pop(f"model{i}" if vp_on else "model") for sd in state_dicts]
                for name in list(v_state_dicts[0].keys()):
                    if name.endswith("._extra_state"):
                        continue
                    weights = [sd.get(name) for sd in v_state_dicts]
                    converted = converter.convert_to_hf({name: weights}, vp_stage=i, **kwargs)
                    if converted:
                        for hf_name, hf_weight in converted.items():
                            # log(f"Converted and yielded: {name} -> {hf_name}")
                            yield hf_name, hf_weight

    def _finalize(self):
        """Saves configs, tokenizer, processor, and releases resources."""
        os.makedirs(self.save_directory, exist_ok=True)
        self.hf_config.save_pretrained(self.save_directory)
        self.mca_config.save_hf_auto_map_files(self.save_directory)

        tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_path, trust_remote_code=True)
        try:
            processor = AutoProcessor.from_pretrained(self.checkpoint_path, trust_remote_code=True)
        except Exception as e:
            if self.verbose:
                logger.info(f"Processor was not found: {e}.")
            processor = tokenizer

        if processor is not None and "Processor" not in processor.__class__.__name__:
            processor = None

        if processor is not None:
            setattr(processor, "tokenizer", tokenizer)
        else:
            processor = tokenizer
        processor.save_pretrained(self.save_directory)

        logger.info(f"Model successfully converted and saved to {self.save_directory}")

    @abstractmethod
    def convert(self):
        """The main conversion method to be implemented by subclasses."""
        raise NotImplementedError


class HFConverter(BaseHFConverter):
    """Converts the model by loading all weights into memory (standard method)."""

    def convert(self):
        logger.info("Starting in-memory conversion...")
        hf_state_dict = {}
        for hf_name, hf_weight in tqdm(self._stream_hf_weights(self.checkpoint_path), desc="Converting mca to hf"):
            if hf_name in hf_state_dict:
                if not hf_weight.equal(hf_state_dict[hf_name]):
                    raise ValueError(
                        f"weight of hf_name:{hf_name} in "
                        f"diff max:{torch.abs(hf_weight - hf_state_dict[hf_name]).max()}, please check the checkpoint"
                    )
            hf_state_dict[hf_name] = hf_weight

        model_class = self._get_hf_model_class()
        model = model_class.from_pretrained(
            None, config=self.hf_config, state_dict=hf_state_dict, torch_dtype=self.torch_dtype, trust_remote_code=True
        )
        model.save_pretrained(self.save_directory, max_shard_size=MAX_SHARD_SIZE, save_original_format=False)
        self._finalize()

    def _get_hf_model_class(self):
        has_remote_code = hasattr(self.hf_config, "auto_map") and "AutoModelForCausalLM" in self.hf_config.auto_map
        model_class = AutoModelForCausalLM

        if type(self.hf_config) in AutoModelForImageTextToText._model_mapping:
            model_class = AutoModelForImageTextToText
        elif type(self.hf_config) in AutoModelForTextToWaveform._model_mapping:
            model_class = AutoModelForTextToWaveform

        if has_remote_code:
            class_ref = self.hf_config.auto_map["AutoModelForCausalLM"]
            model_class = get_class_from_dynamic_module(class_ref, self.mca_config.name_or_path)
        else:
            model_class = _get_model_class(self.hf_config, model_class._model_mapping)

        return model_class


class StreamingHFConverter(BaseHFConverter):
    """Converts the model using a streaming, low-memory approach."""

    def __init__(self, *args, max_shard_bytes: int = MAX_SHARD_SIZE, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_shard_bytes = max_shard_bytes

    def convert(self):
        logger.info(f"Starting streaming conversion (max shard size: {self.max_shard_bytes / 1e9:.2f}GB)...")
        os.makedirs(self.save_directory, exist_ok=True)
        weight_map, total_size, shard_count, current_shard, current_shard_size = {}, 0, 0, {}, 0
        for hf_name, hf_weight in tqdm(
            self._stream_hf_weights(self.checkpoint_path, use_mmap=True), desc="Converting mca to hf"
        ):
            # the hf_name may be replicated
            if hf_name in weight_map:
                # check the weight in the current chard
                if hf_name in current_shard and not hf_weight.equal(current_shard[hf_name]):
                    raise ValueError(
                        f"weight of hf_name:{hf_name} in "
                        f"diff max:{torch.abs(hf_weight - current_shard[hf_name]).max()}, please check the checkpoint"
                    )
                continue
            current_shard[hf_name] = hf_weight
            current_shard_size += hf_weight.nelement() * hf_weight.element_size()

            if current_shard_size >= self.max_shard_bytes:
                shard_name = f"model-{shard_count + 1:05d}.safetensors"
                save_file(current_shard, os.path.join(self.save_directory, shard_name), metadata={"format": "pt"})
                for k in current_shard:
                    weight_map[k] = shard_name

                total_size += current_shard_size
                shard_count += 1
                current_shard = {}
                current_shard_size = 0
                gc.collect()

        if current_shard:
            shard_name = f"model-{shard_count + 1:05d}.safetensors"
            save_file(current_shard, os.path.join(self.save_directory, shard_name), metadata={"format": "pt"})
            for k in current_shard:
                weight_map[k] = shard_name
            total_size += current_shard_size

        with open(os.path.join(self.save_directory, "model.safetensors.index.json"), "w") as f:
            json.dump({"metadata": {"total_size": total_size}, "weight_map": weight_map}, f, indent=2)

        self._finalize()


class LoRAHFConverter(HFConverter):
    """Converts Mca LoRA adapters to Hugging Face PEFT format."""

    def __init__(self, hf_base_model_path: str, adapter_name_or_path: str, *args, **kwargs):
        self.hf_base_model_path = hf_base_model_path
        self.adapter_name_or_path = adapter_name_or_path
        super().__init__(adapter_name_or_path, *args, **kwargs)

    def convert(self):
        if not is_peft_available():
            raise ImportError("PEFT is not installed. Run `pip install peft` to convert LoRA adapters.")

        adapter_names = (
            [
                folder_name
                for folder_name in os.listdir(self.adapter_name_or_path)
                if os.path.isdir(os.path.join(self.adapter_name_or_path, folder_name))
                and os.path.isfile(os.path.join(self.adapter_name_or_path, folder_name, ADAPTER_CONFIG_NAME))
            ]
            if os.path.isdir(self.adapter_name_or_path)
            else []
        )

        if not adapter_names:
            raise ValueError(f"No LoRA adapters found in '{self.adapter_name_or_path}'")

        peft_configs = {
            adapter_name: PeftConfig.from_pretrained(os.path.join(self.adapter_name_or_path, adapter_name))
            for adapter_name in adapter_names
        }

        hf_state_dicts = defaultdict(dict)
        for adapter_name, peft_config in peft_configs.items():
            logger.info(f"Converting adapter: {adapter_name}")

            stream = self._stream_hf_weights(
                checkpoint_path=os.path.join(self.adapter_name_or_path, adapter_name), lora_rank=peft_config.r
            )
            for hf_name, hf_weight in stream:
                if self.hf_config.model_type in ["qwen3_5_moe"]:
                    if "mlp.experts.gate_up_proj" in hf_name or "mlp.experts.down_proj" in hf_name:
                        if "lora_A" in hf_name:
                            # lora_A = nn.Linear(in_features, r * num_experts), shape = (r * num_experts, in_features)
                            hf_weight = hf_weight.transpose(0, 1).reshape(hf_weight.shape[0], -1).contiguous()
                        if "lora_B" in hf_name:
                            # lora_B = nn.Linear(r * num_experts, out_features), shape = (out_features, r * num_experts)
                            hf_weight = hf_weight.permute(1, 2, 0).reshape(-1, hf_weight.shape[-1]).contiguous()
                    if "mlp.experts.gate_up_proj" in hf_name:
                        hf_name = hf_name.replace(".gate_up_proj", ".base_layer")
                    if "mlp.experts.down_proj" in hf_name:
                        hf_name = hf_name.replace(".down_proj", "")
                hf_state_dicts[adapter_name][hf_name] = hf_weight

        model_class = self._get_hf_model_class()
        model = model_class.from_pretrained(
            self.hf_base_model_path, config=self.hf_config, torch_dtype=self.torch_dtype, trust_remote_code=True
        )

        def get_lora_config(adapter_name):
            peft_cfg = peft_configs[adapter_name]

            target_modules = [
                name[: name.find(".lora")].split(".")[-1]
                for name in hf_state_dicts[adapter_name].keys()
                if ".lora_A." in name or ".lora_B." in name
            ]
            target_modules = list(set(target_modules))

            kwargs = {}
            if self.hf_config.model_type in ["qwen3_5_moe"]:
                kwargs["target_parameters"] = ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]
            if self.mca_config.num_moe_experts is not None:
                kwargs["rank_pattern"] = {
                    p: peft_cfg.r // self.mca_config.moe_router_topk
                    for p in [
                        "down_proj",
                        "up_proj",
                        "gate_proj",
                        "w1",
                        "w2",
                        "w3",
                        "mlp.experts.gate_up_proj",
                        "mlp.experts.down_proj",
                    ]
                }

            return LoraConfig(
                r=peft_cfg.r,
                target_modules=target_modules,
                lora_alpha=peft_cfg.lora_alpha,
                lora_dropout=peft_cfg.lora_dropout,
                use_rslora=peft_cfg.use_rslora,
                modules_to_save=peft_cfg.modules_to_save,
                **kwargs,
            )

        adapter0_name = "default" if "default" in hf_state_dicts else sorted(hf_state_dicts.keys())[0]
        model = get_peft_model(model, get_lora_config(adapter0_name), adapter_name=adapter0_name)
        set_peft_model_state_dict(model.base_model.model, hf_state_dicts[adapter0_name], adapter_name=adapter0_name)

        for adapter_name, state_dict in hf_state_dicts.items():
            if adapter_name == adapter0_name:
                continue
            model.add_adapter(adapter_name, get_lora_config(adapter_name))
            set_peft_model_state_dict(model.base_model.model, state_dict, adapter_name=adapter_name)

        model.save_pretrained(self.save_directory, max_shard_size=MAX_SHARD_SIZE)
        self._finalize()


class DistHFConverter(HFConverter):
    """
    Megatron dist -> hf
    """
    def __init__(self, dist_model_dir: str, *args, **kwargs):
        self.dist_model_dir = dist_model_dir
        super().__init__(*args, **kwargs)
    
    def _setup(self):
        super()._setup()
        self._reset_parallel_config()

    def _reset_parallel_config(self):
        """Reset parallel config to 1 for dist checkpoint conversion."""
        # Set mca_config parallel sizes to 1
        self.mca_config.tensor_model_parallel_size = 1
        self.mca_config.expert_tensor_parallel_size = 1
        self.mca_config.pipeline_model_parallel_size = 1
        self.mca_config.expert_model_parallel_size = 1
        self.mca_config.virtual_pipeline_model_parallel_size = None
        self.mca_config.pipeline_model_parallel_layout = None
        # Reset mpu parallel state
        mpu.set_expert_model_parallel_world_size(1)
        mpu.set_pipeline_model_parallel_world_size(1)
        mpu.set_tensor_model_parallel_world_size(1)
        mpu.set_virtual_pipeline_model_parallel_world_size(None)

    def convert(self):
        models = self._init_model()
        sharded_state_dict = models.sharded_state_dict()

        sharded_state_dict = self._sharded_meta_to_cpu(sharded_state_dict)
        state_dict = dist_checkpointing.load(sharded_state_dict, self.dist_model_dir)
        model_converter = ModelConverter(
            mca_config=self.mca_config,
            verbose=self.verbose,
            to_hf=True,
        )

        hf_state_dict = {}
        for weight_name, weight_tensor in state_dict.items():
            if "._extra_state" in weight_name:
                continue
            converted_state_dict = model_converter.convert_to_hf(
                mca_state_dict={weight_name: [weight_tensor]},
                vp_stage=None,
            )
            hf_state_dict.update(converted_state_dict)
        
        model_class = self._get_hf_model_class()
        model = model_class.from_pretrained(
            None, config=self.hf_config, state_dict=hf_state_dict, torch_dtype=self.torch_dtype, trust_remote_code=True
        )
        model.save_pretrained(self.save_directory, max_shard_size=MAX_SHARD_SIZE, save_original_format=False)

        self._finalize()

    def _init_model(self):
        torch.distributed.init_process_group(
            backend="cpu:gloo",
            rank=0,
            world_size=1,
        )
        mpu.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=1,
            expert_tensor_parallel_size=None,
            virtual_pipeline_model_parallel_size=None,
        )

        model_parallel_cuda_manual_seed(42)
        model_class = get_model_cls(self.mca_config.hf_model_type)
        
        # Use meta device for low memory initialization
        # Skip .to(device) calls for vision/audio models
        self.mca_config.init_model_with_meta_device = True
        with torch.device("meta"):
            model = model_class(self.mca_config)
        return model

    def _sharded_meta_to_cpu(self, sharded_state_dict):
        from megatron.core.dist_checkpointing.dict_utils import dict_list_map_inplace
        from megatron.core.dist_checkpointing.mapping import ShardedTensor, ShardedTensorFactory, ShardedObject

        def meta_to_cpu_empty(v):
            if isinstance(v, ShardedTensorFactory):
                if v.data is not None and v.data.device.type == 'meta':
                    v.data = torch.empty(v.data.shape, dtype=v.data.dtype, device='cpu')
            elif isinstance(v, ShardedTensor):
                if v.data is not None and v.data.device.type == 'meta':
                    v = v.without_data()
                    v.init_data('cpu')
            elif isinstance(v, ShardedObject):
                pass
            return v
        
        dict_list_map_inplace(meta_to_cpu_empty, sharded_state_dict)
        return sharded_state_dict


class DistStreamingHFConverter(DistHFConverter):
    """
    Converts Megatron distributed checkpoint to Hugging Face format using streaming approach.
    """
    MEGATRON_LAYER_PATTERN = r'^(.*\.layers)\.(\d+)\.'

    def __init__(self, dist_model_dir: str, max_shard_bytes: int = MAX_SHARD_SIZE, **kwargs):
        super().__init__(dist_model_dir=dist_model_dir, **kwargs)
        self.max_shard_bytes = max_shard_bytes
        self.layer_prefix_cache = None

    def _get_layer_key(self, key: str) -> tuple[str, int] | None:
        """
        Extract layer prefix and layer index from a weight key.
        """
        if self.layer_prefix_cache is None:
            self.layer_prefix_cache = re.compile(self.MEGATRON_LAYER_PATTERN)
        # Match patterns like: decoder.layers.0, encoder.layers.1, etc.
        match = self.layer_prefix_cache.match(key)
        if match:
            layer_prefix = match.group(1)
            layer_idx = int(match.group(2))
            return (layer_prefix, layer_idx)
        return None

    def _split_sharded_state_dict(self, sharded_state_dict: dict) -> list[dict]:
        """
        Split sharded_state_dict into chunks by transformer layer.        
        """
        from collections import defaultdict
        
        # Group weights by layer
        layer_chunks: dict[tuple[str, int], dict] = defaultdict(dict)
        non_layer_chunk: dict = {}
        
        for key, value in sharded_state_dict.items():
            # Skip _extra_state entries
            if "._extra_state" in key:
                continue
            
            layer_key = self._get_layer_key(key)
            if layer_key is not None:
                layer_chunks[layer_key][key] = value
            else:
                non_layer_chunk[key] = value
        
        # Build final chunks list
        chunks = []
        
        # Add non-layer weights first (embeddings, final norm, etc.)
        if non_layer_chunk:
            chunks.append(non_layer_chunk)
        
        # Add each layer as a separate chunk, sorted by layer index
        for layer_key in sorted(layer_chunks.keys()):
            chunks.append(layer_chunks[layer_key])
        
        return chunks

    def _stream_dist_weights_from_shards(self, sharded_state_dict: dict):
        """
        Stream weights by processing sharded_state_dict in chunks.
        """
        import copy
        
        # Split into chunks based on estimated memory size
        chunks = self._split_sharded_state_dict(sharded_state_dict)
        logger.info(f"Split sharded_state_dict into {len(chunks)} chunks for streaming conversion")
        
        model_converter = ModelConverter(
            mca_config=self.mca_config,
            pipeline_model_parallel_rank=0,
            expert_model_parallel_rank=0,
            tensor_model_parallel_rank=0,
            verbose=self.verbose,
            to_hf=True,
        )
        
        for chunk_idx, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {chunk_idx + 1}/{len(chunks)} with {len(chunk)} weights...")
            
            chunk_copy = copy.deepcopy(chunk)            
            chunk_copy = self._sharded_meta_to_cpu(chunk_copy)
            
            loaded_chunk = dist_checkpointing.load(chunk_copy, self.dist_model_dir)
            
            # Convert each weight to HF format
            for weight_name, weight_tensor in loaded_chunk.items():
                if "._extra_state" in weight_name:
                    continue
                    
                # Convert to HF format
                converted_state_dict = model_converter.convert_to_hf(
                    mca_state_dict={weight_name: [weight_tensor]},
                    vp_stage=None,
                )
                
                for hf_name, hf_weight in converted_state_dict.items():
                    yield hf_name, hf_weight
            
            # Clean up to save memory
            del chunk_copy
            del loaded_chunk
            gc.collect()

    def convert(self):
        logger.info(f"Starting streaming conversion from dist checkpoint (max shard size: {self.max_shard_bytes / 1e9:.2f}GB)...")
        os.makedirs(self.save_directory, exist_ok=True)

        model = self._init_model()
        sharded_state_dict = model.sharded_state_dict()
        
        weight_map = {}
        total_size = 0
        shard_count = 0
        current_shard = {}
        current_shard_size = 0

        for hf_name, hf_weight in tqdm(
            self._stream_dist_weights_from_shards(sharded_state_dict),
            desc="Converting dist checkpoint to hf"
        ):
            # The hf_name may be replicated (e.g., shared embeddings)
            if hf_name in weight_map:
                # Check consistency
                if hf_name in current_shard and not hf_weight.equal(current_shard[hf_name]):
                    raise ValueError(
                        f"weight of hf_name:{hf_name} in "
                        f"diff max:{torch.abs(hf_weight - current_shard[hf_name]).max()}, "
                        "please check the checkpoint"
                    )
                continue

            current_shard[hf_name] = hf_weight
            current_shard_size += hf_weight.nelement() * hf_weight.element_size()

            # Save shard if it exceeds max size
            if current_shard_size >= self.max_shard_bytes:
                shard_name = f"model-{shard_count + 1:05d}.safetensors"
                save_file(current_shard, os.path.join(self.save_directory, shard_name), metadata={"format": "pt"})
                for k in current_shard:
                    weight_map[k] = shard_name

                total_size += current_shard_size
                shard_count += 1
                current_shard = {}
                current_shard_size = 0
                gc.collect()

        # Save the last shard
        if current_shard:
            shard_name = f"model-{shard_count + 1:05d}.safetensors"
            save_file(current_shard, os.path.join(self.save_directory, shard_name), metadata={"format": "pt"})
            for k in current_shard:
                weight_map[k] = shard_name
            total_size += current_shard_size

        # Save index file
        with open(os.path.join(self.save_directory, "model.safetensors.index.json"), "w") as f:
            json.dump({"metadata": {"total_size": total_size}, "weight_map": weight_map}, f, indent=2)

        self._finalize()


def convert_checkpoint_to_hf(
    model_name_or_path: str,
    save_directory: str,
    adapter_name_or_path: Optional[str] = None,
    torch_dtype: Optional["torch.dtype"] = None,
    low_mem: bool = True,
    verbose: bool = True,
    max_shard_bytes: int = MAX_SHARD_SIZE,
):
    """
    Converts a Mca checkpoint to Hugging Face format using the appropriate strategy.

    Args:
        model_name_or_path (str): For full model conversion, path to the Mca checkpoint.
                                  For adapter conversion, path to the base Hugging Face model.
        save_directory (str): Directory to save the converted HF model/adapter.
        adapter_name_or_path (Optional[str]): Path to the Mca LoRA adapter checkpoint directory.
        torch_dtype (Optional[torch.dtype]): The torch dtype for the converted model.
        low_mem (bool): If True, use streaming conversion to save memory (not for adapters).
        verbose (bool): Whether to print detailed conversion logs.
        max_shard_bytes (int): Max size of each shard in bytes for low_mem mode.
    """
    # for dist ckpt
    dist_model_dir = find_dist_ckpt(model_name_or_path)
    if dist_model_dir:
        logger.info(f"The model_path: {dist_model_dir} is dist ckpt...")
        if low_mem:
            converter = DistStreamingHFConverter(
                dist_model_dir=dist_model_dir,
                checkpoint_path=model_name_or_path,
                save_directory=save_directory,
                torch_dtype=torch_dtype,
                verbose=verbose,
            )
        else:
            converter = DistHFConverter(
                dist_model_dir=dist_model_dir,
                checkpoint_path=model_name_or_path,
                save_directory=save_directory,
                torch_dtype=torch_dtype,
                verbose=verbose,
            )
        converter.convert()
        return

    # for legacy ckpt
    if adapter_name_or_path:
        if low_mem:
            raise ValueError("There is no need using `low_mem` mode for lora convert.")
        converter = LoRAHFConverter(
            hf_base_model_path=model_name_or_path,
            adapter_name_or_path=adapter_name_or_path,
            save_directory=save_directory,
            torch_dtype=torch_dtype,
            verbose=verbose,
        )
    elif low_mem:
        converter = StreamingHFConverter(
            checkpoint_path=model_name_or_path,
            save_directory=save_directory,
            torch_dtype=torch_dtype,
            verbose=verbose,
            max_shard_bytes=max_shard_bytes,
        )
    else:
        converter = HFConverter(
            checkpoint_path=model_name_or_path,
            save_directory=save_directory,
            torch_dtype=torch_dtype,
            verbose=verbose,
        )

    converter.convert()


def convert_checkpoint_to_mca(
    model_name_or_path: str,
    save_directory: str,
    dist_args: "DistributingParallelArguments",
    bf16: bool = False,
    fp16: bool = False,
    verbose: bool = True,
):
    dist_args.pipeline_model_parallel_size = dist_args.pipeline_model_parallel_size or 1
    dist_args.tensor_model_parallel_size = dist_args.tensor_model_parallel_size or 1
    dist_args.expert_model_parallel_size = dist_args.expert_model_parallel_size or 1
    hf_config = HfAutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    template: "Template" = get_template(hf_config.model_type)
    mca_config = template.convert_hf_to_mca_config(hf_config, bf16=bf16, fp16=fp16, **dist_args.get_config_dict())
    template.set_mca_config_for_ops(mca_config)
    mpu.set_tensor_model_parallel_world_size(dist_args.tensor_model_parallel_size)
    mpu.set_pipeline_model_parallel_world_size(dist_args.pipeline_model_parallel_size)
    mpu.set_expert_model_parallel_world_size(dist_args.expert_model_parallel_size)
    if dist_args.virtual_pipeline_model_parallel_size is not None:
        mpu.set_virtual_pipeline_model_parallel_world_size(dist_args.virtual_pipeline_model_parallel_size)

    for tp_rank, pp_rank, ep_rank in tqdm(
        product(
            range(dist_args.tensor_model_parallel_size),
            range(dist_args.pipeline_model_parallel_size),
            range(dist_args.expert_model_parallel_size),
        ),
        total=(
            dist_args.tensor_model_parallel_size
            * dist_args.pipeline_model_parallel_size
            * dist_args.expert_model_parallel_size
        ),
        desc="Converting",
        disable=not verbose,
    ):
        mpu.set_tensor_model_parallel_rank(tp_rank)
        mpu.set_pipeline_model_parallel_rank(pp_rank)
        mpu.set_expert_model_parallel_rank(ep_rank)
        model_parallel_cuda_manual_seed(42)
        model_converter = ModelConverter(
            mca_config=mca_config,
            verbose=verbose,
            tensor_model_parallel_rank=tp_rank,
            pipeline_model_parallel_rank=pp_rank,
            expert_model_parallel_rank=ep_rank,
        )

        mca_state_dict = {}
        for i in range(mca_config.virtual_pipeline_model_parallel_size or 1):
            key = "model"
            if dist_args.virtual_pipeline_model_parallel_size is not None:
                key = f"model{i}"
                mpu.set_virtual_pipeline_model_parallel_rank(i)
            mca_state_dict[key] = model_converter.load_mca_state_dict_from_hf(model_name_or_path, vp_stage=i)
        if verbose:
            logger.info(f"Saving ({tp_rank=} {pp_rank=} {ep_rank=}) model to {save_directory}")
        save_config_and_state_dict(save_directory, mca_config, mca_state_dict)
        template.release()

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    try:
        processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
    except Exception as e:
        if verbose:
            logger.info(f"Processor was not found: {e}.")
        processor = tokenizer
    if processor is not None and "Processor" not in processor.__class__.__name__:
        processor = None

    if processor is not None:
        setattr(processor, "tokenizer", tokenizer)
    else:
        processor = tokenizer
    processor.save_pretrained(save_directory)
