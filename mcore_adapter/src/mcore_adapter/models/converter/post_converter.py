import os
from collections import defaultdict
from itertools import product
from typing import TYPE_CHECKING, Optional

import torch
from megatron.core import mpu
from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed
from tqdm import tqdm
from transformers import (
    AutoConfig as HfAutoConfig,
)
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForVision2Seq,
    AutoModelForTextToWaveform,
    AutoProcessor,
    AutoTokenizer,
)
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.models.auto.auto_factory import _get_model_class
from transformers.utils import is_peft_available

from ...checkpointing import get_checkpoint_name, save_config_and_state_dict
from ...constants import ADAPTER_CONFIG_NAME
from ...training_args import DistributingParallelArguments
from ...utils import get_logger
from ..auto.config_auto import AutoConfig
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


def _add_mca_state_dicts_to_hf(
    model_converter: "ModelConverter",
    state_dicts: list[dict[str, torch.Tensor] | dict[str, dict[str, torch.Tensor]]],
    hf_state_dict: dict | dict[str, dict],
    vp_stage: int,
    verbose: bool = True,
    **kwargs,
):
    def log(msg):
        if verbose:
            logger.info(msg)

    tp_rank, pp_rank, ep_rank = (
        model_converter.dist_converter.tensor_model_parallel_rank,
        model_converter.dist_converter.pipeline_model_parallel_rank,
        model_converter.dist_converter.expert_model_parallel_rank,
    )
    for mca_name in state_dicts[0].keys():
        if mca_name.endswith("._extra_state"):
            continue
        weights = [state_dict[mca_name] if mca_name in state_dict else None for state_dict in state_dicts]
        converted_state_dict = model_converter.convert_to_hf({mca_name: weights}, vp_stage=vp_stage, **kwargs)
        if converted_state_dict is not None and len(converted_state_dict) > 0:
            for hf_name, hf_weight in converted_state_dict.items():
                if hf_name in hf_state_dict:
                    if not hf_weight.equal(hf_state_dict[hf_name]):
                        raise ValueError(
                            f"weight of hf_name:{hf_name} mca_name:{mca_name} in "
                            f"tp_rank, pp_rank, ep_rank, vp_rank:{tp_rank} {pp_rank} {ep_rank} {vp_stage} "
                            f"diff max:{torch.abs(hf_weight - hf_state_dict[hf_name]).max()}"
                        )
                hf_state_dict[hf_name] = hf_weight
                log(f"mca_name: {mca_name} -> hf_name: {hf_name}")
        else:
            log(f"mca_name: {mca_name} added but not converted")


def _load_mca_config_and_setup(checkpoint_path: str):
    mca_config = AutoConfig.from_pretrained(checkpoint_path)
    if mca_config is None:
        raise ValueError("No mca config found in checkpoint")
    if mca_config.hf_model_type is None:
        raise ValueError("No hf model type found in mca config")

    template: "Template" = get_template(mca_config.hf_model_type)
    hf_config = template.convert_mca_to_hf_config(mca_config)
    template.set_mca_config_for_ops(mca_config)

    mpu.set_expert_model_parallel_world_size(mca_config.expert_model_parallel_size)
    mpu.set_pipeline_model_parallel_world_size(mca_config.pipeline_model_parallel_size)
    mpu.set_tensor_model_parallel_world_size(mca_config.tensor_model_parallel_size)
    if mca_config.virtual_pipeline_model_parallel_size is not None:
        mpu.set_virtual_pipeline_model_parallel_world_size(mca_config.virtual_pipeline_model_parallel_size)

    return mca_config, hf_config


def _convert_state_dicts(
    checkpoint_path: str,
    mca_config: "McaModelConfig",
    target_state_dict: dict,
    verbose: bool = True,
    adapter_name: str | None = None,
    **kwargs,
):
    for pp_rank, ep_rank in product(
        range(mca_config.pipeline_model_parallel_size), range(mca_config.expert_model_parallel_size)
    ):
        state_dicts = []
        for tp_rank in range(mca_config.tensor_model_parallel_size):
            ckpt_name = get_checkpoint_name(
                checkpoint_path,
                tensor_rank=tp_rank,
                pipeline_rank=pp_rank,
                pipeline_parallel=mca_config.pipeline_model_parallel_size > 1,
                expert_rank=ep_rank,
                expert_parallel=mca_config.expert_model_parallel_size > 1,
            )
            state_dicts.append(torch.load(ckpt_name, map_location="cpu"))

        virtual_pipe_on = (mca_config.virtual_pipeline_model_parallel_size or 1) > 1
        mpu.set_pipeline_model_parallel_rank(pp_rank)
        mpu.set_expert_model_parallel_rank(ep_rank)
        mpu.set_tensor_model_parallel_rank(0)

        model_converter = ModelConverter(
            mca_config=mca_config,
            pipeline_model_parallel_rank=pp_rank,
            expert_model_parallel_rank=ep_rank,
            tensor_model_parallel_rank=0,
            verbose=verbose,
            to_hf=True,
        )

        for i in range(mca_config.virtual_pipeline_model_parallel_size or 1):
            if virtual_pipe_on:
                mpu.set_virtual_pipeline_model_parallel_rank(i)
            key = "model" + (str(i) if virtual_pipe_on else "")
            virtual_state_dicts = [sd.pop(key) for sd in state_dicts]
            _add_mca_state_dicts_to_hf(
                model_converter,
                virtual_state_dicts,
                target_state_dict,
                vp_stage=i,
                verbose=verbose,
                **kwargs,
            )


def _get_hf_model_class(hf_config: "PretrainedConfig", mca_config: "McaModelConfig"):
    has_remote_code = hasattr(hf_config, "auto_map") and "AutoModelForCausalLM" in hf_config.auto_map
    model_class = AutoModelForCausalLM

    if type(hf_config) in AutoModelForVision2Seq._model_mapping.keys():
        model_class = AutoModelForVision2Seq
    elif type(hf_config) in AutoModelForImageTextToText._model_mapping.keys():
        model_class = AutoModelForImageTextToText
    elif type(hf_config) in AutoModelForTextToWaveform._model_mapping.keys():
        model_class = AutoModelForTextToWaveform
    if has_remote_code:
        class_ref = hf_config.auto_map["AutoModelForCausalLM"]
        model_class = get_class_from_dynamic_module(class_ref, mca_config.name_or_path)
    else:
        model_class = _get_model_class(hf_config, model_class._model_mapping)

    return model_class


def _save_tokenizer_and_processor(checkpoint_path: str, save_directory: str, verbose: bool = True):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    try:
        processor = AutoProcessor.from_pretrained(checkpoint_path, trust_remote_code=True)
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


def convert_adapter_to_hf(
    model_name_or_path: str,
    adapter_name_or_path: str,
    save_directory: str,
    torch_dtype: Optional["torch.dtype"] = None,
    verbose: bool = True,
):
    adapter_names = (
        [
            folder_name
            for folder_name in os.listdir(adapter_name_or_path)
            if os.path.isdir(os.path.join(adapter_name_or_path, folder_name))
            and os.path.isfile(os.path.join(adapter_name_or_path, folder_name, ADAPTER_CONFIG_NAME))
        ]
        if os.path.isdir(adapter_name_or_path)
        else []
    )
    if not adapter_names:
        raise ValueError(f"No LoRA adapters found in {adapter_name_or_path}")

    peft_configs = {
        adapter_name: PeftConfig.from_pretrained(os.path.join(adapter_name_or_path, adapter_name))
        for adapter_name in adapter_names
    }

    mca_config, hf_config = _load_mca_config_and_setup(adapter_name_or_path)
    hf_state_dict = defaultdict(dict)

    # 转换每个 adapter 的权重
    for adapter_name, peft_config in peft_configs.items():
        adapter_checkpoint_path = os.path.join(adapter_name_or_path, adapter_name)
        _convert_state_dicts(
            adapter_checkpoint_path,
            mca_config,
            hf_state_dict[adapter_name],
            verbose=verbose,
            adapter_name=adapter_name,
            lora_rank=peft_config.r,
        )

    # 创建模型并加载 adapter
    model_class = _get_hf_model_class(hf_config, mca_config)
    hf_config.save_pretrained(save_directory)

    model = model_class.from_pretrained(
        model_name_or_path,
        config=hf_config,
        torch_dtype=torch_dtype if torch_dtype is not None else mca_config.params_dtype,
        trust_remote_code=True,
    )

    # 加载第一个 adapter
    adapter0_name = "default" if "default" in hf_state_dict else sorted(hf_state_dict.keys())[0]
    target_modules = [
        name[: name.find(".lora")].split(".")[-1]
        for name in hf_state_dict[adapter0_name].keys()
        if ".lora_A." in name or ".lora_B." in name
    ]
    target_modules = list(set(target_modules))
    kwargs = {}
    if mca_config.num_moe_experts is not None: # MoE model
        rank_pattern = {
            "down_proj": peft_configs[adapter0_name].r // mca_config.moe_router_topk,
            "up_proj": peft_configs[adapter0_name].r // mca_config.moe_router_topk,
            "gate_proj": peft_configs[adapter0_name].r // mca_config.moe_router_topk,
            "w1": peft_configs[adapter0_name].r // mca_config.moe_router_topk,
            "w2": peft_configs[adapter0_name].r // mca_config.moe_router_topk,
            "w3": peft_configs[adapter0_name].r // mca_config.moe_router_topk,
        }
        kwargs["rank_pattern"] = rank_pattern

    lora_config = LoraConfig(
        r=peft_configs[adapter0_name].r,
        target_modules=target_modules,
        lora_alpha=peft_configs[adapter0_name].lora_alpha,
        lora_dropout=peft_configs[adapter0_name].lora_dropout,
        use_rslora=peft_configs[adapter0_name].use_rslora,
        modules_to_save=peft_configs[adapter0_name].modules_to_save,
        **kwargs,
    )
    model = get_peft_model(model, lora_config, adapter_name=adapter0_name)
    set_peft_model_state_dict(model.base_model.model, hf_state_dict[adapter0_name], adapter_name=adapter0_name)

    # 加载其他 adapter
    for adapter_name, state_dict in hf_state_dict.items():
        if adapter_name == adapter0_name:
            continue
        target_modules = [
            name[: name.find(".lora")].split(".")[-1]
            for name in state_dict.keys()
            if ".lora_A." in name or ".lora_B." in name
        ]
        target_modules = list(set(target_modules))
        kwargs = {}
        if mca_config.num_moe_experts is not None: # MoE model
            rank_pattern = {
                "down_proj": peft_configs[adapter_name].r // mca_config.moe_router_topk,
                "up_proj": peft_configs[adapter_name].r // mca_config.moe_router_topk,
                "gate_proj": peft_configs[adapter_name].r // mca_config.moe_router_topk,
                "w1": peft_configs[adapter_name].r // mca_config.moe_router_topk,
                "w2": peft_configs[adapter_name].r // mca_config.moe_router_topk,
                "w3": peft_configs[adapter_name].r // mca_config.moe_router_topk,
            }
            kwargs["rank_pattern"] = rank_pattern

        lora_config = LoraConfig(
            r=peft_configs[adapter_name].r,
            target_modules=target_modules,
            lora_alpha=peft_configs[adapter_name].lora_alpha,
            lora_dropout=peft_configs[adapter_name].lora_dropout,
            use_rslora=peft_configs[adapter_name].use_rslora,
            modules_to_save=peft_configs[adapter_name].modules_to_save,
            **kwargs,
        )
        model.add_adapter(adapter_name, lora_config)
        set_peft_model_state_dict(model.base_model.model, state_dict, adapter_name=adapter_name)

    model.save_pretrained(save_directory)
    mca_config.save_hf_auto_map_files(save_directory)
    _save_tokenizer_and_processor(adapter_name_or_path, save_directory, verbose)


def convert_checkpoint_to_hf(
    model_name_or_path: str,
    save_directory: str,
    adapter_name_or_path: Optional[str] = None,
    torch_dtype: Optional["torch.dtype"] = None,
    verbose: bool = True,
):
    if adapter_name_or_path is not None:
        if not is_peft_available():
            raise ImportError("PEFT is not installed. Please install it with `pip install peft`")
        convert_adapter_to_hf(
            model_name_or_path=model_name_or_path,
            adapter_name_or_path=adapter_name_or_path,
            save_directory=save_directory,
            torch_dtype=torch_dtype,
            verbose=verbose,
        )
        return

    ckpt_path = model_name_or_path
    mca_config, hf_config = _load_mca_config_and_setup(ckpt_path)
    hf_state_dict = {}

    # 转换权重
    _convert_state_dicts(ckpt_path, mca_config, hf_state_dict, verbose=verbose)

    # 创建并保存模型
    model_class = _get_hf_model_class(hf_config, mca_config)
    model = model_class.from_pretrained(
        None,
        config=hf_config,
        state_dict=hf_state_dict,
        torch_dtype=torch_dtype if torch_dtype is not None else mca_config.params_dtype,
        trust_remote_code=True,
    )
    model.save_pretrained(save_directory)
    mca_config.save_hf_auto_map_files(save_directory)
    _save_tokenizer_and_processor(ckpt_path, save_directory, verbose)


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
