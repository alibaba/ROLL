from itertools import product
from typing import TYPE_CHECKING, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
)
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from ...checkpointing import get_checkpoint_name
from ...utils import get_logger
from ..auto.config_auto import AutoConfig
from .dist_converter import DistConverter
from .template import get_template


if TYPE_CHECKING:
    from .template import Template

logger = get_logger(__name__)


def _add_mca_state_dicts_to_hf(
    state_dicts, dist_reverter: "DistConverter", template: "Template", hf_state_dict, verbose: bool = True
):
    def log(msg):
        if verbose:
            logger.info(msg)
    tp_rank, pp_rank, ep_rank, vp_rank = (
        dist_reverter.tensor_model_parallel_rank,
        dist_reverter.pipeline_model_parallel_rank,
        dist_reverter.expert_model_parallel_rank,
        dist_reverter.virtual_pipeline_model_parallel_rank,
    )
    for mca_name in state_dicts[0].keys():
        if mca_name.endswith("._extra_state"):
            continue
        weights = [state_dict[mca_name] if mca_name in state_dict else None for state_dict in state_dicts]
        mca_named_weights = dist_reverter(mca_name, weights)
        converted_state_dict = {}
        if mca_named_weights is not None:
            for mca_name, mca_weight in mca_named_weights.items():
                converted = template.add_mca_weight(mca_name, mca_weight)
                assert (
                    len(set(converted_state_dict.keys()).intersection(converted.keys())) == 0
                ), f"converted_state_dict: {converted_state_dict.keys()} converted: {converted.keys()}"
                converted_state_dict.update(converted)
        if converted_state_dict is not None and len(converted_state_dict) > 0:
            for hf_name, hf_weight in converted_state_dict.items():
                if hf_name in hf_state_dict:
                    if not hf_weight.equal(hf_state_dict[hf_name]):
                        raise ValueError(
                            f"weight of hf_name:{hf_name} mca_name:{mca_name} in "
                            f"tp_rank, pp_rank, ep_rank, vp_rank:{tp_rank} {pp_rank} {ep_rank} {vp_rank} "
                            f"diff max:{torch.abs(hf_weight - hf_state_dict[hf_name]).max()}"
                        )
                hf_state_dict[hf_name] = hf_weight
                log(f"mca_name: {mca_name} -> hf_name: {hf_name}")
        else:
            log(f"mca_name: {mca_name} added but not converted")


def convert_checkpoint_to_hf(model_name_or_path: str, save_directory: str, torch_dtype: Optional["torch.dtype"] = None, verbose: bool = True):
    mca_config = AutoConfig.from_pretrained(model_name_or_path)
    if mca_config is None:
        raise ValueError("No mca config found in checkpoint")
    if mca_config.hf_model_type is None:
        raise ValueError("No hf model type found in mca config")
    template: "Template" = get_template(mca_config.hf_model_type)
    hf_config = template.convert_mca_to_hf_config(mca_config)
    template.set_mca_config_for_ops(mca_config)
    hf_state_dict = {}

    for pp_rank, ep_rank in product(
        range(mca_config.pipeline_model_parallel_size), range(mca_config.expert_model_parallel_size)
    ):
        state_dicts = []
        # TODO: use loader and support low_mem
        for tp_rank in range(mca_config.tensor_model_parallel_size):
            ckpt_name = get_checkpoint_name(
                model_name_or_path,
                tensor_rank=tp_rank,
                pipeline_rank=pp_rank,
                pipeline_parallel=mca_config.pipeline_model_parallel_size > 1,
                expert_rank=ep_rank,
                expert_parallel=mca_config.expert_model_parallel_size > 1,
            )
            state_dicts.append(torch.load(ckpt_name, map_location="cpu"))
        virtual_pipe_on = (mca_config.virtual_pipeline_model_parallel_size or 1) > 1
        for i in range(mca_config.virtual_pipeline_model_parallel_size or 1):
            dist_reverter = DistConverter(
                mca_config=mca_config,
                revert=True,
                pipeline_model_parallel_rank=pp_rank,
                expert_model_parallel_rank=ep_rank,
                virtual_pipeline_model_parallel_rank=i if virtual_pipe_on else 0,
            )
            key = "model" + (str(i) if virtual_pipe_on else "")
            virtual_state_dicts = [sd.pop(key) for sd in state_dicts]
            _add_mca_state_dicts_to_hf(virtual_state_dicts, dist_reverter, template, hf_state_dict, verbose=verbose)

    has_remote_code = hasattr(hf_config, "auto_map") and "AutoModelForCausalLM" in hf_config.auto_map
    model_class = AutoModelForCausalLM
    if type(hf_config) in AutoModelForVision2Seq._model_mapping.keys():
        model_class = AutoModelForVision2Seq
    elif type(hf_config) in AutoModelForImageTextToText._model_mapping.keys():
        model_class = AutoModelForImageTextToText
    if has_remote_code:
        class_ref = hf_config.auto_map["AutoModelForCausalLM"]
        model_class = get_class_from_dynamic_module(class_ref, mca_config.name_or_path)
    model = model_class.from_pretrained(
        None,
        config=hf_config,
        state_dict=hf_state_dict,
        torch_dtype=torch_dtype if torch_dtype is not None else mca_config.params_dtype,
        trust_remote_code=True,
    )
    model.save_pretrained(save_directory)
    mca_config.save_hf_auto_map_files(save_directory)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    try:
        processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
    except Exception as e:
        logger.info(f"Processor was not found: {e}.")
        processor = tokenizer
    if processor is not None and "Processor" not in processor.__class__.__name__:
        processor = None

    if processor is not None:
        setattr(processor, "tokenizer", tokenizer)
    else:
        processor = tokenizer
    processor.save_pretrained(save_directory)
