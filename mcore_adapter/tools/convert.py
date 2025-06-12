import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from megatron.core import mpu
from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor, AutoTokenizer, HfArgumentParser

from mcore_adapter.models import AutoModel as AutoMcaModel
from mcore_adapter.models.converter.dist_converter import DistConverter
from mcore_adapter.models.converter.model_converter import ModelConverter
from mcore_adapter.models.converter.post_converter import convert_checkpoint_to_hf
from mcore_adapter.models.converter.template import get_template
from mcore_adapter.training_args import DistributingParallelArguments
from mcore_adapter.utils import get_logger


if TYPE_CHECKING:
    from mcore_adapter.models.converter.template import Template

logger = get_logger(__name__)


@dataclass
class ConvertArguments:
    checkpoint_path: str
    output_path: str = field(default="./output")
    bf16: bool = field(default=False)
    fp16: bool = field(default=False)


def convert_hf_to_mca(convert_args: ConvertArguments, dist_args: DistributingParallelArguments):
    dist_args.pipeline_model_parallel_size = dist_args.pipeline_model_parallel_size or 1
    dist_args.tensor_model_parallel_size = dist_args.tensor_model_parallel_size or 1
    dist_args.expert_model_parallel_size = dist_args.expert_model_parallel_size or 1
    hf_config = AutoConfig.from_pretrained(convert_args.checkpoint_path, trust_remote_code=True)
    template: "Template" = get_template(hf_config.model_type)
    mca_config = template.convert_hf_to_mca_config(
        hf_config,
        bf16=convert_args.bf16,
        fp16=convert_args.fp16,
        **dist_args.get_config_dict()
    )
    template.set_mca_config_for_ops(mca_config)
    mpu.set_tensor_model_parallel_world_size(dist_args.tensor_model_parallel_size)
    mpu.set_pipeline_model_parallel_world_size(dist_args.pipeline_model_parallel_size)
    mpu.set_expert_model_parallel_world_size(dist_args.expert_model_parallel_size)
    if dist_args.virtual_pipeline_model_parallel_size is not None:
        mpu.set_virtual_pipeline_model_parallel_world_size(dist_args.virtual_pipeline_model_parallel_size)

    model_converter = ModelConverter(mca_config=mca_config, verbose=True)

    for dist_converter in tqdm(
        DistConverter.dist_converter_iter(mca_config=mca_config),
        total=dist_args.tensor_model_parallel_size
        * dist_args.pipeline_model_parallel_size
        * dist_args.expert_model_parallel_size,
        desc="Converting",
    ):
        mpu.set_tensor_model_parallel_rank(dist_converter.tensor_model_parallel_rank)
        mpu.set_pipeline_model_parallel_rank(dist_converter.pipeline_model_parallel_rank)
        mpu.set_expert_model_parallel_rank(dist_converter.expert_model_parallel_rank)
        model_parallel_cuda_manual_seed(42)
        mca_config.use_cpu_initialization = True
        mca_config.perform_initialization = False
        mca_model = AutoMcaModel.from_config(config=mca_config)
        mca_state_dict = {}
        for i in range(len(mca_model.models)):
            key = "model"
            dist_converter = DistConverter(
                mca_config=mca_config,
                tensor_model_parallel_rank=dist_converter.tensor_model_parallel_rank,
                pipeline_model_parallel_rank=dist_converter.pipeline_model_parallel_rank,
                expert_model_parallel_rank=dist_converter.expert_model_parallel_rank,
                virtual_pipeline_model_parallel_rank=i
            )
            if dist_args.virtual_pipeline_model_parallel_size is not None:
                key = f"model{i}"
                mpu.set_virtual_pipeline_model_parallel_rank(i)
            mca_state_dict[key] = model_converter.get_mca_state_dict(
                dist_converter, model_converter.hf_state_dict_iter(convert_args.checkpoint_path, dist_converter)
            )

        missing_keys, unexpected_keys = mca_model.load_state_dict(mca_state_dict, strict=False)
        if missing_keys:  # something about fp8 ignored for now
            missing_keys = [key for key in missing_keys if not key.endswith("._extra_state")]
        assert unexpected_keys is None or len(unexpected_keys) == 0, f"unexpected_keys: {unexpected_keys}"
        assert missing_keys is None or len(missing_keys) == 0, f"missing_keys: {missing_keys}"
        logger.info(
            f"Saving model tp_rank: {dist_converter.tensor_model_parallel_rank} "
            f"pp_rank: {dist_converter.pipeline_model_parallel_rank} "
            f"ep_rank: {dist_converter.expert_model_parallel_rank} to {convert_args.output_path}"
        )
        mca_config.use_cpu_initialization = False
        mca_model.save_pretrained(convert_args.output_path)
        del mca_model
        template.release()

    tokenizer = AutoTokenizer.from_pretrained(convert_args.checkpoint_path, trust_remote_code=True)
    try:
        processor = AutoProcessor.from_pretrained(convert_args.checkpoint_path, trust_remote_code=True)
    except Exception as e:
        logger.info(f"Processor was not found: {e}.")
        processor = tokenizer
    if processor is not None and "Processor" not in processor.__class__.__name__:
        processor = None

    if processor is not None:
        setattr(processor, "tokenizer", tokenizer)
    else:
        processor = tokenizer
    processor.save_pretrained(convert_args.output_path)

def convert_mca_to_hf(convert_args: ConvertArguments):
    torch_dtype = None
    if convert_args.bf16:
        torch_dtype = torch.bfloat16
    elif convert_args.fp16:
        torch_dtype = torch.float16
    convert_checkpoint_to_hf(convert_args.checkpoint_path, convert_args.output_path, torch_dtype=torch_dtype)

def main():
    convert_args, dist_args = HfArgumentParser(
        [ConvertArguments, DistributingParallelArguments]
    ).parse_args_into_dataclasses()

    mca_config_path = os.path.join(convert_args.checkpoint_path, "mca_config.json")
    from_mca = os.path.exists(mca_config_path)

    if not from_mca:
        convert_hf_to_mca(convert_args, dist_args)
    else:
        convert_mca_to_hf(convert_args)


if __name__ == "__main__":
    main()
