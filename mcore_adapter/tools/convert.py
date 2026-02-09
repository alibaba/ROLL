import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import AutoConfig, HfArgumentParser

from mcore_adapter.models.converter.post_converter import convert_checkpoint_to_hf, convert_checkpoint_to_mca
from mcore_adapter.training_args import DistributingParallelArguments
from mcore_adapter.utils import get_logger


logger = get_logger(__name__)


@dataclass
class ConvertArguments:
    checkpoint_path: str
    adapter_path: str | None = field(default=None)
    output_path: str = field(default="./output")
    bf16: bool = field(default=False)
    fp16: bool = field(default=False)
    convert_model_max_length: Optional[int] = field(
        default=None, metadata={"help": "Change the model_max_length in hf config.json ."}
    )

    def __post_init__(self):
        if self.bf16 and self.fp16:
            raise ValueError("bf16 and fp16 cannot be both True.")


def convert_mca_to_hf(convert_args: ConvertArguments):
    torch_dtype = None
    if convert_args.bf16:
        torch_dtype = torch.bfloat16
    elif convert_args.fp16:
        torch_dtype = torch.float16
    convert_checkpoint_to_hf(
        convert_args.checkpoint_path,
        convert_args.output_path,
        adapter_name_or_path=convert_args.adapter_path,
        torch_dtype=torch_dtype,
    )

    config = AutoConfig.from_pretrained(convert_args.output_path, trust_remote_code=True)
    if convert_args.convert_model_max_length is not None:
        config.model_max_length = convert_args.convert_model_max_length
        config.max_position_embeddings = convert_args.convert_model_max_length
        config.save_pretrained(convert_args.output_path)
    logger.info(f"\n ==============HF config===========: \n {config}")


def main():
    convert_args, dist_args = HfArgumentParser(
        [ConvertArguments, DistributingParallelArguments]
    ).parse_args_into_dataclasses()

    if convert_args.adapter_path is not None:
        mca_config_path = os.path.join(convert_args.adapter_path, "mca_config.json")
    else:
        mca_config_path = os.path.join(convert_args.checkpoint_path, "mca_config.json")
    from_mca = os.path.exists(mca_config_path)

    if not from_mca:
        convert_checkpoint_to_mca(
            convert_args.checkpoint_path,
            convert_args.output_path,
            dist_args,
            bf16=convert_args.bf16,
            fp16=convert_args.fp16,
        )
    else:
        convert_mca_to_hf(convert_args)


if __name__ == "__main__":
    main()
