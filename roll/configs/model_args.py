from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional

import torch


# Inspired by: https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/hparams/finetuning_args.py
@dataclass
class LoraArguments:
    r"""
    Arguments pertaining to the LoRA training.
    """

    additional_target: Optional[str] = field(
        default=None,
        metadata={
            "help": "Name(s) of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint."
        },
    )
    lora_alpha: Optional[int] = field(
        default=None,
        metadata={"help": "The scale factor for LoRA fine-tuning (default: lora_rank * 2)."},
    )
    lora_dropout: Optional[float] = field(
        default=0.0,
        metadata={"help": "Dropout rate for the LoRA fine-tuning."},
    )
    lora_rank: Optional[int] = field(
        default=8,
        metadata={"help": "The intrinsic dimension for LoRA fine-tuning."},
    )
    lora_target: str = field(
        default=None,
        metadata={
            "help": (
                "Name(s) of target modules to apply LoRA. "
                "Use commas to separate multiple modules. "
                "Use `all` to specify all the linear modules."
            )
        },
    )


@dataclass
class ModelArguments(LoraArguments):
    r"""
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune.
    """

    model_name_or_path: str = field(
        default=None,
        metadata={
            "help": "Path to the model weight or identifier from huggingface.co/models or modelscope.cn/models."
        },
    )
    adapter_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the adapter weight or identifier from huggingface.co/models."},
    )
    attn_implementation: Optional[Literal["sdpa", "fa2", "auto"]] = field(
        default=None,
        metadata={"help": "Enable FlashAttention for faster training and inference."},
    )
    moe_aux_loss_coef: Optional[float] = field(
        default=None,
        metadata={"help": "Coefficient of the auxiliary router loss in mixture-of-experts model."},
    )
    disable_gradient_checkpointing: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to disable gradient checkpointing."},
    )
    device_map: Optional[str] = field(
        default="balanced", metadata={"help": "transformer's from_pretrained device map"}
    )
    dtype: Optional[Literal["fp32", "bf16", "fp16"]] = field(
        default="bf16", metadata={"help": "Set model dtype as fp32, bf16, or fp16, otherwise use config's torch_dtype"}
    )
    model_type: Optional[Literal["auto_sequence_classification", "auto_token_classification", "trl", "diffusion_module"]] = field(
        default=None,
        metadata={
            "help": "reward model type."
        },
    )
    num_labels: Optional[int] = field(
        default=1,
        metadata={
            "help": "The number of labels for AutoModelForTokenClassification and "
            "AutoModelForSequenceClassification."
        },
    )
    model_config_kwargs: dict = field(
        default_factory=lambda: {},
        metadata={"help": "Additional keyword arguments to pass to the model config"},
    )
    freeze_module_prefix: Optional[str] = field(
        default=None,
        metadata={
            "help": "Prefix of frozen modules for partial-parameter (freeze) fine-tuning. Use commas to separate multiple modules."
        },
    )
    ulysses_size: Optional[int] = field(
        default=1,
        metadata={
            "help": "The group size for Ulysses attention."
        },
    )

    def __post_init__(self):
        def split_arg(arg):
            if isinstance(arg, str):
                return [item.strip() for item in arg.split(",")]
            return arg

        self.lora_alpha = self.lora_alpha or self.lora_rank * 2
        if self.lora_target is not None and not any(c in self.lora_target for c in ["*", "$", "|", "("]):
            # split when lora_target is not regex expression
            self.lora_target = split_arg(self.lora_target)
        self.freeze_module_prefix: Optional[List[str]] = split_arg(self.freeze_module_prefix)
        self.additional_target: Optional[List[str]] = split_arg(self.additional_target)

        dtype_mapping = {
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }

        self.compute_dtype = dtype_mapping[self.dtype]
        self.model_max_length = None

        if self.attn_implementation == "fa2":
            self.attn_implementation = "flash_attention_2"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
