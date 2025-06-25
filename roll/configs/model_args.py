# Copyright (c) 2025, ALIBABA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Literal, Optional

import torch


@dataclass
class ModelArguments:
    r"""
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune.
    """

    model_name_or_path: str = field(
        default=None,
        metadata={
            "help": "Path to the model weight or identifier from huggingface.co/models or modelscope.cn/models."
        },
    )
    attn_implementation: Optional[str] = field(
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
    model_type: Optional[Literal["auto_sequence_classification", "auto_token_classification", "trl"]] = field(
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

    def __post_init__(self):
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
