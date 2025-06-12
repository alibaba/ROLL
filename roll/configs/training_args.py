import contextlib
import math
from dataclasses import field, dataclass, asdict
from typing import Optional, Dict, Any


@dataclass
class TrainingArguments:
    """
    The TrainingArguments of Transformers will automatically initialize the distributed environment, which should be avoided.
    """

    output_dir: str = field(
        default="",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    max_steps: int = field(
        default=-1,
        metadata={"help": "If > 0: set total number of training steps to perform. Override num_train_epochs."},
    )
    num_train_epochs: float = field(
        default=1.0,
        metadata={"help": "Total number of training epochs to perform."}
    )
    learning_rate: float = field(
        default=5e-5,
        metadata={"help": "The initial learning rate for AdamW."}
    )
    lr_scheduler_type: str = field(
        default="linear",
        metadata={"help": "The scheduler type to use. Refer to HF"})
    weight_decay: float = field(
        default=0.0,
        metadata={"help": "Weight decay for AdamW if we apply some."}
    )
    adam_beta1: float = field(
        default=0.9,
        metadata={"help": "Beta1 for AdamW optimizer"}
    )
    adam_beta2: float = field(
        default=0.999,
        metadata={"help": "Beta2 for AdamW optimizer"}
    )
    per_device_train_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU/MPS/NPU core/CPU for training."}
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed that will be set at the beginning of training."}
    )
    data_seed: Optional[int] = field(
        default=None,
        metadata={"help": "Random seed to be used with data samplers."}
    )
    dataloader_num_workers: int = field(
        default=0,
        metadata={
            "help": (
                "Number of subprocesses to use for data loading (PyTorch only). 0 means that the data will be loaded"
                " in the main process."
            )
        },
    )
    fp16: bool = field(
        default=False,
        metadata={"help": "Whether to use fp16 (mixed) precision instead of 32-bit"},
    )
    bf16: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA"
                " architecture or using CPU (use_cpu) or Ascend NPU. This is an experimental API and it may change."
            )
        },
    )
    tf32: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Whether to enable tf32 mode, available in Ampere and newer GPU architectures. This is an experimental"
                " API and it may change."
            )
        },
    )
    warmup_ratio: float = field(
        default=0.03,
        metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."}
    )
    warmup_steps: int = field(default=0, metadata={"help": "Linear warmup over warmup_steps."})

    def get_warmup_steps(self, num_training_steps: int):
        """
        Get number of steps used for a linear warmup.
        """
        warmup_steps = (
            self.warmup_steps if self.warmup_steps > 0 else math.ceil(num_training_steps * self.warmup_ratio)
        )
        return warmup_steps

    @contextlib.contextmanager
    def main_process_first(self, local=True, desc="work"):
        yield

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
