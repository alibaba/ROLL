from dataclasses import dataclass, field
from typing import Optional

from roll.configs.base_config import BaseConfig
from roll.configs.worker_config import WorkerConfig


@dataclass
class SFTConfig(BaseConfig):
    global_template: str = field(
        default=None,
        metadata={"help": "The template of the global."}
    )
    
    pretrain: str = field(
        default=None,
        metadata={"help": "Path to pretrain model directory, if available."}
    )

    # sft data related
    system_key: str = field(
        default=None,
        metadata={"help": "the key of system prompt in dataset, use the default system prompt in the tokenizer tmplate if not provided"}
    )
    prompt_key: str = field(
        default="instruction",
        metadata={"help": "the key of prompt in dataset"},
    )
    query_key: Optional[str] = field(
        default=None,
        metadata={"help": "(optional)the key of query in dataset"},
    )
    response_key: str = field(
        default="output",
        metadata={"help": "the key of response in dataset"}
    )
    
    # role related
    validation: WorkerConfig = field(
        default=None,
        metadata={"help": "Configuration for the validation."}
    )
    sft_train: WorkerConfig = field(
        default_factory=WorkerConfig,
        metadata={"help": "Configuration for the sft's training role."}
    )

    max_grad_norm: float = field(
        default=1.0, 
        metadata={"help": "Maximum norm"}
    )

    def __post_init__(self):
        super().__post_init__()
        self.sft_train.model_args.model_name_or_path = self.pretrain

        if self.sft_train.worker_cls is None:
            self.sft_train.worker_cls = "roll.pipeline.sft.sft_worker.SFTWorker"

        self.sft_train.name = "sft_train"

        self.validate_worker_config()

    def set_max_steps(self, max_steps: int):
        self.sft_train.training_args.max_steps = max_steps
