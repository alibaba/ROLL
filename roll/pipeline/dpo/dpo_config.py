import dataclasses
from dataclasses import dataclass, field

from roll.configs.base_config import BaseConfig
from roll.configs.worker_config import WorkerConfig
from roll.utils.logging import get_logger

logger = get_logger()


@dataclass
class DPOConfig(BaseConfig):
    # global
    global_template: str = field(default=None, metadata={"help": "The template of the global."})

    train_batch_size: int = field(
        default=16,
        metadata={"help": "batch_size for one train step"},
    )
    val_batch_size: int = field(
        default=32,
        metadata={"help": "batch_size for validate step"},
    )
    max_grad_norm: float = field(default=1.0, metadata={"help": "Maximum norm"})

    # role related
    pretrain: str = field(default=None, metadata={"help": "Path to pretrain model directory, if available."})
    validation: WorkerConfig = field(
        default_factory=WorkerConfig, metadata={"help": "Configuration for the validation."}
    )
    actor_train: WorkerConfig = field(
        default_factory=WorkerConfig, metadata={"help": "Configuration for the actor's training role."}
    )
    reference: WorkerConfig = field(
        default_factory=WorkerConfig, metadata={"help": "Configuration for the reference role."}
    )

    # dpo related
    ipo: bool = field(
        default=False, metadata={"help": "Whether to use ipo."}  # IPO https://arxiv.org/pdf/2310.12036v2.pdf
    )
    beta: float = field(default=0.1, metadata={"help": "beta for dpo."})
    label_smoothing: float = field(
        default=0.0, metadata={"help": "label_smoothing for dpo."}  # cDPO https://arxiv.org/pdf/2305.18290.pdf
    )

    # data related
    chosen_key: str = field(
        default = "chosen",
        metadata = {"help": "the key of chosen response in dataset"},
    )
    rejected_key: str = field(
        default = "rejected",
        metadata = {"help": "the key of rejected response in dataset"},
    )

    def __post_init__(self):
        BaseConfig.__post_init__(self)

        if (
            self.actor_train.model_args.model_name_or_path is None
            or self.reference.model_args.model_name_or_path is None
        ):
            self.actor_train.model_args.model_name_or_path = self.pretrain
            self.reference.model_args.model_name_or_path = self.pretrain

        # default worker_cls
        if self.actor_train.worker_cls is None:
            self.actor_train.worker_cls = "roll.pipeline.dpo.actor_worker.ActorWorker"
        if self.reference.worker_cls is None:
            self.reference.worker_cls = "roll.pipeline.dpo.actor_worker.ActorWorker"

        self.actor_train.training_args.output_dir = self.output_dir

        self.actor_train.name = "actor_train"
        self.reference.name = "reference"

        self.validate_worker_config()

    def set_max_steps(self, max_steps: int):
        self.max_steps = max_steps
        self.actor_train.training_args.max_steps = max_steps

        logger.info(f"pipeline max_steps: {self.max_steps} to {max_steps}")
        logger.info(f"actor train max_steps without dp_size: {self.actor_train.training_args.max_steps}")

    def to_dict(self):
        return dataclasses.asdict(self)
