import dataclasses
from dataclasses import dataclass, field

from roll.configs.base_config import BaseConfig
from roll.configs.worker_config import WorkerConfig
from roll.utils.logging import get_logger

logger = get_logger()


@dataclass
class RewardFLConfig(BaseConfig):
    # global
    global_template: str = field(default=None, metadata={"help": "The template of the global."})

    train_batch_size: int = field(
        default=8,
        metadata={"help": "batch_size for one train step"},
    )

    max_grad_norm: float = field(default=1.0, metadata={"help": "Maximum norm"})

    actor_train: WorkerConfig = field(
        default_factory=WorkerConfig, metadata={"help": "Configuration for the actor's training role."}
    )

    # reward_fl related
    def __post_init__(self):
        BaseConfig.__post_init__(self)

        # default worker_cls
        if self.actor_train.worker_cls is None:
            self.actor_train.worker_cls = "roll.pipeline.diffusion.reward_fl.actor_worker.ActorWorker"

        self.actor_train.training_args.output_dir = self.output_dir

        self.actor_train.name = "actor_train"

    def set_max_steps(self, max_steps: int):
        self.max_steps = max_steps
        self.actor_train.training_args.max_steps = max_steps

        logger.info(f"pipeline max_steps: {self.max_steps} to {max_steps}")
        logger.info(f"actor train max_steps without dp_size: {self.actor_train.training_args.max_steps}")

    def to_dict(self):
        return dataclasses.asdict(self)
