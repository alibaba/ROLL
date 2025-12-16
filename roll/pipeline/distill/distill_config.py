import dataclasses
from dataclasses import dataclass, field
from typing import Optional, Literal, List, Dict, Any

from roll.configs.base_config import BaseConfig, ScheduleConfig
from roll.configs.worker_config import WorkerConfig
from roll.utils.logging import get_logger

logger = get_logger()

@dataclass
class DistillConfig(BaseConfig):
    # role related
    student_pretrain: str = field(
        default=None,
        metadata={"help": "Path to pretrain model directory for the student, if available."})
    teacher_pretrain: str = field(
        default=None,
        metadata={"help": "Path to pretrain model directory for the teacher, if available."}
    )
    student: WorkerConfig = field(
        default_factory=WorkerConfig,
        metadata={"help": "Configuration for the student's role."}
    )
    teacher: WorkerConfig = field(
        default_factory=WorkerConfig,
        metadata={"help": "Configuration for the teacher's role."}
    )
    validation: WorkerConfig = field(
        default=None,
        metadata={"help": "Configuration for the validation."}
    )

    # data related
    question_key: str = field(
        default='question',
        metadata={"help": "The user’s question"})
    answer_key: str = field(
        default='answer',
        metadata={"help": "The assistant’s answer corresponding to that question."}
    )

    # Distillation related
    logits_topk: int = field(
        default=64,
        metadata={
            "help": (
                "Top-k of logits used to caculate distill loss"
            )
        }
    )
    distill_loss_weight: float = field(
        default=0.5,
        metadata={
            "help": (
                "Distillation loss ratio versus soft target loss ratio in distillation training"
                "SFT loss ratio is 1 - distill_loss_weight"
            )
        },
    )
    kd_temperature: float = field(
        default=1,
        metadata={
            "help": (
                "kd_temperature"
            )
        },
    )
    teacher_temperature: float = field(
        default=1,
        metadata={
            "help": (
                "teacher_temperature"
            )
        },
    )
    kd_objective: Optional[Literal[
        "forward_kl", "reverse_kl", "adaptive_kl", "skewed_forward_kl", "skewed_reverse_kl", "js_divergence"]] = field(
        default="forward_kl",
        metadata={
            "help":
                ("kd_objective.")
        },
    )
    adaptive_kl_alpha: Optional[float] = field(
        default=0.5,
        metadata={
            "help":
                ("adaptive_kl_alpha.")
        },
    )
    skew_lambda: Optional[float] = field(
        default=0.1,
        metadata={
            "help":
                ("skew_lambda.")
        },
    )
    distill_on_prompt: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to distill on the prompt or not."},
    )

    max_grad_norm: Optional[float] = field(
        default=0,
        metadata={"help": "Maximum grad norm"}
    )

    logits_transfer_backend: Optional[str] = field(
        default="ipc+nccl",
        metadata={"help": "Backend used to transfer logits from teacher to student, available options are"
                          " 'ipc+nccl', 'nccl-only', 'ray'"}
    )

    def __post_init__(self):
        super().__post_init__()

        if (
                self.student.model_args.model_name_or_path is None
                or self.teacher.model_args.model_name_or_path is None
        ):
            self.student.model_args.model_name_or_path = self.student_pretrain
            self.teacher.model_args.model_name_or_path = self.teacher_pretrain

        # default worker_cls
        if self.student.worker_cls is None:
            self.student.worker_cls = "roll.pipeline.distill.distill_worker.StudentWorker"
        if self.teacher.worker_cls is None:
            self.teacher.worker_cls = "roll.pipeline.distill.distill_worker.TeacherWorker"

        self.student.training_args.output_dir = self.output_dir
        self.teacher.training_args.output_dir = self.output_dir

        self.teacher.name = "teacher"
        self.student.name = "student"

        self.target_vocab_size = None

        self.validate_worker_config()

    def to_dict(self):
        return dataclasses.asdict(self)

    def set_max_steps(self, max_steps: int):
        self.student.training_args.max_steps = max_steps