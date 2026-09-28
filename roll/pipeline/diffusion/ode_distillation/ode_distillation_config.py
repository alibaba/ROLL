"""Configuration for ODE trajectory distillation."""

from __future__ import annotations

from dataclasses import dataclass, field

from roll.configs.base_config import BaseConfig
from roll.configs.worker_config import StrategyArguments, WorkerConfig


@dataclass
class ODEDistillationConfig(BaseConfig):
    """ROLL config for teacher ODE trajectory regression."""

    pretrain: str | None = None
    diffusion_model_variant: str = field(
        default="wan2_1",
        metadata={"help": "Diffusion model variant used to select the model adapter and Self-Forcing builder."},
    )
    sequence_length: int | None = 1
    latent_shape: list[int] = field(default_factory=lambda: [21, 16, 60, 104])
    num_train_timesteps: int = 1000
    teacher_num_steps: int = 48
    denoising_step_list: list[int] = field(default_factory=lambda: [1000, 750, 500, 250])
    timestep_shift: float = 5.0
    guidance_scale: float = 6.0
    negative_prompt: str = ""
    num_frame_per_block: int = 3
    num_ode_pairs: int = 16000
    ode_pair_cache_dir: str | None = None
    gpu_tensor_transfer: bool = True
    max_grad_norm: float = 10.0
    teacher: WorkerConfig = field(default_factory=WorkerConfig)
    student: WorkerConfig = field(default_factory=WorkerConfig)

    def __post_init__(self) -> None:
        if (
            not self.denoising_step_list
            or self.denoising_step_list[0] != self.num_train_timesteps
            or any(step <= 0 or step > self.num_train_timesteps for step in self.denoising_step_list)
            or any(left <= right for left, right in zip(self.denoising_step_list, self.denoising_step_list[1:]))
        ):
            raise ValueError(
                "denoising_step_list must start at num_train_timesteps and contain strictly decreasing positive steps"
            )
        if any(
            (self.num_train_timesteps - step) * self.teacher_num_steps % self.num_train_timesteps
            for step in self.denoising_step_list
        ):
            raise ValueError("denoising_step_list must lie exactly on the configured teacher ODE integration grid")
        super().__post_init__()
        role_defaults = (
            (
                self.teacher,
                "teacher",
                "roll.pipeline.diffusion.ode_distillation.ode_distillation_worker.ODETrajectoryTeacherWorker",
                "veomni_infer",
            ),
            (
                self.student,
                "student",
                "roll.pipeline.diffusion.ode_distillation.ode_distillation_worker.ODEDistillationStudentWorker",
                "fsdp2_train",
            ),
        )
        for worker_config, name, worker_cls, strategy_name in role_defaults:
            worker_config.name = worker_config.name or name
            worker_config.worker_cls = worker_config.worker_cls or worker_cls
            worker_config.model_args.model_name_or_path = worker_config.model_args.model_name_or_path or self.pretrain
            worker_config.training_args.output_dir = self.output_dir
            worker_config.system_envs.update(
                {key: value for key, value in self.system_envs.items() if key not in worker_config.system_envs}
            )
            if worker_config.strategy_args is None:
                worker_config.strategy_args = StrategyArguments(strategy_name=strategy_name)

        self.student.model_args.model_config_kwargs.setdefault("local_attn_size", -1)
        self.student.model_args.model_config_kwargs.setdefault("sink_size", 0)
        self.student.model_args.model_config_kwargs["num_frame_per_block"] = self.num_frame_per_block
