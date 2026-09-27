"""Deterministic SFT pipeline/worker step planning."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SFTStepPlan:
    pipeline_steps: int
    worker_max_steps: int
    steps_per_epoch: int
    epochs: int


def resolve_sft_step_plan(
    *,
    configured_max_steps: int,
    dataset_size: int,
    data_parallel_size: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    num_train_epochs: float,
) -> SFTStepPlan:
    """Resolve pipeline steps and the pre-DP worker step budget.

    Megatron divides ``training_args.max_steps`` by data parallel size inside
    each worker.  The pipeline, however, consumes global batches, so its
    explicit max-step budget must be multiplied by DP before being handed to
    the worker.  A non-positive pipeline budget retains the historical
    epoch-derived behavior.
    """
    values = {
        "dataset_size": dataset_size,
        "data_parallel_size": data_parallel_size,
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
    }
    for name, value in values.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if num_train_epochs <= 0:
        raise ValueError(f"num_train_epochs must be positive, got {num_train_epochs!r}")

    global_batch = data_parallel_size * per_device_train_batch_size * gradient_accumulation_steps
    steps_per_epoch = max(1, dataset_size // global_batch)
    if configured_max_steps > 0:
        pipeline_steps = configured_max_steps
        epochs = max(1, math.ceil(pipeline_steps / steps_per_epoch))
    else:
        epochs = max(1, math.ceil(num_train_epochs))
        pipeline_steps = epochs * steps_per_epoch
    return SFTStepPlan(
        pipeline_steps=pipeline_steps,
        worker_max_steps=pipeline_steps * data_parallel_size,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
    )
