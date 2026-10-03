from roll.pipeline.sft.step_planning import resolve_sft_step_plan


def test_explicit_pipeline_max_steps_overrides_epoch_derived_steps():
    plan = resolve_sft_step_plan(
        configured_max_steps=2,
        dataset_size=12,
        data_parallel_size=4,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
    )

    assert plan.pipeline_steps == 2
    assert plan.worker_max_steps == 8
    # Epoch-derived behavior would produce three steps here; the explicit
    # pipeline cap must win and shorten the run to two updates.
    assert plan.steps_per_epoch == 3
    assert plan.epochs == 1


def test_non_positive_pipeline_max_steps_keeps_epoch_derived_behavior():
    plan = resolve_sft_step_plan(
        configured_max_steps=-1,
        dataset_size=10,
        data_parallel_size=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_train_epochs=2,
    )

    assert plan.pipeline_steps == 10
    assert plan.worker_max_steps == 20
    assert plan.steps_per_epoch == 5
    assert plan.epochs == 2


def test_rejects_dataset_without_a_complete_global_batch():
    import pytest

    with pytest.raises(ValueError, match="complete global batch"):
        resolve_sft_step_plan(
            configured_max_steps=2,
            dataset_size=3,
            data_parallel_size=4,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            num_train_epochs=1,
        )
