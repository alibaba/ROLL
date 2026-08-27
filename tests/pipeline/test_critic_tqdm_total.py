"""Test that CriticWorker tqdm total matches actual dataloader iteration count.

Regression test for a bug where the CriticWorker's tqdm progress bar used
``ppo_epochs`` in its total calculation, but the dataloader was created with
``epochs=1``. This caused the progress bar to never reach 100% when
``ppo_epochs > 1``.
"""

import torch
import pytest
from tensordict import TensorDict

from roll.distributed.scheduler.protocol import DataProto


@pytest.mark.parametrize("batch_size,mini_batch_size,ppo_epochs", [
    (128, 32, 1),
    (128, 32, 2),
    (128, 32, 4),
    (64, 16, 3),
    (256, 64, 2),
])
def test_critic_tqdm_total_matches_iteration_count(batch_size, mini_batch_size, ppo_epochs):
    """Verify that the tqdm total for CriticWorker equals the actual number of iterations.

    CriticWorker uses ``epochs=1`` in ``make_iterator``, so the tqdm total
    should be ``batch_size // mini_batch_size`` (without multiplying by
    ``ppo_epochs``).
    """
    data = DataProto.from_dict(
        tensors={"values": torch.randn(batch_size, 1)},
    )

    # CriticWorker creates the dataloader with epochs=1
    dataloader = data.make_iterator(
        mini_batch_size=mini_batch_size,
        epochs=1,
        seed=42,
        dataloader_kwargs={"shuffle": True},
    )

    actual_iterations = sum(1 for _ in dataloader)

    # The correct tqdm total for critic (epochs=1)
    correct_total = batch_size // mini_batch_size

    # The old buggy total that used ppo_epochs
    buggy_total = batch_size * ppo_epochs // mini_batch_size

    assert actual_iterations == correct_total, (
        f"Actual iterations ({actual_iterations}) != correct total ({correct_total})"
    )

    # When ppo_epochs > 1, the buggy total would be wrong
    if ppo_epochs > 1:
        assert buggy_total != actual_iterations, (
            f"Buggy total ({buggy_total}) should differ from actual iterations "
            f"when ppo_epochs={ppo_epochs} > 1"
        )


@pytest.mark.parametrize("batch_size,mini_batch_size,ppo_epochs", [
    (128, 32, 1),
    (128, 32, 2),
    (128, 32, 4),
])
def test_actor_tqdm_total_matches_iteration_count(batch_size, mini_batch_size, ppo_epochs):
    """Verify that ActorWorker tqdm total is correct (uses ppo_epochs in both places).

    ActorWorker uses ``epochs=ppo_epochs`` in ``make_iterator``, so the tqdm
    total ``batch_size * ppo_epochs // mini_batch_size`` is correct.
    """
    data = DataProto.from_dict(
        tensors={"values": torch.randn(batch_size, 1)},
    )

    # ActorWorker creates the dataloader with epochs=ppo_epochs
    dataloader = data.make_iterator(
        mini_batch_size=mini_batch_size,
        epochs=ppo_epochs,
        seed=42,
        dataloader_kwargs={"shuffle": True},
    )

    actual_iterations = sum(1 for _ in dataloader)
    expected_total = batch_size * ppo_epochs // mini_batch_size

    assert actual_iterations == expected_total, (
        f"Actual iterations ({actual_iterations}) != expected total ({expected_total})"
    )
