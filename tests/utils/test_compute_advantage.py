from collections.abc import Callable

import pytest
import torch

from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.agentic.utils import agentic_compute_advantage
from roll.utils.functionals import compute_advantage


@pytest.mark.parametrize("compute_fn", [compute_advantage, agentic_compute_advantage])
@pytest.mark.parametrize(
    ("response_mask", "expected_advantages"),
    [
        (
            torch.tensor([[0, 0, 0]], dtype=torch.long),
            torch.tensor([[0.0, 0.0, 0.0]]),
        ),
        (
            torch.tensor([[0, 1, 0]], dtype=torch.long),
            torch.tensor([[0.0, 3.0, 0.0]]),
        ),
        (
            torch.tensor([[1, 1, 0]], dtype=torch.long),
            torch.tensor([[-0.70710677, 0.70710677, 0.0]]),
        ),
    ],
)
def test_compute_advantage_handles_short_response_masks(
    compute_fn: Callable[..., DataProto],
    response_mask: torch.Tensor,
    expected_advantages: torch.Tensor,
) -> None:
    """Whitening is skipped below two valid tokens and retained otherwise."""
    data = DataProto.from_dict(
        tensors={
            "token_level_rewards": torch.tensor([[1.0, 3.0, 0.0]]),
        }
    )

    result = compute_fn(
        data=data,
        gamma=0.0,
        lambd=1.0,
        adv_estimator="reinforce",
        whiten_rewards=True,
        whiten_advantages=True,
        response_mask=response_mask,
    )

    assert torch.isfinite(result.batch["token_level_rewards"]).all()
    assert torch.isfinite(result.batch["advantages"]).all()
    torch.testing.assert_close(result.batch["advantages"], expected_advantages)
