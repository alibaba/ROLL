import numpy as np
import pytest
import torch

from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.agentic.agentic_config import RewardNormalizationConfig
from roll.pipeline.agentic.utils import agentic_reward_norm, step_reward_normalization


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        # deprecated `method` field: unchanged behaviour
        (dict(grouping="traj_group_id", method="mean_std"), ("group", "group")),
        (dict(grouping="traj_group_id", method="mean"), ("group", None)),
        (dict(grouping="batch", method="mean_std"), ("group", "group")),
        (dict(grouping="traj_group_id", method="identity"), (None, None)),
        # new fields only: step rewards are normalized the same way
        (dict(grouping="traj_group_id", norm_mean_type="group", norm_std_type="group"), ("group", "group")),
        (dict(grouping="batch", norm_mean_type="batch", norm_std_type=None), ("group", None)),
    ],
)
def test_step_reward_normalization_follows_episode_config(config, expected):
    step_cfg = step_reward_normalization(RewardNormalizationConfig(**config))

    assert step_cfg.grouping == "state_group_id"
    assert (step_cfg.norm_mean_type, step_cfg.norm_std_type) == expected


def test_step_rewards_are_normalized_within_state_groups():
    episode_cfg = RewardNormalizationConfig(grouping="traj_group_id", norm_mean_type="group", norm_std_type="group")
    step_rewards = torch.tensor([1.0, 0.0, 2.0, 2.0, 4.0])
    batch = DataProto.from_dict(
        tensors={"scores": step_rewards},
        non_tensors={"state_group_id": np.array(["a", "a", "b", "b", "b"], dtype=object)},
    )

    normalized = agentic_reward_norm(batch, reward_normalization=step_reward_normalization(episode_cfg))

    a, b = step_rewards[:2], step_rewards[2:]
    expected = torch.cat([(a - a.mean()) / (a.std() + 1e-6), (b - b.mean()) / (b.std() + 1e-6)])
    torch.testing.assert_close(normalized, expected)
