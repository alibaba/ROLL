from types import SimpleNamespace

import pytest
import torch

from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.base_worker import CriticWorker


def _critic(value_clip):
    worker = object.__new__(CriticWorker)  # loss_func only needs the pipeline config
    worker.pipeline_config = SimpleNamespace(value_clip=value_clip)
    return worker


@pytest.mark.parametrize("value_clip", [None, 0.2])
def test_critic_clipfrac_is_a_fraction(value_clip):
    # response_mask covers the full sequence; loss_func drops the first position
    data = DataProto.from_dict(
        tensors={
            "response_mask": torch.tensor([[1, 1, 1, 0]]),
            "values": torch.tensor([[0.0, 0.0, 0.0]]),
            "returns": torch.tensor([[2.0, 2.0, 2.0]]),
        }
    )
    # predicted values after forward_func_values: output_tensor[:, :-1]
    output_tensor = torch.tensor([[[1.0], [1.0], [1.0], [1.0]]])

    loss, metrics = _critic(value_clip).loss_func(data, output_tensor)

    torch.testing.assert_close(loss, torch.tensor(0.5 * (1.0 if value_clip is None else 1.8**2)))
    assert 0.0 <= metrics["critic/clipfrac"] <= 1.0
    # with value_clip=0.2 the prediction 1.0 is clipped to 0.2 for every valid token
    assert metrics["critic/clipfrac"] == (0.0 if value_clip is None else 1.0)
