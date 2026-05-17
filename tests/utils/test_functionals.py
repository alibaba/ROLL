from typing import Tuple

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from roll.distributed.scheduler.protocol import DataProto
from roll.utils.functionals import (
    compute_advantage,
    divide_by_chunk_size,
    pad_to_length,
    traverse_obj,
)


def visitor(obj: object, path: Tuple):
    if torch.is_tensor(obj):
        print(f"Tensor found: {obj}, shape: {obj.shape}, dtype: {obj.dtype}")
        return True
    return False


def test_traverse_obj():
    class CustomObject2:
        def __init__(self):
            self.attr1 = torch.tensor([1, 2, 3])
            self.attr2 = {
                "nested_key1": torch.tensor([[1, 2], [3, 4]]),
                "nested_key2": [torch.tensor(5), np.array([6, 7])],
            }

    class CustomObject:
        def __init__(self):
            self.attr1 = torch.tensor([1, 2, 3])
            self.attr2 = {
                "nested_key1": torch.tensor([[1, 2], [3, 4]]),
                "nested_key2": [torch.tensor(5), np.array([6, 7])],
            }
            self.attr3 = CustomObject2()

    custom_obj = CustomObject()

    traverse_obj(custom_obj, visitor, path=(str(custom_obj),))


def test_divide_by_chunk_size_valid():
    array = np.arange(51)
    chunk_sizes = [7, 7, 7, 7, 7, 7, 7, 2]
    result = divide_by_chunk_size(array, chunk_sizes)

    assert len(result) == len(chunk_sizes)
    assert all(isinstance(chunk, np.ndarray) for chunk in result)
    assert [len(chunk) for chunk in result] == chunk_sizes


def test_pad_to_length():
    tensor = torch.tensor([[1, 2, 3, 4, 5, 6, 7], [4, 5, 6, 1, 2, 3, 7]])
    length = 5
    pad_value = 0

    padded_tensor = pad_to_length(tensor, length, pad_value, dim=-1)
    print(padded_tensor)


def test_compute_advantage_masks_values_before_gae_bootstrap():
    response_mask = torch.tensor([[1.0, 1.0, 0.0]])
    token_level_rewards = torch.tensor([[0.0, 1.0, 0.0]])
    values = torch.tensor([[0.0, 0.0, 100.0]])
    data = DataProto(
        batch=TensorDict(
            {
                "response_mask": response_mask.clone(),
                "token_level_rewards": token_level_rewards.clone(),
                "values": values.clone(),
            },
            batch_size=[1],
        ),
        meta_info={},
    )

    compute_advantage(
        data=data,
        gamma=torch.tensor(1.0),
        lambd=torch.tensor(0.95),
        adv_estimator="gae",
        response_mask=response_mask,
    )

    expected = torch.tensor([[0.95, 1.0, 0.0]])
    torch.testing.assert_close(data.batch["values"], values * response_mask)
    torch.testing.assert_close(data.batch["advantages"], expected)
    torch.testing.assert_close(data.batch["returns"], expected)


if __name__ == "__main__":
    pytest.main()
