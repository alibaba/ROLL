from typing import Tuple

import numpy as np
import pytest
import torch

from roll.utils.functionals import (
    agg_loss,
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


def test_agg_loss_token_mean_masks_loss_numerator_and_gradients():
    loss_mat = torch.tensor([[1.0, 100.0], [1.0, -50.0]], requires_grad=True)
    loss_mask = torch.tensor([[1, 0], [1, 0]])

    loss = agg_loss(
        loss_mat=loss_mat,
        loss_mask=loss_mask,
        loss_agg_mode="token-mean",
        batch_num_tokens=int(loss_mask.sum().item()),
    )
    loss.backward()

    assert loss.item() == pytest.approx(1.0)
    torch.testing.assert_close(
        loss_mat.grad,
        torch.tensor([[0.5, 0.0], [0.5, 0.0]]),
    )


if __name__ == "__main__":
    pytest.main()
