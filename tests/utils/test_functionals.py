from typing import Tuple

import numpy as np
import pytest
import torch

from roll.utils.functionals import (
    agg_loss,
    compute_approx_kl,
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


def test_compute_approx_kl_k3_clamps_before_exp():
    log_probs = torch.tensor([-1000.0, 1000.0, 0.0])
    log_probs_base = torch.tensor([1000.0, -1000.0, 1.0])

    result = compute_approx_kl(
        log_probs=log_probs,
        log_probs_base=log_probs_base,
        kl_penalty="k3",
    )

    clamped_kl = torch.clamp(log_probs_base - log_probs, min=-20, max=20)
    expected = torch.clamp(torch.exp(clamped_kl) - clamped_kl - 1, min=-10, max=10)
    torch.testing.assert_close(result, expected)
    assert torch.isfinite(result).all()


@pytest.mark.parametrize("loss_agg_mode", ["token-mean", "seq-mean-token-sum", "seq-mean-token-mean"])
def test_agg_loss_ignores_masked_tokens(loss_agg_mode):
    # Token 2 of the first sequence is masked out (e.g. by a train-infer correction filter) and the last
    # token of the second one is padding; their losses must not change the result or receive gradient.
    loss_mask = torch.tensor([[1, 1, 0, 1], [1, 1, 1, 0]])
    loss_mat = torch.tensor([[1.0, 2.0, 100.0, 3.0], [4.0, 5.0, 6.0, -100.0]], requires_grad=True)

    loss = agg_loss(loss_mat, loss_mask, loss_agg_mode, batch_num_tokens=loss_mask.sum(), global_valid_samples=2)
    loss.backward()

    unmasked = loss_mat.detach() * loss_mask
    expected = agg_loss(unmasked, loss_mask, loss_agg_mode, batch_num_tokens=loss_mask.sum(), global_valid_samples=2)
    torch.testing.assert_close(loss.detach(), expected)
    assert (loss_mat.grad[loss_mask == 0] == 0).all()


def test_agg_loss_token_mean_uses_global_token_count():
    # On one data-parallel rank the local sum is divided by the global number of valid tokens.
    loss_mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
    loss_mat = torch.tensor([[1.0, 2.0, 7.0], [3.0, 7.0, 7.0]])
    weights = torch.tensor([1.0, 2.0])

    loss = agg_loss(loss_mat, loss_mask, "token-mean", batch_num_tokens=torch.tensor(6), weights=weights)

    torch.testing.assert_close(loss, torch.tensor((1.0 + 2.0 + 2 * 3.0) / 6))


if __name__ == "__main__":
    pytest.main()
