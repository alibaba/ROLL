import pytest
import torch

from roll.utils import functionals


class _Batch:
    def __init__(self, attention_mask: torch.Tensor) -> None:
        self.batch = {
            "attention_mask": attention_mask,
            "sample_id": torch.arange(attention_mask.shape[0]),
        }

    def reorder(self, indices: torch.Tensor) -> None:
        self.batch = {key: value[indices] for key, value in self.batch.items()}


@pytest.mark.parametrize("keep_minibatch", [False, True])
def test_batch_balance_uses_host_lists_without_changing_results(
    monkeypatch: pytest.MonkeyPatch, keep_minibatch: bool
) -> None:
    lengths = [3, 8, 5, 5, 2, 7, 4, 6]
    attention_mask = torch.arange(max(lengths)).unsqueeze(0) < torch.tensor(lengths).unsqueeze(1)
    batch = _Batch(attention_mask)

    workloads = [24576 * seq_len + seq_len * seq_len for seq_len in lengths]
    minibatch_size = 4 if keep_minibatch else len(lengths)
    expected_partitions = [[] for _ in range(2)]
    for offset in range(0, len(lengths), minibatch_size):
        minibatch_partitions = functionals.get_seqlen_balanced_partitions(
            workloads[offset : offset + minibatch_size], 2, equal_size=True
        )
        for index, partition in enumerate(minibatch_partitions):
            expected_partitions[index].extend(item + offset for item in partition)
    for index, partition in enumerate(expected_partitions):
        partition.sort(key=lambda item: (workloads[item], item))
        expected_partitions[index] = partition[::2] + partition[1::2][::-1]
    expected_order = [item for partition in expected_partitions for item in partition]
    expected_metrics = functionals.log_seqlen_unbalance(lengths, expected_partitions, "global_seqlen")

    original_partition = functionals.get_seqlen_balanced_partitions

    def check_host_list(seqlen_list: list[int], k_partitions: int, equal_size: bool = False) -> list[list[int]]:
        assert isinstance(seqlen_list, list)
        return original_partition(seqlen_list, k_partitions, equal_size)

    monkeypatch.setattr(functionals, "get_seqlen_balanced_partitions", check_host_list)

    metrics = functionals.batch_balance(batch, dp_size=2, minibatch_size=minibatch_size, keep_minibatch=keep_minibatch)

    assert batch.batch["sample_id"].tolist() == expected_order
    assert metrics == expected_metrics
