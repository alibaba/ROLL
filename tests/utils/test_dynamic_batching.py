import pytest
import torch

from roll.distributed.scheduler.protocol import DataProto
from roll.utils.dynamic_batching import (
    dynamic_batching_shard,
    make_micro_batch_iter_for_dynamic_batching,
    make_mini_batch_iter_for_dynamic_batching,
)
from roll.utils.functionals import batch_balance


def test_dynamic_batching():
    dp_size = 2
    num_seq = 6
    max_seq_len = 20
    seqs_len = [2, 4, 7, 6, 3, 4]
    input_ids = torch.arange(num_seq).unsqueeze(1).expand(num_seq, max_seq_len)
    attention_mask = (torch.arange(max_seq_len) < torch.tensor(seqs_len)[:, None]).int()
    data = DataProto.from_dict(
        tensors={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
    )
    max_tokens_per_microbatch = 20
    sequence_length_round = 2

    # test dynamic_batching_shard
    data, _ = dynamic_batching_shard(data, dp_size, max_tokens_per_microbatch, sequence_length_round)
    assert data.meta_info["global_micro_batch_indices"] == [[0, 2], [2, 3]]
    assert data.meta_info["global_micro_batch_lengths"] == [4, 8]

    # test make_mini_batch_iter_for_dynamic_batching
    data_dp0 = data.slice(0, num_seq // dp_size)
    mini_batch_iter = make_mini_batch_iter_for_dynamic_batching(data_dp0, 1, 1)
    mini_batch0 = next(mini_batch_iter)
    assert mini_batch0.meta_info["micro_batch_indices"] == [[0, 2]]
    assert data_dp0.meta_info["global_micro_batch_indices"] == [[0, 2], [2, 3]]
    assert data_dp0.meta_info["global_micro_batch_lengths"] == [4, 8]
    assert data_dp0.meta_info["micro_batch_lengths"] == [4, 8]

    # test make_mini_batch_iter_for_dynamic_batching
    micro_batch_iter = make_micro_batch_iter_for_dynamic_batching(mini_batch0)
    micro_batch0 = next(micro_batch_iter)
    assert tuple(micro_batch0.batch["input_ids"].shape) == (2, 4)


def test_dynamic_batching_with_vpp():
    torch.manual_seed(42)
    dp_size = 4
    num_seq = 256
    max_seq_len = 8192
    seqs_len = torch.randint(low=128, high=8192, size=(256,)).tolist()
    input_ids = torch.arange(num_seq).unsqueeze(1).expand(num_seq, max_seq_len)
    attention_mask = (torch.arange(max_seq_len) < torch.tensor(seqs_len)[:, None]).int()
    data = DataProto.from_dict(
        tensors={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
    )
    max_tokens_per_microbatch = 8192
    sequence_length_round = 128

    # test dynamic_batching_shard
    pipeline_model_parallel_size = 4
    virtual_pipeline_model_size = 2
    data, _ = dynamic_batching_shard(data, dp_size, max_tokens_per_microbatch, sequence_length_round,
                                     pipeline_model_parallel_size=pipeline_model_parallel_size,
                                     virtual_pipeline_model_parallel_size=virtual_pipeline_model_size)
    assert data.meta_info["global_micro_batch_indices"].__len__() % pipeline_model_parallel_size == 0
    assert data.meta_info["global_micro_batch_lengths"].__len__() == data.meta_info["global_micro_batch_indices"].__len__()


def _make_dynamic_batch(max_tokens_per_microbatch):
    num_samples, sequence_length = 8, 8
    return dynamic_batching_shard(
        DataProto.from_dict(
            tensors={
                "input_ids": torch.arange(num_samples * sequence_length).reshape(num_samples, sequence_length),
                "attention_mask": torch.ones((num_samples, sequence_length), dtype=torch.long),
            }
        ),
        dp_size=1,
        max_tokens_per_microbatch=max_tokens_per_microbatch,
        sequence_length_round=sequence_length,
    )[0]


def test_issue_442_keep_mini_batch_with_more_dynamic_micro_batches():
    data = _make_dynamic_batch(max_tokens_per_microbatch=8)

    mini_batches = list(
        make_mini_batch_iter_for_dynamic_batching(
            data,
            epochs=3,
            ga_steps=2,
            keep_mini_batch=True,
            mini_batch_size=4,
        )
    )

    assert len(mini_batches) == 6
    assert all(len(mini_batch) == 4 for mini_batch in mini_batches)
    assert all(mini_batch.meta_info["num_micro_batchs"] == 4 for mini_batch in mini_batches)
    assert all(
        mini_batch.meta_info["micro_batch_indices"] == [[0, 1], [1, 2], [2, 3], [3, 4]]
        for mini_batch in mini_batches
    )


def test_issue_442_keep_mini_batch_with_fewer_dynamic_micro_batches():
    data = _make_dynamic_batch(max_tokens_per_microbatch=64)
    source_ranges = [range(start, end) for start, end in data.meta_info["global_micro_batch_indices"]]

    mini_batches = list(
        make_mini_batch_iter_for_dynamic_batching(
            data,
            epochs=3,
            ga_steps=2,
            keep_mini_batch=True,
            mini_batch_size=4,
        )
    )

    assert len(mini_batches) == 6
    assert all(len(mini_batch) == 4 for mini_batch in mini_batches)
    assert [
        mini_batch.meta_info["micro_batch_indices"] for mini_batch in mini_batches[:2]
    ] == [[[0, 4]], [[0, 4]]]
    assert [
        mini_batch.batch["input_ids"][:, 0].tolist() for mini_batch in mini_batches[:2]
    ] == [list(range(0, 32, 8)), list(range(32, 64, 8))]
    assert [sample for sample_range in source_ranges for sample in sample_range] == list(range(8))


def test_issue_442_dynamic_mini_batch_metadata_is_independent():
    num_samples, sequence_length = 8, 8
    data = DataProto.from_dict(
        tensors={
            "input_ids": torch.arange(num_samples * sequence_length).reshape(num_samples, sequence_length),
            "attention_mask": torch.ones((num_samples, sequence_length), dtype=torch.long),
        },
        meta_info={
            "global_micro_batch_indices": [[0, 2], [2, 4], [4, 5], [5, 8]],
            "global_micro_batch_lengths": [8, 8, 8, 8],
        },
    )

    mini_batches = list(
        make_mini_batch_iter_for_dynamic_batching(
            data,
            epochs=1,
            ga_steps=2,
            keep_mini_batch=True,
            mini_batch_size=4,
        )
    )

    assert [mini_batch.meta_info["micro_batch_indices"] for mini_batch in mini_batches] == [
        [[0, 2], [2, 4]],
        [[0, 1], [1, 4]],
    ]
    assert [mini_batch.meta_info["micro_batch_lengths"] for mini_batch in mini_batches] == [
        [8, 8],
        [8, 8],
    ]
    assert mini_batches[0].meta_info is not mini_batches[1].meta_info
    mini_batches[0].meta_info["micro_batch_indices"][0][0] = 99
    assert mini_batches[1].meta_info["micro_batch_indices"] == [[0, 1], [1, 4]]
    assert data.meta_info["global_micro_batch_indices"] == [[0, 2], [2, 4], [4, 5], [5, 8]]
    assert data.meta_info["global_micro_batch_lengths"] == [8, 8, 8, 8]
    assert "micro_batch_indices" not in data.meta_info
    assert "micro_batch_lengths" not in data.meta_info


def test_issue_442_keep_mini_batch_splits_ranges_at_every_static_boundary():
    """A dynamic range may span more than one static logical mini-batch."""
    data = DataProto.from_dict(
        tensors={
            "input_ids": torch.arange(12).reshape(6, 2),
            "attention_mask": torch.ones((6, 2), dtype=torch.long),
        },
        meta_info={
            "global_micro_batch_indices": [[0, 6]],
            "global_micro_batch_lengths": [2],
        },
    )

    mini_batches = list(
        make_mini_batch_iter_for_dynamic_batching(
            data,
            epochs=1,
            ga_steps=8,
            keep_mini_batch=True,
            mini_batch_size=2,
        )
    )

    assert [len(mini_batch) for mini_batch in mini_batches] == [2, 2, 2]
    assert [mini_batch.meta_info["micro_batch_indices"] for mini_batch in mini_batches] == [
        [[0, 2]],
        [[0, 2]],
        [[0, 2]],
    ]
    assert [mini_batch.meta_info["micro_batch_lengths"] for mini_batch in mini_batches] == [[2], [2], [2]]


def test_issue_442_keep_mini_batch_splits_each_group_for_vpp():
    data = DataProto.from_dict(
        tensors={
            "input_ids": torch.arange(16).reshape(8, 2),
            "attention_mask": torch.ones((8, 2), dtype=torch.long),
        },
        meta_info={
            "global_micro_batch_indices": [[0, 4], [4, 8]],
            "global_micro_batch_lengths": [2, 2],
        },
    )

    mini_batches = list(
        make_mini_batch_iter_for_dynamic_batching(
            data,
            epochs=1,
            ga_steps=2,
            keep_mini_batch=True,
            mini_batch_size=4,
            pipeline_model_parallel_size=4,
            virtual_pipeline_model_parallel_size=2,
        )
    )

    assert len(mini_batches) == 2
    assert all(mini_batch.meta_info["num_micro_batchs"] == 4 for mini_batch in mini_batches)
    assert all(
        mini_batch.meta_info["micro_batch_indices"] == [[0, 1], [1, 2], [2, 3], [3, 4]]
        for mini_batch in mini_batches
    )
    assert all(mini_batch.meta_info["micro_batch_lengths"] == [2, 2, 2, 2] for mini_batch in mini_batches)


def test_issue_442_keep_mini_batch_rejects_undersized_vpp_group():
    data = DataProto.from_dict(
        tensors={
            "input_ids": torch.arange(4).reshape(2, 2),
            "attention_mask": torch.ones((2, 2), dtype=torch.long),
        },
        meta_info={
            "global_micro_batch_indices": [[0, 1], [1, 2]],
            "global_micro_batch_lengths": [2, 2],
        },
    )

    with pytest.raises(ValueError, match="cannot be split into a multiple"):
        list(
            make_mini_batch_iter_for_dynamic_batching(
                data,
                epochs=1,
                ga_steps=2,
                keep_mini_batch=True,
                mini_batch_size=2,
                pipeline_model_parallel_size=4,
                virtual_pipeline_model_parallel_size=2,
            )
        )


def test_issue_442_keep_mini_batch_handles_large_dynamic_range_with_vpp():
    """The shard-level VPP padding must tolerate a range that needs repeats."""
    data = _make_dynamic_batch(max_tokens_per_microbatch=64)
    data, _ = dynamic_batching_shard(
        data,
        dp_size=1,
        max_tokens_per_microbatch=64,
        sequence_length_round=8,
        pipeline_model_parallel_size=4,
        virtual_pipeline_model_parallel_size=2,
    )

    assert data.meta_info["global_micro_batch_indices"] == [[0, 2], [2, 4], [4, 6], [6, 8]]
    mini_batches = list(
        make_mini_batch_iter_for_dynamic_batching(
            data,
            epochs=1,
            ga_steps=2,
            keep_mini_batch=True,
            mini_batch_size=4,
            pipeline_model_parallel_size=4,
            virtual_pipeline_model_parallel_size=2,
        )
    )

    assert len(mini_batches) == 2
    assert all(mini_batch.meta_info["num_micro_batchs"] == 4 for mini_batch in mini_batches)
    assert all(
        mini_batch.meta_info["micro_batch_indices"] == [[0, 1], [1, 2], [2, 3], [3, 4]]
        for mini_batch in mini_batches
    )


@pytest.mark.parametrize(
    ("ranges", "message"),
    [
        ([[0, 2], [3, 4]], "contiguous"),
        ([[0, 2]], "complete local batch"),
        ([[0, 3], [2, 4]], "contiguous"),
    ],
)
def test_issue_442_keep_mini_batch_rejects_non_covering_layouts(ranges, message):
    data = DataProto.from_dict(
        tensors={
            "input_ids": torch.zeros((4, 2), dtype=torch.long),
            "attention_mask": torch.ones((4, 2), dtype=torch.long),
        },
        meta_info={
            "global_micro_batch_indices": ranges,
            "global_micro_batch_lengths": [2] * len(ranges),
        },
    )

    with pytest.raises(AssertionError, match=message):
        list(
            make_mini_batch_iter_for_dynamic_batching(
                data,
                epochs=1,
                ga_steps=2,
                keep_mini_batch=True,
                mini_batch_size=2,
            )
        )


@pytest.mark.parametrize("minibatch_size", [0, 3, 8])
def test_issue_442_batch_balance_rejects_incomplete_static_batch(minibatch_size):
    data = DataProto.from_dict(
        tensors={
            "attention_mask": torch.ones((4, 2), dtype=torch.long),
        }
    )

    with pytest.raises(ValueError, match="minibatch_size|divisible"):
        batch_balance(data, dp_size=2, minibatch_size=minibatch_size, keep_minibatch=True)


if __name__ == "__main__":
    # test_dynamic_batching()
    test_dynamic_batching_with_vpp()
