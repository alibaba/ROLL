import pytest
import torch
import numpy as np
from tensordict import TensorDict

from roll.distributed.scheduler.protocol import DataProto, custom_np_concatenate
from roll.utils.functionals import reduce_metrics


@pytest.fixture
def create_data_proto():
    tensors = {
        "a": torch.randn(5, 2),
        "b": torch.randn(5, 3),
    }
    non_tensors = {
        "c": np.array([{'id': i} for i in range(5)], dtype=object)
    }
    return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)



def test_data_proto_initialization(create_data_proto):
    dp = create_data_proto
    assert len(dp) == 5
    assert "a" in dp.batch.keys()
    assert "c" in dp.non_tensor_batch


def test_data_proto_get_item(create_data_proto):
    dp = create_data_proto[0]
    print(dp)


def test_data_proto_check_consistency(create_data_proto):
    dp = create_data_proto
    dp.check_consistency()


def test_data_proto_select(create_data_proto):
    dp = create_data_proto.select(batch_keys=["a"], non_tensor_batch_keys=["c"])
    assert "a" in dp.batch.keys()
    assert "c" in dp.non_tensor_batch.keys()
    assert len(dp) == 5


def test_data_proto_chunk(create_data_proto):
    chunks = create_data_proto.chunk(5)
    assert len(chunks) == 5


def test_data_proto_concat(create_data_proto):
    list_to_concat = [create_data_proto, create_data_proto]
    concatenated_dp = DataProto.concat(list_to_concat)
    assert len(concatenated_dp) == 10


def test_data_proto_rename(create_data_proto):
    dp = create_data_proto.rename(old_keys="a", new_keys="alpha")
    assert "alpha" in dp.batch.keys()
    assert "a" not in dp.batch.keys()


@pytest.fixture
def sample_proto():
    tensor_data = TensorDict(
        {"group1": torch.tensor([0, 0, 1, 1]), "group2": torch.tensor([10, 20, 20, 30])}, batch_size=[4]
    )

    non_tensor_data = {
        "category": np.array(["A", "B", "A", "B"], dtype=object),
        "flag": np.array(["1", "2", "3", "4"], dtype=object),
    }

    return DataProto(batch=tensor_data, non_tensor_batch=non_tensor_data)


def test_single_non_tensor_key(sample_proto):
    groups = sample_proto.group_by("category")
    expected_categories = {"A", "B"}
    assert set(groups.keys()) == expected_categories

    # 验证 category=A 的分组
    group_a = groups["A"]
    assert len(group_a) == 2


def test_multi_key_grouping(sample_proto):
    groups = sample_proto.group_by(["group1", "category"])


def test_mixed_type_keys(sample_proto):
    groups = sample_proto.group_by(["group2", "flag"])


def test_invalid_key(sample_proto):
    with pytest.raises(KeyError) as excinfo:
        sample_proto.group_by("invalid_key")
    assert "Grouping key 'invalid_key'" in str(excinfo.value)


def test_all_same_group():
    proto = DataProto(
        batch=TensorDict({"key": torch.tensor([5, 5, 5])}, [3]),
        non_tensor_batch={"category": np.array(["1", "1", "1"], dtype=object)},
    )

    groups = proto.group_by(["key", "category"])
    assert len(groups) == 1


def test_np_concat():
    import numpy as np

    array1 = np.random.rand(1, 8, 128, 128, 3).astype(np.float32)
    array2 = np.random.rand(1, 8, 256, 256, 3).astype(np.float32)
    array3 = np.random.rand(1, 8, 256, 256, 3).astype(np.float32)
    val = [array1, array2, array3]
    t = custom_np_concatenate(val)
    print(t.shape)

def test_concat_global_keys_dict():
    """Test dict-valued metrics aggregation across ranks."""
    dp0 = DataProto.from_single_dict(
        {"x": torch.randn(3)},  # shape (3,) keeps batch_size=3
        meta_info={"metrics": {"acc": 0.8, "loss": np.array([0.1, 0.2])}}
    )
    dp1 = DataProto.from_single_dict(
        {"x": torch.randn(2)},
        meta_info={"metrics": {"acc": 0.9, "loss": np.array([0.15, 0.25])}}
    )

    merged = DataProto.concat([dp0, dp1], global_keys={"metrics"})

    # total batch length = 3 + 2 = 5 (flattened)
    assert len(merged) == 5
    np.testing.assert_array_equal(merged.meta_info["metrics"]["acc"], [0.8, 0.9])
    np.testing.assert_array_equal(
        merged.meta_info["metrics"]["loss"],
        [0.1, 0.2, 0.15, 0.25]
    )




def test_concat_global_keys_scalar():
    """Test scalar metrics aggregation."""
    dp0 = DataProto.from_single_dict(
        {"dummy": torch.randn(1)}, meta_info={"lr": 1e-4}
    )
    dp1 = DataProto.from_single_dict(
        {"dummy": torch.randn(1)}, meta_info={"lr": 2e-4}
    )
    merged = DataProto.concat([dp0, dp1], global_keys={"lr"})
    assert merged.meta_info["lr"] == [1e-4, 2e-4]


def test_concat_non_global_remain_rank0():
    """Test non-global keys retain rank-0 value only."""
    dp0 = DataProto.from_single_dict(
        {"dummy": torch.randn(1)}, meta_info={"seed": 42}
    )
    dp1 = DataProto.from_single_dict(
        {"dummy": torch.randn(1)}, meta_info={"seed": 123}
    )
    merged = DataProto.concat([dp0, dp1])
    assert merged.meta_info["seed"] == 42


def test_concat_empty_global_keys():
    """Test no aggregation when global_keys is empty/default."""
    dp0 = DataProto.from_single_dict(
        {"dummy": torch.randn(1)}, meta_info={"metrics": {"a": 1}}
    )
    dp1 = DataProto.from_single_dict(
        {"dummy": torch.randn(1)}, meta_info={"metrics": {"a": 2}}
    )
    merged = DataProto.concat([dp0, dp1])  # no global_keys provided
    np.testing.assert_array_equal(merged.meta_info["metrics"]["a"], [1, 2])

def test_concat_global_keys_dict_missing_subkeys():
    """
    Test dict-valued metrics aggregation when some ranks have missing sub-keys.
    Ensures missing keys are skipped (not filled with None) and calculations like np.mean work.
    """
    # Rank 0: has both acc and loss
    dp0 = DataProto.from_single_dict(
        {"x": torch.randn(2)},
        meta_info={"metrics": {"acc": 0.8, "loss": np.array([0.1, 0.2])}}
    )
    # Rank 1: missing acc
    dp1 = DataProto.from_single_dict(
        {"x": torch.randn(1)},
        meta_info={"metrics": {"loss": np.array([0.15])}}
    )
    # Rank 2: has acc and extra precision, missing loss
    dp2 = DataProto.from_single_dict(
        {"x": torch.randn(2)},
        meta_info={"metrics": {"acc": 0.9, "precision": np.array([0.7, 0.75])}}
    )

    # Merge all ranks, aggregate metrics across them
    merged = DataProto.concat([dp0, dp1, dp2], global_keys={"metrics"})
    metrics = merged.meta_info["metrics"]

    # 1. Total batch length = 2 + 1 + 2 = 5
    assert len(merged) == 5

    # 2. Check merged keys
    assert set(metrics.keys()) == {"acc", "loss", "precision"}

    # 3. Validate individual metric arrays
    expected_loss = np.array([0.1, 0.2, 0.15])  # Rank 0 (2x) + Rank 1 (1x)
    expected_acc = np.array([0.8, 0.9])         # Rank 0 (1x) + Rank 2 (1x)
    expected_precision = np.array([0.7, 0.75])  # Rank 2 (2x)

    np.testing.assert_array_equal(metrics["loss"], expected_loss)
    np.testing.assert_array_equal(metrics["acc"], expected_acc)
    np.testing.assert_array_equal(metrics["precision"], expected_precision)

    # 4. Reduce metrics (using project function) and check mean calculation
    reduced = reduce_metrics(metrics)
    assert np.isclose(reduced["loss"], expected_loss.mean())
    assert np.isclose(reduced["acc"], expected_acc.mean())
    assert np.isclose(reduced["precision"], expected_precision.mean())

def test_clone_independence(create_data_proto):
    """Test that clone() returns an independent copy with the same content."""
    dp = create_data_proto
    dp_clone = dp.clone()

    # 1. Not the same object
    assert dp is not dp_clone

    # 2. Batch contents equal
    for key in dp.batch.keys():
        assert torch.equal(dp.batch[key], dp_clone.batch[key])

    # 3. Non-tensor contents equal
    for key in dp.non_tensor_batch.keys():
        np.testing.assert_array_equal(dp.non_tensor_batch[key], dp_clone.non_tensor_batch[key])

    # 4. Meta-info equal but mutable objects are independent
    assert dp.meta_info == dp_clone.meta_info
    dp_clone.meta_info["new_key"] = "value"
    assert "new_key" not in dp.meta_info

    # 5. Changing clone's tensor doesn't affect original
    dp_clone.batch["a"][0] += 100.0
    assert not torch.equal(dp.batch["a"], dp_clone.batch["a"])



if __name__ == "__main__":
    pytest.main()
