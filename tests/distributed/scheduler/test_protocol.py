# Copyright (c) 2025, ALIBABA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pytest
import torch
import numpy as np
from tensordict import TensorDict

from roll.distributed.scheduler.protocol import DataProto, custom_np_concatenate


@pytest.fixture
def create_data_proto():
    tensors = {
        "a": torch.randn(5, 2),
        "b": torch.randn(5, 3),
    }
    non_tensors = {"c": np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], dtype=object)}
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


if __name__ == "__main__":
    pytest.main()
