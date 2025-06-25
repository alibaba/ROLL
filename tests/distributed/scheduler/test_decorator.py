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
from unittest.mock import Mock, PropertyMock

from roll.distributed.scheduler.decorator import (
    dispatch_one_to_all,
    collect_all_to_all,
    dispatch_all_to_all,
    dispatch_dp_mp_compute,
    collect_dp_mp_compute,
    register,
    BIND_WORKER_METHOD_FLAG,
)


def test_dispatch_one_to_all():
    cluster = Mock()
    cluster.world_size = 4

    args = (1, 2)
    kwargs = {"key1": "value1"}

    dispatched_args, dispatched_kwargs = dispatch_one_to_all(cluster, *args, **kwargs)
    print(dispatched_args)
    print(dispatched_kwargs)
    assert dispatched_args == ([1, 1, 1, 1], [2, 2, 2, 2])
    assert dispatched_kwargs == {"key1": ["value1", "value1", "value1", "value1"]}


def test_collect_all_to_all():
    cluster = Mock()
    cluster.world_size = 4

    output = [1, 2, 3, 4]
    collected_output = collect_all_to_all(cluster, output)
    print(collected_output)

    assert collected_output == output


def test_dispatch_all_to_all():
    cluster = Mock()
    cluster.world_size = 4

    args = (["data1", "data2", "data3", "data4"],)
    kwargs = {"key1": ["value1", "value2", "value3", "value4"]}

    dispatched_args, dispatched_kwargs = dispatch_all_to_all(cluster, *args, **kwargs)

    assert dispatched_args == (["data1", "data2", "data3", "data4"],)
    assert dispatched_kwargs == {"key1": ["value1", "value2", "value3", "value4"]}


def test_dispatch_dp_mp_compute():
    cluster = Mock()
    cluster.dp_size = 2
    cluster.world_size = 4

    cluster.get_rank_info = Mock(side_effect=lambda rank: Mock(dp_rank=rank % 2))

    args = (
        ["arg1_chunk0", "arg1_chunk1", "arg1_chunk2", "arg1_chunk3"],
        ["arg2_chunk0", "arg2_chunk1", "arg1_chunk2", "arg1_chunk3"],
    )
    kwargs = {
        "key1": ["val1_chunk0", "val1_chunk1", "val1_chunk2", "val1_chunk3"],
        "key2": ["val2_chunk0", "val2_chunk1", "val1_chunk2", "val1_chunk3"],
    }

    splitted_args, splitted_kwargs = dispatch_dp_mp_compute(cluster, *args, **kwargs)

    assert len(splitted_args) == len(args)
    assert len(splitted_kwargs) == len(kwargs)

    for arg in splitted_args:
        assert isinstance(arg, (tuple, list))
        assert len(arg) == cluster.world_size

    for k, v in splitted_kwargs.items():
        assert isinstance(v, (tuple, list))
        assert len(v) == cluster.world_size


def test_collect_dp_mp_compute():
    cluster = Mock()
    cluster.world_size = 4

    cluster.get_rank_info = Mock(side_effect=lambda rank: Mock(tp_rank=rank % 2))

    output = [[0], [1], [2], [3]]
    collected_output = collect_dp_mp_compute(cluster, output)

    assert collected_output == [0, 2]


def test_register():

    @register()
    def process(a, b):
        return a + b

    registered = process
    flag = getattr(registered, BIND_WORKER_METHOD_FLAG, None)
    print(flag)

    assert flag is not None


if __name__ == "__main__":
    pytest.main()
