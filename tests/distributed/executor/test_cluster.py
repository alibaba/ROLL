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
import copy
from typing import Any, List

import pytest
import ray

from roll.configs.worker_config import WorkerConfig
from roll.distributed.executor.cluster import Cluster
from roll.distributed.executor.worker import Worker, RankInfo
from roll.distributed.scheduler.decorator import register, Dispatch
from roll.distributed.scheduler.resource_manager import ResourceManager


@ray.remote
class TestWorker(Worker):

    def __init__(self, worker_config: WorkerConfig):
        super().__init__(worker_config=worker_config)

        self.value = self.rank

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def add(self, x):
        self.value = self.value + x
        return self.value


@ray.remote
class TestDPWorker(Worker):

    def __init__(self, worker_config: WorkerConfig):
        super().__init__(worker_config)
        self.tp_size = 2
        self.pp_size = 2
        self.dp_size = self.world_size // self.tp_size // self.pp_size
        self.tp_rank = self.rank % self.tp_size
        self.pp_rank = self.rank // (self.dp_size * self.tp_size)
        self.dp_rank = (self.rank % (self.dp_size * self.tp_size)) // self.tp_size

        self.rank_info = RankInfo(
            world_size=self.world_size,
            tp_size=self.tp_size,
            dp_size=self.dp_size,
            pp_size=self.pp_size,
            rank=self.rank,
            tp_rank=self.tp_rank,
            dp_rank=self.dp_rank,
            pp_rank=self.pp_rank,
        )

        self.value = [self.dp_rank, self.tp_rank, self.pp_rank]

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE)
    def add(self, x: List):
        res = []
        for val in x:
            v = copy.deepcopy(self.value)
            v.append(val)
            res.append(v)
        return res


def test_cluster_run():
    ray.init(log_to_driver=True)

    resource_manager = ResourceManager()

    test_worker_config = WorkerConfig(name="test_worker", world_size=2)
    test_cluster: Any = Cluster(
        name=test_worker_config.name,
        resource_manager=resource_manager,
        worker_cls=TestWorker,
        worker_config=test_worker_config,
    )
    x = 1
    res = test_cluster.add(x=x)
    print(res)
    assert res == [1, 2]


def test_cluster_dp_mp_compute():
    ray.init(log_to_driver=True)

    resource_manager = ResourceManager()

    test_worker_config = WorkerConfig(name="test_worker", world_size=8)
    test_cluster: Any = Cluster(
        name=test_worker_config.name,
        resource_manager=resource_manager,
        worker_cls=TestDPWorker,
        worker_config=test_worker_config,
    )
    x = [1, 2, 3, 4, 5, 6, 7, 8]
    res = test_cluster.add(x=x)
    print(res)
    assert res == [
        [0, 0, 1, 1],
        [0, 0, 1, 2],
        [0, 0, 1, 3],
        [0, 0, 1, 4],
        [1, 0, 1, 5],
        [1, 0, 1, 6],
        [1, 0, 1, 7],
        [1, 0, 1, 8],
    ]


if __name__ == "__main__":
    pytest.main()
