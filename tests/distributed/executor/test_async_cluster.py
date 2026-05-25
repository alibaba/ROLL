from typing import Any

import pytest
import ray
import asyncio

from roll.configs.worker_config import WorkerConfig
from roll.distributed.executor.cluster import Cluster
from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.decorator import register, Dispatch
from roll.distributed.scheduler.resource_manager import ResourceManager


@ray.remote
class TestWorker(Worker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    async def test_one_to_all(self):
        return 1

    @register(dispatch_mode=Dispatch.ONE_TO_ALL_ONE)
    async def test_one_to_all_one(self):
        return 1

    @register(dispatch_mode=Dispatch.ALL_TO_ALL)
    async def test_all_to_all(self):
        return 1

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE)
    async def test_dp_mp_compute(self):
        return 1

    @register(dispatch_mode=Dispatch.DP_MP_DISPATCH_FIRST)
    async def test_dp_mp_dispatch_first(self):
        return 1

async def _assert_async_cluster_calls(cluster):
    ret = await asyncio.gather(*cluster.test_one_to_all(blocking=False))
    assert ret == [1, 1]

    ret = await asyncio.gather(*[ref.obj_ref for ref in cluster.test_one_to_all_one(blocking=False)])
    assert ret == [1, 1]

    ret = await asyncio.gather(*cluster.test_all_to_all(blocking=False))
    assert ret == [1, 1]

    ret = await asyncio.gather(*[ref.obj_ref for ref in cluster.test_dp_mp_compute(blocking=False)])
    assert ret == [1, 1]

    ret = await asyncio.gather(*[ref.obj_ref for ref in cluster.test_dp_mp_dispatch_first(blocking=False)])
    assert ret == [1, 1]

def test_async_cluster():
    ray.shutdown()
    ray.init()
    try:
        resource_manager = ResourceManager(0, 1)
        worker_config = WorkerConfig(name="test_worker", world_size=2)

        cluster: Any = Cluster(
            name=worker_config.name,
            resource_manager=resource_manager,
            worker_cls=TestWorker,
            worker_config=worker_config,
        )

        asyncio.run(_assert_async_cluster_calls(cluster))
    finally:
        ray.shutdown()

if __name__ == "__main__":
    test_async_cluster()
