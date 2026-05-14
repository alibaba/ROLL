import os
from types import SimpleNamespace

from ray.runtime_env import RuntimeEnv

os.environ["RAY_DEDUP_LOGS"] = "0"
import pytest
import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from roll.distributed.scheduler.driver_utils import get_driver_world_size
from roll.distributed.scheduler.initialize import init
from roll.distributed.scheduler.resource_manager import ResourceManager


@ray.remote
class TestResourceManagerActor:
    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size

    def say_hello(self):
        msg = f"Hello from {self.world_size}_{self.rank}! get_gpu_ids: {ray.get_gpu_ids()} current GPU: {os.environ['CUDA_VISIBLE_DEVICES']}"
        print(msg)
        return msg


def test_resource_manager():
    init()

    resource_manager = ResourceManager(num_nodes=get_driver_world_size())
    num_gpus_per_worker = 1
    device_mapping = eval("list(range(0,8))")
    print(f"device_mapping: {device_mapping}")
    world_size = len(device_mapping) // num_gpus_per_worker
    pgs = resource_manager.allocate_placement_group(world_size=world_size, device_mapping=device_mapping)

    actor_list = []
    world_size = len(pgs)
    for rank, pg in enumerate(pgs):
        runtime_env = RuntimeEnv(
            env_vars={
                "CUDA_VISIBLE_DEVICES": ",".join(map(str, pg.gpu_ranks)),
                "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
            }
        )
        actor_list.append(
            TestResourceManagerActor.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg.placement_group),
                num_gpus=0.01,
                runtime_env=runtime_env,
            ).remote(rank=rank, world_size=world_size)
        )

    refs = []
    for actor in actor_list:
        refs.append(actor.say_hello.remote())
    res = ray.get(refs)
    print(res)


def test_resource_manager_num_gpus_per_worker_gt_1():
    init()

    resource_manager = ResourceManager(num_nodes=get_driver_world_size())
    num_gpus_per_worker = 2
    device_mapping = eval("list(range(0,8))")
    print(f"device_mapping: {device_mapping}")
    world_size = len(device_mapping) // num_gpus_per_worker
    pgs = resource_manager.allocate_placement_group(world_size=world_size, device_mapping=device_mapping)

    actor_list = []
    world_size = len(pgs)
    for rank, pg in enumerate(pgs):
        runtime_env = RuntimeEnv(
            env_vars={
                "CUDA_VISIBLE_DEVICES": ",".join(map(str, pg.gpu_ranks)),
                "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
            }
        )
        actor_list.append(
            TestResourceManagerActor.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg.placement_group),
                num_gpus=0.01,
                runtime_env=runtime_env,
            ).remote(rank=rank, world_size=world_size)
        )

    refs = []
    for actor in actor_list:
        refs.append(actor.say_hello.remote())
    res = ray.get(refs)
    print(res)


def test_resource_manager_reports_missing_ray_gpu_resources(monkeypatch):
    monkeypatch.setattr(
        "roll.distributed.scheduler.resource_manager.ray.available_resources",
        lambda: {"CPU": 8.0},
    )
    monkeypatch.setattr(
        "roll.distributed.scheduler.resource_manager.ray.nodes",
        lambda: [{"Alive": True, "Resources": {"CPU": 8.0}}],
    )
    monkeypatch.setattr(
        "roll.distributed.scheduler.resource_manager.current_platform",
        SimpleNamespace(ray_device_key="GPU", device_type="cuda", device_name="Iluvatar"),
    )
    monkeypatch.setattr(
        "roll.distributed.scheduler.resource_manager.torch",
        SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 16)),
    )

    with pytest.raises(AssertionError, match="Ray reports 0 GPU resources"):
        ResourceManager(num_gpus_per_node=1, num_nodes=1)


if __name__ == "__main__":
    """
    RANK=0 WORLD_SIZE=2 MASTER_ADDR='33.195.52.67' MASTER_PORT=54893 python tests/distributed/scheduler/test_resource_manager.py
    RANK=1 WORLD_SIZE=2 MASTER_ADDR='33.195.52.67' MASTER_PORT=54893 python tests/distributed/scheduler/test_resource_manager.py
    """
    # test_resource_manager()
    test_resource_manager_num_gpus_per_worker_gt_1()
