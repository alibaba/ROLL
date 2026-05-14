from types import SimpleNamespace

import ray

from roll.distributed.scheduler.initialize import _get_ray_start_resource_args, init


@ray.remote
class MyActor:
    def __init__(self, name):
        self.name = name

    def say_hello(self):
        msg = f"Hello from {self.name}! current node: {ray.get_runtime_context().get_node_id()}"
        print(msg)
        return msg


def test_ray_cluster_func():
    init()
    placement_group = ray.util.placement_group(bundles=[{"CPU": 1}, {"CPU": 1}], strategy="STRICT_SPREAD")
    ray.get(placement_group.ready())

    actor1 = MyActor.options(placement_group=placement_group, placement_group_bundle_index=0, num_cpus=1).remote(
        "Actor on Node 1"
    )
    actor2 = MyActor.options(placement_group=placement_group, placement_group_bundle_index=1, num_cpus=1).remote(
        "Actor on Node 2"
    )

    hello_msg1 = ray.get(actor1.say_hello.remote())
    hello_msg2 = ray.get(actor2.say_hello.remote())

    print(hello_msg1)
    print(hello_msg2)


def test_get_ray_start_resource_args_for_gpu(monkeypatch):
    fake_platform = SimpleNamespace(device_type="cuda", ray_device_key="GPU")
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 4))

    monkeypatch.setattr("roll.distributed.scheduler.initialize.current_platform", fake_platform)
    monkeypatch.setattr("roll.distributed.scheduler.initialize.torch", fake_torch)

    assert _get_ray_start_resource_args() == " --num-gpus=4"


def test_get_ray_start_resource_args_for_custom_accelerator(monkeypatch):
    fake_platform = SimpleNamespace(device_type="npu", ray_device_key="NPU")
    fake_torch = SimpleNamespace(npu=SimpleNamespace(device_count=lambda: 8))

    monkeypatch.setattr("roll.distributed.scheduler.initialize.current_platform", fake_platform)
    monkeypatch.setattr("roll.distributed.scheduler.initialize.torch", fake_torch)

    assert _get_ray_start_resource_args() == """ --resources='{"NPU": 8}'"""


if __name__ == "__main__":
    """
    RANK=0 WORLD_SIZE=2 MASTER_ADDR='33.197.137.224' MASTER_PORT=54893 python tests/distributed/scheduler/test_initialize.py
    RANK=1 WORLD_SIZE=2 MASTER_ADDR='33.197.137.224' MASTER_PORT=54893 python tests/distributed/scheduler/test_initialize.py
    """
    test_ray_cluster_func()
