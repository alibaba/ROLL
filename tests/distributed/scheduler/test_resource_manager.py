from types import SimpleNamespace

import ray

from roll.distributed.scheduler.resource_manager import ResourceManager


def _make_resource_manager(num_nodes=2, num_gpus_per_node=4):
    resource_manager = ResourceManager.__new__(ResourceManager)
    resource_manager.num_nodes = num_nodes
    resource_manager.gpu_per_node = num_gpus_per_node
    resource_manager.node2pg = {
        node_rank: f"pg-{node_rank}"
        for node_rank in range(num_nodes)
    }
    return resource_manager


def _mock_runtime_context(monkeypatch):
    monkeypatch.setattr(
        ray,
        "get_runtime_context",
        lambda: SimpleNamespace(gcs_address="127.0.0.1:6379"),
    )


def _placement(node_rank, gpu_rank, placement_group):
    return {
        "node_rank": node_rank,
        "gpu_rank": gpu_rank,
        "placement_group": placement_group,
        "ray_address": "127.0.0.1:6379",
    }


def test_allocate_placement_group_single_device_per_worker(monkeypatch):
    _mock_runtime_context(monkeypatch)
    resource_manager = _make_resource_manager()

    allocated = resource_manager.allocate_placement_group(
        world_size=8,
        device_mapping=list(range(8)),
    )

    assert len(allocated) == 8
    assert allocated[0] == [_placement(node_rank=0, gpu_rank=0, placement_group="pg-0")]
    assert allocated[3] == [_placement(node_rank=0, gpu_rank=3, placement_group="pg-0")]
    assert allocated[4] == [_placement(node_rank=1, gpu_rank=0, placement_group="pg-1")]
    assert allocated[7] == [_placement(node_rank=1, gpu_rank=3, placement_group="pg-1")]


def test_allocate_placement_group_multi_device_per_worker(monkeypatch):
    _mock_runtime_context(monkeypatch)
    resource_manager = _make_resource_manager()

    allocated = resource_manager.allocate_placement_group(
        world_size=4,
        device_mapping=list(range(8)),
    )

    assert len(allocated) == 4
    assert allocated[0] == [
        _placement(node_rank=0, gpu_rank=0, placement_group="pg-0"),
        _placement(node_rank=0, gpu_rank=1, placement_group="pg-0"),
    ]
    assert allocated[2] == [
        _placement(node_rank=1, gpu_rank=0, placement_group="pg-1"),
        _placement(node_rank=1, gpu_rank=1, placement_group="pg-1"),
    ]


def test_allocate_placement_group_without_device_mapping_spreads_workers(monkeypatch):
    _mock_runtime_context(monkeypatch)
    resource_manager = _make_resource_manager(num_nodes=2)

    allocated = resource_manager.allocate_placement_group(world_size=3)

    assert allocated == [
        [_placement(node_rank=0, gpu_rank=None, placement_group="pg-0")],
        [_placement(node_rank=1, gpu_rank=None, placement_group="pg-1")],
        [_placement(node_rank=0, gpu_rank=None, placement_group="pg-0")],
    ]
