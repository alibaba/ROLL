import pytest

from roll.distributed.strategy.vllm_topology import resolve_vllm_mp_topology


def placements(*node_gpu_ranks):
    return [
        {"node_rank": node_rank, "gpu_rank": gpu_rank, "placement_group": f"pg-{node_rank}"}
        for node_rank, gpu_ranks in enumerate(node_gpu_ranks)
        for gpu_rank in gpu_ranks
    ]


def test_external_dp_uses_logical_nodes_on_one_physical_node():
    topology = resolve_vllm_mp_topology(
        worker_rank=1,
        worker_world_size=4,
        data_parallel_size=2,
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
        resource_placements=placements([2, 3]),
    )

    assert topology["deployment_id"] == 0
    assert topology["data_parallel_rank"] == 1
    assert topology["nodes_per_engine"] == 1
    assert topology["nnodes"] == 2
    assert topology["node_rank"] == 1


def test_cross_node_tp_assigns_head_and_headless_node_ranks_per_dp_rank():
    rank_zero = resolve_vllm_mp_topology(
        worker_rank=0,
        worker_world_size=2,
        data_parallel_size=2,
        tensor_parallel_size=16,
        pipeline_parallel_size=1,
        resource_placements=placements(range(8), range(8)),
    )
    rank_one = resolve_vllm_mp_topology(
        worker_rank=1,
        worker_world_size=2,
        data_parallel_size=2,
        tensor_parallel_size=16,
        pipeline_parallel_size=1,
        resource_placements=placements(range(8), range(8)),
    )

    assert rank_zero["nnodes"] == rank_one["nnodes"] == 4
    assert rank_zero["node_rank"] == 0
    assert rank_one["node_rank"] == 2
    assert len(rank_zero["node_groups"]) == len(rank_one["node_groups"]) == 2


def test_cross_node_tp_without_dp_uses_one_head_and_one_headless_node():
    topology = resolve_vllm_mp_topology(
        worker_rank=0,
        worker_world_size=1,
        data_parallel_size=1,
        tensor_parallel_size=16,
        pipeline_parallel_size=1,
        resource_placements=placements(range(8), range(8)),
    )

    assert topology["deployment_id"] == 0
    assert topology["data_parallel_rank"] == 0
    assert topology["nnodes"] == 2
    assert topology["node_rank"] == 0


@pytest.mark.parametrize(
    ("worker_rank", "deployment_id", "data_parallel_rank", "node_rank"),
    [(15, 0, 15, 15), (16, 1, 0, 0), (31, 1, 15, 15)],
)
def test_dp16_tp4_forms_independent_external_dp_deployments(
    worker_rank, deployment_id, data_parallel_rank, node_rank
):
    topology = resolve_vllm_mp_topology(
        worker_rank=worker_rank,
        worker_world_size=32,
        data_parallel_size=16,
        tensor_parallel_size=4,
        pipeline_parallel_size=1,
        resource_placements=placements(range(4)),
    )

    assert topology["deployment_id"] == deployment_id
    assert topology["data_parallel_rank"] == data_parallel_rank
    assert topology["nnodes"] == 16
    assert topology["node_rank"] == node_rank


def test_cross_node_mp_rejects_uneven_local_world_sizes():
    with pytest.raises(ValueError, match="same number of GPUs"):
        resolve_vllm_mp_topology(
            worker_rank=0,
            worker_world_size=1,
            data_parallel_size=1,
            tensor_parallel_size=3,
            pipeline_parallel_size=1,
            resource_placements=placements([0, 1], [0]),
        )


def test_external_dp_must_divide_roll_worker_world_size():
    with pytest.raises(ValueError, match="must be divisible"):
        resolve_vllm_mp_topology(
            worker_rank=0,
            worker_world_size=3,
            data_parallel_size=2,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            resource_placements=placements([0]),
        )
