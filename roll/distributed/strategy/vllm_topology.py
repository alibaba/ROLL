def resolve_vllm_mp_topology(
    worker_rank,
    worker_world_size,
    data_parallel_size,
    tensor_parallel_size,
    pipeline_parallel_size,
    resource_placements,
):
    model_parallel_size = tensor_parallel_size * pipeline_parallel_size
    if len(resource_placements) != model_parallel_size:
        raise ValueError(
            f"vLLM requires TP*PP={model_parallel_size} placements per ROLL "
            f"InferWorker, got {len(resource_placements)}."
        )
    if worker_world_size % data_parallel_size != 0:
        raise ValueError(
            f"worker_world_size={worker_world_size} must be divisible by "
            f"data_parallel_size={data_parallel_size}."
        )
    if any(placement["gpu_rank"] is None for placement in resource_placements):
        raise ValueError("vLLM mp requires GPU placement groups.")

    placements_by_node = {}
    for placement in resource_placements:
        placements_by_node.setdefault(placement["node_rank"], []).append(placement)
    node_groups = list(placements_by_node.values())
    local_world_sizes = {len(node_group) for node_group in node_groups}
    if len(local_world_sizes) != 1:
        raise ValueError(
            "vLLM mp requires each node in an engine to own the same number "
            f"of GPUs, got {sorted(local_world_sizes)}."
        )

    data_parallel_rank = worker_rank % data_parallel_size
    nodes_per_engine = len(node_groups)
    return {
        "node_groups": node_groups,
        "deployment_id": worker_rank // data_parallel_size,
        "data_parallel_rank": data_parallel_rank,
        "nodes_per_engine": nodes_per_engine,
        "nnodes": data_parallel_size * nodes_per_engine,
        "node_rank": data_parallel_rank * nodes_per_engine,
    }
