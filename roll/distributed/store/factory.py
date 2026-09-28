"""Backend selection: pick an offload store implementation from worker config."""
from roll.distributed.store.base import OffloadBackend

OFFLOAD_BACKENDS = ("local", "local_dedup")


def get_offload_backend(worker, dp_rank: int = 0, dp_group=None) -> OffloadBackend:
    """
    Get the offload backend based on worker config.

    Backends:
        local:       per-rank pinned CPU memory, no dedup. Exact-size pinned
                     via HostMemPool's cudaHostRegister (never torch's
                     power-of-two pinned allocator), direct DMA both ways.
        local_dedup: each rank keeps a 1/world pinned chunk, NCCL all-gather
                     on get (dedup savings without a store service).

    Args:
        worker: Worker instance with pipeline_config.
        dp_rank: Data-parallel rank (used by dedup-aware backends).
        dp_group: Torch distributed process group for DP dedup/all-gather.
    """
    pipeline_config = getattr(worker, 'pipeline_config', None)
    backend = getattr(pipeline_config, 'offload_backend', None) or "local"
    if backend not in OFFLOAD_BACKENDS:
        raise ValueError(f"Unknown offload_backend '{backend}', expected one of {OFFLOAD_BACKENDS}")
    if backend == "local_dedup":
        from roll.distributed.store.local.backend_with_dedup import CPUDedupOffloadBackend
        return CPUDedupOffloadBackend(dp_rank=dp_rank, dp_group=dp_group)
    from roll.distributed.store.local.backend import CPUOffloadBackend
    return CPUOffloadBackend()
