# GPU Time-Division Multiplexing Control Guide

The ROLL framework implements GPU time-division multiplexing functionality, which allows flexible sharing of GPU resources between different roles through offload/reload capabilities. This document will provide detailed instructions on how to use this feature.

## Time-Division Multiplexing Overview

In the ROLL framework, different roles (such as actor_train, actor_infer, critic, reference, and rewards) may need to use the same GPU resources. To improve resource utilization, the framework implements GPU time-division multiplexing functionality, which allows model states to be switched between GPU and CPU at different time points.

## Offload/Reload Control Mechanism

### Automatic Control

Taking RLVRPipeline as an example, the framework automatically manages the offload and reload of model states:

```python
# Example in rlvr_pipeline.py
ref_log_probs = self.reference.compute_log_probs(batch, blocking=True)
```

By default, when executing RPC calls to a worker, the framework will first reload the GPU-related state of the current worker onto the GPU, and after execution is completed, it will offload the state to memory.

### Manual Control

You can also manually intervene in model state management by setting `batch.meta_info["is_offload_states"]`:

```python
# Example in rlvr_pipeline.py
self.actor_train.offload_states(blocking=True)
```

When `is_offload_states` is set to `False`, the model state will not be automatically offloaded to CPU after the RPC call is completed, and the model will continue to remain on the GPU.

You can also directly use `worker.offload_states()` and `worker.reload_states()` for more direct control over offload and reload timing.

## Offload Backend Configuration

The offload/reload of training states (FSDP2 and Megatron actor/critic) goes through an offload store, selected by the top-level `offload_backend` option:

```yaml
is_offload_states: true
offload_backend: local  # or local_dedup
```

- `local` (default): each rank offloads to its own pinned CPU memory. Buffers are page-locked with `cudaHostRegister` at exact sizes and recycled across steps, so repeated offload cycles reuse the same pinned memory instead of re-pinning.
- `local_dedup`: tensors replicated across data-parallel ranks (HSDP dense shards, Megatron DDP buffers, and non-distributed master weights) are split into one chunk per rank (~1/world of the host memory) and rebuilt with a single NCCL/RCCL all-gather on reload. Use it when host memory is the bottleneck and ranks of a replication group share the node.

Notes:

- FSDP2 DTensor local shards remain views into the restored flat buffer, including nonzero storage offsets. Async checkpoint staging handles their real local storage without requiring independent GPU shard copies or extra reload collectives.
- With the `local` backend, FSDP2 DCP checkpoints use the existing CPU store buffers directly. The buffers are recycled only after an active asynchronous checkpoint finishes, avoiding a GPU reload and a second model copy in CPU staging. This direct path does not apply to `local_dedup`, whose per-rank chunks require reconstruction.
- Expert-parallel parameters dedup over their own replication group (expert-DP) in Megatron, so EP/ETP settings never corrupt deduplication.
- In FSDP2, MoE expert shards dedup over the expert-DP (`eddp`) mesh dim instead of the dense `ddp` dim, so expert weights also shrink to 1/eddp_size of the host memory when `eddp_size > 1`.
- Pinned host memory is recycled but never released back to the OS during the worker's lifetime; the high-water mark of an offload cycle stays resident. Enable roll debug mode to log store stats (`host_pinned_gb`) via `[OffloadDebug]` entries.

## Usage Example

The following is an example of using offload/reload control in `rlvr_pipeline.py`:

```python
# After the inference phase, manually offload reward model states
if not self.pipeline_config.async_pipeline:
    for reward_cluster in self.rewards.values():
        reward_cluster.offload_states()

# When computing reference model log probs, control whether to offload states
if self.is_lora:
    batch.meta_info["disable_adapter"] = True
    batch.meta_info["is_offload_states"] = False
    ref_log_probs = self.actor_train.compute_log_probs(batch, blocking=True)
else:
    ref_log_probs = self.reference.compute_log_probs(batch, blocking=True)
```

## Context Manager Support

The ROLL framework also provides the `state_offload_manager` context manager to simplify state management:

```python
from roll.utils.context_managers import state_offload_manager

with state_offload_manager(strategy, metrics, metric_infix, is_offload_states=True):
    # Execute operations that require GPU state within this context
    yield
```

This context manager automatically handles:
1. Loading model states to GPU
2. Executing operations
3. Deciding whether to offload states to CPU based on the `is_offload_states` parameter

## Memory Monitoring

The framework also provides memory usage monitoring functionality:

```python
from roll.utils.context_managers import log_gpu_memory_usage

# Record GPU memory usage
log_gpu_memory_usage(head="model_loading", logger=logger, rank=None)
```

## Usage Recommendations

### Native NCCL communicator offload

Set `actor_train.offload_nccl: true` and/or `reference.offload_nccl: true` to
release idle PyTorch/Megatron communicator memory with `ncclCommSuspend` and restore
it with `ncclCommResume`. The default is `false`. ProcessGroups are preserved;
the legacy destroy/recreate implementation has been removed. Cross-role
`model_update/*` groups remain resident because their peers do not offload together.

Each enabled Worker validates the **loaded** NCCL library before loading models:
`ncclGetVersion()` must report `>=2.29.7`, and both Suspend/Resume symbols
must exist. An unsupported runtime raises an error instead of silently disabling
offload. This check does not rely on the installed wheel version or Torch's build
metadata. No requirements or installation commands are changed by this feature;
provide a compatible NCCL in the deployment environment on every node.
Native memory offload requires cuMem allocations to be enabled and NCCL's internal
memory manager not to be disabled. NCCL normally auto-enables cuMem on supported
systems and enables the memory manager by default, so explicit settings are usually
unnecessary. If the deployment environment overrides these defaults, set
`NCCL_CUMEM_ENABLE=1` and `NCCL_DISABLE_MEM_MANAGER=0` before any NCCL communicator
is created.

1. In resource-constrained situations, properly using the offload/reload feature can significantly improve GPU utilization
2. In pipeline implementation, arrange the execution order of different roles to maximize resource utilization efficiency, such as parallel computation of ref/reward models
3. In asynchronous training, properly arrange the execution order of different roles to maximize resource utilization efficiency
