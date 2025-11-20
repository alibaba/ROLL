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

1. In resource-constrained situations, properly using the offload/reload feature can significantly improve GPU utilization
2. In pipeline implementation, arrange the execution order of different roles to maximize resource utilization efficiency, such as parallel computation of ref/reward models
3. In asynchronous training, properly arrange the execution order of different roles to maximize resource utilization efficiency