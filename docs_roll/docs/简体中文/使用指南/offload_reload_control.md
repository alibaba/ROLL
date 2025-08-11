# GPU 时分复用控制指南

ROLL 框架实现了 GPU 时分复用功能，通过 offload/reload 能力，可以在不同角色间灵活共享 GPU 资源。本文档将详细介绍如何使用这一功能。

## 时分复用概述

在 ROLL 框架中，不同的角色（如 actor_train、actor_infer、critic、reference 和 rewards）可能需要使用相同的 GPU 资源。为了提高资源利用率，框架实现了 GPU 时分复用功能，允许在不同时间点将模型状态在 GPU 和 CPU 之间进行切换。

## Offload/Reload 控制机制

### 自动控制

以RLVRPipeline为例，框架会自动管理模型状态的 offload 和 reload：

```python
# 在 rlvr_pipeline.py 中的示例
ref_log_probs = self.reference.compute_log_probs(batch, blocking=True)
```

默认情况下，执行对worker的RPC调用时，框架会先将当前 worker 的GPU有关的state reload 到 GPU 上，执行完成后会将state offload 到内存上。

### 手动控制

您也可以通过设置 `batch.meta_info["is_offload_states"]` 来手动干预模型状态：

```python
# 在 rlvr_pipeline.py 中的示例
self.actor_train.offload_states(blocking=True)
```

当设置 `is_offload_states` 为 `False` 时，RPC 调用完成后不会自动 offload 模型状态到 CPU，模型会继续保留在 GPU 上。

也可以直接使用`worker.offload_states()`和`worker.reload_states()`来更加直接地控制offload和reload时机。

## 使用示例

以下是在 `rlvr_pipeline.py` 中使用 offload/reload 控制的示例：

```python
# 在推理阶段结束后，手动 offload reward 模型状态
if not self.pipeline_config.async_pipeline:
    for reward_cluster in self.rewards.values():
        reward_cluster.offload_states()

# 在计算参考模型 log probs 时，控制是否 offload 状态
if self.is_lora:
    batch.meta_info["disable_adapter"] = True
    batch.meta_info["is_offload_states"] = False
    ref_log_probs = self.actor_train.compute_log_probs(batch, blocking=True)
else:
    ref_log_probs = self.reference.compute_log_probs(batch, blocking=True)
```

## Context Manager 支持

ROLL 框架还提供了 `state_offload_manager` 上下文管理器来简化状态管理：

```python
from roll.utils.context_managers import state_offload_manager

with state_offload_manager(strategy, metrics, metric_infix, is_offload_states=True):
    # 在此上下文中执行需要 GPU 状态的操作
    yield
```

这个上下文管理器会自动处理：
1. 加载模型状态到 GPU
2. 执行操作
3. 根据 `is_offload_states` 参数决定是否将状态 offload 到 CPU

## 内存监控

框架还提供了内存使用情况的监控功能：

```python
from roll.utils.context_managers import log_gpu_memory_usage

# 记录 GPU 内存使用情况
log_gpu_memory_usage(head="model_loading", logger=logger, rank=None)
```

## 使用建议

1. 在资源紧张的情况下，合理使用 offload/reload 功能可以显著提高 GPU 利用率
2. 在pipeline的实现中，安排不同角色的执行顺序，最大化资源利用效率，如ref/reward model可并行计算等
3. 在异步训练中，合理安排不同角色的执行顺序，最大化资源利用效率
