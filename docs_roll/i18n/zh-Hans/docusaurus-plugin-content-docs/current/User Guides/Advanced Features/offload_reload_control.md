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

## Offload 后端配置

训练侧状态（FSDP2 与 Megatron 的 actor/critic）的 offload/reload 统一经由 offload 存储层完成，通过顶层 `offload_backend` 配置项选择：

```yaml
is_offload_states: true
offload_backend: local  # 或 local_dedup
```

- `local`（默认）：每个 rank 独立 offload 到本进程的 pinned CPU 内存。缓冲区通过 `cudaHostRegister` 按精确尺寸锁页并在各 step 间循环复用，因此反复的 offload 周期会复用同一批 pinned 内存，无需重复锁页。
- `local_dedup`：在数据并行各 rank 间复制的数据（HSDP 的 dense 分片、Megatron 的 DDP buffer、非分布式优化器的 master weights）按 rank 切分，每个 rank 只保存约 1/world 的内存，reload 时通过一次 NCCL/RCCL all-gather 重建。适合 host 内存成为瓶颈、且同一复制组的各 rank 位于同一节点的场景。

注意事项：

- FSDP2 DTensor 的 local shards 保持为恢复后的 flat buffer 的 views，允许非零 storage offset。异步 checkpoint staging 处理真实的 local storage，无需在 GPU 上复制独立 shards 或增加 reload 通信。
- 使用 `local` backend 时，FSDP2 DCP checkpoint 直接使用已有的 CPU store buffer。存在异步 checkpoint 时，Strategy 会在写盘完成后再回收 buffer，从而避免为 DCP reload 到 GPU，也避免 CPU staging 再复制一份模型参数。`local_dedup` 每个 rank 只存储一段数据，需要重建，因此不使用该直接路径。
- Megatron 中专家并行（EP）参数在其自身的复制组（expert-DP）上去重，因此 EP/ETP 配置不会破坏去重的正确性。
- FSDP2 中 MoE 专家分片改在 expert-DP（`eddp`）mesh 维度上去重（而非 dense 的 `ddp` 维度），因此 `eddp_size > 1` 时专家权重同样只需 1/eddp_size 的 host 内存。
- pinned host 内存只循环复用、不会在 worker 生命周期内归还给操作系统；offload 周期的历史高水位会常驻。可开启 roll debug 模式，通过 `[OffloadDebug]` 日志查看存储统计（`host_pinned_gb`）。

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

### 原生 NCCL communicator offload

设置 `actor_train.offload_nccl: true` 和/或 `reference.offload_nccl: true`，即可通过
`ncclCommSuspend` 释放空闲 PyTorch/Megatron communicator 的动态显存，再通过
`ncclCommResume` 恢复。默认值为 `false`。ProcessGroup 保持不变，旧的销毁/重建实现已移除。
跨角色 `model_update/*` 通信组的对端不会同时卸载，因此保持常驻。

开启功能的每个 Worker 会在加载模型前校验**实际加载**的 NCCL：
`ncclGetVersion()` 必须返回 `>=2.29.7` 的版本，且同时存在 Suspend/Resume 符号。
不满足要求时直接报错，不会静默关闭功能。校验不依赖已安装 wheel 的版本或 Torch 编译时元数据。
本功能不修改 requirements，也不增加依赖安装命令；需由部署环境在每个节点提供兼容的 NCCL。
原生内存 offload 要求启用 cuMem 分配，并且不能禁用 NCCL 内部 memory manager。
在支持的系统上，NCCL 通常会自动启用 cuMem，memory manager 也默认开启，因此一般无需显式设置。
如果部署环境覆盖了这些默认值，需要在创建任何 NCCL communicator 前设置
`NCCL_CUMEM_ENABLE=1`、`NCCL_DISABLE_MEM_MANAGER=0`。

1. 在资源紧张的情况下，合理使用 offload/reload 功能可以显著提高 GPU 利用率
2. 在pipeline的实现中，安排不同角色的执行顺序，最大化资源利用效率，如ref/reward model可并行计算等
3. 在异步训练中，合理安排不同角色的执行顺序，最大化资源利用效率
