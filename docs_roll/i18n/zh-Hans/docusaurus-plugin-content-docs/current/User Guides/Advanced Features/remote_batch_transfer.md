# ROLL RemoteBatch 与传输后端

ROLL 框架支持 **RemoteBatch**，一种惰性数据传输机制，将数据存储与数据消费解耦。通过集成可插拔的传输后端（如 **TransferQueue**），大数据批次可以在 Ray Worker 之间传输，而无需通过 Ray 的 Object Store 序列化整个数据负载。本文档详细介绍如何使用这一功能。

## 简介

在 RL 训练流水线中（特别是 VLM 和 Agentic 场景），`DataProto` 批次可能包含大型张量（如图片、多模态 embedding）和非张量数据（如对话历史）。通过 Ray 默认的序列化方式在 RolloutScheduler 和训练 Worker 之间传输这些数据存在以下问题：

1. **内存开销高**：完整数据需要通过 Ray Object Store 进行序列化和反序列化，峰值内存使用量翻倍。
2. **传输延迟大**：大数据批次（如 VLM 场景中的图片数据）需要在 Worker 之间完整传输，导致数据搬运耗时显著。

**RemoteBatch** 通过将数据存储在外部键值存储中，仅通过 Ray 传递轻量级元数据（键/引用）来解决这些问题。实际数据在消费侧按需**惰性物化（lazily materialized）**，且仅获取请求的字段。

### 核心概念

- **RemoteBatch**：表示远程存储数据批次的抽象基类。它支持与 `TensorDict` 相同的切片、索引、选择、拼接和重复操作，但将实际数据访问延迟到调用 `materialize()` 时执行。
- **RowRemoteBatch**：以**行 ID** 为键存储数据的具体 `RemoteBatch` 实现。每行（样本）有一个唯一 ID，传输后端以行粒度存储/检索数据。**TransferQueue** 后端使用此实现。
- **ColumnRemoteBatch**：以**列 ID** 为键存储数据的具体 `RemoteBatch` 实现（每个字段/列一个键）。**RayMemoryStore** 后端使用此实现。
- **BatchProxy**：包装本地 `TensorDict`（或 `dict`）和 `RemoteBatch` 的代理对象，支持透明的回退查找。访问键时，先检查本地 batch，再回退到远程 batch。
- **传输后端（Transfer Backend）**：负责 `put`、`get` 和 `delete` 操作的可插拔存储后端。目前支持的后端：
  - `None`（Dummy）：无远程存储，数据保留在本地（默认）。
  - `TransferQueue`：使用 [TransferQueue](https://github.com/kvcache-ai/TransferQueue) 库进行高性能分布式键值传输。

### 工作原理

1. **上传 (`to_remote`)**：`DataProto.to_remote()` 类方法将本地 `DataProto` 转换为远程支持的 `DataProto`。它将所有张量和非张量字段上传到传输后端，并返回一个包含 `RemoteBatch` 引用的新 `DataProto`（无本地数据）。
2. **传输**：轻量级的 `DataProto`（仅包含 `RemoteBatch` 元数据）通过 Ray 在 Worker 之间传输。由于元数据很小，序列化速度很快。
3. **物化（惰性）**：在消费侧，当需要特定字段时，调用 `RemoteBatch.materialize(fields)` 仅从后端获取请求的列。获取的数据会缓存在本地供后续访问。
4. **清理（Drop）**：数据消费完成后，可以调用 `RemoteBatch.drop()` 从后端存储中删除数据。

## 配置

传输后端通过 ROLL 顶层配置中的 `transfer_backend` 字段进行配置：

```yaml
transfer_backend:
  backend_name: TransferQueue
  backend_config:
    backend:
      SimpleStorage:
        num_data_storage_units: 16
```

- `backend_name`：要使用的传输后端名称。
  - `null`（默认）：禁用远程传输，所有数据保留在本地。未配置 `transfer_backend` 时的默认行为。
  - `TransferQueue`：使用 TransferQueue 库进行高性能数据传输。
- `backend_config`：后端特定的配置字典。对于 TransferQueue，对应 TransferQueue 的初始化配置。
  - `backend.SimpleStorage.num_data_storage_units`：数据分片的存储单元数量。可以根据 CPU 核数和集群节点数进行配置。`msgpack` 序列化单个对象有最大 4GB 的限制，因此传输大数据时需要更多的 storage unit 来将 `non_tensor_batch` 分片成更小的块。

### Agentic Pipeline 优化

在 Agentic Pipeline 中，默认在 RolloutScheduler 层面调用 `to_remote`。如果要完全避免从 env worker 汇总数据到 RolloutScheduler 的开销，可以在 env manager 将数据放入 output queue 之前手动调用 `to_remote`：

```python
batch = DataProto.to_remote(batch)
output_queue.put(batch)
```

:::caution
在环境 Worker 中手动调用 `to_remote` 与 filter 不兼容。当数据被 filter 过滤掉时，Scheduler 不会对被过滤的数据调用 `drop()`，导致远程存储中的数据泄漏。仅在不需要 filter 时才在 env worker 中使用手动 `to_remote`。（TODO：后续将支持 Scheduler 对被 filter 的 RemoteBatch 自动调用 `drop()`）
:::

## 开发状态

| 后端 | 状态 | 说明 |
|------|------|------|
| TransferQueue | 端到端已测试 | 生产可用。已在 RLVR、VLM 和 Agentic Pipeline 中测试通过。 |
| RayMemoryStore | 仅作示例 | 未经测试。仅作为 `ColumnRemoteBatch` 模式的参考实现提供。 |

### TODO

- 避免在 Trainer 侧全量物化：当前 Trainer 会对整个 RemoteBatch 调用 `materialize()`，后续可优化为仅物化实际需要的字段，避免不必要的数据拉取。
- Driver 侧选择性预取：在 Pipeline Driver 中实现选择性 prefetch，根据后续步骤的需求批量预取所需字段，减少多次小规模拉取的开销。
- Scheduler 对被 filter 的 RemoteBatch 自动调用 `drop()`，避免远程存储泄漏。

