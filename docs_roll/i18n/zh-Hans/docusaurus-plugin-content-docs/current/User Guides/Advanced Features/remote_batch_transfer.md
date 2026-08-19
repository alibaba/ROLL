# ROLL RemoteBatch 与传输后端

ROLL 框架支持 **RemoteBatch**，一种将数据存储与数据消费解耦的惰性传输机制。使用可插拔传输后端时，跨节点的大数据 payload 由后端存储和传输，Ray 只携带轻量引用。node scope 后端仍可能使用本机 Ray Object Store，在 node actor 和 Worker 进程之间交接物化后的数据。本文档介绍各后端的配置和使用方法。

## 简介

在 RL 训练流水线中（特别是 VLM 和 Agentic 场景），`DataProto` 批次可能包含大型张量（如图片、多模态 embedding）和非张量数据（如对话历史）。通过 Ray 默认的序列化方式在 RolloutScheduler 和训练 Worker 之间传输这些数据存在以下问题：

1. **内存开销高**：完整数据需要通过 Ray Object Store 进行序列化和反序列化，会产生额外副本并抬高峰值内存。
2. **传输延迟大**：大数据批次（如 VLM 场景中的图片数据）需要在 Worker 之间完整传输，导致数据搬运耗时显著。

**RemoteBatch** 通过将数据存储在外部键值存储中，仅通过 Ray 传递轻量级元数据（键/引用）来解决这些问题。实际数据在消费侧按需**惰性物化（lazily materialized）**，且仅获取请求的字段。

### 核心概念

- **RemoteBatch**：表示远程存储数据批次的抽象基类。它支持与 `TensorDict` 相同的切片、索引、选择、拼接和重复操作，但将实际数据访问延迟到调用 `materialize()` 时执行。
- **RowRemoteBatch**：以**行 ID** 为键存储数据的具体 `RemoteBatch` 实现。每行（样本）有一个唯一 ID，传输后端以行粒度存储/检索数据。**TransferQueue** 后端使用此实现。
- **ColumnRemoteBatch**：使用列式字段元数据的 `RemoteBatch` 实现。**RayMemoryStore** 和 **Mooncake** 后端使用此实现。
- **BatchProxy**：包装本地 `TensorDict`（或 `dict`）和 `RemoteBatch` 的代理对象，支持透明的回退查找。访问键时，先检查本地 batch，再回退到远程 batch。
- **传输后端（Transfer Backend）**：负责 `put`、`get` 和 `delete` 操作的可插拔存储后端。目前支持的后端：
  - `None`（Dummy）：无远程存储，数据保留在本地（默认）。
  - `TransferQueue`：使用 [TransferQueue](https://github.com/kvcache-ai/TransferQueue) 库进行高性能分布式键值传输。
  - `Mooncake`：作为可选的结构化 `DataProto` 传输后端，用于大规模 tensor、non-tensor 和多模态 rollout payload。

### 工作原理

1. **上传 (`to_remote`)**：`DataProto.to_remote()` 类方法将本地 `DataProto` 转换为远程支持的 `DataProto`。它将所有张量和非张量字段上传到传输后端，并返回一个包含 `RemoteBatch` 引用的新 `DataProto`。生产进程可以保留本地 cache；序列化后的消费端只收到轻量引用，并按需物化数据。
2. **传输**：Ray 传递 `RemoteBatch` 引用和原有的轻量 `meta_info`，体积较大的 `batch` 与 `non_tensor_batch` 字段保留在传输后端。
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

### Mooncake

#### 1. 安装 Mooncake

ROLL 需要 Mooncake 提供统一 DataProto API、schema 支持、GET 结果显式释放能力和 typed-ragged ndarray layout。请使用基于 `86b21ccf` 或更新 main commit 构建的 Mooncake wheel。下面的命令遵循 Mooncake 的源码构建流程：

```bash
git clone https://github.com/kvcache-ai/Mooncake.git
cd Mooncake
git checkout 86b21ccf
git submodule update --init --recursive
sudo bash dependencies.sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_UNIT_TESTS=OFF -DBUILD_EXAMPLES=OFF
cmake --build build -j
PYTHON_VERSION="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
PYTHON_VERSION="${PYTHON_VERSION}" BUILD_DIR=build bash scripts/build_wheel.sh "${PYTHON_VERSION}" dist
python -m pip install dist/mooncake_transfer_engine-*.whl
```

#### 2. 启动 metadata 服务

启动 ROLL 前，需要保证各节点能够访问 Mooncake metadata 服务。ROLL 只负责初始化 Store client，不负责启动或管理 `mooncake_master`、etcd 或 Redis。使用独立 master 时，在一个所有节点可访问的地址启动：

```bash
mooncake_master --rpc_address=<master-ip> --rpc_port=<master-port>
```

使用 etcd 或 Redis 时，请按 Mooncake 的部署说明准备 metadata 服务。

#### 3. 配置 ROLL 集群

ROLL 会在集群内共享同一份 `backend_config`，因此 YAML 只放公共配置：

```yaml
transfer_backend:
  backend_name: Mooncake
  backend_config:
    client_scope: node
    metadata_server: P2PHANDSHAKE
    global_segment_size: 8589934592
    local_buffer_size: 8589934592
    protocol: tcp
    master_server_addr: <master-host:port>
```

使用 RDMA 时，只需修改共享的传输协议：

```yaml
transfer_backend:
  backend_name: Mooncake
  backend_config:
    client_scope: node
    metadata_server: P2PHANDSHAKE
    global_segment_size: 8589934592
    local_buffer_size: 8589934592
    protocol: rdma
    master_server_addr: <master-host:port>
```

本机地址和 RDMA 设备属于节点本地配置，需要在每个节点分别设置。TCP 不需要 `MOONCAKE_DEVICE`：

```bash
export MOONCAKE_LOCAL_HOSTNAME=<本机-ip>
export MOONCAKE_DEVICE=<本机-rdma-device>
```

`field_schemas` 是可选配置。ROLL 会把每个条目转换成 Mooncake `FieldSchema`，并传给 `put(type="dataproto")`。未配置时，Mooncake 根据字段的实际值推断数据表示。对于结构已知的 object-array 字段，建议由框架提供 schema，这样不同 batch 会稳定使用同一种表示方式，全空采样也不会改变编码路径。例如，一个 `int64` ragged 字段可以这样声明：

```yaml
transfer_backend:
  backend_name: Mooncake
  backend_config:
    field_schemas:
      token_rows:
        codec: typed_ragged
        nullable: true
        metadata:
          section: non_tensor_batch
          dtype: int64
```

实际使用时，需要把 `token_rows` 和 schema 替换成当前 workload 的字段。多模态 processor 输出的 key 由模型决定，因此 schema 必须匹配 ROLL collator 的实际输出，不能假定只有固定的 `image` 成员。

Mooncake 使用的 `backend_config` 选项如下：

| 配置项 | 说明 |
| --- | --- |
| `client_scope` | `node` 在每个 Ray 节点复用一个 Mooncake client 和 BufferPool，也是当前唯一支持的 scope。 |
| `key_prefix` | Mooncake Store key 的前缀，默认为 `roll`。 |
| `transfer_policy` | 可选的 Mooncake `BundleTransferPolicy` 参数，例如 `copy_mode`、`put_mode` 和 `max_inflight_put`。 |
| `field_schemas` | 可选的字段级 `codec`、`nullable` 和 `metadata`。无法识别的 schema 参数会在 client 初始化时直接报错。 |
| Store 初始化参数 | 包括本机地址、metadata server、segment 大小、传输协议、RDMA 设备和 metadata backend 地址。 |

Store 参数也可以通过 `setup_args` / `setup_kwargs` 传入，或者使用 Mooncake 标准环境变量：

```bash
export MOONCAKE_MASTER=<master-host:port>
export MOONCAKE_LOCAL_HOSTNAME=<worker-ip>
export MOONCAKE_TE_META_DATA_SERVER=P2PHANDSHAKE
export MOONCAKE_PROTOCOL=rdma  # 或 tcp
export MOONCAKE_DEVICE=<rdma-device>
```

如果节点相关的 Store 参数由环境变量提供，共享的 ROLL 配置只需要包含 backend policy：

```yaml
transfer_backend:
  backend_name: Mooncake
  backend_config:
    client_scope: node
    key_prefix: roll
    transfer_policy:
      copy_mode: auto
```

假设上述配置位于某个 YAML 文件中，可以使用配置目录和不带 `.yaml` 的文件名启动：

```bash
export PYTHONPATH="$(pwd):${PYTHONPATH}"
python examples/start_rlvr_pipeline.py \
  --config_path <config-directory> \
  --config_name <config-filename-without-yaml>
```

其他 ROLL pipeline launcher 使用相同的配置字段。初始化成功后，Driver 日志中会出现 `Initialized Mooncake transfer backend`，首次使用 RemoteBatch 的进程会输出 `Initialized transfer client`。

#### 4. 数据和 BufferPool 生命周期

Mooncake 将每次上传的 ROLL DataProto fragment 保存为一个结构化对象；经过 union 或 concat 后，一个逻辑 DataProto 可以引用多个物理对象。tensor 字段保留在 `batch` section，NumPy 字段保留在 `non_tensor_batch`，ROLL 原有的 `meta_info` 继续沿用轻量控制路径传递。`ColumnRemoteBatch.materialize(fields)` 只读取指定字段，`ColumnRemoteBatch.drop()` 负责删除这些远端对象。

Mooncake GET lease 属于 node actor。actor 将可序列化的数据视图写入本机 Ray Object Store 后释放 BufferPool lease；如果释放失败，会在后续 Mooncake 操作中重试。select、slice、union 和 concat 会保留原始 row 顺序。

同一进程中的派生视图共享 cleanup 状态，但 ROLL 不提供跨进程引用计数或失效通知。框架必须指定唯一的清理路径，并在所有 consumer 完成后调用 `drop()`。该调用会删除物理远端对象；其他进程持有的旧 handle 不会自动失效，但之后不能再使用。

- `backend_name`：要使用的传输后端名称。
  - `null`（默认）：禁用远程传输，所有数据保留在本地。未配置 `transfer_backend` 时的默认行为。
  - `TransferQueue`：使用 TransferQueue 库进行高性能数据传输。
  - `Mooncake`：使用 Mooncake 结构化 `DataProto` 传输 tensor batch 和 `non_tensor_batch`；原有 `meta_info` 仍沿用 ROLL 的轻量控制路径。
- `backend_config`：后端特定的配置字典。
  - 对于 TransferQueue，对应 TransferQueue 的初始化配置。
  - 对于 Mooncake，使用上面介绍的显式配置或标准环境变量。
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
| Mooncake | 实验性 | 可选的结构化 `DataProto` backend，支持 tensor、non-tensor 和多模态 rollout payload。 |
| RayMemoryStore | 仅作示例 | 未经测试。仅作为 `ColumnRemoteBatch` 模式的参考实现提供。 |

### TODO

- 避免在 Trainer 侧全量物化：当前 Trainer 会对整个 RemoteBatch 调用 `materialize()`，后续可优化为仅物化实际需要的字段，避免不必要的数据拉取。
- Driver 侧选择性预取：在 Pipeline Driver 中实现选择性 prefetch，根据后续步骤的需求批量预取所需字段，减少多次小规模拉取的开销。
- Scheduler 对被 filter 的 RemoteBatch 自动调用 `drop()`，避免远程存储泄漏。
