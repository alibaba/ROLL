# vLLM 推理后端配置指南

vLLM 是一个快速且易于使用的大型语言模型推理库，通过 PagedAttention 技术高效管理注意力键值缓存。本文档将详细介绍如何在 ROLL 框架中配置和使用 vLLM 推理后端。

## vLLM 简介

vLLM 是一个高性能的推理引擎，具有以下特点：
1. **快速推理**：通过 PagedAttention 技术高效管理注意力键值缓存
2. **内存高效**：通过量化和优化减少内存使用
3. **易于使用**：提供简单的 API 接口
4. **可扩展性**：支持分布式推理

## 配置 vLLM 策略

在 ROLL 框架中，可以通过在 YAML 配置文件中设置 `strategy_args` 来配置 vLLM 推理策略。

### 配置示例

以下是一个典型的 vLLM 配置示例（来自 `examples/qwen2.5-7B-rlvr_megatron/rlvr_config.yaml`）：

```yaml
actor_infer:
  model_args:
    disable_gradient_checkpointing: true
    dtype: bf16
  generating_args:
    max_new_tokens: ${response_length}
    top_p: 0.99
    top_k: 100
    num_beams: 1
    temperature: 0.99
    num_return_sequences: ${num_return_sequences_in_group}
  strategy_args:
    strategy_name: vllm
    strategy_config:
      gpu_memory_utilization: 0.8
      block_size: 16
      max_model_len: 8000
  device_mapping: list(range(0,12))
  infer_batch_size: 1
```

### 配置参数详解

1. **strategy_name**: 设置为 `vllm` 以使用 vLLM 推理后端

2. **strategy_config**: vLLM 特定的配置参数，更多vllm优化配置，请参考[vLLM官方文档](https://docs.vllm.ai/en/latest/), strategy_config透传处理。
   - `gpu_memory_utilization`: 用于模型执行器的 GPU 内存占比
     - 例如 0.8 表示使用 80% 的 GPU 内存
     - 根据模型大小和硬件配置调整此值
   - `block_size`: token 块大小，用于连续的 token 块
     - 影响 vLLM 内部的内存管理效率
     - 通常设置为 16 或 32
   - `max_model_len`: 模型上下文长度
     - 如果未指定，将从模型配置中自动推导
     - 确保不超过硬件限制
   - `load_format`: 加载模型权重的格式
     - 由于模型会在开始时进行"更新"，此值可以设置为 `dummy`
   - `sleep_level`: sleep model时的级别
     - 1 默认值，仅销毁 KV 缓存，会将模型权重保留
     - 2 将在生成后销毁模型权重与 KV 缓存，从而节省内存
   - `enable_expert_parallel`: 为 MoE 层启用 vLLM Expert Parallel，要求 vLLM 0.16.x 或更高版本
   - `data_parallel_size`: 每个自动推导出的 vLLM deployment 中的 DP rank 数量
3. **device_mapping**: 指定使用的 GPU 设备 ID 列表

4. **infer_batch_size**: 推理时的批次大小

### 分布式 TP、DP 与 EP

ROLL 默认使用 vLLM multiprocessing executor。Ray 仍负责资源放置和 actor 生命周期；在受支持的
vLLM 0.11.1 之前版本中，它还用于保留跨节点 TP/PP 能力。用户无需配置
`distributed_executor_backend` 或 `data_parallel_backend`，这两个后端由 ROLL 自动管理。

定义：

```text
N = len(device_mapping)
T = tensor_parallel_size（默认 1）
P = pipeline_parallel_size（默认 1）
D = data_parallel_size（默认 1）
W = N / (T * P)（ROLL InferWorker 数量）
K = W / D（相互独立的 vLLM deployment 数量）
```

`N` 必须能被 `T * P` 整除，`W` 必须能被 `D` 整除。每个 InferWorker 对应一个 vLLM external-DP
rank，并占用 `T * P` 个 GPU placements；每连续 `D` 个 InferWorkers 组成一个 deployment。ROLL
自动推导 DP rank、rendezvous 地址和端口，不向用户暴露 internal/external DP 模式。

在 vLLM 0.16.x 或更高版本中，对 MoE 模型设置 `enable_expert_parallel: true` 后，ROLL 使用 MP backend，
并在每个 deployment 内形成 Expert Parallel group。更早版本会提前报错，并提示升级 vLLM 或关闭 EP：

```text
EP size = T * D
```

EP 只改变 MoE 通信排布，不改变 InferWorker 或 deployment 数量。在 `P=1` 时，常见组合如下：

| DP | TP | EP 关闭 | EP 开启 |
| --- | --- | --- | --- |
| `D=1` | `T=1` | `N` 个独立 engines | 独立的 EP1 engines |
| `D=1` | `T>1` | `N/T` 个独立 TP engines | `N/T` 个 deployments，每个 EP size 为 `T` |
| `D>1` | `T=1` | `N/D` 个 external-DP deployments | `N/D` 个 deployments，每个 EP size 为 `D` |
| `D>1` | `T>1` | `N/(T*D)` 个 external-DP + TP deployments | `N/(T*D)` 个 deployments，每个 EP size 为 `T*D` |

#### 物理节点与逻辑节点

如果一个 InferWorker 的 `T * P` 张 GPU 能放入一个物理节点，其 actor 会启动全部本地 vLLM MP
进程。从 vLLM 0.11.1 开始，engine 跨越 `S` 个物理节点时，ROLL 在第一个节点启动 MP head actor，
并在其余节点各启动一个轻量 headless launcher。每个节点必须提供相同数量的 GPU。

对于 vLLM 0.11.0，如果一个 TP/PP engine 跨节点，ROLL 会自动使用原有 Ray executor，用户不需要
选择 executor backend。

从 vLLM 0.11.1 开始，vLLM 为整个 external-DP deployment 使用一套逻辑节点编号：

```text
logical nnodes = D * S
logical node rank = data_parallel_rank * S + node_offset
```

即使多个 external-DP ranks 位于同一个物理节点，它们仍各自对应不同的逻辑节点。从 vLLM 0.11.1
开始，同一条 MP 路径可以覆盖同节点 DP、跨节点 DP，以及单个 TP engine 跨节点的场景；在该 MP
拓扑上启用 EP 要求 vLLM 0.16.x 或更高版本。
在 vLLM 0.11.0 中，只要每个 TP/PP engine 能放进单节点，ROLL 沿用 external DP 的 rank/address/RPC
启动方式，不传递逻辑节点参数。

例如，在 vLLM 0.16.x 或更高版本中开启 EP 后，4 个 8 卡节点上的 DP2/TP16 排布为：

```text
DP rank 0: logical node 0（head，GPU 0-7）+ logical node 1（headless，GPU 8-15）
DP rank 1: logical node 2（head，GPU 0-7）+ logical node 3（headless，GPU 8-15）
EP size: 2 * 16 = 32
```

受支持的更早版本在相同 GPU placements 上启动 Ray workers，而不是 MP head/headless 进程。

在 128 张 GPU 上开启 DP16/TP4/EP 时，ROLL 创建 32 个 InferWorkers 和两个独立 deployments。
每个 deployment 包含 16 个 TP4 engines 并形成 EP64；两个 EP groups 不会合并成 EP128。

省略 `data_parallel_size` 时默认值为 1。例如 32 张 GPU 配置 TP16 会创建两个相互独立的 TP16
engines；再配置 DP2，则把这两个 engines 组成一个 external-DP deployment。

external DP 要求 vLLM 0.11.0 或更高版本。跨节点 TP/PP engine 在 vLLM 0.11.0 上使用原有 Ray
executor，从 vLLM 0.11.1 开始使用 MP executor。EP 要求 vLLM 0.16.x 或更高版本，并使用 MP
backend。跨节点吞吐受互联网络和 vLLM All2All 实现影响，应在目标集群上实测，不能默认认为开启 EP
一定更快。

## 与其他组件的集成

在配置示例中，我们可以看到：

1. `actor_infer` 使用 vLLM 作为推理后端
2. `actor_train` 使用 Megatron 进行训练
3. `reference` 使用 Megatron 进行推理
4. 奖励模型使用不同的推理后端（如 `hf_infer`）

这种设计允许不同组件根据其需求选择最适合的推理引擎。

## 性能优化建议

1. **内存管理**：
   - 合理设置 `gpu_memory_utilization` 参数以平衡性能和内存使用
   - 监控 GPU 内存使用情况，避免内存溢出

2. **批处理优化**：
   - 根据模型大小和硬件能力调整 `infer_batch_size`
   - 考虑序列长度对批处理大小的影响

3. **上下文长度**：
   - 合理设置 `max_model_len` 以匹配任务需求
   - 避免设置过大的上下文长度导致内存不足

## 注意事项

1. vLLM 需要特定版本的依赖库，请确保安装了兼容的版本
2. 在资源受限的环境中，需要仔细平衡不同组件的资源分配
3. vLLM 与 Megatron 等训练框架的集成可能需要额外的配置

通过合理配置 vLLM 推理后端，您可以充分发挥 ROLL 框架在大规模语言模型推理方面的性能优势。
