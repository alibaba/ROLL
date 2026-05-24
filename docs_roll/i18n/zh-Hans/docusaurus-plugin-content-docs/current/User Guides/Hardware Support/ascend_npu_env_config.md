# 昇腾 NPU 环境变量配置指南

最后更新：2026/04/27。

本文档说明在华为昇腾 NPU 上运行 ROLL 时涉及的关键环境变量，涵盖设备管理、HCCL 通信、显存优化、CPU 调度、vLLM-Ascend 推理及调试日志等方面。

## ROLL 自动设置的环境变量

ROLL 在运行时自动注入以下环境变量（定义在 `roll/platforms/npu.py` 中）：

| 变量 | 值 | 说明 |
| ---- | -- | ---- |
| `ASCEND_RT_VISIBLE_DEVICES` | 如 `"0,1,2,3"` | 控制 NPU 设备可见性，类似 GPU 的 `CUDA_VISIBLE_DEVICES` |
| `RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES` | `"1"` | 阻止 Ray 自动覆盖 `ASCEND_RT_VISIBLE_DEVICES` |
| `VLLM_ALLOW_INSECURE_SERIALIZATION` | `"1"` | 允许 vLLM 使用非安全序列化，用于 Ray 跨进程传输张量 |
| `RAY_get_check_signal_interval_milliseconds` | `"1"` | 缩短 Ray plasma lock 持有时间，避免多 Worker 场景下锁饥饿 |
| `RAY_CGRAPH_get_timeout` | `"600"` | Ray 计算图获取超时时间（秒） |

## Docker 镜像中的环境变量

在 [昇腾 NPU Docker 使用指南](ascend_docker_usage.md) 中说明的预构建镜像内，包含以下环境设置：

| 变量 | 值 | 说明 |
| ---- | -- | ---- |
| `ASCEND_HOME_PATH` | `/usr/local/Ascend/ascend-toolkit/latest` | CANN 工具包根路径 |
| `LD_LIBRARY_PATH` | 包含多个 Ascend lib64 路径 | 动态库搜索路径，确保 `libascendcl.so` 等可被加载 |

预构建镜像会通过 `/root/.bashrc` 自动加载以下 CANN 环境脚本：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
```

## Ray 集群环境变量（多机）

这些变量控制 ROLL 如何在多个 NPU 节点间组建 Ray 集群。它们定义在 `roll/distributed/scheduler/driver_utils.py` 中，由 `roll/distributed/scheduler/initialize.py` 消费：

| 变量 | 默认值 | 说明 |
| ---- | ------ | ---- |
| `RANK` | `0` | 节点编号，`0` = 主节点，`1, 2, 3...` = 工作节点 |
| `WORLD_SIZE` | `1` | 集群总节点数 |
| `MASTER_ADDR` | `127.0.0.1` | 主节点 IP 地址 |
| `MASTER_PORT` | `6379` | Ray 主节点端口（也是 Ray 默认端口） |
| `DASHBOARD_PORT` | `8265` | Ray Dashboard Web UI 端口 |
| `WORKER_ID` | `<MASTER_ADDR>:<RANK>` | Ray 集群中的节点名称，未设置时自动生成 |

当 `RANK=0` 时，ROLL 自动执行 `ray start --head --port=<MASTER_PORT>`。当 `RANK>0` 时，ROLL 会休眠 5 秒后执行 `ray start --address=<MASTER_ADDR>:<MASTER_PORT>` 加入集群。所有节点加入后，工作节点退出（`sys.exit(0)`），仅主节点执行训练流水线。

示例（主节点，在启动流水线前设置）：

```bash
export RANK=0
export WORLD_SIZE=2
export MASTER_ADDR=10.0.0.1
export MASTER_PORT=6379
export DASHBOARD_PORT=8265
```

示例（工作节点，在加入前设置）：

```bash
export RANK=1
export WORLD_SIZE=2
export MASTER_ADDR=10.0.0.1
export MASTER_PORT=6379
```

你也可以在运行 ROLL 之前手动启动 Ray（`ray start --head` / `ray start --address=...`）。ROLL 会检测到已存在的集群并跳过自动启动。

## HCCL 通信相关变量

这些变量控制 HCCL（Huawei Collective Communication Library）的行为，HCCL 是 NPU 上的分布式通信后端（等同于 GPU 上的 NCCL）：

| 变量 | 推荐值 | 说明 |
| ---- | ------ | ---- |
| `HCCL_CONNECT_TIMEOUT` | `3600` | 建链超时时间（秒），默认 120 秒，大模型训练场景需增大 |
| `HCCL_EXEC_TIMEOUT` | `3600` | 集合通信算子执行超时时间（秒），长步长训练需增大 |
| `HCCL_DETERMINISTIC` | `false` | 关闭确定性计算，开启会显著降低通信性能 |
| `HCCL_OP_EXPANSION_MODE` | `"AIV"` | 通信算法展开位置，`AIV` 使用 Vector Core，性能优于 `AI_CPU`/`HOST`/`HOST_TS` |
| `HCCL_BUFFSIZE` | 如 `"2147483648"` | HCCL 通信缓冲区大小（字节），大数据量场景可增大 |
| `HCCL_IF_IP` | 节点 IP 地址 | 指定 HCCL 跨节点通信使用的 IP 地址，多机训练必需 |
| `HCCL_SOCKET_IFNAME` | 如 `"enp194s0f0"` | HCCL Socket 通信使用的网卡名称，所有节点必须一致 |
| `HCCL_IF_BASE_PORT` | 如 `23456` | HCCL 跨节点通信基础端口，确保端口未被防火墙拦截 |
| `HCCL_WHITELIST_DISABLE` | `1` | 禁用 HCCL 白名单检查，某些环境下遇到通信错误时可能需要设置 |

示例（单机）：

```bash
export HCCL_CONNECT_TIMEOUT=3600
export HCCL_DETERMINISTIC=false
export HCCL_OP_EXPANSION_MODE="AIV"
```

示例（多机）：

```bash
export HCCL_CONNECT_TIMEOUT=3600
export HCCL_EXEC_TIMEOUT=3600
export HCCL_DETERMINISTIC=false
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_IF_IP=$(hostname -I | awk '{print $1}')
export HCCL_SOCKET_IFNAME="enp194s0f0"
export HCCL_IF_BASE_PORT=23456
```

## NPU 显存相关变量

| 变量 | 推荐值 | 说明 |
| ---- | ------ | ---- |
| `NPU_MEMORY_FRACTION` | `0.96` | NPU 显存可用比例，默认 0.8，大模型推理建议调到 0.95+ |
| `PYTORCH_NPU_ALLOC_CONF` | `expandable_segments:True` | 启用 PyTorch NPU 内存池可扩展段，减少内存碎片和 OOM 风险 |
| `MULTI_STREAM_MEMORY_REUSE` | `1` | 多流内存复用，减少显存占用 |
| `TASK_QUEUE_ENABLE` | `2` | 任务下发优化，非图模式设为 2，图模式设为 1 |
| `COMBINED_ENABLE` | `1` | 启用算子组合优化，将多个小算子融合为一个大算子以减少内核启动开销 |

示例：

```bash
export NPU_MEMORY_FRACTION=0.96
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export MULTI_STREAM_MEMORY_REUSE=1
export TASK_QUEUE_ENABLE=2
export COMBINED_ENABLE=1
```

## CPU 调度相关变量

| 变量 | 推荐值 | 说明 |
| ---- | ------ | ---- |
| `CPU_AFFINITY_CONF` | `2` | CPU 绑核优化，避免跨 NUMA 节点内存访问。`1`=粗粒度，`2`=细粒度（推荐） |
| `OMP_NUM_THREADS` | `1` | OpenMP 线程数，分布式训练中建议设为 1 避免过度竞争 |

示例：

```bash
export CPU_AFFINITY_CONF=2
export OMP_NUM_THREADS=1
```

也支持按 NPU 卡自定义绑核范围：

```bash
export CPU_AFFINITY_CONF=1,npu0:0-1,npu1:2-3,npu2:4-5,npu3:6-7
```

## vLLM-Ascend 推理相关变量

| 变量 | 推荐值 | 说明 |
| ---- | ------ | ---- |
| `VLLM_USE_V1` | `1` | 启用 vLLM V1 架构，vLLM-Ascend 必需 |
| `VLLM_ATTENTION_BACKEND` | `XFORMERS` | vLLM 注意力计算后端 |
| `VLLM_ASCEND_ENABLE_FLASHCOMM` | `1` | 启用昇腾 FlashComm 高速通信优化 |
| `VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE` | `1` | 启用大模型稠密计算优化 |
| `VLLM_ASCEND_ENABLE_PREFETCH_MLP` | `1` | 启用 MLP 层权重预取 |
| `VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE` | `1` | 启用 TopK 算子融合优化，提升生成解码性能 |
| `VLLM_ASCEND_MODEL_EXECUTE_TIME_OBSERVE` | `1` | 打印 prefill/decode 阶段耗时详情（调试用） |
| `VLLM_ASCEND_TRACE_RECOMPILES` | `1` | 追踪算子重编译，用于调试性能问题 |
| `VLLM_ENABLE_MC2` | `1` | 启用 MC2 通信优化，用于多机推理 |

示例：

```bash
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_ASCEND_ENABLE_FLASHCOMM=1
export VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE=1
export VLLM_ASCEND_ENABLE_PREFETCH_MLP=1
```

## CANN 日志与调试变量

| 变量 | 推荐值 | 说明 |
| ---- | ------ | ---- |
| `ASCEND_GLOBAL_LOG_LEVEL` | `3`（ERROR） | CANN 日志级别：0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR |
| `ASCEND_SLOG_PRINT_TO_STDOUT` | `1` | 将 CANN 日志输出到标准输出（调试用） |
| `ASDOPS_LOG_LEVEL` | `ERROR` | 算子库日志级别 |
| `ATB_LOG_LEVEL` | `ERROR` | ATB 加速库日志级别 |
| `ASCEND_LAUNCH_BLOCKING` | `1` | 启用同步执行以定位错误。仅在调试 NPU 错误时设为 `1`，会禁用异步执行并严重降低性能 |

:::caution
生产环境中开启 DEBUG/INFO 日志级别会显著降低性能，请务必将日志级别设为 ERROR。
:::

调试示例：

```bash
export ASCEND_GLOBAL_LOG_LEVEL=0
export ASCEND_SLOG_PRINT_TO_STDOUT=1
export ASCEND_LAUNCH_BLOCKING=1
```

生产示例：

```bash
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASDOPS_LOG_LEVEL=ERROR
export ATB_LOG_LEVEL=ERROR
```

## CANN 算子编译与精度变量

| 变量 | 推荐值 | 说明 |
| ---- | ------ | ---- |
| `ACL_OP_COMPILER_CACHE_MODE` | `enable` | 启用算子编译缓存，避免重复运行时重新编译 |
| `ACL_OP_COMPILER_CACHE_DIR` | 如 `/tmp/npu_cache` | 算子编译缓存存储目录 |
| `ASCEND_MAX_OP_CACHE_SIZE` | 如 `5000` | 最大算子缓存数量，增大可防止长训练中缓存淘汰导致性能下降 |
| `ACL_PRECISION_MODE` | `allow_fp32_to_fp16` | 允许不支持的 FP32 算子自动转换为 FP16 精度 |

示例：

```bash
export ACL_OP_COMPILER_CACHE_MODE=enable
export ACL_OP_COMPILER_CACHE_DIR=/tmp/npu_cache
export ASCEND_MAX_OP_CACHE_SIZE=5000
export ACL_PRECISION_MODE=allow_fp32_to_fp16
```

## 生产环境推荐配置

### 单机

多 NPU 分布式 RL 训练场景，建议在启动脚本或 ROLL YAML 配置中添加以下环境变量：

```bash
# HCCL 通信
export HCCL_CONNECT_TIMEOUT=3600
export HCCL_EXEC_TIMEOUT=3600
export HCCL_DETERMINISTIC=false
export HCCL_OP_EXPANSION_MODE="AIV"

# NPU 显存
export NPU_MEMORY_FRACTION=0.96
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export MULTI_STREAM_MEMORY_REUSE=1
export TASK_QUEUE_ENABLE=2
export COMBINED_ENABLE=1

# CPU 调度
export CPU_AFFINITY_CONF=2
export OMP_NUM_THREADS=1

# vLLM-Ascend 推理
export VLLM_USE_V1=1
export VLLM_ASCEND_ENABLE_FLASHCOMM=1
export VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE=1
export VLLM_ASCEND_ENABLE_PREFETCH_MLP=1

# 算子编译缓存
export ACL_OP_COMPILER_CACHE_MODE=enable
export ACL_OP_COMPILER_CACHE_DIR=/tmp/npu_cache
export ASCEND_MAX_OP_CACHE_SIZE=5000

# 日志（生产环境）
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASDOPS_LOG_LEVEL=ERROR
export ATB_LOG_LEVEL=ERROR
```

### 多机

多机训练需在单机配置基础上增加 Ray 集群变量：

```bash
# Ray 集群（多机）
export RANK=0                        # 0=主节点, 1/2/3=工作节点
export WORLD_SIZE=2                  # 集群总节点数
export MASTER_ADDR=10.0.0.1          # 主节点 IP
export MASTER_PORT=6379              # Ray 通信端口
export DASHBOARD_PORT=8265           # Ray Dashboard 端口

# HCCL 多机通信
export HCCL_CONNECT_TIMEOUT=3600
export HCCL_EXEC_TIMEOUT=3600
export HCCL_DETERMINISTIC=false
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_IF_IP=$(hostname -I | awk '{print $1}')
export HCCL_SOCKET_IFNAME="enp194s0f0"
export HCCL_IF_BASE_PORT=23456

# ...（其余 NPU 显存、CPU、vLLM、缓存、日志变量同上）
```

或通过 ROLL YAML 配置：

```yaml
system_envs:
  HCCL_CONNECT_TIMEOUT: "3600"
  HCCL_EXEC_TIMEOUT: "3600"
  HCCL_DETERMINISTIC: "false"
  HCCL_OP_EXPANSION_MODE: "AIV"
  HCCL_IF_IP: "10.0.0.1"
  HCCL_SOCKET_IFNAME: "enp194s0f0"
  HCCL_IF_BASE_PORT: "23456"
  NPU_MEMORY_FRACTION: "0.96"
  PYTORCH_NPU_ALLOC_CONF: "expandable_segments:True"
  CPU_AFFINITY_CONF: "2"
  OMP_NUM_THREADS: "1"
  COMBINED_ENABLE: "1"
  VLLM_USE_V1: "1"
  ACL_OP_COMPILER_CACHE_MODE: "enable"
  ACL_OP_COMPILER_CACHE_DIR: "/tmp/npu_cache"
```

## 声明

ROLL 中提供的 Ascend 支持代码皆为参考样例，生产环境使用请通过官方正式途径沟通。
