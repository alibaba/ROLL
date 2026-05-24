# Ascend NPU Environment Configuration Guide

Last updated: 04/27/2026.

This document describes the key environment variables for running ROLL on Huawei Ascend NPU, covering device management, HCCL communication, memory optimization, CPU scheduling, vLLM-Ascend inference, and debugging.

## Environment Variables Set by ROLL

ROLL automatically injects the following environment variables at runtime (defined in `roll/platforms/npu.py`):

| Variable | Value | Description |
| -------- | ----- | ----------- |
| `ASCEND_RT_VISIBLE_DEVICES` | e.g. `"0,1,2,3"` | Controls NPU device visibility, analogous to `CUDA_VISIBLE_DEVICES` for GPU |
| `RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES` | `"1"` | Prevents Ray from overriding `ASCEND_RT_VISIBLE_DEVICES` |
| `VLLM_ALLOW_INSECURE_SERIALIZATION` | `"1"` | Allows vLLM to use insecure serialization for cross-process tensor transfer via Ray |
| `RAY_get_check_signal_interval_milliseconds` | `"1"` | Reduces Ray plasma lock hold time to avoid lock starvation under multi-worker load |
| `RAY_CGRAPH_get_timeout` | `"600"` | Ray compute graph fetch timeout in seconds |

## Docker Image Environment Variables

The pre-built Ascend images described in [Ascend NPU Docker Usage Guide](ascend_docker_usage.md) include the following environment settings:

| Variable | Value | Description |
| -------- | ----- | ----------- |
| `ASCEND_HOME_PATH` | `/usr/local/Ascend/ascend-toolkit/latest` | CANN toolkit root path |
| `LD_LIBRARY_PATH` | Includes multiple Ascend `lib64` paths | Dynamic library search path, ensures `libascendcl.so` etc. can be loaded |

The following CANN environment scripts are automatically sourced via `/root/.bashrc` in the pre-built images:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
```

## Ray Cluster Environment Variables (Multi-Node)

These variables control how ROLL forms a Ray cluster across multiple NPU nodes. They are defined in `roll/distributed/scheduler/driver_utils.py` and consumed by `roll/distributed/scheduler/initialize.py`:

| Variable | Default | Description |
| -------- | ------- | ----------- |
| `RANK` | `0` | Node rank. `0` = head node, `1, 2, 3...` = worker nodes |
| `WORLD_SIZE` | `1` | Total number of nodes in the cluster |
| `MASTER_ADDR` | `127.0.0.1` | IP address of the head node |
| `MASTER_PORT` | `6379` | Ray head node port (also default Ray port) |
| `DASHBOARD_PORT` | `8265` | Ray dashboard web UI port |
| `WORKER_ID` | `<MASTER_ADDR>:<RANK>` | Node name used in Ray cluster, auto-derived if not set |

When `RANK=0`, ROLL automatically runs `ray start --head --port=<MASTER_PORT>`. When `RANK>0`, ROLL sleeps 5 seconds then runs `ray start --address=<MASTER_ADDR>:<MASTER_PORT>` to join the cluster. After all nodes join, worker nodes exit (`sys.exit(0)`), leaving only the head node to execute the training pipeline.

Example (head node, set before launching the pipeline):

```bash
export RANK=0
export WORLD_SIZE=2
export MASTER_ADDR=10.0.0.1
export MASTER_PORT=6379
export DASHBOARD_PORT=8265
```

Example (worker node, set before joining):

```bash
export RANK=1
export WORLD_SIZE=2
export MASTER_ADDR=10.0.0.1
export MASTER_PORT=6379
```

You can also pre-start Ray manually (`ray start --head` / `ray start --address=...`) before running ROLL. ROLL will detect the existing cluster and skip auto-start.

## HCCL Communication Variables

These variables control the behavior of HCCL (Huawei Collective Communication Library), the distributed communication backend for NPU (equivalent to NCCL on GPU):

| Variable | Recommended Value | Description |
| -------- | ----------------- | ----------- |
| `HCCL_CONNECT_TIMEOUT` | `3600` | Link establishment timeout in seconds (default 120s). Increase for large model training |
| `HCCL_EXEC_TIMEOUT` | `3600` | Collective operation execution timeout in seconds. Increase for long-running training steps |
| `HCCL_DETERMINISTIC` | `false` | Disable deterministic computation. Enabling it significantly reduces communication performance |
| `HCCL_OP_EXPANSION_MODE` | `"AIV"` | Communication algorithm dispatch location. `AIV` uses Vector Core, outperforms `AI_CPU`/`HOST`/`HOST_TS` |
| `HCCL_BUFFSIZE` | e.g. `"2147483648"` | HCCL communication buffer size in bytes. Increase for large data volume scenarios |
| `HCCL_IF_IP` | Node's IP address | Specify the IP address used by HCCL for inter-node communication. Required for multi-node training |
| `HCCL_SOCKET_IFNAME` | e.g. `"enp194s0f0"` | Network interface name for HCCL socket communication. Must be consistent across all nodes |
| `HCCL_IF_BASE_PORT` | e.g. `23456` | Base port for HCCL inter-node communication. Ensure ports are not blocked by firewall |
| `HCCL_WHITELIST_DISABLE` | `1` | Disable HCCL whitelist check. May be needed when encountering communication errors in certain environments |

Example (single-node):

```bash
export HCCL_CONNECT_TIMEOUT=3600
export HCCL_DETERMINISTIC=false
export HCCL_OP_EXPANSION_MODE="AIV"
```

Example (multi-node):

```bash
export HCCL_CONNECT_TIMEOUT=3600
export HCCL_EXEC_TIMEOUT=3600
export HCCL_DETERMINISTIC=false
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_IF_IP=$(hostname -I | awk '{print $1}')
export HCCL_SOCKET_IFNAME="enp194s0f0"
export HCCL_IF_BASE_PORT=23456
```

## NPU Memory Variables

| Variable | Recommended Value | Description |
| -------- | ----------------- | ----------- |
| `NPU_MEMORY_FRACTION` | `0.96` | Fraction of NPU memory available for use (default 0.8). Increase to 0.95+ for large model inference |
| `PYTORCH_NPU_ALLOC_CONF` | `expandable_segments:True` | Enable PyTorch NPU memory pool expandable segments, reducing memory fragmentation and OOM risk |
| `MULTI_STREAM_MEMORY_REUSE` | `1` | Enable multi-stream memory reuse to reduce memory footprint |
| `TASK_QUEUE_ENABLE` | `2` | Task dispatch optimization. Set to `2` for non-graph mode, `1` for graph mode |
| `COMBINED_ENABLE` | `1` | Enable operator combination optimization. Fuses multiple small operators into larger ones to reduce kernel launch overhead |

Example:

```bash
export NPU_MEMORY_FRACTION=0.96
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export MULTI_STREAM_MEMORY_REUSE=1
export TASK_QUEUE_ENABLE=2
export COMBINED_ENABLE=1
```

## CPU Scheduling Variables

| Variable | Recommended Value | Description |
| -------- | ----------------- | ----------- |
| `CPU_AFFINITY_CONF` | `2` | CPU core affinity optimization to avoid cross-NUMA memory access. `1`=coarse-grained, `2`=fine-grained (recommended) |
| `OMP_NUM_THREADS` | `1` | OpenMP thread count. Set to 1 in distributed training to avoid over-subscription |

Example:

```bash
export CPU_AFFINITY_CONF=2
export OMP_NUM_THREADS=1
```

Custom per-NPU affinity is also supported:

```bash
export CPU_AFFINITY_CONF=1,npu0:0-1,npu1:2-3,npu2:4-5,npu3:6-7
```

## vLLM-Ascend Inference Variables

| Variable | Recommended Value | Description |
| -------- | ----------------- | ----------- |
| `VLLM_USE_V1` | `1` | Enable vLLM V1 architecture. Required for vLLM-Ascend |
| `VLLM_ATTENTION_BACKEND` | `XFORMERS` | vLLM attention computation backend |
| `VLLM_ASCEND_ENABLE_FLASHCOMM` | `1` | Enable Ascend FlashComm high-speed communication optimization |
| `VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE` | `1` | Enable dense computation optimization for large model inference |
| `VLLM_ASCEND_ENABLE_PREFETCH_MLP` | `1` | Enable MLP layer weight prefetching |
| `VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE` | `1` | Enable TopK operator fusion optimization for generation decoding |
| `VLLM_ASCEND_MODEL_EXECUTE_TIME_OBSERVE` | `1` | Print prefill/decode phase timing details (for debugging) |
| `VLLM_ASCEND_TRACE_RECOMPILES` | `1` | Trace operator recompilation for debugging performance issues |
| `VLLM_ENABLE_MC2` | `1` | Enable MC2 communication optimization for multi-node inference |

Example:

```bash
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_ASCEND_ENABLE_FLASHCOMM=1
export VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE=1
export VLLM_ASCEND_ENABLE_PREFETCH_MLP=1
```

## CANN Logging & Debugging Variables

| Variable | Recommended Value | Description |
| -------- | ----------------- | ----------- |
| `ASCEND_GLOBAL_LOG_LEVEL` | `3` (ERROR) | CANN log level: 0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR |
| `ASCEND_SLOG_PRINT_TO_STDOUT` | `1` | Print CANN logs to stdout (for debugging) |
| `ASDOPS_LOG_LEVEL` | `ERROR` | Operator library log level |
| `ATB_LOG_LEVEL` | `ERROR` | ATB acceleration library log level |
| `ASCEND_LAUNCH_BLOCKING` | `1` | Enable synchronous execution for error localization. Set to `1` only when debugging NPU errors, as it disables async execution and severely degrades performance |

:::caution
Leaving debug/info log levels enabled in production will significantly degrade performance. Always set log levels to ERROR for production workloads.
:::

Example (debugging):

```bash
export ASCEND_GLOBAL_LOG_LEVEL=0
export ASCEND_SLOG_PRINT_TO_STDOUT=1
export ASCEND_LAUNCH_BLOCKING=1
```

Example (production):

```bash
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASDOPS_LOG_LEVEL=ERROR
export ATB_LOG_LEVEL=ERROR
```

## CANN Operator Compilation & Precision Variables

| Variable | Recommended Value | Description |
| -------- | ----------------- | ----------- |
| `ACL_OP_COMPILER_CACHE_MODE` | `enable` | Enable operator compilation cache to avoid recompilation on repeated runs |
| `ACL_OP_COMPILER_CACHE_DIR` | e.g. `/tmp/npu_cache` | Directory to store operator compilation cache |
| `ASCEND_MAX_OP_CACHE_SIZE` | e.g. `5000` | Maximum operator cache size. Increase to prevent performance degradation from cache eviction during long training |
| `ACL_PRECISION_MODE` | `allow_fp32_to_fp16` | Allow automatic FP32-to-FP16 precision conversion for unsupported FP32 operators |

Example:

```bash
export ACL_OP_COMPILER_CACHE_MODE=enable
export ACL_OP_COMPILER_CACHE_DIR=/tmp/npu_cache
export ASCEND_MAX_OP_CACHE_SIZE=5000
export ACL_PRECISION_MODE=allow_fp32_to_fp16
```

## Recommended Production Configuration

### Single-Node

For single-node multi-NPU distributed RL training, add the following to your startup script or ROLL YAML config:

```bash
# HCCL communication
export HCCL_CONNECT_TIMEOUT=3600
export HCCL_EXEC_TIMEOUT=3600
export HCCL_DETERMINISTIC=false
export HCCL_OP_EXPANSION_MODE="AIV"

# NPU memory
export NPU_MEMORY_FRACTION=0.96
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export MULTI_STREAM_MEMORY_REUSE=1
export TASK_QUEUE_ENABLE=2
export COMBINED_ENABLE=1

# CPU scheduling
export CPU_AFFINITY_CONF=2
export OMP_NUM_THREADS=1

# vLLM-Ascend inference
export VLLM_USE_V1=1
export VLLM_ASCEND_ENABLE_FLASHCOMM=1
export VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE=1
export VLLM_ASCEND_ENABLE_PREFETCH_MLP=1

# Operator compilation cache
export ACL_OP_COMPILER_CACHE_MODE=enable
export ACL_OP_COMPILER_CACHE_DIR=/tmp/npu_cache
export ASCEND_MAX_OP_CACHE_SIZE=5000

# Logging (production)
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASDOPS_LOG_LEVEL=ERROR
export ATB_LOG_LEVEL=ERROR
```

### Multi-Node

For multi-node training, add the Ray cluster variables on top of the single-node configuration:

```bash
# Ray cluster (multi-node)
export RANK=0                        # 0=head, 1/2/3=worker
export WORLD_SIZE=2                  # Total number of nodes
export MASTER_ADDR=10.0.0.1          # Head node IP
export MASTER_PORT=6379              # Ray communication port
export DASHBOARD_PORT=8265           # Ray dashboard port

# HCCL multi-node communication
export HCCL_CONNECT_TIMEOUT=3600
export HCCL_EXEC_TIMEOUT=3600
export HCCL_DETERMINISTIC=false
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_IF_IP=$(hostname -I | awk '{print $1}')
export HCCL_SOCKET_IFNAME="enp194s0f0"
export HCCL_IF_BASE_PORT=23456

# ... (rest of NPU memory, CPU, vLLM, cache, logging variables as above)
```

Or configure via ROLL YAML:

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

## Disclaimer

The Ascend support provided in ROLL is intended as a reference example. For production use, please consult official channels.
