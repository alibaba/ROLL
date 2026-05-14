# ROLL x CoreX

最后更新：2026年3月20日

本文档记录了当前本地 ROLL 工作区中为象帝先设备（如 `Iluvatar BI-V150`）集成的 CoreX 特定适配。

## 当前范围

当前适配针对的是类似 CUDA 的 CoreX 软件栈，在该栈中：

- `torch.cuda` 可用
- Ray 将加速器暴露为 `GPU`
- 由 `libixml.so` 提供兼容 NVML 的监控接口
- 供应商打过补丁的 `torch`、`megatron-core` 和 `vllm` 的行为可能与其上游版本不同

这是一个在当前机器上运行 ROLL 的实用兼容层。它还不是一个完整的官方上游硬件支持包。

## 已适配内容

### 1. 平台检测

ROLL 此前将 `Iluvatar BI-V150` 视为未知的 CUDA 设备。平台初始化逻辑现在能检测 CoreX 风格的设备名称，并创建一个专用的 `CorexPlatform`，而不是回退到 `UnknownPlatform`。

当前检测关键词包括：

- `ILUVATAR`
- `COREX`
- `BI-V`

实现：

- `roll/platforms/corex.py`
- `roll/platforms/__init__.py`

### 2. 安全的 CUDA 平台初始化

在此供应商栈上，子进程可能遇到以下状态：

- `torch.cuda.is_available()` 实际可用
- 但在当前可见性范围内 `device_count() == 0`

在该状态下直接调用 `torch.cuda.get_device_name()` 可能会引发 `AssertionError: Invalid device id`。

平台引导程序现在会首先检查 `device_count()`，并且仅在存在至少一个可见设备时才查询 CUDA 设备名称。

实现：

- `roll/platforms/__init__.py`

### 3. Ray GPU 资源注册

即使 `torch` 可以看到设备，Ray 在此机器上也不会自动将 CoreX GPU 注册为 `GPU` 资源。这导致调度器认为集群拥有零个可用 GPU 节点。

ROLL 现在在启动 Ray 时会显式指定加速器资源，以便集群暴露预期的 `GPU` 计数。

实现：

- `roll/distributed/scheduler/initialize.py`
- `roll/distributed/scheduler/resource_manager.py`

### 4. 通过 `libixml.so` 实现 NVML 兼容的内存监控

供应商栈未提供 `libnvidia-ml.so.1`，因此上游的 `torch.cuda.device_memory_used()` 在尝试初始化 NVML 时会失败。但是，CoreX 通过 `libixml.so` 暴露了一个兼容 NVML 的 API。

ROLL 现在会：

1. 首先尝试使用上游的 `torch.cuda.device_memory_used()`
2. 如果失败，则尝试加载一个兼容 NVML 的库
3. 按以下顺序回退：
   - 标准 NVML（如果存在）
   - `libixml.so`
4. 调用：
   - `nvmlInit_v2` / `nvmlInit`
   - `nvmlDeviceGetHandleByIndex_v2` / `nvmlDeviceGetHandleByIndex`
   - `nvmlDeviceGetMemoryInfo`
5. 在查询物理句柄之前，通过 `CUDA_VISIBLE_DEVICES` 映射逻辑设备索引

也支持通过以下方式覆盖库路径：

```bash
export ROLL_NVML_COMPAT_LIB=/path/to/libixml.so
```

实现：

- `roll/platforms/platform.py`

### 5. Megatron 优化器兼容性

此机器上安装的供应商打过补丁的 `megatron-core` 使用了比之前 ROLL 兼容层所期望的更新的 `_get_param_groups_and_buffers()` 签名。

ROLL 现在会检查运行时函数的签名，并在需要时传递新的必需参数，同时保持与旧版本的兼容性。

实现：

- `roll/third_party/megatron/optimizer.py`

### 6. vLLM 休眠/卸载兼容性

当前的 CoreX 栈未暴露 vLLM 休眠模式所需的分配器后端（`cumem`）。当 ROLL 在此机器上强制启用休眠模式时，vLLM 会在初始化期间崩溃。

ROLL 现在会：

- 当分配器后端不可用时，自动禁用 vLLM 休眠/卸载功能
- 当这意味着 `actor_infer` 将常驻 GPU 时，发出明确的警告
- 如果用户仍在使用了共置的训练/推理布局，再次发出警告

实现：

- `roll/third_party/vllm/__init__.py`
- `roll/distributed/strategy/vllm_strategy.py`

### 7. 单节点演示布局调整

在此 CoreX 栈上，在 vLLM 休眠模式变得不可用之后，将 `actor_train` 和 `actor_infer` 共置于同一 GPU 上对于默认的 frozen-lake 代理演示来说不稳定。主要故障模式是在 Megatron 优化器步骤中发生 OOM。

frozen-lake 单节点演示调整为 2-GPU 分离布局：

- `actor_train` 在 GPU 0 上
- `actor_infer` 在 GPU 1 上
- 降低 vLLM 的 `gpu_memory_utilization`

实现：

- `examples/agentic_demo/agent_val_frozen_lake_single_node_demo.yaml`

## 已执行的验证

在当前机器上执行了以下检查：

- `torch.cuda.get_device_name(0)` 返回 `Iluvatar BI-V150`
- `ldconfig -p` 暴露了 `libixml.so`
- 对 `libixml.so` 的直接 `ctypes` 调用成功执行了：
  - `nvmlInit_v2`
  - `nvmlDeviceGetHandleByIndex_v2`
  - `nvmlDeviceGetMemoryInfo`
- `current_platform.device_memory_used()` 成功通过 NVML 兼容路径报告了内存使用情况
- 在更改为分离布局后，frozen-lake 单节点流水线成功运行了第 0 步和第 1 步

## 新增或更新的测试

- `tests/platforms/test_platform_init.py`
- `tests/platforms/test_platform_memory.py`
- `tests/distributed/scheduler/test_initialize.py`
- `tests/distributed/scheduler/test_resource_manager.py`
- `tests/third_party/megatron/test_optimizer_compat.py`

## 已知限制

- CoreX 目前作为类似 CUDA 的平台集成，而非具有供应商特定内核或调度行为的完全独立后端。
- vLLM 休眠模式在当前栈上仍被禁用，因为所需的分配器后端不可用。
- 当前的适配优先保证可靠执行，而不是保留原始的单 GPU 共置演示拓扑。
- 必须重启现有的长时间运行进程，才能应用最新的平台和监控更改。

## 推荐运行命令

```bash
conda activate ROLL
ray stop
bash examples/agentic_demo/run_agentic_pipeline_frozen_lake_single_node_demo.sh
```