---
sidebar_position: 5
---

# Rollout Dump Mock 使用指南

## 概述

Rollout Dump Mock是ROLL框架提供的强大调试工具，用于**消除RL训练中rollout阶段的随机性**，实现数值级精度对齐验证。它通过保存和回放rollout数据，帮助开发者快速验证计算优化的正确性。

### 核心价值

- **消除随机性**：实现数值级精度对齐验证
- **快速迭代**：Mock模式下跳过昂贵的环境rollout
- **可复现调试**：捕获问题rollout数据，反复调试
- **架构透明**：在Scheduler层实现，对Pipeline完全无感知

### 适用场景

| 场景 | 说明 |
|------|------|
| **计算优化验证** | 验证dynamic_batching、sequence_packing等优化的数值一致性 |
| **模型并行验证** | 验证TP、PP、EP等并行策略的精度对齐 |
| **回归测试** | CI/CD中自动化精度测试 |

---

## 快速开始

### 典型工作流

```
[1. Dump模式] → [2. 修改代码] → [3. Mock模式] → [4. 精度验证]
    ↓              ↓                 ↓               ↓
 捕获基准数据    优化计算逻辑      确定性回放      数值对比
```

### Step 1: Dump模式 - 捕获基准数据

在修改代码前，先捕获正确的rollout数据作为基准。

**配置文件** (`agentic_sokoban_rollout_mock_dump.yaml`)：
```yaml
exp_name: "sokoban_precision_test_dump"
max_steps: 50

# Rollout Mock Configuration - DUMP MODE
rollout_mock:
  enable: true
  mode: dump
  dump_dir: ./output/rollout_dumps/baseline_v1

# 用于确定性执行的环境变量
system_envs:
  NCCL_ALGO: Ring
  NVTE_ALLOW_NONDETERMINISTIC_ALGO: '0'
  CUBLAS_WORKSPACE_CONFIG: ':4096:8'
  DETERMINISTIC_MODE: '1'

# ... 其他配置 ...
```

**命令**：
```bash
python examples/start_agentic_pipeline.py \
  --config_name agentic_sokoban_rollout_mock_dump \
  --config_path examples/qwen2.5-0.5B-agentic
```

**输出**：
```
./output/rollout_dumps/baseline_v1/
  └── train/
      ├── step_000000.pkl  (~5MB)
      ├── step_000001.pkl
      ├── step_000002.pkl
      ├── ...
      └── step_000049.pkl
```

**日志示例**：
```
[Rollout Mock] Rollout Mock enabled: mode=dump, dir=./output/rollout_dumps/baseline_v1
[Rollout Mock] Dumped step 0: ./output/rollout_dumps/baseline_v1/train/step_000000.pkl (samples=128, size=4.82MB)
[Rollout Mock] Dumped step 1: ./output/rollout_dumps/baseline_v1/train/step_000001.pkl (samples=128, size=4.85MB)
```

### Step 2: 修改代码

实现你的计算优化，例如：
- 添加dynamic_batching
- 实现sequence_packing
- 迁移到新的并行策略

### Step 3: Mock模式 - 确定性回放

使用预录制的rollout数据，验证修改后的代码是否保持数值一致性。

**配置文件** (`agentic_sokoban_rollout_mock_mock.yaml`)：
```yaml
exp_name: "sokoban_precision_test_mock"
max_steps: 50

# Rollout Mock Configuration - MOCK MODE
rollout_mock:
  enable: true
  mode: mock
  dump_dir: ./output/rollout_dumps/baseline_v1  # 与dump模式相同路径

# 用于确定性执行的环境变量（保持与dump模式一致）
system_envs:
  NCCL_ALGO: Ring
  NVTE_ALLOW_NONDETERMINISTIC_ALGO: '0'
  CUBLAS_WORKSPACE_CONFIG: ':4096:8'
  DETERMINISTIC_MODE: '1'

# ... 其他配置（保持与dump模式一致）...
```

**命令**：
```bash
python examples/start_agentic_pipeline.py \
  --config_name agentic_sokoban_rollout_mock_mock \
  --config_path examples/qwen2.5-0.5B-agentic
```

**行为**：
- ✅ 直接从磁盘加载每步的DataProto
- ✅ 后续所有计算（advantages, losses, gradients）完全确定

**日志示例**：
```
[Rollout Mock] Rollout Mock enabled: mode=mock, dir=./output/rollout_dumps/baseline_v1
[Rollout Mock] Loaded step 0: ./output/rollout_dumps/baseline_v1/train/step_000000.pkl (samples=128)
[Rollout Mock] Loaded step 1: ./output/rollout_dumps/baseline_v1/train/step_000001.pkl (samples=128)
```

### Step 4: 数值精度验证

对比baseline和优化版本的训练指标，确保数值完全一致。可以通过查看日志中的关键指标（如pg_loss、total_loss、value_loss、approx_kl、grad_norm等）来验证两次运行的结果是否一致。

---

## 配置参数

### 配置Schema

在你的YAML配置文件中添加 `rollout_mock` 段：

```yaml
rollout_mock:
  enable: bool              # 启用rollout dump/mock机制
  mode: "dump" | "mock"     # dump: 保存数据, mock: 加载数据
  dump_dir: str             # 数据存储目录
```

### 配置示例

**Dump模式配置**：
```yaml
rollout_mock:
  enable: true
  mode: dump
  dump_dir: ./rollout_dumps/precision_test_v1
```

**Mock模式配置**：
```yaml
rollout_mock:
  enable: true
  mode: mock
  dump_dir: ./rollout_dumps/precision_test_v1  # 与dump模式相同路径
```

### 确定性执行的环境变量

为确保完全的数值可复现性，需要配置以下环境变量：

```yaml
system_envs:
  NCCL_ALGO: Ring                           # 使用Ring算法进行NCCL通信
  NVTE_ALLOW_NONDETERMINISTIC_ALGO: '0'     # 禁用Transformer Engine中的非确定性算法
  CUBLAS_WORKSPACE_CONFIG: ':4096:8'        # 启用确定性的CUDA操作
  DETERMINISTIC_MODE: '1'                   # 启用PyTorch确定性模式
```

**DETERMINISTIC_MODE 的作用**：
- 设置 `torch.backends.cudnn.deterministic = True` 以确保cuDNN操作的可复现性
- 设置 `torch.backends.cudnn.benchmark = False` 禁用导致非确定性的自动调优
- 调用 `torch.use_deterministic_algorithms(True)` 强制使用确定性的PyTorch算法

**重要提示**：这些环境变量在dump和mock模式之间必须保持一致，以确保数值精度对齐。

### 关键注意事项

1. **dump_dir必须一致**：Dump和Mock模式必须使用相同的`dump_dir`路径
2. **mode必须匹配**：Scheduler的mode（train/val）必须与dump时一致
3. **max_steps不能超过**：Mock模式的`max_steps`不能超过Dump模式时的值
4. **system_envs必须一致**：确定性执行的环境变量在dump和mock模式之间必须保持一致

---

## 常见问题与排查

### 问题1: Mock文件不存在

**错误信息**：
```
FileNotFoundError: [Rollout Mock] Mock文件不存在: ./dumps/baseline/train/step_000005.pkl
可能的原因:
  1. 未在dump模式下运行过step 5
  2. dump_dir配置不正确: ./dumps/baseline
  3. mode不匹配(当前: train)
请先以dump模式运行,确保生成了所有步骤的数据。
```

**排查步骤**：

1. 检查dump模式下是否运行了足够的步骤：
   ```bash
   ls -lh ./output/rollout_dumps/baseline_v1/train/
   # 应该看到 step_000000.pkl ~ step_000049.pkl
   ```

2. 确认`max_steps`一致：
   ```bash
   # Dump时: max_steps=50
   # Mock时: max_steps=50 (必须一致或更小)
   ```

3. 确认`dump_dir`路径正确：
   ```yaml
   # Dump时
   dump_dir: ./output/rollout_dumps/baseline_v1

   # Mock时 (必须相同)
   dump_dir: ./output/rollout_dumps/baseline_v1
   ```

### 问题2: Mode不匹配

**问题**：Dump时使用train mode，Mock时误用val mode。

**文件结构**：
```
dumps/baseline/
  ├── train/       # Dump时生成
  │   └── step_*.pkl
  └── val/         # 空目录
      └── (无文件)
```

**解决**：确保dump和mock使用相同的scheduler mode（train/val）。

### 问题3: 磁盘空间不足

**症状**：Dump过程中报错：
```
OSError: [Errno 28] No space left on device
```

**估算磁盘占用**：
```
单步文件大小 ≈ batch_size × seq_len × 数据类型大小
             ≈ 128 × 512 × 4 bytes (float32)
             ≈ 256KB ~ 10MB (取决于序列长度和metadata)

总磁盘占用 ≈ 单步大小 × max_steps
          ≈ 5MB × 100 steps = 500MB
```

**解决**：
- 增加磁盘空间
- 减少`max_steps`
- 使用网络存储（OSS等）

### 问题4: Pickle版本不兼容

**症状**：在不同Python版本间加载报错：
```
pickle.UnpicklingError: invalid load key, '\x00'
```

**原因**：Pickle在不同Python版本间的兼容性问题。

**解决**：
- 确保dump和mock使用相同Python版本
- 或在dump时使用较低的protocol版本（需修改源码）
