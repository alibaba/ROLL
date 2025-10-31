# Off-Policy 算法配置指南

ROLL 框架支持多种 Off-Policy 算法变体，用于async-training。本文档详细介绍各种算法的配置方法和使用示例。

## 支持的算法变体

ROLL 框架目前支持以下 Off-Policy 算法：

1. **vanilla** - 基础 Policy Gradient 算法
2. **ppo** - Proximal Policy Optimization
3. **tis** - Truncated Importance Sampling
4. **topr** - Tapered off-policy REINFORCE
5. **cispo** - Clipped Importance Sampling Policy Optimization
6. **kimi15** - Kimi15 算法

## 基础配置

### 核心参数

在配置文件中设置 Off-Policy 算法的基本参数：

```yaml
# 选择算法变体
pg_variant: topr  # 可选: vanilla, tis, topr, cispo, kimi15, ppo

# 训练配置
max_steps: 500
save_steps: 100
logging_steps: 1
eval_steps: 10

# 数据配置
rollout_batch_size: 128
prompt_length: 2048
response_length: 8192
num_return_sequences_in_group: 8

# 通用训练参数
ppo_epochs: 1
adv_estimator: "reinforce"
whiten_advantages: true
```

### Worker 配置

使用专门的 ActorPGWorker 来处理 Off-Policy 算法：

```yaml
actor_train:
  worker_cls: roll.pipeline.rlvr.actor_pg_worker.ActorPGWorker
  pg_variant: topr  # 与全局配置保持一致
  model_args:
    flash_attn: fa2
    disable_gradient_checkpointing: false
    dtype: bf16
  training_args:
    learning_rate: 1.0e-6
    weight_decay: 0
    per_device_train_batch_size: 1
    gradient_accumulation_steps: 64
    warmup_steps: 20
    num_train_epochs: 50
  strategy_args:
    strategy_name: megatron_train
    strategy_config:
      tensor_model_parallel_size: 1
      pipeline_model_parallel_size: 1
      use_distributed_optimizer: true
      recompute_granularity: full
  device_mapping: list(range(0,16))
```

## 各算法详细配置

### 1. Vanilla Policy Gradient

最基础的策略梯度算法，直接使用 log 概率和优势函数的乘积作为损失。

**配置特点：**
- 无需额外参数
- 计算效率高
- 适合简单的强化学习任务

```yaml
pg_variant: vanilla

# 无需额外配置参数
```

### 2. PPO (Proximal Policy Optimization)

近端策略优化算法，通过裁剪重要性采样比率来稳定训练。

**关键参数：**
```yaml
pg_variant: ppo

# PPO 特定参数
pg_clip: 0.2                    # 裁剪范围
pg_clip_low: 0.2               # 下界裁剪（可选）
pg_clip_high: 0.2              # 上界裁剪（可选）
use_pg_clip_range: false       # 是否使用不对称裁剪
dual_clip_loss: true           # 是否启用双重裁剪
```

**配置示例：**
```yaml
pg_variant: ppo
pg_clip: 0.2
dual_clip_loss: true
```

### 3. TIS (Truncated Importance Sampling)

截断重要性采样算法，将重要性采样比率限制在 [0, 1] 范围内。

**关键参数：**
```yaml
pg_variant: tis

# TIS 特定参数
tis_lower_bound: 0.0           # 下界
tis_upper_bound: 1.0           # 上界
```

**配置示例：**
```yaml
pg_variant: tis
tis_lower_bound: 0.0
tis_upper_bound: 1.0
```

### 4. TOPR (Tapered off-policy REINFORCE)

锥形离策略强化学习算法，根据奖励的正负分别采用不同的更新策略。

**算法特点：**
- 正样本：直接 SFT 更新，不使用重要性采样
- 负样本：使用 TIS 更新，限制重要性采样比率在 [0, 1]

**关键参数：**
```yaml
pg_variant: topr

# TOPR 特定参数
topr_positive_weight: 1.0      # 正样本权重
topr_negative_weight: 1.0      # 负样本权重
```

**配置示例：**
```yaml
pg_variant: topr
topr_positive_weight: 1.0
topr_negative_weight: 1.0
```

### 5. CISPO (Clipped Importance Sampling Policy Optimization)

截断重要性采样策略优化算法，使用截断的重要性采样权重和 stop-gradient 操作。

**关键参数：**
```yaml
pg_variant: cispo

# CISPO 特定参数
cispo_epsilon_low: 0.1         # 下界裁剪参数
cispo_epsilon_high: 0.1        # 上界裁剪参数
cispo_use_unified_mask: false  # 是否使用统一掩码
```

**配置示例：**
```yaml
pg_variant: cispo
cispo_epsilon_low: 0.1
cispo_epsilon_high: 0.1
cispo_use_unified_mask: false
```

### 6. Kimi15

基于 KL 正则化的策略梯度算法，在策略梯度项中加入 KL 散度正则化项。

**关键参数：**
```yaml
pg_variant: kimi15

# Kimi15 特定参数
kimi15_tau: 0.1               # 正则化参数
```

**配置示例：**
```yaml
pg_variant: kimi15
kimi15_tau: 0.1
```

## 完整配置示例

RLVR Off-Policy 配置示例参考: examples/qwen2.5-7B-rlvr-offpolicy/rlvr_config.yaml

