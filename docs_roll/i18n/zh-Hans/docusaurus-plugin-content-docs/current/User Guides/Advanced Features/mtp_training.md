# MTP (Multi-Token Prediction) 训练指南

## 概述

MTP (Multi-Token Prediction) 是一种通过并行预测多个未来 token 来加速推理的技术。ROLL 框架支持 MTP 模型的训练，可用于 SFT（监督微调）和 RL（强化学习）场景。

## 投机采样原理

### 自回归生成的瓶颈

大语言模型的文本生成是自回归过程：每生成一个 token，都需要完整的前向传播。对于长序列生成（如数学推理），这成为主要的性能瓶颈。

```
传统自回归生成：
  Step 1: 前向传播 → Token 1
  Step 2: 前向传播 → Token 2
  Step 3: 前向传播 → Token 3
  ...
  每个 token 都需要一次完整的前向传播
```

### 投机采样的思想

投机采样（Speculative Decoding）通过"预测-验证"的方式打破这个瓶颈：

1. **Draft（草稿）阶段**：使用一个小型模型快速生成 K 个候选 token
2. **Verify（验证）阶段**：主模型一次前向传播并行验证这 K 个 token
3. **Accept/Reject**：接受符合主模型概率分布的 token，拒绝不符合的

```
投机采样：
  Draft: 小模型快速生成 [Token 1, Token 2, Token 3, Token 4]
  Verify: 主模型一次前向传播验证所有候选
  结果: 接受前 3 个，拒绝第 4 个

  等效于：用 2 次前向传播（1次draft + 1次verify）生成了 3 个 token
```

### 为什么能加速？

关键洞察：**主模型的前向传播可以并行计算多个位置的 logits**。

传统方式下，生成 token 时只计算最后一个位置的 logits，其余位置的计算被浪费了。投机采样利用这一点，用一次主模型前向传播验证多个候选 token，从而提高计算效率。

### 加速效果取决于什么？

- **接受率**：draft model 的输出分布与主模型越接近，接受率越高
- **投机步数**：每次投机生成的候选 token 数量
- **Draft model 效率**：draft model 的推理速度

理想的 draft model 应该：
1. 输出分布接近主模型（高接受率）
2. 推理速度快（低 draft 开销）
3. 参数量小（低内存开销）

## 什么是 MTP？

MTP (Multi-Token Prediction) 是一种高效的 draft model 实现。与使用独立小模型不同，MTP 与主模型共享权重，具有以下优势：

### 与普通 LM 的区别

- **普通 LM**：用位置 t 的 hidden state 预测位置 t+1 的 token
- **MTP**：用位置 t 的 hidden state + 位置 t+1 的 token embedding 预测位置 t+2 的 token

```
普通 LM:    H(t) → predict(t+1)
MTP:        H(t) + E(t+1) → predict(t+2)
             ↑         ↑
        hidden state  embedding
```

### MTP 的优势

1. **权重共享**：MTP 共享主模型的 embedding 和 output layer，参数量增加很小（约 5-10%）
2. **高接受率**：MTP 直接利用主模型的 hidden states，输出分布自然接近主模型
3. **训练简单**：可以与主模型联合训练，无需单独训练 draft model

### MTP 的用途

1. **推理加速**：作为投机采样的 draft model，加速文本生成
2. **RL 训练加速**：在 RLVR 等场景中加速 rollout 生成，提高训练吞吐量

### 为什么 RL 训练需要 MTP？

在 RL 训练（如 RLVR）中，rollout 生成是主要瓶颈：

1. **大量生成需求**：每轮训练需要生成大量样本
2. **长序列生成**：数学推理等任务需要长 response
3. **推理引擎负载高**：actor_infer worker 往往是训练瓶颈

使用 MTP 投机采样可以显著加速 rollout 过程，提高训练吞吐量。

## 训练模式

ROLL 支持三种 MTP 训练模式，通过 `mtp_training_mode` 参数配置：

### 1. disabled（默认）

MTP 权重被加载但不参与训练。

```yaml
actor_train:
  mtp_training_mode: disabled  # 或不配置
```

**适用场景**：
- 只想使用预训练的 MTP 进行推理加速
- 不需要更新 MTP 权重

### 2. standalone（推荐用于 RL）

MTP 独立训练，梯度被截断，不影响主模型。

```yaml
actor_train:
  mtp_training_mode: standalone
```

**特点**：
- MTP 的梯度流被 `detach()` 截断
- 主模型梯度不受 MTP 训练影响
- 主模型和 MTP 使用不同的学习信号

**适用场景**：
- **RL 训练**：主模型需要根据 reward 优化，MTP 需要学习主模型的生成分布
- 避免强化学习的不稳定性影响 MTP

**工作原理**：

在 standalone 模式下，MTP 和主模型的梯度流完全隔离：
- 主模型根据 RL reward 进行优化，梯度正常反向传播
- MTP 根据 cross-entropy loss 进行优化，但梯度不会回传到主模型
- 这样 MTP 可以稳定地学习主模型的生成分布，而不受 RL 训练波动的影响

### 3. joint（推荐用于 SFT）

MTP 与主模型联合训练，梯度完整流动。

```yaml
actor_train:
  mtp_training_mode: joint
```

**特点**：
- 主模型和 MTP 共享梯度流
- MTP 的 loss 会影响主模型参数
- 主模型和 MTP 协同优化

**适用场景**：
- **SFT 训练**：希望主模型和 MTP 同时学习目标任务
- MTP 作为辅助训练目标

## 配置参数

### 训练参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `mtp_training_mode` | `str` | `disabled` | MTP 训练模式：`disabled`、`standalone`、`joint` |
| `mtp_loss_scaling_factor` | `float` | 见下文 | MTP loss 缩放系数 |

**mtp_loss_scaling_factor**：
- 默认值通常为 `0.3`（参考 DeepSeek-V3）
- 在 `standalone` 模式下，MTP loss 直接乘以该系数
- 在 `joint` 模式下，MTP loss 参与主模型的梯度更新

### 推理引擎配置（投机采样）

目前 ROLL 只有 vLLM 支持 MTP 投机采样功能。在 `actor_infer` 的 `strategy_config` 中配置：

```yaml
actor_infer:
  strategy_args:
    strategy_name: vllm
    strategy_config:
      tensor_parallel_size: 4
      # MTP 投机采样配置
      speculative_config:
        method: mtp
        num_speculative_tokens: 4
```

另外注意，无论使用哪种模式只要使用 MTP 都需要在 `actor_train` 的 `strategy_config` 中配置 `mtp_num_layers`（模型 config.json 中的相应值）

## 训练示例

### RLVR Pipeline with MTP

在 RLVR 训练中启用 MTP，需要在 `actor_train` 中配置 `mtp_training_mode: standalone`，在 `actor_infer` 中配置 `speculative_config`：

```yaml
actor_train:
  strategy_args:
    strategy_name: megatron_train
    strategy_config:
      tensor_model_parallel_size: 4
      pipeline_model_parallel_size: 2
      # ... 其他配置
  # MTP 训练配置（取消注释启用）
  #mtp_training_mode: standalone

actor_infer:
  strategy_args:
    strategy_name: vllm
    strategy_config:
      tensor_parallel_size: 4
      # ... 其他配置
      # 投机采样配置（取消注释启用）
      #speculative_config:
        # method: mtp
        # num_speculative_tokens: 3
```

完整配置示例请参考：
- `examples/qwen3.5-27B-rlvr_megatron/rlvr_megatron_80GB.yaml` - Qwen3.5-27B Dense 模型
- `examples/qwen3.5-35BA3-rlvr_megatron/rlvr_megatron_80GB.yaml` - Qwen3.5-35B-A3B MoE 模型

### SFT Pipeline with MTP

SFT 训练使用 `joint` 模式，让主模型和 MTP 协同学习：

```yaml
actor_train:
  model_args:
    model_name_or_path: Qwen/Qwen3.5-7B
    flash_attn: sdpa
    dtype: bf16
  training_args:
    learning_rate: 2.0e-5
    per_device_train_batch_size: 2
    gradient_accumulation_steps: 8
  data_args:
    file_name:
      - data/sft_data.jsonl
  strategy_args:
    strategy_name: megatron_train
    strategy_config:
      tensor_model_parallel_size: 2
      pipeline_model_parallel_size: 1
  # MTP 联合训练
  mtp_training_mode: joint
  mtp_loss_scaling_factor: 0.3
```

## 支持的模型

目前 ROLL 支持 Qwen3.5 系列模型的 MTP 训练：

| 模型 | MTP 层数 | 备注 |
|------|---------|------|
| Qwen3.5-7B | 1 | Dense 模型 |
| Qwen3.5-27B | 1 | Dense 模型 |
| Qwen3.5-35B-A3B | 1 | MoE 模型 |

MTP 相关配置在模型 checkpoint 中：

```json
// config.json
{
    "mtp_num_hidden_layers": 1,
    "mtp_use_dedicated_embeddings": false
}
```

## 注意事项

### 1. 模式选择

| 场景 | 推荐模式 | 原因 |
|------|---------|------|
| RL 训练 | `standalone` | 隔离 RL 梯度，MTP 学习主模型分布 |
| SFT 训练 | `joint` | 协同优化，MTP 作为辅助目标 |
| 仅推理加速 | `disabled` | 使用预训练 MTP，无需训练 |

### 2. 性能监控

监控以下指标评估 MTP 效果：
- **接受率（Acceptance Rate）**：投机采样的 token 接受比例
- **平均接受长度**：每次投机平均接受的 token 数
- **吞吐量提升**：相比非投机采样的加速比

## 相关文档

- [vLLM 配置指南](../Configuration/vllm.md) - vLLM 推理引擎详细配置
- [RLVR Pipeline](../Pipeline/rlvr_pipeline_start.md) - RLVR 训练流程
- [SFT Pipeline](../Pipeline/sft_pipeline_start.md) - SFT 训练流程
