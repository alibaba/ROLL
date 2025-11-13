# DeepSpeed 训练后端配置指南

DeepSpeed 是微软开发的高效深度学习优化库，提供了内存优化、分布式训练和性能优化等功能。本文档将详细介绍如何在 ROLL 框架中配置和使用 DeepSpeed 训练后端。

## DeepSpeed 简介

DeepSpeed 提供了多种优化技术，包括：
1. **ZeRO 优化**：通过分区优化器状态、梯度和参数来减少内存使用
2. **内存高效训练**：支持大规模模型的训练
3. **高性能通信**：优化分布式训练中的通信效率
4. **灵活的配置**：支持多种优化级别的配置

## 配置 DeepSpeed 策略

在 ROLL 框架中，可以通过在 YAML 配置文件中设置 `strategy_args` 来配置 DeepSpeed 训练策略。

### 配置示例

以下是一个典型的 DeepSpeed 配置示例（来自 `examples/qwen2.5-7B-rlvr_megatron/rlvl_lora_zero3.yaml`）：

```yaml
defaults:
  - ../config/deepspeed_zero3@_here_
  - ../config/deepspeed_zero3_cpuoffload@_here_

actor_train:
  model_args:
    attn_implementation: fa2
    disable_gradient_checkpointing: true
    dtype: bf16
    model_type: ~
  training_args:
    learning_rate: 1.0e-5
    weight_decay: 0
    per_device_train_batch_size: 1
    gradient_accumulation_steps: 32
    warmup_steps: 20
    num_train_epochs: 50
  strategy_args:
    strategy_name: deepspeed_train
    strategy_config: ${deepspeed_zero3}
  device_mapping: list(range(0,16))
  infer_batch_size: 4
```

### 配置参数详解

1. **strategy_name**: 设置为 `deepspeed_train` 以使用 DeepSpeed 训练后端

2. **strategy_config**: DeepSpeed 特定的配置参数
   - 可以引用预定义的配置文件，如 `${deepspeed_zero3}`
   - 在 `./examples/config/` 目录中有多种 DeepSpeed 配置文件可供选择：
     - `deepspeed_zero3.yaml`: ZeRO-3 配置
     - `deepspeed_zero3_cpuoffload.yaml`: 带 CPU 卸载的 ZeRO-3 配置

3. **defaults 部分**: 引入预定义的 DeepSpeed 配置
   ```yaml
   defaults:
     - ../config/deepspeed_zero@_here_
     - ../config/deepspeed_zero2@_here_
     - ../config/deepspeed_zero3@_here_
     - ../config/deepspeed_zero3_cpuoffload@_here_
   ```

4. **device_mapping**: 指定使用的 GPU 设备 ID 列表

## DeepSpeed 配置文件

在 `./examples/config/` 目录中提供了多种预定义的 DeepSpeed 配置文件：

1. **deepspeed_zero3.yaml**: ZeRO-3 配置，优化器状态、梯度和参数分区
2. **deepspeed_zero3_cpuoffload.yaml**: 带 CPU 卸载的 ZeRO-3 配置

### 使用预定义配置

要使用预定义的 DeepSpeed 配置，可以在 YAML 文件中这样引用：

```yaml
defaults:
  - ../config/deepspeed_zero3@_here_

actor_train:
  strategy_args:
    strategy_name: deepspeed_train
    strategy_config: ${deepspeed_zero3}
```

## 与其他组件的集成

在配置示例中，我们可以看到：

1. `actor_train` 使用 DeepSpeed 进行训练
2. `actor_infer` 可能使用其他推理后端（如 vLLM）
3. `reference` 使用 Hugging Face 推理后端
4. 奖励模型使用不同的推理后端

这种设计允许不同组件根据其需求选择最适合的后端。


## 注意事项

1. DeepSpeed 需要特定版本的依赖库，请确保安装了兼容的版本
2. 不同的 ZeRO 级别有不同的内存和性能特征，需要根据具体需求选择
3. 在使用 LoRA 微调时，需要注意与 DeepSpeed 的兼容性
