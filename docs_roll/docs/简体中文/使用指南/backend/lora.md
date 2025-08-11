# LoRA 微调配置指南

LoRA (Low-Rank Adaptation) 是一种高效的参数高效微调方法，通过在预训练模型中添加低秩矩阵来实现参数高效的微调。本文档将详细介绍如何在 ROLL 框架中配置和使用 LoRA 微调。

## LoRA 简介

LoRA 通过以下方式实现参数高效微调：
1. **低秩矩阵分解**：将权重更新矩阵分解为两个低秩矩阵的乘积
2. **参数效率**：只训练少量的新增参数，而不是全部模型参数
3. **易于部署**：微调后的模型可以轻松合并到原始模型中

## 配置 LoRA 微调

在 ROLL 框架中，可以通过在 YAML 配置文件中设置相关参数来配置 LoRA 微调。

### 配置示例

以下是一个典型的 LoRA 配置示例（来自 `examples/qwen2.5-7B-rlvr_megatron/rlvl_lora_zero3.yaml`）：

```yaml
# LoRA 全局配置
lora_target: o_proj,q_proj,k_proj,v_proj
lora_rank: 32
lora_alpha: 32

actor_train:
  model_args:
    attn_implementation: fa2
    disable_gradient_checkpointing: true
    dtype: bf16
    lora_target: ${lora_target}
    lora_rank: ${lora_rank}
    lora_alpha: ${lora_alpha}
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

actor_infer:
  model_args:
    attn_implementation: fa2
    disable_gradient_checkpointing: true
    dtype: bf16
    lora_target: ${lora_target}
    lora_rank: ${lora_rank}
    lora_alpha: ${lora_alpha}
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
      gpu_memory_utilization: 0.6
      enforce_eager: false
      block_size: 16
      max_model_len: 8000
  device_mapping: list(range(0,12))
  infer_batch_size: 1
```

### 配置参数详解

1. **lora_target**: 指定应用 LoRA 的模型层
   - 例如: `o_proj,q_proj,k_proj,v_proj` 表示在注意力机制的输出投影和查询、键、值投影层应用 LoRA
   - 可以根据具体模型结构进行调整

2. **lora_rank**: LoRA 矩阵的秩
   - 控制 LoRA 矩阵的大小
   - 较小的秩可以减少参数数量，但可能影响性能
   - 通常设置为 8、16、32、64 等

3. **lora_alpha**: LoRA 缩放因子
   - 控制 LoRA 更新的幅度
   - 通常设置为与 `lora_rank` 相同或为其倍数

4. **model_args 中的 LoRA 参数**:
   - `lora_target`: 指定应用 LoRA 的层
   - `lora_rank`: LoRA 矩阵的秩
   - `lora_alpha`: LoRA 缩放因子

## LoRA 与训练后端的兼容性

目前，LoRA 微调仅支持 DeepSpeed 训练后端：

```yaml
actor_train:
  strategy_args:
    strategy_name: deepspeed_train  # LoRA 仅支持 deepspeed_train
```

这是因为 DeepSpeed 提供了与 LoRA 良好集成的优化功能。

## 性能优化建议

1. **选择合适的 LoRA 层**：
   - 通常在注意力机制的相关层应用 LoRA 效果较好
   - 可以通过实验确定最佳的 LoRA 层组合

2. **调整 LoRA 参数**：
   - `lora_rank`: 根据模型大小和任务复杂度调整
   - `lora_alpha`: 通常设置为 `lora_rank` 或其倍数

3. **学习率设置**：
   - LoRA 微调通常需要较高的学习率
   - 在示例中设置为 `1.0e-5`

## 注意事项

1. LoRA 微调目前仅支持 DeepSpeed 训练后端
2. 需要确保模型支持 LoRA 微调
3. 在使用梯度检查点时需要注意与 LoRA 的兼容性
4. LoRA 微调的性能可能与全参数微调有所不同，需要根据具体任务进行评估

通过合理配置 LoRA 微调，您可以在保持模型性能的同时显著减少训练参数数量和计算资源消耗。