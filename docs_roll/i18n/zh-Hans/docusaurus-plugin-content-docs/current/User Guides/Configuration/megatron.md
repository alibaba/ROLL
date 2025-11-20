# Megatron 推理和训练后端配置指南

Megatron 是 NVIDIA 开发的大规模语言模型训练和推理框架，支持高效的分布式训练和推理。本文档将详细介绍如何在 ROLL 框架中配置和使用 Megatron 后端。

## Megatron 简介

Megatron 提供了高效的模型并行和数据并行策略，特别适合训练和推理超大规模的语言模型。它支持张量并行、流水线并行和专家并行等多种并行策略。

## 配置 Megatron 策略

在 ROLL 框架中，可以通过在 YAML 配置文件中设置 `strategy_args` 来配置 Megatron 训练和推理策略。

### 训练配置示例

以下是一个典型的 Megatron 训练配置示例（来自 `examples/qwen3-30BA3B-rlvr_megatron/rlvr_config_sglang.yaml`）：

```yaml
actor_train:
  model_args:
    disable_gradient_checkpointing: false
    dtype: bf16
    model_type: ~
  training_args:
    learning_rate: 1.0e-6
    weight_decay: 0
    per_device_train_batch_size: 1
    gradient_accumulation_steps: 32
    warmup_steps: 20
    num_train_epochs: 50
  strategy_args:
    strategy_name: megatron_train
    strategy_config:
      tensor_model_parallel_size: 2
      pipeline_model_parallel_size: 2
      virtual_pipeline_model_parallel_size: 8
      expert_model_parallel_size: 4
      context_parallel_size: 1
      use_distributed_optimizer: true
      sequence_parallel: true
      moe_token_dispatcher_type: "alltoall"
      moe_grouped_gemm: true
      moe_layer_recompute: true
  device_mapping: list(range(0,32))
  infer_batch_size: 2
```

### 推理配置示例

以下是一个典型的 Megatron 推理配置示例：

```yaml
reference:
  model_args:
    disable_gradient_checkpointing: true
    dtype: bf16
    model_type: ~
  strategy_args:
    strategy_name: megatron_infer
    strategy_config:
      tensor_model_parallel_size: 1
      pipeline_model_parallel_size: 1
      expert_model_parallel_size: 4
      moe_token_dispatcher_type: "alltoall"
      moe_grouped_gemm: true
  device_mapping: list(range(0,32))
  infer_batch_size: 2
```

### 配置参数详解

1. **strategy_name**: 
   - `megatron_train` 用于训练
   - `megatron_infer` 用于推理

2. **strategy_config**: mcore提供的并行优化配置都可以使用，具体可详细了解[megatron](https://github.com/NVIDIA/Megatron-LM/blob/main/README.md)的使用。 这里列举一些Megatron常用配置参数
   - `tensor_model_parallel_size`: 张量模型并行度，将模型的层内计算和内存分割到多个 GPU 上
   - `pipeline_model_parallel_size`: 流水线模型并行度，将模型的不同层分配到不同的 GPU 上
   - `virtual_pipeline_model_parallel_size`: 虚拟流水线并行度，用于改善流水线效率
   - `expert_model_parallel_size`: 专家模型并行度，在 MoE 模型中将不同专家分配到不同 GPU 上
   - `context_parallel_size`: 上下文并行度，用于处理超长序列
   - `use_distributed_optimizer`: 是否使用分布式优化器
   - `sequence_parallel`: 是否启用序列并行优化
   - `moe_token_dispatcher_type`: MoE 模型中的 token 调度器类型（'allgather' 或 'alltoall'）
   - `moe_grouped_gemm`: 是否为 MoE 专家启用分组 GEMM
   - `moe_layer_recompute`: 是否对 MoE 层进行检查点以节省激活内存
   - `recompute_granularity`: 激活值重计算粒度（'full' 或 'selective'）
   - `overlap_grad_reduce`: 是否在分布式优化器中将梯度 All-reduce 过程与反向传播计算重叠

3. **device_mapping**: 指定使用的 GPU 设备 ID 列表

4. **infer_batch_size**: 推理时的批次大小

## 与其他组件的集成

在配置示例中，我们可以看到：

1. `actor_train` 使用 Megatron 进行训练
2. `actor_infer` 可能使用其他推理后端（如 vLLM 或 SGLang）
3. `reference` 使用 Megatron 进行推理
4. 奖励模型可能使用不同的推理后端

这种设计允许不同组件根据其需求选择最适合的后端。

## 注意事项

1. Megatron 需要特定版本的依赖库，请确保安装了兼容的版本
2. 并行策略的设置需要根据硬件配置和模型大小进行调整
3. 在资源受限的环境中，需要仔细平衡不同组件的资源分配

通过合理配置 Megatron 后端，您可以充分发挥 ROLL 框架在大规模语言模型训练和推理方面的性能优势。