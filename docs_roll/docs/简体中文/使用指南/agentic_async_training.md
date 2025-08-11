# Agentic 异步训练功能使用指南

ROLL 框架支持 Agentic 异步训练功能，可以显著提高训练效率。本文档将详细介绍如何使用这一功能。

## 异步训练概述

在传统的同步训练中，训练和推理过程是串行执行的，即必须等待一批推理完成并收集到奖励后才能开始下一批推理。而在异步训练中，训练和推理可以并行进行，推理过程可以提前生成多个批次的数据，训练过程可以使用这些预先生成的数据进行学习。

## 开启异步训练

要开启 Agentic 异步训练功能，需要在在配置文件中设置 `async_generation_ratio` 参数。

### 配置参数

在 `roll/configs/base_config.py` 中定义了 `async_generation_ratio` 参数：

```python
async_generation_ratio: float = field(
    default=0,
    metadata={
        "help": "The ratio of ahead generation requests in pipeline, "
        "0 means synchronous pipeline. currently only integer is supported."
    },
)
```

### 示例配置

以下是一个完整的异步训练配置示例（来自 `examples/qwen2.5-7B-agentic_megatron/agentic_val_webshop_async.yaml`）：

```yaml
# 开启异步训练
async_generation_ratio: 1

# 其他相关配置
rollout_batch_size: 64
val_batch_size: 64
sequence_length: 8192

# 训练参数
max_steps: 1024
save_steps: 10000
logging_steps: 1
eval_steps: 10

# PPO 参数
ppo_epochs: 1
adv_estimator: "grpo"
whiten_advantages: true

# 模型配置
pretrain: Qwen/Qwen2.5-7B-Instruct
reward_pretrain: Qwen/Qwen2.5-7B-Instruct

# 角色配置
actor_train:
  model_args:
    attn_implementation: fa2
    disable_gradient_checkpointing: false
    dtype: bf16
  training_args:
    learning_rate: 1.0e-6
    weight_decay: 0
    per_device_train_batch_size: 1
    gradient_accumulation_steps: 16
    warmup_steps: 10
  strategy_args:
    strategy_name: megatron_train
    strategy_config:
      tensor_model_parallel_size: 1
      context_parallel_size: 1
      pipeline_model_parallel_size: 1
      expert_model_parallel_size: 1
      use_distributed_optimizer: true
      recompute_granularity: full
  device_mapping: list(range(0,4))
  infer_batch_size: 1

actor_infer:
  model_args:
    disable_gradient_checkpointing: true
    dtype: bf16
  generating_args:
    max_new_tokens: 1024
    top_p: 0.99
    top_k: 100
    num_beams: 1
    temperature: 0.99
    num_return_sequences: 1
  strategy_args:
    strategy_name: vllm
    strategy_config:
      gpu_memory_utilization: 0.8
      block_size: 16
      load_format: auto
  device_mapping: list(range(4,8))
  infer_batch_size: 1
```

## 异步训练的工作原理

1. 当 `async_generation_ratio` 设置为大于 0 的值时，框架会启动异步训练模式
2. 推理过程会提前生成 `async_generation_ratio` 倍于训练所需的数据
3. 训练过程可以使用这些预先生成的数据进行学习，而不需要等待当前批次的推理完成
4. 这种并行化处理可以显著提高训练效率，特别是在推理耗时较长的情况下

## 使用建议

1. 根据硬件资源和任务特点调整 `async_generation_ratio` 的值
2. 确保训练和推理角色分离部署
3. 监控训练过程中的资源使用情况，避免资源瓶颈
4. 在验证阶段会暂停异步生成，验证完成后恢复
