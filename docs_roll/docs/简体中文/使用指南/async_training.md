# ROLL 异步训练功能使用指南

ROLL 框架现已支持 RLVR 和 Agentic pipeline 的异步训练功能，可以显著提高训练效率。本文档将详细介绍如何使用这一功能。

## 异步训练概述

在传统的同步训练中，训练和推理过程是串行执行的，即必须等待一批推理完成并收集到奖励后才能开始下一批推理。而在异步训练中，训练和推理可以并行进行，推理过程可以提前生成多个批次的数据，训练过程可以使用这些预先生成的数据进行学习。

## 开启异步训练

要开启异步训练功能，需要在配置文件中设置 `async_generation_ratio` 参数。该参数在 RLVR 和 Agentic pipeline 中的含义和使用方式完全一致。

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

#### Agentic 异步训练配置

以下是一个完整的 Agentic 异步训练配置示例（来自 `examples/qwen2.5-0.5B-agentic/agent_val_frozen_lake_async.yaml`）：

```yaml
# 开启异步训练
async_generation_ratio: 1

# 其他相关配置
rollout_batch_size: 1024
val_batch_size: 1024
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
pretrain: Qwen/Qwen2.5-0.5B-Instruct
reward_pretrain: Qwen/Qwen2.5-0.5B-Instruct

# 角色配置
actor_train:
  model_args:
    attn_implementation: fa2
    disable_gradient_checkpointing: false
    dtype: bf16
  training_args:
    learning_rate: 1.0e-6
    weight_decay: 0
    per_device_train_batch_size: 2
    gradient_accumulation_steps: 128
    warmup_steps: 10
  strategy_args:
    strategy_name: megatron_train
    strategy_config:
      tensor_model_parallel_size: 1
      pipeline_model_parallel_size: 1
      expert_model_parallel_size: 1
      use_distributed_optimizer: true
      recompute_granularity: full
  device_mapping: list(range(0,4))
  infer_batch_size: 2

actor_infer:
  model_args:
    disable_gradient_checkpointing: true
    dtype: bf16
  generating_args:
    max_new_tokens: 128
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
```

#### RLVR 异步训练配置

以下是一个完整的 RLVR 异步训练配置示例（来自 `examples/qwen2.5-7B-rlvr_megatron/rlvr_config_async.yaml`）：

```yaml
# 开启异步训练
async_generation_ratio: 1

# 其他相关配置
rollout_batch_size: 64
prompt_length: 2048
response_length: 8192

# 训练参数
max_steps: 1000
save_steps: 100
logging_steps: 1

# RLVR 特定参数
is_num_return_sequences_expand: true
num_return_sequences_in_group: 8
ppo_epochs: 1
adv_estimator: "reinforce"

# 模型配置
pretrain: /data/cpfs_0/common/models/Qwen2.5-7B
reward_pretrain: /data/cpfs_0/common/models/Qwen2.5-7B

# 角色配置
actor_train:
  model_args:
    dtype: bf16
  training_args:
    learning_rate: 1.0e-6
    weight_decay: 0
    per_device_train_batch_size: 1
    gradient_accumulation_steps: 64
    warmup_steps: 1
  data_args:
    template: qwen2_5
    file_name:
      - data/math_deepmath_deal.jsonl
  strategy_args:
    strategy_name: megatron_train
    strategy_config:
      tensor_model_parallel_size: 2
      pipeline_model_parallel_size: 1
      sequence_parallel: true
      use_distributed_optimizer: true
  device_mapping: list(range(0,16))
  infer_batch_size: 2

actor_infer:
  model_args:
    dtype: fp16
  generating_args:
    max_new_tokens: ${response_length}
    top_p: 0.99
    top_k: 100
    num_beams: 1
    temperature: 0.99
    num_return_sequences: ${num_return_sequences_in_group}
  strategy_args:
    strategy_name: sglang
    strategy_config:
      mem_fraction_static: 0.85
      load_format: dummy
  device_mapping: list(range(16,24))
  infer_batch_size: 1
```

## 异步训练的工作原理

1. 当 `async_generation_ratio` 设置为大于 0 的值时，框架会启动异步训练模式
2. 推理过程会提前生成 `async_generation_ratio` 倍于训练所需的数据
3. 训练过程可以使用这些预先生成的数据进行学习，而不需要等待当前批次的推理完成
4. 这种并行化处理可以显著提高训练效率，特别是在推理耗时较长的情况下

## 支持的算法
除了sync training默认支持的PPO/GRPO/REINFORCE++/LitePPO等设置外，ROLL 还提供了多种 Off-Policy 算法实现，详细信息可参考：`docs_roll/docs/简体中文/使用指南/algorithms/offpolicy_setting.md`

配置示例：`examples/qwen2.5-7B-rlvr-offpolicy/rlvr_config.yaml`

支持的算法变体包括：
- `topr`
- `vanilla`
- `tis`
- `cispo`
- `kimi15`

## 使用建议

1. 根据硬件资源和任务特点调整 `async_generation_ratio` 的值
2. 确保训练和推理角色分离部署
3. 监控训练过程中的资源使用情况，避免资源瓶颈
4. 在验证阶段会暂停异步生成，验证完成后恢复
5. 对于 RLVR 任务，可以结合 `is_num_return_sequences_expand` 和 `num_return_sequences_in_group` 参数进一步优化性能
6. 对于 Off-Policy 算法，注意配置正确的 `pg_variant` 参数和对应的 worker 类
