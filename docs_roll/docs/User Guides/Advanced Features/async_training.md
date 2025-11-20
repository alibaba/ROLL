# ROLL Asynchronous Training User Guide

The ROLL framework now supports asynchronous training for both RLVR and Agentic pipelines, significantly improving training efficiency. This document provides detailed instructions on how to use this feature.

## Asynchronous Training Overview

In traditional synchronous training, the training and inference processes run serially, meaning that the next batch of inference can only start after the current batch completes and rewards are collected. In asynchronous training, however, training and inference can run in parallel. The inference process can generate multiple batches of data in advance, and the training process can use this pre-generated data for learning.

## Enabling Asynchronous Training

To enable asynchronous training, set the `async_generation_ratio` parameter in your configuration file. This parameter has consistent meaning and usage across both RLVR and Agentic pipelines.

### Configuration Parameters

The `async_generation_ratio` parameter is defined in `roll/configs/base_config.py`:

```python
async_generation_ratio: float = field(
    default=0,
    metadata={
        "help": "The ratio of ahead generation requests in pipeline, "
        "0 means synchronous pipeline. currently only integer is supported."
    },
)
```

### Configuration Examples

#### Agentic Asynchronous Training Configuration

Here is a complete Agentic asynchronous training configuration example (from `examples/qwen2.5-0.5B-agentic/agent_val_frozen_lake_async.yaml`):

```yaml
# Enable asynchronous training
async_generation_ratio: 1

# Other related configurations
rollout_batch_size: 1024
val_batch_size: 1024
sequence_length: 8192

# Training parameters
max_steps: 1024
save_steps: 10000
logging_steps: 1
eval_steps: 10

# PPO parameters
ppo_epochs: 1
adv_estimator: "grpo"
whiten_advantages: true

# Model configuration
pretrain: Qwen/Qwen2.5-0.5B-Instruct
reward_pretrain: Qwen/Qwen2.5-0.5B-Instruct

# Actor configuration
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

#### RLVR Asynchronous Training Configuration

Here is a complete RLVR asynchronous training configuration example (from `examples/qwen2.5-7B-rlvr_megatron/rlvr_config_async.yaml`):

```yaml
# Enable asynchronous training
async_generation_ratio: 1

# Other related configurations
rollout_batch_size: 64
prompt_length: 2048
response_length: 8192

# Training parameters
max_steps: 1000
save_steps: 100
logging_steps: 1

# RLVR specific parameters
is_num_return_sequences_expand: true
num_return_sequences_in_group: 8
ppo_epochs: 1
adv_estimator: "reinforce"

# Model configuration
pretrain: /data/cpfs_0/common/models/Qwen2.5-7B
reward_pretrain: /data/cpfs_0/common/models/Qwen2.5-7B

# Actor configuration
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

## How Asynchronous Training Works

1. When `async_generation_ratio` is set to a value greater than 0, the framework starts asynchronous training mode
2. The inference process generates `async_generation_ratio` times more data than needed for training in advance
3. The training process can use this pre-generated data for learning without waiting for the current batch of inference to complete
4. This parallel processing can significantly improve training efficiency, especially when inference is time-consuming

## Supported Algorithms

### Agentic Pipeline
- Supports GRPO and other policy gradient algorithms
- Suitable for environment interaction tasks, such as games, dialogues, etc.
- Configuration example: `examples/qwen2.5-0.5B-agentic/agent_val_frozen_lake_async.yaml`

### RLVR Pipeline
- Supports Reinforce and other algorithms
- Suitable for language modeling tasks, such as mathematical reasoning, code generation, etc.
- Configuration example: `examples/qwen2.5-7B-rlvr_megatron/rlvr_config_async.yaml`

### Off-Policy Algorithms
ROLL also supports various Off-Policy algorithms. For detailed information, please refer to: `docs_roll/docs/UserGuide/algorithms/offpolicy_setting.md`

Configuration example: `examples/qwen2.5-7B-rlvr-offpolicy/rlvr_config.yaml`

Supported algorithm variants include:
- `topr`
- `vanilla`
- `tis`
- `cispo`
- `kimi15`
- `ppo`

## Usage Recommendations

1. Adjust the value of `async_generation_ratio` according to hardware resources and task characteristics
2. Ensure separate deployment of training and inference roles
3. Monitor resource usage during training to avoid resource bottlenecks
4. Asynchronous generation is paused during validation and resumes after validation is complete
5. For RLVR tasks, you can further optimize performance by combining `is_num_return_sequences_expand` and `num_return_sequences_in_group` parameters
6. For Off-Policy algorithms, ensure correct configuration of the `pg_variant` parameter and corresponding worker class