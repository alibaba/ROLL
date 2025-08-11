# Agentic Asynchronous Training Feature Usage Guide

The ROLL framework supports Agentic asynchronous training functionality, which can significantly improve training efficiency. This document will provide detailed instructions on how to use this feature.

## Asynchronous Training Overview

In traditional synchronous training, the training and inference processes are executed sequentially, meaning that the next batch of inference cannot begin until a batch of inference is completed and rewards are collected. In asynchronous training, training and inference can be performed in parallel. The inference process can generate multiple batches of data in advance, and the training process can use these pre-generated data for learning.

## Enabling Asynchronous Training

To enable Agentic asynchronous training functionality, you need to set the `async_generation_ratio` parameter in the configuration file.

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

### Example Configuration

The following is a complete asynchronous training configuration example (from `examples/qwen2.5-7B-agentic_megatron/agentic_val_webshop_async.yaml`):

```yaml
# Enable asynchronous training
async_generation_ratio: 1

# Other related configurations
rollout_batch_size: 64
val_batch_size: 64
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
pretrain: Qwen/Qwen2.5-7B-Instruct
reward_pretrain: Qwen/Qwen2.5-7B-Instruct

# Role configurations
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

## How Asynchronous Training Works

1. When `async_generation_ratio` is set to a value greater than 0, the framework will start asynchronous training mode
2. The inference process will generate data ahead by `async_generation_ratio` times the amount needed for training
3. The training process can use these pre-generated data for learning without waiting for the current batch of inference to complete
4. This parallelized processing can significantly improve training efficiency, especially when inference takes a long time

## Usage Recommendations

1. Adjust the value of `async_generation_ratio` based on hardware resources and task characteristics
2. Ensure that training and inference roles are deployed separately
3. Monitor resource usage during the training process to avoid resource bottlenecks
4. Asynchronous generation will be paused during the validation phase and resumed after validation is complete