# DeepSpeed Training Backend Configuration Guide

DeepSpeed is Microsoft's efficient deep learning optimization library that provides memory optimization, distributed training, and performance optimization features. This document will provide detailed instructions on how to configure and use the DeepSpeed training backend in the ROLL framework.

## DeepSpeed Introduction

DeepSpeed provides multiple optimization techniques, including:
1. **ZeRO Optimization**: Reduces memory usage by partitioning optimizer states, gradients, and parameters
2. **Memory-Efficient Training**: Supports training of large-scale models
3. **High-Performance Communication**: Optimizes communication efficiency in distributed training
4. **Flexible Configuration**: Supports configuration of multiple optimization levels

## Configuring DeepSpeed Strategy

In the ROLL framework, DeepSpeed training strategy can be configured by setting `strategy_args` in the YAML configuration file.

### Configuration Example

The following is a typical DeepSpeed configuration example (from `examples/qwen2.5-7B-rlvr_megatron/rlvl_lora_zero3.yaml`):

```yaml
defaults:
  - ../config/deepspeed_zero@_here_
  - ../config/deepspeed_zero2@_here_
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

### Configuration Parameter Details

1. **strategy_name**: Set to `deepspeed_train` to use the DeepSpeed training backend

2. **strategy_config**: DeepSpeed-specific configuration parameters
   - Can reference predefined configuration files, such as `${deepspeed_zero3}`
   - Multiple DeepSpeed configuration files are available in the `./examples/config/` directory:
     - `deepspeed_zero.yaml`: Basic ZeRO configuration
     - `deepspeed_zero2.yaml`: ZeRO-2 configuration
     - `deepspeed_zero3.yaml`: ZeRO-3 configuration
     - `deepspeed_zero3_cpuoffload.yaml`: ZeRO-3 configuration with CPU offloading

3. **defaults section**: Import predefined DeepSpeed configurations
   ```yaml
   defaults:
     - ../config/deepspeed_zero@_here_
     - ../config/deepspeed_zero2@_here_
     - ../config/deepspeed_zero3@_here_
     - ../config/deepspeed_zero3_cpuoffload@_here_
   ```

4. **device_mapping**: Specify the list of GPU device IDs to use

## DeepSpeed Configuration Files

Multiple predefined DeepSpeed configuration files are provided in the `./examples/config/` directory:

1. **deepspeed_zero.yaml**: Basic ZeRO configuration
2. **deepspeed_zero2.yaml**: ZeRO-2 configuration with optimizer state partitioning
3. **deepspeed_zero3.yaml**: ZeRO-3 configuration with optimizer state, gradient, and parameter partitioning
4. **deepspeed_zero3_cpuoffload.yaml**: ZeRO-3 configuration with CPU offloading

### Using Predefined Configurations

To use predefined DeepSpeed configurations, you can reference them in the YAML file like this:

```yaml
defaults:
  - ../config/deepspeed_zero3@_here_

actor_train:
  strategy_args:
    strategy_name: deepspeed_train
    strategy_config: ${deepspeed_zero3}
```

## Integration with Other Components

In the configuration example, we can see:

1. `actor_train` uses DeepSpeed for training
2. `actor_infer` may use other inference backends (such as vLLM)
3. `reference` uses the Hugging Face inference backend
4. Reward models use different inference backends

This design allows different components to choose the most suitable backend according to their needs.

## Notes

1. DeepSpeed requires specific versions of dependency libraries, please ensure compatible versions are installed
2. Different ZeRO levels have different memory and performance characteristics, choose according to specific needs
3. When using LoRA fine-tuning, pay attention to compatibility with DeepSpeed