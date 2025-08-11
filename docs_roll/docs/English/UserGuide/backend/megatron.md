# Megatron Inference and Training Backend Configuration Guide

Megatron is NVIDIA's large-scale language model training and inference framework that supports efficient distributed training and inference. This document will provide detailed instructions on how to configure and use the Megatron backend in the ROLL framework.

## Megatron Introduction

Megatron provides efficient model parallel and data parallel strategies, particularly suitable for training and inference of ultra-large-scale language models. It supports multiple parallel strategies including tensor parallelism, pipeline parallelism, and expert parallelism.

## Configuring Megatron Strategy

In the ROLL framework, Megatron training and inference strategies can be configured by setting `strategy_args` in the YAML configuration file.

### Training Configuration Example

The following is a typical Megatron training configuration example (from `examples/qwen3-30BA3B-rlvr_megatron/rlvr_config_sglang.yaml`):

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

### Inference Configuration Example

The following is a typical Megatron inference configuration example:

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

### Configuration Parameter Details

1. **strategy_name**: 
   - `megatron_train` for training
   - `megatron_infer` for inference

2. **strategy_config**: All parallel optimization configurations provided by mcore can be used. For detailed information, please refer to the usage of [megatron](https://github.com/NVIDIA/Megatron-LM/blob/main/README.md). Here are some common Megatron configuration parameters:
   - `tensor_model_parallel_size`: Tensor model parallelism degree, partitioning intra-layer computation and memory across multiple GPUs
   - `pipeline_model_parallel_size`: Pipeline model parallelism degree, assigning different layers of the model to different GPUs
   - `virtual_pipeline_model_parallel_size`: Virtual pipeline parallelism degree, used to improve pipeline efficiency
   - `expert_model_parallel_size`: Expert model parallelism degree, assigning different experts to different GPUs in MoE models
   - `context_parallel_size`: Context parallelism degree, used for processing ultra-long sequences
   - `use_distributed_optimizer`: Whether to use distributed optimizer
   - `sequence_parallel`: Whether to enable sequence parallel optimization
   - `moe_token_dispatcher_type`: Token dispatcher type in MoE models ('allgather' or 'alltoall')
   - `moe_grouped_gemm`: Whether to enable grouped GEMM for MoE experts
   - `moe_layer_recompute`: Whether to checkpoint MoE layers to save activation memory
   - `recompute_granularity`: Activation value recomputation granularity ('full' or 'selective')
   - `overlap_grad_reduce`: Whether to overlap gradient All-reduce process with backward propagation computation in distributed optimizer

3. **device_mapping**: Specify the list of GPU device IDs to use

4. **infer_batch_size**: Batch size during inference

## Integration with Other Components

In the configuration example, we can see:

1. `actor_train` uses Megatron for training
2. `actor_infer` may use other inference backends (such as vLLM or SGLang)
3. `reference` uses Megatron for inference
4. Reward models may use different inference backends

This design allows different components to choose the most suitable backend according to their needs.

## Notes

1. Megatron requires specific versions of dependency libraries, please ensure compatible versions are installed
2. Parallel strategy settings need to be adjusted according to hardware configuration and model size
3. In resource-constrained environments, carefully balance resource allocation among different components

By properly configuring the Megatron backend, you can fully leverage the performance advantages of the ROLL framework in large-scale language model training and inference.