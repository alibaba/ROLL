# vLLM Inference Backend Configuration Guide

vLLM is a fast and easy-to-use large language model inference library that efficiently manages attention key-value cache through PagedAttention technology. This document will provide detailed instructions on how to configure and use the vLLM inference backend in the ROLL framework.

## vLLM Introduction

vLLM is a high-performance inference engine with the following features:
1. **Fast Inference**: Efficiently manages attention key-value cache through PagedAttention technology
2. **Memory Efficient**: Reduces memory usage through quantization and optimization
3. **Easy to Use**: Provides simple API interfaces
4. **Scalability**: Supports distributed inference

## Configuring vLLM Strategy

In the ROLL framework, vLLM inference strategy can be configured by setting `strategy_args` in the YAML configuration file.

### Configuration Example

The following is a typical vLLM configuration example (from `examples/qwen2.5-7B-rlvr_megatron/rlvr_config.yaml`):

```yaml
actor_infer:
  model_args:
    disable_gradient_checkpointing: true
    dtype: bf16
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
      gpu_memory_utilization: 0.8
      block_size: 16
      max_model_len: 8000
  device_mapping: list(range(0,12))
  infer_batch_size: 1
```

### Configuration Parameter Details

1. **strategy_name**: Set to `vllm` to use the vLLM inference backend

2. **strategy_config**: vLLM-specific configuration parameters. For more vLLM optimization configurations, please refer to the [vLLM official documentation](https://docs.vllm.ai/en/latest/). The strategy_config is passed through directly.
   - `gpu_memory_utilization`: GPU memory utilization ratio for the model executor
     - For example, 0.8 means using 80% of GPU memory
     - Adjust this value according to model size and hardware configuration
   - `block_size`: Token block size for contiguous chunks of tokens
     - Affects vLLM's internal memory management efficiency
     - Usually set to 16 or 32
   - `max_model_len`: Model context length
     - If not specified, it will be automatically derived from the model configuration
     - Ensure it does not exceed hardware limitations
   - `load_format`: Format for loading model weights
     - Since the model will be "updated" at the beginning, this value can be set to `dummy`
   - `sleep_level`: Sleep level when sleeping the model
     - 1 (default): Only destroys KV cache, retains model weights
     - 2: Destroys both model weights and KV cache after generation, thus saving memory
3. **device_mapping**: Specify the list of GPU device IDs to use

4. **infer_batch_size**: Batch size during inference

## Integration with Other Components

In the configuration example, we can see:

1. `actor_infer` uses vLLM as the inference backend
2. `actor_train` uses Megatron for training
3. `reference` uses Megatron for inference
4. Reward models use different inference backends (such as `hf_infer`)

This design allows different components to choose the most suitable inference engine according to their needs.

### beam_search Configuration
RLVRPipeline supports vllm beam_search generation method, configured as follows:
```yaml
generate_opt_level: 0 # Degrades to batch_generate generation method, generate_opt_level=1 is prompt-level parallel method
num_return_sequences_in_group: 8 
actor_infer:
  generating_args:
    num_beams: ${num_return_sequences_in_group}
    num_return_sequences: ${num_return_sequences_in_group}
```
Note:
- generating_args.num_beams and generating_args.num_return_sequences must be set to the same value.
- The generating_args configuration in validate is also configured in the same way.

## Performance Optimization Recommendations

1. **Memory Management**:
   - Properly set the `gpu_memory_utilization` parameter to balance performance and memory usage
   - Monitor GPU memory usage to avoid memory overflow

2. **Batch Processing Optimization**:
   - Adjust `infer_batch_size` according to model size and hardware capabilities
   - Consider the impact of sequence length on batch size

3. **Context Length**:
   - Properly set `max_model_len` to match task requirements
   - Avoid setting excessively large context lengths that could cause memory insufficiency

## Notes

1. vLLM requires specific versions of dependency libraries, please ensure compatible versions are installed
2. In resource-constrained environments, carefully balance resource allocation among different components
3. Integration of vLLM with training frameworks like Megatron may require additional configuration

By properly configuring the vLLM inference backend, you can fully leverage the performance advantages of the ROLL framework in large-scale language model inference.