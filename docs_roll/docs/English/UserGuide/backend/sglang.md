# SGLang Inference Backend Configuration Guide

SGLang is a fast and easy-to-use inference engine, particularly suitable for inference tasks of large-scale language models. This document will provide detailed instructions on how to configure and use the SGLang inference backend in the ROLL framework.

## SGLang Introduction

SGLang is a structured generation language specifically designed for inference of large language models. It provides efficient inference performance and flexible programming interfaces.

## Configuring SGLang Strategy

In the ROLL framework, SGLang inference strategy can be configured by setting `strategy_args` in the YAML configuration file.

### Basic Configuration Example

The following is a typical SGLang configuration example (from `examples/qwen3-30BA3B-rlvr_megatron/rlvr_config_sglang.yaml`):

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
    strategy_name: sglang
    strategy_config:
      mem_fraction_static: 0.7
      load_format: dummy
  num_gpus_per_worker: 2
  device_mapping: list(range(0,24))
```

### Configuration Parameter Details

1. **strategy_name**: Set to `sglang` to use the SGLang inference backend

2. **strategy_config**: SGLang-specific configuration parameters. For more SGLang configuration parameters, see the [official documentation](https://docs.sglang.ai/). The strategy_config is passed through directly to SGLang.
   - `mem_fraction_static`: GPU memory utilization ratio for static memory such as model weights and KV cache
     - Increase this value if KV cache building fails
     - Decrease this value if CUDA memory is insufficient
   - `load_format`: Format for loading model weights
     - Since the model will be "updated" at the beginning, this value can be set to `dummy`

3. **num_gpus_per_worker**: Number of GPUs allocated per worker
   - SGLang can utilize multiple GPUs for parallel inference

4. **device_mapping**: Specify the list of GPU device IDs to use

5. **infer_batch_size**: Batch size during inference

## Integration with Other Components

In the above example, we can see:

1. `actor_infer` uses SGLang as the inference backend
2. `actor_train` uses Megatron for training
3. `reference` uses Megatron for inference
4. Reward models use different inference backends (such as `hf_infer`)

This design allows different components to choose the most suitable inference engine according to their needs.

## Performance Optimization Recommendations

1. **Memory Management**:
   - Properly set the `mem_fraction_static` parameter to balance performance and memory usage
   - Monitor GPU memory usage to avoid memory overflow

2. **Parallel Processing**:
   - Appropriately increase `num_gpus_per_worker` to utilize multiple GPUs for model loading and parallel inference
   - Adjust `device_mapping` according to hardware configuration. The number of SGLang engines is `len(device_mapping) // num_gpus_per_worker`

3. **Batch Processing Optimization**:
   - `infer_batch_size` is not effective, as continuous batching is automatically performed
   - Consider the impact of sequence length on batch size

## Notes

1. SGLang requires specific versions of dependency libraries, please ensure compatible versions are installed
2. In resource-constrained environments, carefully balance resource allocation among different components
3. Integration of SGLang with training frameworks like Megatron may require additional configuration