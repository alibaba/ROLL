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
   - `enable_expert_parallel`: Enables vLLM expert parallelism for MoE layers; requires vLLM 0.16.x or later
   - `data_parallel_size`: Number of DP ranks in each inferred vLLM deployment
3. **device_mapping**: Specify the list of GPU device IDs to use

4. **infer_batch_size**: Batch size during inference

### Distributed TP, DP, and EP

ROLL uses vLLM's multiprocessing executor by default. Ray also provides resource placement and actor lifecycle; on
supported vLLM versions earlier than 0.11.1, it additionally preserves cross-node TP/PP execution. Do not configure
`distributed_executor_backend` or `data_parallel_backend`; ROLL manages them automatically.

Define:

```text
N = len(device_mapping)
T = tensor_parallel_size   (default: 1)
P = pipeline_parallel_size (default: 1)
D = data_parallel_size     (default: 1)
W = N / (T * P)            (ROLL InferWorker count)
K = W / D                  (independent vLLM deployment count)
```

`N` must be divisible by `T * P`, and `W` must be divisible by `D`. Each InferWorker is one vLLM external-DP rank
and owns `T * P` GPU placements. Every `D` consecutive InferWorkers form one deployment. ROLL derives the DP rank,
rendezvous address, and ports, so no internal/external DP mode is exposed to users.

For MoE models on vLLM 0.16.x or later, `enable_expert_parallel: true` uses the MP backend and forms an expert
group within each deployment. Earlier versions fail early with a message to upgrade vLLM or disable EP:

```text
EP size = T * D
```

EP changes the MoE communication layout, not the number of InferWorkers or deployments. With `P=1`, the common
combinations are:

| DP | TP | EP off | EP on |
| --- | --- | --- | --- |
| `D=1` | `T=1` | `N` independent engines | independent EP1 engines |
| `D=1` | `T>1` | `N/T` independent TP engines | `N/T` deployments, each EP size `T` |
| `D>1` | `T=1` | `N/D` external-DP deployments | `N/D` deployments, each EP size `D` |
| `D>1` | `T>1` | `N/(T*D)` external-DP + TP deployments | `N/(T*D)` deployments, each EP size `T*D` |

#### Physical and logical node layout

If one InferWorker's `T * P` GPUs fit on one physical node, its actor launches all local vLLM MP processes. With
vLLM 0.11.1 or later, an engine spanning `S` physical nodes uses the actor as the MP head on the first node and one
lightweight headless launcher on each remaining node. Every node must contribute the same number of GPUs.

For vLLM 0.11.0, ROLL automatically uses its legacy Ray executor when one TP/PP engine spans nodes. Users do not
need to select an executor backend.

With vLLM 0.11.1 or later, vLLM uses a logical node space for the complete external-DP deployment:

```text
logical nnodes = D * S
logical node rank = data_parallel_rank * S + node_offset
```

This rule also applies when several external-DP ranks share one physical node: each rank is still a distinct
logical node. With vLLM 0.11.1 or later, the same MP path covers same-node DP, cross-node DP, and TP engines that
span nodes. EP on this MP topology requires vLLM 0.16.x or later.
With vLLM 0.11.0, ROLL keeps the legacy external-DP rank/address/RPC launch and omits logical-node arguments when
every TP/PP engine fits on one node.

For example, with vLLM 0.16.x or later and EP enabled, DP2/TP16 on four 8-GPU nodes is laid out as:

```text
DP rank 0: logical node 0 (head, GPUs 0-7) + logical node 1 (headless, GPUs 8-15)
DP rank 1: logical node 2 (head, GPUs 0-7) + logical node 3 (headless, GPUs 8-15)
EP size: 2 * 16 = 32
```

The supported earlier versions use Ray workers on the same GPU placements instead of MP head/headless processes.

On 128 GPUs with DP16/TP4/EP enabled, ROLL creates 32 InferWorkers and two independent deployments. Each deployment
contains 16 TP4 engines and forms EP64; the two EP groups do not merge into EP128.

When `data_parallel_size` is omitted, it defaults to 1. For example, 32 GPUs with TP16 create two independent TP16
engines. Setting DP2 as well groups those two engines into one external-DP deployment.

External DP requires vLLM 0.11.0 or later. A cross-node TP/PP engine uses the legacy Ray executor on vLLM 0.11.0
and the MP executor on vLLM 0.11.1 or later. EP requires vLLM 0.16.x or later and uses the MP backend. Cross-node
throughput depends on the interconnect and the selected vLLM All2All implementation; benchmark the target cluster
rather than assuming EP is always faster.

## Integration with Other Components

In the configuration example, we can see:

1. `actor_infer` uses vLLM as the inference backend
2. `actor_train` uses Megatron for training
3. `reference` uses Megatron for inference
4. Reward models use different inference backends (such as `hf_infer`)

This design allows different components to choose the most suitable inference engine according to their needs.

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
