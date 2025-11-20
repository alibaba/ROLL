# FP8 Quantization Configuration Guide

This document describes how to use FP8 quantization in ROLL to optimize inference performance and VRAM usage.

## Overview

FP8 quantization is an efficient numerical precision optimization technique that can significantly reduce model VRAM footprint and improve inference speed. ROLL supports FP8 quantization configuration for actor_infer and llm_judge components.

## actor_infer FP8 Configuration

### Basic Configuration

```yaml
actor_infer:
  strategy_args:
    strategy_name: vllm
    strategy_config:
      quantization: fp8
```

### Dense Model Configuration

For Dense models, configuration requirements differ based on quantization method:

**Dense + Per Tensor Quantization (Default)**
```yaml
actor_infer:
  strategy_args:
    strategy_name: vllm
    strategy_config:
      quantization: fp8
```

**Dense + Per Block Quantization**
```yaml
actor_infer:
  strategy_args:
    strategy_name: vllm
    strategy_config:
      quantization: fp8
      hf_overrides:
        quantization_config:
          activation_scheme: dynamic
          fmt: e4m3
          quant_method: fp8
          weight_block_size: [128, 128]  # Required: per block quantization
```

**Configuration Description:**
- `activation_scheme: dynamic`: Use dynamic activation scheme
- `fmt: e4m3`: Specify FP8 format as E4M3
- `quant_method: fp8`: Set quantization method to FP8
- `weight_block_size: [128, 128]`: Required for per block quantization, specifies weight block size

**Note:** When specifying `weight_block_size`, you must also provide `activation_scheme`, `fmt`, and `quant_method` parameters, otherwise an error will occur.

### MoE Model Configuration

For MoE (Mixture of Experts) models, `hf_overrides/quantization_config` must be configured, and only per block quantization is supported:

```yaml
actor_infer:
  strategy_args:
    strategy_name: vllm
    strategy_config:
      quantization: fp8
      hf_overrides:
        quantization_config:
          activation_scheme: dynamic
          fmt: e4m3
          quant_method: fp8
          weight_block_size: [128, 128]  # Required: MoE models must use per block quantization
```

**Note:** MoE models must use per block quantization. The `weight_block_size` parameter is required, and you must also provide `activation_scheme`, `fmt`, and `quant_method` parameters.

## llm_judge FP8 Configuration

LLM as judge model also supports FP8 quantization. Note that the judge model requires independent GPU resources and cannot share GPU with actor_infer:

```yaml
llm_judge:
  # NOTE: llm as judge also needs GPU, cannot share GPU with actor infer
  worker_cls: roll.pipeline.rlvr.rewards.llm_judge_reward_worker.LLMJudgeRewardWorker
  judge_prompt: Qwen2.5-7B-Instruct-RLVR-prompt
  judge_model_type: inference
  tag_included: [RLVR]  
  strategy_args:
    strategy_name: vllm
    strategy_config:
      gpu_memory_utilization: 0.8
      quantization: fp8
      max_model_len: 8000
      load_format: auto
```

**Configuration Description:**
- `gpu_memory_utilization: 0.8`: Set VRAM utilization to 80%
- `quantization: fp8`: Enable FP8 quantization
- `max_model_len: 8000`: Maximum model length limit
- `load_format: auto`: Automatically select loading format

## Configuration Notes

1. **GPU Resource Isolation**: llm_judge requires independent GPU and cannot share with actor_infer
2. **MoE Model Limitations**: MoE models must use per block quantization, per tensor quantization is not supported
3. **Memory Optimization**: FP8 quantization can significantly reduce memory usage, recommended for VRAM-constrained scenarios
4. **Performance Trade-off**: While FP8 quantization improves performance, it may slightly affect model accuracy, requiring trade-offs based on specific scenarios

## Complete Example

```yaml
# Configuration example: FP8 quantization for actor_infer and llm_judge
actor_infer:
  strategy_args:
    strategy_name: vllm
    strategy_config:
      quantization: fp8
      hf_overrides:
        quantization_config:
          activation_scheme: dynamic
          fmt: e4m3
          quant_method: fp8
          weight_block_size: [128, 128]

llm_judge:
  worker_cls: roll.pipeline.rlvr.rewards.llm_judge_reward_worker.LLMJudgeRewardWorker
  judge_prompt: Qwen2.5-7B-Instruct-RLVR-prompt
  judge_model_type: inference
  tag_included: [RLVR]  
  strategy_args:
    strategy_name: vllm
    strategy_config:
      gpu_memory_utilization: 0.8
      quantization: fp8
      max_model_len: 8000
      load_format: auto
```

With the above configuration, you can successfully enable FP8 quantization in ROLL to achieve better inference performance and VRAM efficiency.