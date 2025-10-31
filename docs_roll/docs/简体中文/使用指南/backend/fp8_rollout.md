# FP8 量化配置指南

本文档介绍如何在 ROLL 中使用 FP8 量化来优化推理性能和显存使用。

## 概述

FP8 量化是一种高效的数值精度优化技术，可以显著减少模型显存占用并提升推理速度。ROLL 支持 FP8 量化配置，适用于 actor_infer 和 llm_judge 组件。

## actor_infer FP8 配置

### 基本配置

```yaml
actor_infer:
  strategy_args:
    strategy_name: vllm
    strategy_config:
      quantization: fp8
```

### Dense 模型配置

对于 Dense 模型，根据量化方式不同，配置要求也不同：

**Dense + Per Tensor 量化（默认）**
```yaml
actor_infer:
  strategy_args:
    strategy_name: vllm
    strategy_config:
      quantization: fp8
```

**Dense + Per Block 量化**
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
          weight_block_size: [128, 128]  # 必填：per block 量化
```

**配置说明：**
- `activation_scheme: dynamic`：使用动态激活方案
- `fmt: e4m3`：指定 FP8 格式为 E4M3
- `quant_method: fp8`：量化方法设置为 FP8
- `weight_block_size: [128, 128]`：per block 量化时必填，指定权重块大小

**注意：** 当指定 `weight_block_size` 时，必须同时填写 `activation_scheme`、`fmt` 和 `quant_method` 三个参数，否则会报错。

### MoE 模型配置

对于 MoE (Mixture of Experts) 模型，必须配置 `hf_overrides/quantization_config`，且只支持 per block 量化：

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
          weight_block_size: [128, 128]  # 必填：MoE 模型必须使用 per block 量化
```

**注意：** MoE 模型必须使用 per block 量化，`weight_block_size` 参数必填，同时必须填写 `activation_scheme`、`fmt` 和 `quant_method` 三个参数。

## llm_judge FP8 配置

LLM 作为评判模型同样支持 FP8 量化，需要注意的是 judge 模型需要独立的 GPU 资源，不能与 actor_infer 共享 GPU：

```yaml
llm_judge:
  # NOTE: llm as judge 也需要gpu, 不能和actor infer共享gpu
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

**配置说明：**
- `gpu_memory_utilization: 0.8`：显存利用率设置为 80%
- `quantization: fp8`：启用 FP8 量化
- `max_model_len: 8000`：最大模型长度限制
- `load_format: auto`：自动选择加载格式

## 配置注意事项

1. **GPU 资源隔离**：llm_judge 需要独立的 GPU，不能与 actor_infer 共享
2. **MoE 模型限制**：MoE 模型必须使用 per block 量化，不支持 per tensor 量化
3. **内存优化**：FP8 量化可以显著减少内存使用，建议在显存受限的场景中使用
4. **性能权衡**：虽然 FP8 量化提升了性能，但可能会轻微影响模型精度，需要根据具体场景权衡

## 完整示例

```yaml
# 配置示例：包含 actor_infer 和 llm_judge 的 FP8 量化
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

通过以上配置，您可以在 ROLL 中成功启用 FP8 量化，获得更好的推理性能和显存效率。