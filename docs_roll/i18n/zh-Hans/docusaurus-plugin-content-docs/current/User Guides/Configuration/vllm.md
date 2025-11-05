# vLLM 推理后端配置指南

vLLM 是一个快速且易于使用的大型语言模型推理库，通过 PagedAttention 技术高效管理注意力键值缓存。本文档将详细介绍如何在 ROLL 框架中配置和使用 vLLM 推理后端。

## vLLM 简介

vLLM 是一个高性能的推理引擎，具有以下特点：
1. **快速推理**：通过 PagedAttention 技术高效管理注意力键值缓存
2. **内存高效**：通过量化和优化减少内存使用
3. **易于使用**：提供简单的 API 接口
4. **可扩展性**：支持分布式推理

## 配置 vLLM 策略

在 ROLL 框架中，可以通过在 YAML 配置文件中设置 `strategy_args` 来配置 vLLM 推理策略。

### 配置示例

以下是一个典型的 vLLM 配置示例（来自 `examples/qwen2.5-7B-rlvr_megatron/rlvr_config.yaml`）：

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

### 配置参数详解

1. **strategy_name**: 设置为 `vllm` 以使用 vLLM 推理后端

2. **strategy_config**: vLLM 特定的配置参数，更多vllm优化配置，请参考[vLLM官方文档](https://docs.vllm.ai/en/latest/), strategy_config透传处理。
   - `gpu_memory_utilization`: 用于模型执行器的 GPU 内存占比
     - 例如 0.8 表示使用 80% 的 GPU 内存
     - 根据模型大小和硬件配置调整此值
   - `block_size`: token 块大小，用于连续的 token 块
     - 影响 vLLM 内部的内存管理效率
     - 通常设置为 16 或 32
   - `max_model_len`: 模型上下文长度
     - 如果未指定，将从模型配置中自动推导
     - 确保不超过硬件限制
   - `load_format`: 加载模型权重的格式
     - 由于模型会在开始时进行"更新"，此值可以设置为 `dummy`
   - `sleep_level`: sleep model时的级别
     - 1 默认值，仅销毁 KV 缓存，会将模型权重保留
     - 2 将在生成后销毁模型权重与 KV 缓存，从而节省内存
3. **device_mapping**: 指定使用的 GPU 设备 ID 列表

4. **infer_batch_size**: 推理时的批次大小

## 与其他组件的集成

在配置示例中，我们可以看到：

1. `actor_infer` 使用 vLLM 作为推理后端
2. `actor_train` 使用 Megatron 进行训练
3. `reference` 使用 Megatron 进行推理
4. 奖励模型使用不同的推理后端（如 `hf_infer`）

这种设计允许不同组件根据其需求选择最适合的推理引擎。

### beam_search 配置方式
RLVRPipeline 支持vllm beam_search 的生成方式，配置方式如下：
```yaml
generate_opt_level: 0 # 退化为batch_generate生成方式，generate_opt_level=1是prompt粒度并行方式
num_return_sequences_in_group: 8 
actor_infer:
  generating_args:
    num_beams: ${num_return_sequences_in_group}
    num_return_sequences: ${num_return_sequences_in_group}
```
注意：
- generating_args.num_beams 和 generating_args.num_return_sequences 必须设置为相同的值。
- validate中配置generating_args也是相同的方式。


## 性能优化建议

1. **内存管理**：
   - 合理设置 `gpu_memory_utilization` 参数以平衡性能和内存使用
   - 监控 GPU 内存使用情况，避免内存溢出

2. **批处理优化**：
   - 根据模型大小和硬件能力调整 `infer_batch_size`
   - 考虑序列长度对批处理大小的影响

3. **上下文长度**：
   - 合理设置 `max_model_len` 以匹配任务需求
   - 避免设置过大的上下文长度导致内存不足

## 注意事项

1. vLLM 需要特定版本的依赖库，请确保安装了兼容的版本
2. 在资源受限的环境中，需要仔细平衡不同组件的资源分配
3. vLLM 与 Megatron 等训练框架的集成可能需要额外的配置

通过合理配置 vLLM 推理后端，您可以充分发挥 ROLL 框架在大规模语言模型推理方面的性能优势。