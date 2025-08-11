# SGLang 推理后端配置指南

SGLang 是一个快速且易于使用的推理引擎，特别适合大规模语言模型的推理任务。本文档将详细介绍如何在 ROLL 框架中配置和使用 SGLang 推理后端。

## SGLang 简介

SGLang 是一种结构化生成语言，专为大型语言模型的推理而设计。它提供了高效的推理性能和灵活的编程接口。

## 配置 SGLang 策略

在 ROLL 框架中，可以通过在 YAML 配置文件中设置 `strategy_args` 来配置 SGLang 推理策略。

### 基本配置示例

以下是一个典型的 SGLang 配置示例（来自 `examples/qwen3-30BA3B-rlvr_megatron/rlvr_config_sglang.yaml`）：

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

### 配置参数详解

1. **strategy_name**: 设置为 `sglang` 以使用 SGLang 推理后端

2. **strategy_config**: SGLang 特定的配置参数，更多sglang的配置参数[官方文档](https://docs.sglang.ai/), strategy_config透传给sglang.
   - `mem_fraction_static`: 用于模型权重和 KV 缓存等静态内存的 GPU 内存占比
     - 如果 KV 缓存构建失败，请增加此值
     - 如果 CUDA 内存不足，请减小此值
   - `load_format`: 加载模型权重的格式
     - 由于模型会在开始时进行"更新"，此值可以设置为 `dummy`

3. **num_gpus_per_worker**: 每个 worker 分配的 GPU 数量
   - SGLang 可以利用多个 GPU 进行并行推理

4. **device_mapping**: 指定使用的 GPU 设备 ID 列表

5. **infer_batch_size**: 推理时的批次大小

## 与其他组件的集成

在上述示例中，我们可以看到：

1. `actor_infer` 使用 SGLang 作为推理后端
2. `actor_train` 使用 Megatron 进行训练
3. `reference` 使用 Megatron 进行推理
4. 奖励模型使用不同的推理后端（如 `hf_infer`）

这种设计允许不同组件根据其需求选择最适合的推理引擎。

## 性能优化建议

1. **内存管理**：
   - 合理设置 `mem_fraction_static` 参数以平衡性能和内存使用
   - 监控 GPU 内存使用情况，避免内存溢出

2. **并行处理**：
   - 适当增加 `num_gpus_per_worker` 以利用多 GPU 加载模型, 并行推理
   - 根据硬件配置调整 `device_mapping`, sglang engine的数量是`len(device_mapping) // num_gpus_per_worker` 

3. **批处理优化**：
   - `infer_batch_size`不生效，会自动进行continue batch
   - 考虑序列长度对批处理大小的影响

## 注意事项

1. SGLang 需要特定版本的依赖库，请确保安装了兼容的版本
2. 在资源受限的环境中，需要仔细平衡不同组件的资源分配
3. SGLang 与 Megatron 等训练框架的集成可能需要额外的配置
