# FSDP2 训练和推理后端配置指南

[FSDP2 (Fully Sharded Data Parallel 2)](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html) 是 PyTorch 最新的分布式训练框架，提供高效的参数分片和 [DTensor](https://docs.pytorch.org/docs/stable/distributed.tensor.html) 支持。本文档将详细介绍如何在 ROLL 框架中配置和使用 FSDP2 后端。

## FSDP2 与 ROLL

ROLL 支持以下 FSDP2 特性：
1. **FSDP2 分片**：使用 FSDP2 [fully_shard](https://docs.pytorch.org/docs/main/distributed.fsdp.fully_shard.html) 分片模型参数、梯度和优化器状态。同时支持使用 [DCP](https://docs.pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html) 进行检查点管理。
2. **上下文并行**：支持与序列并行（Ulysses）集成
3. **模型支持**：支持文本模型、视觉语言（VL）模型和 MoE（混合专家）模型。

## 配置 FSDP2 策略

在 ROLL 框架中，可以通过在 YAML 配置文件中设置 `strategy_args` 来配置 FSDP2 训练和推理策略。

### 训练配置示例

以下是一个典型的 FSDP2 训练配置示例（来自 `examples_lixing/qwen3-8B-rlvr_fsdp2/rlvr_config.yaml`）：

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
    strategy_name: fsdp2_train
    strategy_config:
      fsdp_size: 16
      param_dtype: bf16
      reduce_dtype: float32
      reshard_after_forward: true
      offload_policy: false
  device_mapping: list(range(0,16))
  infer_batch_size: 4
```

### 推理配置示例

以下是一个典型的 FSDP2 推理配置示例：

```yaml
reference:
  model_args:
    disable_gradient_checkpointing: true
    dtype: bf16
    model_type: ~
  strategy_args:
    strategy_name: fsdp2_infer
    strategy_config:
      fsdp_size: 4
      param_dtype: bf16
      reduce_dtype: float32
      reshard_after_forward: true
      offload_policy: false
  device_mapping: list(range(0,8))
  infer_batch_size: 1
```

### FSDP2 + 上下文并行配置示例

以下是一个结合 FSDP2 和序列并行（Ulysses）的配置示例（来自 `examples_lixing/qwen3-4b-vl_fsdp2_lct/vl_fsdp2_lct_cp2.yaml`）：

```yaml
actor_train:
  model_args:
    disable_gradient_checkpointing: false
    dtype: bf16
    model_type: ~
    ulysses_size: 2  # 序列并行大小
  training_args:
    learning_rate: 1.0e-6
    weight_decay: 1.0e-2
    per_device_train_batch_size: 1
    gradient_accumulation_steps: 256
    warmup_steps: 0
    num_train_epochs: 50
  strategy_args:
    strategy_name: fsdp2_train
    strategy_config:
      fsdp_size: 4  # FSDP 分片大小
      param_dtype: bf16
      reduce_dtype: float32
      reshard_after_forward: true
      offload_policy: false
  device_mapping: list(range(0,8))
  infer_batch_size: 1
```

在此示例中：
- 总 GPU 数：8
- 上下文并行（Ulysses）大小：2
- FSDP 大小：4
- 设备网格形状：(2, 4) [ddp, fsdp]
- 2 个副本，每个副本有 4 路参数分片

### 配置参数详解

1. **strategy_name**：
   - `fsdp2_train` 用于训练
   - `fsdp2_infer` 用于推理

2. **strategy_config**：FSDP2 特定的配置参数
   - `fsdp_size`：FSDP 分片数量
     - 如果 `fsdp_size >= world_size` 或 `fsdp_size <= 1`：纯 FSDP2 模式
     - 如果 `fsdp_size < world_size`：带有 DDP 副本的 HSDP 模式
   - `param_dtype`：参数数据类型（例如 `bf16`、`fp16`、`float32`）
   - `reduce_dtype`：梯度归约的数据类型（例如 `float32`）
   - `reshard_after_forward`：是否在前向传播后重新分片参数
     - `true`：前向传播后重新分片
     - `false`：保持参数gathered
   - `offload_policy`：是否启用 CPU 卸载
     - `true`：在不使用时将参数卸载到 CPU（节省 GPU 内存）
     - `false`：将所有参数保留在 GPU 上（更快但使用更多内存）
   - `wrap_policy`：模块包装策略
     - `transformer_layer_cls_to_wrap`：要wrap的 Transformer 层类名列表（例如 `["Qwen3DecoderLayer"]`）
     - `wrap_embeddings`：是否wrap input embedding（默认：`false`）
     - `wrap_lm_output`：是否wrap LM head（默认：`false`）
     - `moe_experts`：要包装的 MoE Expert类名列表（对于 MoE 模型，我们可能希望单独wrap每个expert以避免参数gather时OOM，但需要dummy前向传播以避免程序挂起，请参阅[示例](https://github.com/alibaba/ROLL/blob/main/roll/third_party/fsdp2/qwen3_moe_patch.py)）

      如果未设置 `wrap_policy`，默认将使用 transformers 模型的 `_no_split_modules`。
   - `apply_expert_patch`：是否应用 MoE 专家补丁（用于 MoE 模型）
     - `true`：应用补丁以防止不同 rank 激活不同专家时的死锁
     - `false`：不应用补丁（在 MoE 模型中可能导致死锁）
   - `apply_tiled_mlp`：是否应用 TiledMLP 优化
     - `true`：使用分块 MLP 计算以减少内存使用
     - `false`：使用标准 MLP 计算
   - `tiled_num_shards`：TiledMLP 的分片数量（默认：4）
   - `async_save_ckpt`：是否异步保存checkpoint（默认：`true`）

3. **ulysses_size**：序列并行大小（在 `model_args` 中设置）
   - 在多个 GPU 之间拆分序列维度
   - 与 FSDP2 兼容以实现混合并行
   - 适用于长上下文训练

4. **device_mapping**：指定要使用的 GPU 设备 ID 列表

5. **infer_batch_size**：推理期间的批量大小

## 设备网格配置

FSDP2 根据 `fsdp_size` 和 `ulysses_size` 支持不同的设备网格配置：

### FSDP2 模式

当 `fsdp_size >= world_size` 或 `fsdp_size <= 1` 时：

```yaml
# 示例：16 个 GPU，fsdp_size=16
strategy_config:
  fsdp_size: 16
# 设备网格：(16,) [fsdp]
# 所有 16 个 GPU 分片参数
```

### HSDP 模式

当 `fsdp_size < world_size` 时：

```yaml
# 示例：16 个 GPU，fsdp_size=8
strategy_config:
  fsdp_size: 8
# ddp_size = 16 // 8 = 2
# 设备网格：(2, 8) [ddp, fsdp]
# 2 个副本，每个副本有 8 路参数分片
```

### FSDP2 + 序列并行（Ulysses）

当同时配置 `ulysses_size` 和 `fsdp_size` 时：

```yaml
# 示例：8 个 GPU，ulysses_size=2，fsdp_size=4
model_args:
  ulysses_size: 2
strategy_config:
  fsdp_size: 4
# ddp_size = 8 // 4 = 2
# 设备网格：(2, 4) [ddp, fsdp]
# 2 个副本，每个副本有 4 路参数分片
# Ulysses：2 路序列并行（序列维度拆分）
```

## 模型特定配置

### 文本模型（Qwen2.5、Qwen3、LLaMA）

```yaml
strategy_config:
  fsdp_size: 16
  param_dtype: bf16
  reduce_dtype: float32
  wrap_policy:
    transformer_layer_cls_to_wrap: ["Qwen3DecoderLayer"]
```

### 视觉语言模型（Qwen2.5-VL、Qwen3-VL）

```yaml
actor_train:
  model_args:
    freeze_module_prefix: vision_model  # 冻结
    ulysses_size: 2  # 可选：序列并行
  strategy_args:
    strategy_name: fsdp2_train
    strategy_config:
      fsdp_size: 4
      param_dtype: bf16
      reduce_dtype: float32
      # vision encoder自动禁用 cast_forward_inputs
```

### MoE 模型（Qwen3-MoE）


```yaml
strategy_config:
  fsdp_size: 16
  param_dtype: bf16
  reduce_dtype: float32
  apply_expert_patch: true  # 如果单独wrap每个expert
  wrap_policy:
    moe_experts: ["Qwen3MoeMLP"]
```

## 注意事项

1. **PyTorch 版本**：FSDP2 需要 PyTorch >= 2.4
2. **MoE 模型**：如果单独wrap expert，始终启用 `apply_expert_patch: true` 以防止死锁（目前仅支持Qwen3-MoE）
3. **VL 模型**：对视Vision Encoder将默认`cast_forward_inputs=False`防止可能的精度问题
4. **内存与性能**：
   - `offload_policy: true` 节省内存但速度较慢
   - `reshard_after_forward: true` 节省内存但可能较慢
   - 根据硬件和要求进行平衡