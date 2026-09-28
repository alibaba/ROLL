# 检查点保存与恢复指南

在 ROLL 框架中，检查点（Checkpoint）机制允许您保存训练过程中的模型状态，以便在需要时恢复训练。本文档将详细介绍如何配置和使用检查点保存与恢复功能。

## 检查点保存配置

ROLL 框架通过 `checkpoint_config` 参数来配置检查点保存的相关设置。以下是一个典型的配置示例：

```yaml
checkpoint_config:
  type: file_system
  output_dir: /data/cpfs_0/rl_examples/models/${exp_name}
```

### 配置参数详解

1. **type**: 指定检查点存储的类型
   - 目前支持 `file_system`，表示将检查点保存到文件系统中

2. **output_dir**: 指定检查点保存的目录路径
   - 可以使用变量，如 `${exp_name}` 表示实验名称
   - 框架会在该目录下自动创建时间戳子目录来区分不同的检查点

## 检查点保存机制

ROLL 框架会在以下情况下自动保存检查点：

1. **定期保存**: 根据 `save_steps` 参数设置的间隔自动保存
   ```yaml
   save_steps: 100  # 每100步保存一次检查点
   ```

2. **训练结束时**: 在训练完成时自动保存最终检查点

3. **手动保存**: 在代码中可以调用相应的 API 手动保存检查点

FSDP2 异步 DCP 保存（`strategy_config.async_save_ckpt: true`）在 storage-level staging 时读取 DTensor 的真实 local storage，并保留 mesh、placements、shape 和 view offset。使用 `offload_backend: local` 时，模型参数只 offload 一次，DCP 直接写已有的 CPU flat buffer，不再额外分配一份模型 CPU 快照；异步写盘完成前，Strategy 会保留这些 buffer，并在完成后再回收。优化器、scheduler 以及不在 store 中的 tensor 仍使用常规的独立 CPU staging。`local_dedup` 每个 rank 只保存一段数据，需要先重建 local shard，因此仍走常规 staging 路径。该实现不增加 GPU shard 副本和 checkpoint 通信次数，checkpoint 格式与恢复配置保持不变。

## 恢复训练配置

要从检查点恢复训练，需要设置 `resume_from_checkpoint` 参数：

```yaml
resume_from_checkpoint: false  # 默认不恢复训练
```

要启用恢复训练功能，将该参数设置为检查点路径：

```yaml
resume_from_checkpoint: /data/cpfs_0/rl_examples/models/exp_name/checkpoint-500
```

### 恢复训练的工作原理

1. 当 `resume_from_checkpoint` 设置为有效的检查点路径时，框架会：
   - 加载模型参数
   - 恢复优化器状态
   - 恢复学习率调度器状态
   - 恢复训练步数等其他训练状态

2. 恢复训练会从检查点保存时的训练步数继续训练

## 使用示例

以下是一个完整的配置示例，展示了如何设置检查点保存和恢复功能：

```yaml
exp_name: "qwen2.5-7B-rlvr-config"
seed: 42
logging_dir: ./output/logs
output_dir: ./output

# 检查点配置
checkpoint_config:
  type: file_system
  output_dir: /data/cpfs_0/rl_examples/models/${exp_name}

# 恢复训练配置
resume_from_checkpoint: false  # 设置为检查点路径以恢复训练

# 训练控制参数
max_steps: 500
save_steps: 100  # 每100步保存一次检查点
logging_steps: 1
eval_steps: 10

# 其他训练配置...
```

要从检查点恢复训练，只需将 `resume_from_checkpoint` 设置为相应的检查点路径：

```yaml
resume_from_checkpoint: /data/cpfs_0/rl_examples/models/qwen2.5-7B-rlvr-config/checkpoint-300
```

## 最佳实践

1. **定期保存检查点**: 根据训练时间和资源消耗合理设置 `save_steps`
2. **检查存储空间**: 确保 `output_dir` 有足够的存储空间保存检查点
3. **验证检查点**: 在恢复训练前验证检查点的完整性和有效性
4. **备份重要检查点**: 对重要的检查点进行备份，防止数据丢失

通过合理配置检查点保存与恢复功能，您可以确保训练过程的安全性和可恢复性，避免因意外中断而丢失训练进度。
