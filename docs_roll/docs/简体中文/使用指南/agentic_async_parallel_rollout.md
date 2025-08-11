# Agentic 异步并行 Rollout

## 简介

Agentic 异步并行 rollout 是 ROLL 框架中一种高效的多轮交互处理机制。该机制以环境(env)为粒度管理多轮交互过程，每个 EnvManager 独立执行 `run_rollout_loop`，各环境之间无同步屏障(barrier)，从而实现高效的并行处理。

## 实现原理

Agentic 异步并行 rollout 的核心实现方案如下：

1. **环境粒度管理**：以 env 为粒度管理多轮交互过程，实现在 `roll/pipeline/agentic/env_manager/traj_env_manager.py` 中
2. **独立执行**：每个 EnvManager 独立执行 `run_rollout_loop`，各 env 之间无 barrier
3. **批处理获取**：`AgenticPipeline` 中的 `rollout_scheduler.get_batch()` 函数会阻塞直到获取到所需 `batch_size` 的轨迹

同步训练和异步训练的关键区别在于 `rollout_scheduler.get_batch()` 返回后是否需要暂停 EnvManager.run_rollout_loop() 过程：
- **同步训练**：搜集到 `batch_size` 条轨迹后，会退出 rollout_loop
- **异步训练**：搜集到 `batch_size` 条轨迹后，会继续 pipeline 的后续执行，同时继续执行 EnvManager.run_rollout_loop

## 关键配置参数

在Agentic中，最核心的配置是是EnvManagerConfig，它描述了各环境数量的分布信息。 EnvManager的关键配置参数如下：

```yaml
train_env_manager:
  max_env_num_per_worker: 16
  num_env_groups: 128
  # 在同一个组内，环境配置和环境种子被确保是相同的
  group_size: 8
  tags: [FrozenLake]
  num_groups_partition: [128] # 如果未设置，所有环境名称将平均分配到组中。在同一个组中，环境配置和环境种子（prompt）在每次生成中都是相同的

val_env_manager:
  max_env_num_per_worker: 32
  num_env_groups: 1024
  group_size: 1 # 应该设置为1，因为验证时温度设置为0，相同提示会导致相同输出
  tags: [SimpleSokoban, LargerSokoban, SokobanDifferentGridVocab, FrozenLake]
  num_groups_partition: [256, 256, 256, 256] # 如果未设置，所有环境名称将平均分配到组中。在同一个组中，环境配置和环境种子（prompt）在每次生成中都是相同的
```

### 配置参数详解

#### max_env_num_per_worker
- **含义**：每个 worker（Ray Actor）中可同时运行的最大环境数量
- **作用**：控制单个 worker 的并发环境数，影响内存使用和并行度
- **示例**：`max_env_num_per_worker: 16` 表示每个 worker 最多同时运行 16 个环境实例

#### num_env_groups
- **含义**：训练期间环境组的总数量
- **作用**：定义并行环境组的总数，影响训练的并行度

#### group_size
- **含义**：每个环境组中包含的环境实例数量
- **作用**：控制组内并行度，相同组内的环境具有相同的配置和种子
- **注意事项**：
  - 训练环境中通常设置为大于 1 的值以增加组内多样性
  - 验证环境中应设置为 1，因为验证时温度为 0，相同提示会产生相同输出
- **示例**：
  - `group_size: 8` 表示每个环境组包含 8 个环境实例
  - `num_env_groups: 128` 表示总共创建 128 个环境组
  - env实例的总数量为: `gropu_size * num_env_groups` = 1024 个

#### tags
- **含义**：环境的标签列表，用于标识和选择要使用的环境类型
- **作用**：指定要使用的环境类型，框架会根据标签加载对应的环境实现
- **示例**：`tags: [SimpleSokoban, FrozenLake]` 表示使用 SimpleSokoban 和 FrozenLake 两种环境

#### num_groups_partition
- **含义**：各环境类型的组数量分配
- **作用**：指定不同环境类型在总环境组中的分配比例
- **默认行为**：如果未设置，所有环境名称将平均分配到组中
- **示例**：
  - `num_groups_partition: [128]` 表示单一环境类型占用全部 128 个组
  - `num_groups_partition: [256, 256, 256, 256]` 表示四种环境类型各占用 256 个组

## 使用建议

1. **合理设置并行度**：根据硬件资源（CPU、内存）合理设置 `max_env_num_per_worker` 和 `num_env_groups`
2. **环境组配置**：训练时可适当增加 `group_size` 以提高组内并行度，验证时应设置为 1，这在GRPO类基于group traj计算adv时是必须的
3. **环境类型分配**：通过 `tags` 和 `num_groups_partition` 合理分配不同环境类型的训练资源
4. **资源监控**：监控系统资源使用情况，避免因环境实例过多导致资源耗尽

通过合理配置这些参数，可以充分发挥 Agentic 异步并行 rollout 的性能优势，提高多轮交互任务的训练效率。