# Agentic 工程实践文档

本文档介绍了 ROLL框架 中 Agentic 部分的开发实践经验，包括环境管理器开发协议、GlobalDataset 使用、验证模式配置和轨迹合成功能。

## 1. EnvManager 开发协议

EnvManager 是 Agentic 框架的核心组件，负责环境的管理和轨迹的生成。开发新的 EnvManager 需要遵循以下协议：

### 1.1 核心循环机制

EnvManager 必须实现 `run_rollout_loop` 方法，该方法遵循以下协议：

```python
def run_rollout_loop(self, data: DataProto):
    """
    1. 每次调用 run_rollout_loop 时，会持续执行 episode 直到收到数据收集完成的命令
    2. 需要重置 seed 以确保所有 group 的一致性
    3. episode_id 从 scheduler 端获取

    Seed 更新逻辑:
       group_seed = base_seed + group_id
       episode_seed = group_seed + episode_id

    trajectory_id: f"{group_id}_{episode_id}_{episode_seed}"
    """
    
    # 极简调用示例
    while self.running:
        # 从 scheduler 获取 episode_id
        self.episode_id = ray.get(self.output_queue.get_episode_id.remote(self.env_config["group_id"]))
        if self.episode_id is None:
            break
        
        # 重置环境
        rollout_cache = self.reset()
        
        while rollout_cache is not None and not rollout_cache.terminated and not rollout_cache.truncated:
            # 做出决策
            lm_output = self.make_decision(rollout_cache)
            # 执行环境步骤
            rollout_cache = self.step(lm_output)
        
        # 提交轨迹
        rollout = self.formulate_rollouts(rollout_cache)
        ray.get(self.output_queue.put.remote(self.env_config['group_id'], self.episode_id, start_step, rollout))
```

### 1.2 EnvManager开发约束

- **while loop 无限循环**：EnvManager 通过 while 循环持续执行 episode
- **数据集遍历完成才返回 None 退出**：当数据集遍历完成时，`reset()` 方法返回 None，触发循环退出
- **每个 episode 必须有对应的轨迹 put**：每个完成的 episode 必须通过 `output_queue.put` 提交轨迹数据



## 2. GlobalDataset 使用

### 2.1 设计目的

为了避免每个 env 各自读取数据造成的访存/内存瓶颈，框架层面提供了 GlobalDataset 组件，实现数据集的统一管理和分发。

### 2.2 类定义和位置

```python
# 位置: roll.datasets.global_dataset.GlobalDataset
@ray.remote
class GlobalDataset:
    def __init__(self, dataset_name, split: str = "train", mode="sample", dataset_kwargs: Dict = None):
        # mode: "sample" 或 "traversal"
```

### 2.3 两种工作模式

#### Sample Mode（采样模式）
- **用途**：训练模式下随机采样数据集
- **特点**：每次随机选择数据项
- **配置**：`mode="sample"`

#### Traversal Mode（遍历模式）
- **用途**：验证模式下需要遍历整个数据集
- **特点**：按顺序遍历数据集，确保每个数据项都被访问
- **配置**：`mode="traversal"`

### 2.4 核心特性

- **确定性采样**：`get_data_item` 方法保证相同的 seed 返回相同的数据
- **状态管理**：内部维护索引状态，支持数据集的重置和遍历

### 2.5 使用示例

参考 MathEnv、SWEEnv、TerminalBenchEnv 的实现：

```python
class MathEnv(GEMMathEnv):
    def __init__(self, dataset_name: str = "", mode: str = "train", **kwargs):
        # 转换 train/val mode 为 sample/traversal
        global_dataset_mode = "sample" if self.mode == "train" else "traversal"
        
        self.dataset = GlobalDataset.options(
            name=f"{self.mode}_{dataset_name}",
            get_if_exists=True,
            namespace=RAY_NAMESPACE
        ).remote(
            dataset_name=dataset_name,
            split=split,
            mode=global_dataset_mode
        )
        
        # 创建并注册 dataset_manager，这是实现多次val的必要实现
        self.dataset_manager = GlobalDatasetManager.options(
            name=f"{self.mode}_dataset_manager",
            get_if_exists=True,
            namespace=RAY_NAMESPACE
        ).remote()
        ray.get(self.dataset_manager.register.remote(
            dataset_name=dataset_name, 
            dataset_ref=self.dataset
        ))
```

## 3. 训练过程中 val 遍历数据集配置

### 3.1 配置原则

对于需要遍历数据集的场景（如 math/code/swe 等验证场景），需要特殊配置：

- **env 定义时设置 mode 参数**：val 模式下需要设置 `mode=val`
- **设置 val_batch_size=-1**：这样可以遍历整个 val dataset
- **env.reset 返回 None 时退出**：当数据集遍历完成时，env.reset 会返回 None

### 3.2 MathEnv 实现参考

位置：`roll.pipeline.agentic.env.gem.math_env.MathEnv`

```python
class MathEnv(GEMMathEnv):
    def __init__(self, mode: str = "train", **kwargs):
        # 转换模式
        global_dataset_mode = "sample" if self.mode == "train" else "traversal"
        self.dataset = GlobalDataset.remote(
            dataset_name=dataset_name,
            split=split,
            mode=global_dataset_mode
        )
    
    def reset(self, seed: Optional[None] = None) -> Tuple[str, dict[str, Any]]:
        data = ray.get(self.dataset.get_data_item.remote(seed=seed))
        if data is None:
            return None, None  # 数据集遍历完成
        # 处理数据...
```

### 3.3 YAML 配置示例

```yaml
rollout_batch_size: 128
val_batch_size: -1  # 遍历整个数据集

deep_math:
    env_type: "roll_math"
    env_config:
      mode: val  # 设置为验证模式
      dataset_name: data/math_deepmath_deal.jsonl
      split: train
      question_key: prompt
      answer_key: ground_truth
```

### 3.4 随机采样类评估场景

对于游戏类等随机采样的评估场景，直接按常规方式配置即可，保证相同的 seed 返回相同的数据。随机采样是默认实现，无需特殊配置。

## 4. 轨迹合成遍历数据集配置

### 4.1 AgenticRolloutPipeline 实现

位置：`roll/pipeline/agentic/agentic_rollout_pipeline.py`

### 4.2 启动方式

```shell
python examples/start_agentic_rollout_pipeline.py --config_path $CONFIG_PATH --config_name $CONFIG_NAME
```

### 4.3 核心配置参考

```yaml
# 轨迹存储目录
rollout_dump_dir: /data/oss_bucket_0/lixing/log/swe/${model_name}/rollout_trajectories

# 支持 ODPS 存储
# rollout_dump_dir: odps://odps_project/tables/table_name/ds=${model_name}

# 环境管理器配置
train_env_manager:
  max_env_num_per_worker: 16
  num_env_groups: 32
  group_size: 1  # 支持同一个 prompt rollout 多条轨迹
  tags: [SWEEnvVal]
  num_groups_partition: [32]

# 自定义环境配置
custom_envs:
  SWEEnvVal:
    env_type: "swe_env"
    env_config:
      mode: val  # 验证模式
```

### 4.4 轨迹 Dump 配置

在 EnvManager 的 `formulate_rollouts` 方法中，需要注册需要 dump 的字段及类型：

```python
def formulate_rollouts(self, rollout_cache: RolloutCache):
    # 准备数据
    save = {
        "task_idx": task_idx,
        "episode_score": episode_score,
        "traj_messages": traj_messages,
        "metrics": metrics,
        # ... 其他字段
    }
    
    # 注册 dump 字段
    lm_input.non_tensor_batch["model_name"] = np.array(
        [os.path.basename(self.pipeline_config.base_dir)], dtype=object
    )
    lm_input.non_tensor_batch["save_content"] = np.array([json.dumps(save)], dtype=object)
    lm_input.non_tensor_batch["step"] = np.array([self.current_step], dtype=object)
    lm_input.non_tensor_batch["task_idx"] = np.array([task_idx], dtype=object)
    lm_input.non_tensor_batch["stop_reason"] = np.array([stop_reason], dtype=object)
    lm_input.non_tensor_batch["mode"] = np.array([self.mode], dtype=object)
    lm_input.non_tensor_batch["episode_score"] = np.array([episode_score], dtype=object)
    
    # 配置数据库字段类型
    colummns_config = [
        ["task_idx", "bigint"],
        ["model_name", "string"],
        ["stop_reason", "string"],
        ["episode_score", "double"],
        ["mode", "string"],
        ["save_content", "string"],
    ]
    lm_input.meta_info["COLUMMNS_CONFIG"] = colummns_config
    
    return lm_input
```

### 4.5 重要说明

- **columns_config 中的 key 在 dump 之后会从 data_proto 中移出**
- **save_content 字段包含完整的轨迹信息，以 JSON 格式存储**
- **支持本地文件系统和 ODPS 表存储**
- **每个轨迹都有唯一的 trajectory_id 用于追踪**

## 5. 轨迹过滤

### 5.1 使用方式

轨迹过滤功能通过 `roll.pipeline.agentic.agentic_config.EnvManagerConfig.group_filter_cls` 配置过滤函数实现。`roll.pipeline.agentic.agentic_pipeline.GroupFilter` 是默认实现。

### 5.2 自定义过滤逻辑

可以自定义复杂的轨迹过滤逻辑，例如：

```python
class GroupFilter:
    def __init__(self, config: AgenticConfig, env_manager_config: EnvManagerConfig, mode: str):
        pass

    def filter(self, group_id: int, episode_id: int, group: list[DataProto]):
        for data in group:
            if data.meta_info["drop_flag"]:
                return True
```

通过自定义过滤函数，可以基于轨迹的各种属性（如分数、长度、停止原因等）实现灵活的过滤策略。

## 6. 常见问题

### Q1: 如何处理数据集遍历完成的情况？

A: 在 `reset` 方法中检查 `get_data_item` 返回值，如果返回 None，表示数据集遍历完成，应该返回 None 退出循环。

### Q2: 如何确保实验的可重现性？

A: 通过统一的 seed 管理机制，确保相同的 seed 返回相同的数据。GlobalDataset 的 `get_data_item` 方法保证了这一点。

### Q3: 如何处理大规模轨迹数据的存储？

A: 可以使用 ODPS 表存储，通过配置 `rollout_dump_dir` 为 `odps://` 格式的 URL 来实现。例如：

```yaml
rollout_dump_dir: odps://odps_project/tables/table_name/ds=${model_name}
```

### Q4: 如何调试轨迹生成过程？

A: 可以通过配置日志级别和添加自定义日志来调试轨迹生成过程。轨迹数据会以 JSON 格式完整保存，便于分析。

多轮交互本地调试参考文档：[调试指南](../../QuickStart/debug_guide.md)