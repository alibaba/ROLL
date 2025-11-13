# Agentic Engineering Practice Documentation

This document introduces the development practices of the Agentic component in the ROLL framework, including environment manager development protocols, GlobalDataset usage, validation mode configuration, and trajectory synthesis functionality.

## 1. EnvManager Development Protocol

EnvManager is the core component of the Agentic framework, responsible for environment management and trajectory generation. Developing new EnvManagers requires following the following protocol:

### 1.1 Core Loop Mechanism

EnvManager must implement the `run_rollout_loop` method, which follows the following protocol:

```python
def run_rollout_loop(self, data: DataProto):
    """
    1. Each time run_rollout_loop is called, it continuously executes episodes 
       until receiving a command that data collection is complete
    2. Need to reset seed to ensure consistency across all groups
    3. episode_id is obtained from the scheduler

    Seed update logic:
       group_seed = base_seed + group_id
       episode_seed = group_seed + episode_id

    trajectory_id: f"{group_id}_{episode_id}_{episode_seed}"
    """
    
    # Minimal call example
    while self.running:
        # Get episode_id from scheduler
        self.episode_id = ray.get(self.output_queue.get_episode_id.remote(self.env_config["group_id"]))
        if self.episode_id is None:
            break
        
        # Reset environment
        rollout_cache = self.reset()
        
        while rollout_cache is not None and not rollout_cache.terminated and not rollout_cache.truncated:
            # Make decision
            lm_output = self.make_decision(rollout_cache)
            # Execute environment step
            rollout_cache = self.step(lm_output)
        
        # Submit trajectory
        rollout = self.formulate_rollouts(rollout_cache)
        ray.get(self.output_queue.put.remote(self.env_config['group_id'], self.episode_id, start_step, rollout))
```

### 1.2 EnvManager Development Constraints

- **While loop infinite loop**: EnvManager continuously executes episodes through a while loop
- **Exit only when dataset traversal is complete**: When dataset traversal is complete, the `reset()` method returns None, triggering loop exit
- **Each episode must have corresponding trajectory put**: Each completed episode must submit trajectory data through `output_queue.put`

## 2. GlobalDataset Usage

### 2.1 Design Purpose

To avoid memory access/memory bottlenecks caused by each env reading data independently, the framework provides the GlobalDataset component at the framework level to implement unified management and distribution of datasets.

### 2.2 Class Definition and Location

```python
# Location: roll.datasets.global_dataset.GlobalDataset
@ray.remote
class GlobalDataset:
    def __init__(self, dataset_name, split: str = "train", mode="sample", dataset_kwargs: Dict = None):
        # mode: "sample" or "traversal"
```

### 2.3 Two Working Modes

#### Sample Mode
- **Purpose**: Random sampling of datasets in training mode
- **Features**: Randomly select data items each time
- **Configuration**: `mode="sample"`

#### Traversal Mode
- **Purpose**: Need to traverse the entire dataset in validation mode
- **Features**: Traverse dataset sequentially, ensuring each data item is accessed
- **Configuration**: `mode="traversal"`

### 2.4 Core Features

- **Deterministic sampling**: The `get_data_item` method ensures the same seed returns the same data
- **State management**: Internally maintains index state, supporting dataset reset and traversal

### 2.5 Usage Example

Refer to the implementation of MathEnv, SWEEnv, TerminalBenchEnv:

```python
class MathEnv(GEMMathEnv):
    def __init__(self, dataset_name: str = "", mode: str = "train", **kwargs):
        # Convert train/val mode to sample/traversal
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
        
        # Create and register dataset_manager, this is necessary for implementing multiple val
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

## 3. Validation Dataset Traversal Configuration During Training

### 3.1 Configuration Principles

For scenarios that require dataset traversal (such as math/code/swe validation scenarios), special configuration is required:

- **Set mode parameter when defining env**: In val mode, need to set `mode=val`
- **Set val_batch_size=-1**: This allows traversal of the entire val dataset
- **Exit when env.reset returns None**: When dataset traversal is complete, env.reset will return None

### 3.2 MathEnv Implementation Reference

Location: `roll.pipeline.agentic.env.gem.math_env.MathEnv`

```python
class MathEnv(GEMMathEnv):
    def __init__(self, mode: str = "train", **kwargs):
        # Convert mode
        global_dataset_mode = "sample" if self.mode == "train" else "traversal"
        self.dataset = GlobalDataset.remote(
            dataset_name=dataset_name,
            split=split,
            mode=global_dataset_mode
        )
    
    def reset(self, seed: Optional[None] = None) -> Tuple[str, dict[str, Any]]:
        data = ray.get(self.dataset.get_data_item.remote(seed=seed))
        if data is None:
            return None, None  # Dataset traversal complete
        # Process data...
```

### 3.3 YAML Configuration Example

```yaml
rollout_batch_size: 128
val_batch_size: -1  # Traverse entire dataset

deep_math:
    env_type: "roll_math"
    env_config:
      mode: val  # Set to validation mode
      dataset_name: data/math_deepmath_deal.jsonl
      split: train
      question_key: prompt
      answer_key: ground_truth
```

### 3.4 Random Sampling Evaluation Scenarios

For random sampling evaluation scenarios such as games, simply configure in the conventional way, ensuring the same seed returns the same data. Random sampling is the default implementation, no special configuration required.

## 4. Trajectory Synthesis Dataset Traversal Configuration

### 4.1 AgenticRolloutPipeline Implementation

Location: `roll/pipeline/agentic/agentic_rollout_pipeline.py`

### 4.2 Startup Method

```shell
python examples/start_agentic_rollout_pipeline.py --config_path $CONFIG_PATH --config_name $CONFIG_NAME
```

### 4.3 Core Configuration Reference

```yaml
# Trajectory storage directory
rollout_dump_dir: /data/oss_bucket_0/lixing/log/swe/${model_name}/rollout_trajectories

# Support ODPS storage
# rollout_dump_dir: odps://odps_project/tables/table_name/ds=${model_name}

# Environment manager configuration
train_env_manager:
  max_env_num_per_worker: 16
  num_env_groups: 32
  group_size: 1  # Support multiple trajectories for the same prompt rollout
  tags: [SWEEnvVal]
  num_groups_partition: [32]

# Custom environment configuration
custom_envs:
  SWEEnvVal:
    env_type: "swe_env"
    env_config:
      mode: val  # Validation mode
```

### 4.4 Trajectory Dump Configuration

In the `formulate_rollouts` method of EnvManager, need to register dump fields and types:

```python
def formulate_rollouts(self, rollout_cache: RolloutCache):
    # Prepare data
    save = {
        "task_idx": task_idx,
        "episode_score": episode_score,
        "traj_messages": traj_messages,
        "metrics": metrics,
        # ... other fields
    }
    
    # Register dump fields
    lm_input.non_tensor_batch["model_name"] = np.array(
        [os.path.basename(self.pipeline_config.base_dir)], dtype=object
    )
    lm_input.non_tensor_batch["save_content"] = np.array([json.dumps(save)], dtype=object)
    lm_input.non_tensor_batch["step"] = np.array([self.current_step], dtype=object)
    lm_input.non_tensor_batch["task_idx"] = np.array([task_idx], dtype=object)
    lm_input.non_tensor_batch["stop_reason"] = np.array([stop_reason], dtype=object)
    lm_input.non_tensor_batch["mode"] = np.array([self.mode], dtype=object)
    lm_input.non_tensor_batch["episode_score"] = np.array([episode_score], dtype=object)
    
    # Configure database field types
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

### 4.5 Important Notes

- **Keys in columns_config will be removed from data_proto after dump**
- **save_content field contains complete trajectory information, stored in JSON format**
- **Support local file system and ODPS table storage**
- **Each trajectory has a unique trajectory_id for tracking**

## 5. Trajectory Filtering

### 5.1 Usage Method

The trajectory filtering function is implemented by configuring the filter class through `roll.pipeline.agentic.agentic_config.EnvManagerConfig.group_filter_cls`. `roll.pipeline.agentic.agentic_pipeline.GroupFilter` is the default implementation.

### 5.2 Custom Filtering Logic

Custom complex trajectory filtering logic can be implemented, for example:

```python
class GroupFilter:
    def __init__(self, config: AgenticConfig, env_manager_config: EnvManagerConfig, mode: str):
        pass

    def filter(self, group_id: int, episode_id: int, group: list[DataProto]):
        for data in group:
            if data.meta_info["drop_flag"]:
                return True
```

Through custom filter functions, flexible filtering strategies can be implemented based on various trajectory attributes (such as score, length, stop reason, etc.).

## 6. Frequently Asked Questions

### Q1: How to handle dataset traversal completion?

A: Check the `get_data_item` return value in the `reset` method. If it returns None, it means dataset traversal is complete, and you should return None to exit the loop.

### Q2: How to ensure experiment reproducibility?

A: Through a unified seed management mechanism, ensure the same seed returns the same data. The `get_data_item` method of GlobalDataset guarantees this.

### Q3: How to handle large-scale trajectory data storage?

A: You can use ODPS table storage by configuring `rollout_dump_dir` as an `odps://` format URL. For example:

```yaml
rollout_dump_dir: odps://odps_project/tables/table_name/ds=${model_name}
```

### Q4: How to debug the trajectory generation process?

A: You can debug the trajectory generation process by configuring log levels and adding custom logs. Trajectory data will be completely saved in JSON format for easy analysis.

For multi-round interaction local debugging, refer to the documentation: [Debug Guide](../../QuickStart/debug_guide.md)