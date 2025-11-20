# Agentic Asynchronous Parallel Rollout

## Introduction

Agentic asynchronous parallel rollout is an efficient multi-turn interaction processing mechanism in the ROLL framework. This mechanism manages multi-turn interaction processes at the environment (env) granularity, with each EnvManager independently executing `run_rollout_loop` without synchronization barriers between environments, thus achieving efficient parallel processing.

## Implementation Principle

The core implementation scheme of agentic asynchronous parallel rollout is as follows:

1. **Environment Granularity Management**: Multi-turn interaction processes are managed at the env granularity, implemented in `roll/pipeline/agentic/env_manager/traj_env_manager.py`
2. **Independent Execution**: Each EnvManager independently executes `run_rollout_loop` without barriers between envs
3. **Batch Processing**: The `rollout_scheduler.get_batch()` function in `AgenticPipeline` blocks until the required `batch_size` of trajectories is obtained

The key difference between synchronous and asynchronous training lies in whether the EnvManager.run_rollout_loop() process needs to be paused after `rollout_scheduler.get_batch()` returns:
- **Synchronous Training**: After collecting `batch_size` trajectories, the rollout_loop exits
- **Asynchronous Training**: After collecting `batch_size` trajectories, the pipeline continues with subsequent execution while continuing to execute EnvManager.run_rollout_loop

## Key Configuration Parameters

In Agentic, the most core configuration is EnvManagerConfig, which describes the distribution information of various environment quantities. The key configuration parameters for EnvManager are as follows:

```yaml
train_env_manager:
  max_env_num_per_worker: 16
  num_env_groups: 128
  # Under the same group, the env config and env seed are ensured to be equal
  group_size: 8
  tags: [FrozenLake]
  num_groups_partition: [128] # If not set, all env names divide nums equally. Under the same group, the env config and env seed (prompt) are equal in each generation

val_env_manager:
  max_env_num_per_worker: 32
  num_env_groups: 1024
  group_size: 1 # Should be set to 1 because val temperature is set to 0 and same prompt leads to same output
  tags: [SimpleSokoban, LargerSokoban, SokobanDifferentGridVocab, FrozenLake]
  num_groups_partition: [256, 256, 256, 256] # If not set, all env names divide nums equally. Under the same group, the env config and env seed (prompt) are equal in each generation
```

### Configuration Parameter Details

#### max_env_num_per_worker
- **Meaning**: The maximum number of environments that can run simultaneously per worker (Ray Actor)
- **Purpose**: Controls the concurrency of environments per single worker, affecting memory usage and parallelism
- **Example**: `max_env_num_per_worker: 16` means each worker runs at most 16 environment instances simultaneously

#### num_env_groups
- **Meaning**: The total number of environment groups during training
- **Purpose**: Defines the total number of parallel environment groups, affecting training parallelism

#### group_size
- **Meaning**: The number of environment instances contained in each environment group
- **Purpose**: Controls intra-group parallelism; environments within the same group have the same configuration and seed
- **Notes**:
  - In training environments, typically set to a value greater than 1 to increase intra-group diversity
  - In validation environments, should be set to 1 because validation temperature is 0, and identical prompts produce identical outputs
- **Example**:
  - `group_size: 8` means each environment group contains 8 environment instances
  - `num_env_groups: 128` means a total of 128 environment groups are created
  - Total number of env instances is: `group_size * num_env_groups` = 1024

#### tags
- **Meaning**: List of environment tags used to identify and select environment types
- **Purpose**: Specifies the environment types to use; the framework loads corresponding environment implementations based on tags
- **Example**: `tags: [SimpleSokoban, FrozenLake]` indicates using SimpleSokoban and FrozenLake environment types

#### num_groups_partition
- **Meaning**: Group number allocation for different environment types
- **Purpose**: Specifies the allocation ratio of different environment types in the total environment groups
- **Default Behavior**: If not set, all environment names are equally divided into groups
- **Example**:
  - `num_groups_partition: [128]` means a single environment type occupies all 128 groups
  - `num_groups_partition: [256, 256, 256, 256]` means four environment types each occupy 256 groups

## Usage Recommendations

1. **Reasonable Parallelism Settings**: Set `max_env_num_per_worker` and `num_env_groups` appropriately based on hardware resources (CPU, memory)
2. **Environment Group Configuration**: Increase `group_size` during training to improve intra-group parallelism; set to 1 during validation, which is required for GRPO-like algorithms that calculate advantages based on group trajectories
3. **Environment Type Allocation**: Reasonably allocate training resources for different environment types through `tags` and `num_groups_partition`
4. **Resource Monitoring**: Monitor system resource usage to avoid resource exhaustion due to too many environment instances

By properly configuring these parameters, you can fully leverage the performance advantages of agentic asynchronous parallel rollout and improve training efficiency for multi-turn interaction tasks.