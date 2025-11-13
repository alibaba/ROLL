# TrajWiseLearning——StarPO (State-Thinking-Actions-Reward Policy Optimization)

## Introduction

StarPO (State-Thinking-Actions-Reward Policy Optimization) is a reinforcement learning algorithm for LLM agent training. It optimizes by treating the entire multi-turn interaction trajectory (including observations, reasoning traces, actions, and feedback) as a coherent unit, rather than independently processing each action as in traditional methods.

The core idea of StarPO is trajectory-level optimization, which alternates between two phases:
1. **Rollout Phase**: Generate reasoning-interaction trajectories
2. **Update Phase**: Optimize the model based on complete trajectories


## StarPO Configuration Parameters

In ROLL, the core implementation of StarPO is located at `roll/pipeline/agentic/utils.py`. The specific configuration parameters for the StarPO algorithm are as follows (`roll.pipeline.agentic.agentic_config.AgenticConfig`):

```yaml
# StarPO core config
# StarPO related
adv_estimator: "reinforce"

# rollout_batch_size is the number of trajectories
rollout_batch_size: 1024
val_batch_size: 1024
sequence_length: 1024

advantage_clip: 0.2
ppo_epochs: 1

# pg_clip: 0.1
#dual_clip_loss: True
init_kl_coef: 0.0
whiten_advantages: true
entropy_loss_coef: 0
max_grad_norm: 1.0

reward_normalization:
  grouping: traj_group_id # Can be tags(env_type)/traj_group_id(group)/batch(rollout_batch)... group_by calculates reward/adv
  method: mean # asym_clip / identity / mean_std / mean

train_env_manager:
  max_env_num_per_worker: 16
  num_env_groups: 128
  # under the same group, the env config and env seed are ensured to be equal
  group_size: 8 # grpo's grpo
  tags: [FrozenLake]
  num_groups_partition: [128] # If not set, all env names divide nums equally. Under the same group, the env config and env seed (prompt) are equal in each generation

env_manager_cls: roll.pipeline.agentic.env_manager.traj_env_manager.TrajEnvManager
```

### Core Parameter Descriptions

- `adv_estimator`: Advantage estimator type, set to "reinforce", which is the core configuration of the StarPO algorithm
- `env_manager_cls`: Environment manager class, StarPO needs to use `roll.pipeline.agentic.env_manager.traj_env_manager.TrajEnvManager`

### PPO Related Parameters

The following parameters are common configuration items for PPO-class algorithms:

- `rollout_batch_size`: Number of trajectories per rollout batch, default value is 1024
- `val_batch_size`: Validation batch size, default value is 1024
- `sequence_length`: Maximum sequence length, default value is 1024
- `advantage_clip`: Advantage value clipping range, default value is 0.2
- `ppo_epochs`: Number of optimization epochs per batch of samples, default value is 1
- `init_kl_coef`: Initial coefficient for KL penalty, default value is 0.0
- `whiten_advantages`: Whether to whiten advantage values, default value is true
- `entropy_loss_coef`: Entropy loss coefficient, default value is 0
- `max_grad_norm`: Maximum norm for gradient clipping, default value is 1.0

### Environment Manager Parameters

- `train_env_manager.max_env_num_per_worker`: Maximum number of environments per worker, default value is 16
- `train_env_manager.num_env_groups`: Number of training environment groups, default value is 128
- `train_env_manager.group_size`: Number of environments per group, default value is 8
- `train_env_manager.tags`: List of environment tags, default value is [FrozenLake]
- `train_env_manager.num_groups_partition`: Group allocation for each environment type, default value is [128]

## Reference Examples

You can refer to the following configuration files to set up StarPO training:
- `./examples/qwen2.5-0.5B-agentic/agent_val_frozen_lake.yaml`

## References
[1] Liu, T.; Feng, L.; An, B. StarPO: State-Regularized Policy Optimization for LLM Agent Training. arXiv 2025, 2504.20073.