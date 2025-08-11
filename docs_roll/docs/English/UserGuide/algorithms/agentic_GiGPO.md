# StepWiseLearning——GiGPO (Group-in-Group Policy Optimization)

## Introduction

GiGPO (Group-in-Group Policy Optimization) is a novel reinforcement learning algorithm for LLM agent training. It achieves fine-grained credit assignment for LLM agents while preserving the appealing properties of group-based RL: critic-free, low memory, and stable convergence.

GiGPO introduces a two-level structure for estimating relative advantage:
1. At the episode level, GiGPO computes macro relative advantages based on groups of complete trajectories
2. At the step level, GiGPO introduces an anchor state grouping mechanism that retroactively constructs step-level groups by identifying repeated environment states across trajectories

This hierarchical structure effectively captures both global trajectory quality and local step effectiveness without relying on auxiliary models or additional rollouts.

## GiGPO Configuration Parameters

In ROLL, the core implementation of GiGPO is located at `roll/pipeline/agentic/utils.py`. The specific configuration parameters for the GiGPO algorithm are as follows (`roll.pipeline.agentic.agentic_config.AgenticConfig`):

```yaml
# GiGPO core config
adv_estimator: "gigpo"
batch_adjust_mode: "copy"
step_reward_weight: 1.0
episode_reward_weight: 1.0
step_reward_gamma: 0.95

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
  group_size: 8
  tags: [FrozenLake]
  num_groups_partition: [128] # If not set, all env names divide nums equally. Under the same group, the env config and env seed (prompt) are equal in each generation

env_manager_cls: roll.pipeline.agentic.env_manager.step_env_manager.StepEnvManager
```

### Core Parameter Descriptions

- `adv_estimator`: Advantage estimator type, set to "gigpo", which is the core configuration of the GiGPO algorithm
- `batch_adjust_mode`: Batch adjustment mode, optional values are "copy", "delete", "auto", default value is "copy"
- `step_reward_weight`: Step reward weight, used in the GiGPO algorithm, default value is 1.0
- `episode_reward_weight`: Episode reward weight, used in the GiGPO algorithm, default value is 1.0
- `step_reward_gamma`: Discount factor for step reward calculation, default value is 0.95
- `env_manager_cls`: Environment manager class, GiGPO needs to use `roll.pipeline.agentic.env_manager.step_env_manager.StepEnvManager`

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

You can refer to the following configuration files to set up GiGPO training:
- `./examples/docs_examples/example_gigpo.yaml`

## References
[1] Feng, L.; Xue, Z.; Liu, T.; An, B. Group-in-Group Policy Optimization for LLM Agent Training. arXiv 2025, 2505.10978.