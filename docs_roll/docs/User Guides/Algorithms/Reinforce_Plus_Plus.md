# Reinforce++

## Introduction

Reinforce++ is a policy gradient-based reinforcement learning algorithm that is an enhanced version of the classic REINFORCE algorithm. Reinforce++ works as follows:

1. **Group Sampling**: For a given problem, the model generates multiple possible solutions, forming a "group" of outputs.
2. **Reward Calculation**: Each solution is evaluated and assigned a reward based on its correctness or quality.
3. **Policy Update**: The model updates its parameters based on reward signals and generated sequences, reinforcing strategies that obtain higher rewards.

## Reinforce++ Configuration Parameters

In ROLL, the Reinforce++ algorithm-specific configuration parameters are as follows (`roll.pipeline.rlvr.rlvr_config.RLVRConfig`):

```yaml
# Reinforce++ core config
adv_estimator: "reinforce"

# normalize
norm_mean_type: batch
norm_std_type: batch

# reward
add_token_level_kl: false

# advantage
whiten_advantages: false

# ppo related, other parts are compatible with GRPO/PPO settings
rollout_batch_size: 64  # prompt
num_return_sequences_in_group: 8
prompt_length: 2048
response_length: 4096
ppo_epochs: 1
use_kl_loss: true
kl_loss_coef: 0.001
loss_agg_mode: "seq-mean-token-mean"

# advantage
advantage_clip: 2.0
dual_clip_loss: true
# clip
reward_clip: 10

```

### Core Parameter Descriptions

- `adv_estimator`: Advantage estimator type, set to "reinforce", which is the core configuration of the Reinforce++ algorithm
- `norm_mean_type`: Mean type for reward normalization: the options are "batch", "group", "running", or None; the default is None
- `norm_std_type`: Std type for reward normalization: the options are "batch", "group", "running", or None; the default is None
- `add_token_level_kl`: Whether to add token-level KL penalty, default value is false
- `whiten_advantages`: Whether to whiten advantage values, default value is false

### PPO Related Parameters

The following parameters are common configuration items for PPO-class algorithms:

- `rollout_batch_size`: Number of prompts per rollout_batch_size, default value is 64
- `num_return_sequences_in_group`: Number of responses generated per prompt (group size), the total number of samples trained per pipeline step is (rollout_batch_size * num_return_sequences_in_group), default value is 8
- `prompt_length`: Maximum length of prompts, default value is 2048
- `response_length`: Maximum length of responses, default value is 4096
- `ppo_epochs`: Number of optimization rounds per batch of samples, default value is 1
- `use_kl_loss`: Whether to use KL divergence loss, default value is true
- `kl_loss_coef`: KL-loss coefficient, default value is 0.001
- `loss_agg_mode`: Loss aggregation mode, default is "seq-mean-token-sum", optional values are "token-mean", "seq-mean-token-sum", "seq-mean-token-mean", "seq-mean-token-sum-norm"
- `advantage_clip`: Advantage value clipping range, default value is 2.0
- `dual_clip_loss`: Whether to use dual clipping loss, default value is true
- `reward_clip`: Reward value clipping range, default value is 10

## Reference Example

You can refer to the following configuration file to set up Reinforce++ training:
- `./examples/docs_examples/example_reinforce_pp.yaml`

## References
[1] https://arxiv.org/abs/2504.11343