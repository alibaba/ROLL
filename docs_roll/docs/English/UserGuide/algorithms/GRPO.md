# Group Relative Policy Optimization (GRPO)

## Introduction

Group Relative Policy Optimization (GRPO) is a reinforcement learning algorithm that simplifies the training process by eliminating the need for a value function (critic) model. GRPO works as follows:

1. **Group Sampling**: For a given problem, the model generates multiple possible solutions, forming a "group" of outputs.
2. **Reward Assignment**: Each solution is evaluated and assigned a reward based on its correctness or quality.
3. **Baseline Calculation**: The average reward of the group serves as the baseline.
4. **Policy Update**: The model updates its parameters by comparing each solution's reward to the group baseline, reinforcing solutions that are better than average and suppressing those that are worse than average.

This approach reduces computational overhead by avoiding training a separate value estimation model, making the learning process more efficient.

## GRPO Configuration Parameters

In ROLL, the GRPO algorithm-specific configuration parameters are as follows (`roll.pipeline.rlvr.rlvr_config.RLVRConfig`):

```yaml
# grpo
rollout_batch_size: 64  # prompt
num_return_sequences_in_group: 8
prompt_length: 2048
response_length: 4096

adv_estimator: "grpo"
ppo_epochs: 1
use_kl_loss: true
kl_loss_coef: 0.001
loss_agg_mode: "seq-mean-token-mean"

# ppo related
# advantage
whiten_advantages: true
advantage_clip: 2.0
dual_clip_loss: true

# clip
reward_clip: 10
# normalize
norm_mean_type: ~
norm_std_type: ~

# reward
add_token_level_kl: false
```

### Core Parameter Descriptions

- `rollout_batch_size`: Number of prompts per rollout_batch_size
- `num_return_sequences_in_group`: Number of responses generated per prompt (group size), the total number of samples trained per pipeline step is (rollout_batch_size * num_return_sequences_in_group)
- `prompt_length`: Maximum length of prompts
- `response_length`: Maximum length of responses
- `adv_estimator`: Advantage estimator type, set to "grpo"
- `ppo_epochs`: Number of optimization rounds per batch of samples
- `use_kl_loss`: Whether to use KL divergence loss
- `kl_loss_coef`: KL-loss coefficient
- `loss_agg_mode`: Loss aggregation mode, default is "seq-mean-token-sum", Literal["token-mean", "seq-mean-token-sum", "seq-mean-token-mean", "seq-mean-token-sum-norm"]

### PPO Related Parameters

The following parameters are common in PPO but also apply to GRPO:
- `whiten_advantages`: Whether to whiten advantage values
- `advantage_clip`: Advantage value clipping range
- `dual_clip_loss`: Whether to use dual clipping loss
- `reward_clip`: Reward value clipping range
- `norm_mean_type`: Mean type for reward normalization: the options are "batch", "group", "running", or None; the default is None
- `norm_std_type`: Std type for reward normalization: the options are "batch", "group", "running", or None; the default is None
- `add_token_level_kl`: Whether to add token-level KL penalty

## Differences Between GRPO and PPO

The main differences between GRPO and traditional PPO algorithms are:

1. **No Critic Model Required**: GRPO does not require training a separate value network (critic)
2. **Group Sampling**: GRPO generates multiple completions (responses) for each prompt, rather than evaluating one rollout for each input
3. **Relative Rewards**: Within each group, completions are scored and normalized based on group performance
4. **KL Loss**: GRPO performs regularization by directly adding the KL divergence between the training policy and reference policy to the loss function

## Reference Example

You can refer to the following configuration file to set up GRPO training:
- `./examples/docs_examples/example_grpo.yaml`