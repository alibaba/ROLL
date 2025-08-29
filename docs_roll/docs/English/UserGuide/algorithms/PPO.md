# Proximal Policy Optimization (PPO)

## Introduction

Proximal Policy Optimization (PPO) is a class of policy gradient methods for reinforcement learning introduced by OpenAI in 2017. PPO strikes a balance between simplicity, stability, and performance, making it one of the most widely used algorithms in modern RL applications, including fine-tuning large-scale language models.

Traditional policy gradient methods (such as REINFORCE or Vanilla Policy Gradient) have the following issues:

1. High variance and poor sample efficiency
2. Instability due to large policy updates

PPO addresses these issues by using a clipped surrogate objective function that avoids overly large updates without requiring second-order derivatives.

## PPO Configuration Parameters

In ROLL, the configuration parameters for the PPO algorithm are as follows (`roll.pipeline.rlvr.rlvr_config.RLVRConfig`):

```yaml
# ppo related

rollout_batch_size: 512  # prompt
prompt_length: 2048
response_length: 4096

adv_estimator: "gae"
num_return_sequences_in_group: 1
ppo_epochs: 1
use_kl_loss: true
kl_loss_coef: 0.001
loss_agg_mode: "seq-mean-token-sum"


whiten_advantages: true
advantage_clip: 2.0
reward_clip: ~
dual_clip_loss: true
lambd: 0.95
gamma: 1
pg_clip: 0.2
value_clip: ~
kl_penalty: "kl"
target_kl: ~
init_kl_coef: 0.2
kl_horizon: 10000
add_token_level_kl: false
# normalize
norm_mean_type: None
norm_std_type: None
```

### PPO Parameter Descriptions

| Parameter | Default Value | Options | Description |
|-----------|---------------|---------|-------------|
| `rollout_batch_size` | 512 | Positive integer | Number of prompts per batch |
| `prompt_length` | 2048 | Positive integer | Maximum length of prompts |
| `response_length` | 4096 | Positive integer | Maximum length of responses |
| `adv_estimator` | "gae" | "gae", "reinforce", "grpo" | Advantage estimator type |
| `num_return_sequences_in_group` | 1 | Positive integer | Number of responses generated per prompt |
| `ppo_epochs` | 1 | Positive integer | Number of optimization rounds per batch of samples |
| `use_kl_loss` | true | true, false | Whether to use KL divergence loss |
| `kl_loss_coef` | 0.001 | Float | KL divergence loss coefficient |
| `loss_agg_mode` | "seq-mean-token-sum" | "token-mean", "seq-mean-token-sum", "seq-mean-token-mean", "seq-mean-token-sum-norm" | Loss aggregation mode |
| `whiten_advantages` | true | true, false | Whether to whiten advantage values |
| `advantage_clip` | 2.0 | Float, ~ (means not set) | Advantage value clipping range |
| `reward_clip` | ~ | Float, ~ (means not set) | Reward value clipping range |
| `dual_clip_loss` | true | true, false | Whether to use dual clipping loss |
| `lambd` | 0.95 | Float in [0, 1] range | Lambda parameter in GAE estimator, used to trade off bias and variance |
| `gamma` | 1 | Float in [0, 1] range | Discount factor |
| `pg_clip` | 0.2 | Float | PPO clipping range |
| `value_clip` | ~ | Float, ~ (means not set) | Value function clipping range |
| `kl_penalty` | "kl" | "kl", "abs", "mse", "full" | KL penalty options |
| `target_kl` | ~ | Float, ~ (means not set) | Target KL value for adaptive KL control |
| `init_kl_coef` | 0.2 | Float | Initial KL penalty coefficient |
| `kl_horizon` | 10000 | Positive integer | Range for adaptive KL control |
| `add_token_level_kl` | false | true, false | Whether to add token-level KL penalty |
| `norm_mean_type` | None | "batch", "group", "running", None | Mean type for reward normalization |
| `norm_std_type` | None | "batch", "group", "running", None | Std type for reward normalization |

## Key Components of PPO

1. **Actor-Critic Architecture**: PPO requires an actor model (policy) and a critic model (value function). This is different from algorithms like GRPO and RLOO that don't require a critic model.

2. **Generalized Advantage Estimation (GAE)**: PPO uses GAE to compute advantage values, which helps reduce variance in policy gradient estimates while maintaining low bias.

3. **Clipped Surrogate Objective Function**: The core of PPO is implemented through a clipped surrogate objective function that constrains policy updates.

## KL Divergence Control

PPO provides two mechanisms to prevent the policy from deviating too far from the reference policy:

1. **KL Loss** (GRPO approach, optional):
   - `use_kl_loss`: Whether to use KL loss in the actor
   - `kl_loss_coef`: Coefficient for KL loss
   - `kl_penalty`: KL penalty options

2. **KL Penalty in Rewards**:
   - A KL penalty term can be added to the reward function to control policy updates

## Dual-clip PPO

Dual-Clip PPO introduces a method that applies a lower bound to the policy ratio when the advantage is less than zero, preventing it from exceeding the specified lower bound when multiplied by a large ratio.

## Usage Recommendations

1. **Batch Size**: Adjust `rollout_batch_size` and related parameters according to GPU memory
2. **KL Control**: It is recommended to enable `use_kl_loss` and set an appropriate `kl_loss_coef` value (e.g., 0.001)
3. **Clipping Parameters**: `pg_clip` is typically set to 0.2 and can be adjusted according to specific tasks
4. **Advantage Estimation**: `whiten_advantages` is typically set to true to improve training stability
5. **Loss Aggregation Mode**: Different `loss_agg_mode` options can be tried to optimize training effectiveness

## Reference Example

You can refer to the following configuration file to set up PPO training:

- `/examples/docs_examples/example_ppo.yaml`

This example shows how to configure and run PPO training.