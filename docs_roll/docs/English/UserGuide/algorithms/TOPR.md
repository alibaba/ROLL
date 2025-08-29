# TOPR (Tapered Off-Policy REINFORCE)

## Introduction

TOPR (Tapered Off-Policy REINFORCE) is a stable and efficient reinforcement learning algorithm designed for large language models. TOPR improves training stability and efficiency by combining off-policy mechanisms with tapering techniques. TOPR works as follows:

1. **Off-policy Mechanism**: Utilizes historical data for training to improve sample efficiency.
2. **Tapering Technique**: Stabilizes the training process by gradually reducing dependence on old policies.
3. **Policy Update**: Updates policy parameters using a loss function that combines positive and negative samples.

## TOPR Configuration Parameters

In ROLL, the TOPR algorithm-specific configuration parameters are as follows (`roll.pipeline.rlvr.rlvr_config.RLVRConfig`):

```yaml
# TOPR core config
# TOPR
rl_loss_coef: 0.0
positive_loss_coef: x_1 # x_1 > 0.0
use_topr_neg_loss_coef: x_2 # x_2 > 0.0

# ppo related, other parts are compatible with GRPO/PPO settings
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

### Core Parameter Descriptions

- `rl_loss_coef`: Reinforcement learning loss term coefficient, default value is 0.0
- `positive_loss_coef`: Positive sample loss term coefficient, needs to be set to a value greater than 0.0
- `use_topr_neg_loss_coef`: Negative sample loss term coefficient, needs to be set to a value greater than 0.0

### PPO Related Parameters

The following parameters are common configuration items for PPO-class algorithms:

- `rollout_batch_size`: Number of prompts per rollout_batch_size, default value is 512
- `prompt_length`: Maximum length of prompts, default value is 2048
- `response_length`: Maximum length of responses, default value is 4096
- `adv_estimator`: Advantage estimator type, optional values are "gae", "reinforce", "grpo", default value is "gae"
- `num_return_sequences_in_group`: Number of responses generated per prompt (group size), default value is 1
- `ppo_epochs`: Number of optimization rounds per batch of samples, default value is 1
- `use_kl_loss`: Whether to use KL divergence loss, default value is true
- `kl_loss_coef`: KL-loss coefficient, default value is 0.001
- `loss_agg_mode`: Loss aggregation mode, default is "seq-mean-token-sum", optional values are "token-mean", "seq-mean-token-sum", "seq-mean-token-mean", "seq-mean-token-sum-norm"
- `whiten_advantages`: Whether to whiten advantage values, default value is true
- `advantage_clip`: Advantage value clipping range, default value is 2.0
- `reward_clip`: Reward value clipping range, default value is ~ (means not set)
- `dual_clip_loss`: Whether to use dual clipping loss, default value is true
- `lambd`: Lambda parameter in GAE estimator, used to trade off bias and variance, default value is 0.95
- `gamma`: Discount factor, default value is 1
- `pg_clip`: PPO clipping range, default value is 0.2
- `value_clip`: Value function clipping range, default value is ~ (means not set)
- `kl_penalty`: KL penalty options, optional values are "kl", "abs", "mse", "full", default value is "kl"
- `target_kl`: Target KL value for adaptive KL control, default value is ~ (means not set)
- `init_kl_coef`: Initial KL penalty coefficient, default value is 0.2
- `kl_horizon`: Range for adaptive KL control, default value is 10000
- `add_token_level_kl`: Whether to add token-level KL penalty, default value is false
- `norm_mean_type`: Mean type for reward normalization: the options are "batch", "group", "running", or None; the default is None
- `norm_std_type`: Std type for reward normalization: the options are "batch", "group", "running", or None; the default is None

## Reference Example

You can refer to the following configuration file to set up TOPR training:
- `./examples/docs_examples/example_topr.yaml`

## References
[1] Roux, N. L.; Bellemare, M. G.; Lebensold, J.; Bergeron, A.; Greaves, J.; Fréchette, A.; Pelletier, C.; Thibodeau-Laufer, E.; Toth, S.; Work, S. Tapered Off-Policy REINFORCE: Stable and Efficient Reinforcement Learning for LLMs. arXiv March 19, 2025. https://doi.org/10.48550/arXiv.2503.14286.
