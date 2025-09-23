# Lite PPO

## Introduction

LitePPO is a lightweight proximal policy optimization algorithm designed for efficient training of large language models. LitePPO improves training efficiency and stability through token-level loss computation and "within-group mean + batch standard deviation normalization" only. LitePPO works as follows:

1. **Token-level Loss Computation**: Computes losses at the token level to improve training granularity and efficiency.
2. **Group-level Reward Normalization**: Uses "within-group mean + batch standard deviation normalization" to stabilize the training process.
3. **Redundancy Removal Design**: Removes unnecessary components such as overlong filtering, preserving the original PPO objective function.

## LitePPO Configuration Parameters

In ROLL, the LitePPO algorithm-specific configuration parameters are as follows (`roll.pipeline.rlvr.rlvr_config.RLVRConfig`):

```yaml
# LitePPO core config
## normalization
norm_mean_type: group
norm_std_type: batch

## token-level loss 
token_level_loss: true

# ppo related, other parts are compatible with GRPO/PPO settings
rollout_batch_size: 512  # prompt
prompt_length: 2048
response_length: 4096

adv_estimator: "gae"
num_return_sequences_in_group: 1
ppo_epochs: 1
use_kl_loss: true
kl_loss_coef: 0.001
loss_agg_mode: "seq-mean-token-mean"


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
reward_shift: false
reward_scale: false
```

### Core Parameter Descriptions

- `norm_mean_type`: Mean type for reward normalization: the options are "batch", "group", "running", or None; the default is None
- `norm_std_type`: Std type for reward normalization: the options are "batch", "group", "running", or None; the default is None
- `token_level_loss`: Whether to enable token-level loss computation, default value is true

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
- `reward_shift`: Whether to only subtract mean in reward normalization, default value is false
- `reward_scale`: Whether to only divide by standard deviation in reward normalization, default value is false

## References
[1] Liu, Z.; Liu, J.; He, Y.; Wang, W.; Liu, J.; Pan, L.; Hu, X.; Xiong, S.; Huang, J.; Hu, J.; Huang, S.; Yang, S.; Wang, J.; Su, W.; Zheng, B. Part I: Tricks or Traps? A Deep Dive into RL for LLM Reasoning. arXiv August 11, 2025. https://doi.org/10.48550/arXiv.2508.08221.
