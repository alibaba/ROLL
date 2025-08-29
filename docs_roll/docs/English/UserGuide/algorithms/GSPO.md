# Group Sequence Policy Optimization (GSPO)

## Introduction

Group Sequence Policy Optimization (GSPO) is a reinforcement learning algorithm proposed by Alibaba's Qwen team for training large language models[^1]. GSPO works as follows:

1. **Sequence-Level Optimization**: Unlike algorithms such as GRPO, GSPO performs importance ratio calculation, reward assignment, and optimization at the sequence level rather than the token level.
2. **Group Sampling**: For a given problem, the model generates multiple possible solutions, forming a "group" of outputs.
3. **Reward Assignment**: Each solution is evaluated and assigned a reward based on its correctness or quality.
4. **Baseline Calculation**: The average reward of the group serves as the baseline.
5. **Policy Update**: The model updates its parameters by comparing each solution's reward to the group baseline.

## GSPO Configuration Parameters

In ROLL, the GSPO algorithm-specific configuration parameters are as follows:

```yaml
# GSPO related
adv_estimator: "reinforce"
importance_sampling: seq
rollout_batch_size: 64  # prompt
num_return_sequences_in_group: 8
prompt_length: 2048
response_length: 4096

# ppo related
ppo_epochs: 1
use_kl_loss: true
kl_loss_coef: 0.001
loss_agg_mode: "seq-mean-token-mean"

# advantage
whiten_advantages: true
advantage_clip: 2.0
dual_clip_loss: true
# clip
reward_clip: 10
# normalize
norm_mean_type: None
norm_std_type: None

# reward
add_token_level_kl: false
```

### Core Parameter Descriptions

- `adv_estimator`: Advantage estimator type, set to "reinforce"
- `importance_sampling`: Importance sampling method, set to "seq" for sequence-level sampling
- `rollout_batch_size`: Number of prompts per rollout_batch_size
- `num_return_sequences_in_group`: Number of responses generated per prompt (group size), the total number of samples trained per pipeline step is (rollout_batch_size * num_return_sequences_in_group)
- `prompt_length`: Maximum length of prompts
- `response_length`: Maximum length of responses

### PPO Related Parameters

The following parameters are common in PPO but also apply to GSPO:
- `ppo_epochs`: Number of optimization rounds per batch of samples
- `use_kl_loss`: Whether to use KL divergence loss
- `kl_loss_coef`: KL-loss coefficient
- `loss_agg_mode`: Loss aggregation mode, default is "seq-mean-token-sum", Literal["token-mean", "seq-mean-token-sum", "seq-mean-token-mean", "seq-mean-token-sum-norm"]
- `whiten_advantages`: Whether to whiten advantage values
- `advantage_clip`: Advantage value clipping range
- `dual_clip_loss`: Whether to use dual clipping loss
- `reward_clip`: Reward value clipping range
- `norm_mean_type`: Mean type for reward normalization: the options are "batch", "group", "running", or None; the default is None
- `norm_std_type`: Std type for reward normalization: the options are "batch", "group", "running", or None; the default is None
- `add_token_level_kl`: Whether to add token-level KL penalty

## Differences Between GSPO and GRPO

Main differences between GSPO and GRPO algorithms:

| Comparison Dimension | GRPO (Group Relative Policy Optimization) | GSPO (Group Sequence Policy Optimization) |
|---------------------|------------------------------------------|------------------------------------------|
| **Optimization Granularity** | Token-level optimization | Sequence-level optimization, consistent with reward calculation granularity |
| **Importance Ratio Calculation** | Based on token-level probability ratio calculation, each token independently calculates importance weights | Based on sequence-level probability ratio calculation, using geometric averaging for smoothing, calculating the joint probability ratio for the entire sequence |
| **Mixture of Experts (MoE) Support** | Unstable training in MoE models, requiring additional techniques to maintain expert activation consistency | Naturally supports MoE model training without additional techniques, as it only focuses on sequence-level likelihood |
| **Variance Control** | Due to per-token importance weight calculation, high variance noise is easily introduced | Significantly reduces variance through sequence-level importance sampling and length normalization |
| **Clipping Mechanism** | Clipping at the token level, potentially leading to inconsistent gradient updates | Clipping at the sequence level, providing more consistent and stable gradient updates |

## Reference Example

You can refer to the following configuration file to set up GSPO training:
- `./examples/docs_examples/example_gspo.yaml`

## References
[1]: Qwen Team. "Group Sequence Policy Optimization." arXiv preprint arXiv:2507.18071 (2025). https://arxiv.org/abs/2507.18071