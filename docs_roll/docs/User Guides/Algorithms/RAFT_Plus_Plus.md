# RAFT++ (Reward rAnked Fine-Tuning)

## Introduction

RAFT++ (Reward rAnked Fine-Tuning) is a ranking-based reinforcement learning algorithm that optimizes policies by comparing rewards of different responses. RAFT++ works as follows:

1. **Group Sampling**: For a given problem, the model generates multiple possible solutions, forming a "group" of outputs.
2. **Reward Ranking**: Each solution is evaluated and assigned a reward based on its correctness or quality, then ranked according to the rewards.
3. **Policy Update**: The model updates its parameters by comparing rewards of different solutions within the group, reinforcing strategies that obtain higher rewards.

## RAFT++ Configuration Parameters

In ROLL, the RAFT++ algorithm-specific configuration parameters are as follows (`roll.pipeline.rlvr.rlvr_config.RLVRConfig`):

```yaml
# RAFT++ core config
adv_estimator: "grpo"

# normalize
norm_mean_type: ~
norm_std_type: ~

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

# reward
add_token_level_kl: false
```

### Core Parameter Descriptions

- `adv_estimator`: Advantage estimator type, set to "grpo", which is the core configuration of RAFT++ algorithm
- `norm_mean_type`: Mean type for reward normalization: the options are "batch", "group", "running", or None; the default is None
- `norm_std_type`: Std type for reward normalization: the options are "batch", "group", "running", or None; the default is None
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
- `add_token_level_kl`: Whether to add token-level KL penalty, default value is false

## Reference Example

You can refer to the following configuration file to set up RAFT++ training:
- `./examples/docs_examples/example_raft_pp.yaml`

## References
[1] Xiong, W.; Yao, J.; Xu, Y.; Pang, B.; Wang, L.; Sahoo, D.; Li, J.; Jiang, N.; Zhang, T.; Xiong, C.; Dong, H. A Minimalist Approach to LLM Reasoning: From Rejection Sampling to Reinforce. arXiv April 15, 2025. https://doi.org/10.48550/arXiv.2504.11343.