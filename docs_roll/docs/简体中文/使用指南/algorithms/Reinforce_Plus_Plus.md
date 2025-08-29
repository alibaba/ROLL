# Reinforce++

## 简介

Reinforce++ 是一种基于策略梯度的强化学习算法，它是经典 REINFORCE 算法的增强版本。Reinforce++ 通过以下方式工作：

1. **组采样**：对于给定的问题，模型生成多个可能的解决方案，形成一个"组"的输出。
2. **奖励计算**：每个解决方案都会被评估并分配一个基于其正确性或质量的奖励。
3. **策略更新**：模型根据奖励信号和生成的序列来更新其参数，强化获得更高奖励的策略。

## Reinforce++ 配置参数

在 ROLL 中，使用 Reinforce++ 算法特有的配置参数如下(`roll.pipeline.rlvr.rlvr_config.RLVRConfig`)：

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

# ppo related，其他部分可以和GRPO/PPO等设置兼容
rollout_batch_size: 64  # prompt
num_return_sequences_in_group: 8
prompt_length: 2048
response_length: 4096
ppo_epochs: 1
use_kl_loss: true
kl_loss_coef: 0.001
loss_agg_mode: "seq-mean-token-sum"

# advantage
advantage_clip: 2.0
dual_clip_loss: true
# clip
reward_clip: 10

```

### 核心参数说明

- `adv_estimator`: 优势估计器类型，设置为 "reinforce"，这是 Reinforce++ 算法的核心配置
- `norm_mean_type`: 奖励归一化均值类型，可选值为 "batch", "group", "running", None，默认值为None
- `norm_std_type`: 奖励归一化标准差类型，可选值为 "batch", "group", "running", None，默认值为None
- `add_token_level_kl`: 是否添加 token 级别的 KL 惩罚，默认值为 false
- `whiten_advantages`: 是否对优势值进行白化处理，默认值为 false

### PPO 相关参数

以下参数是PPO类算法通用的配置项：

- `rollout_batch_size`: 每个rollout_batch_size prompt的数量，默认值为 64
- `num_return_sequences_in_group`: 每个prompt生成的response数量（组大小），每个pipeline step训练的总样本数是(rollout_batch_size * num_return_sequences_in_group)，默认值为 8
- `prompt_length`: prompt的最大长度，默认值为 2048
- `response_length`: response的最大长度，默认值为 4096
- `ppo_epochs`: 每个批次样本的优化轮数，默认值为 1
- `use_kl_loss`: 是否使用 KL 散度损失，默认值为 true
- `kl_loss_coef`: KL-loss系数，默认值为 0.001
- `loss_agg_mode`: 损失聚合模式，默认值是"seq-mean-token-sum", 可选值为 "token-mean", "seq-mean-token-sum", "seq-mean-token-mean", "seq-mean-token-sum-norm"
- `advantage_clip`: 优势值裁剪范围，默认值为 2.0
- `dual_clip_loss`: 是否使用双重裁剪损失，默认值为 true
- `reward_clip`: 奖励值裁剪范围，默认值为 10

## 参考示例

可以参考以下配置文件来设置 Reinforce++ 训练：
- `./examples/docs_examples/example_reinforce_pp.yaml`

## 参考文献
[1] https://arxiv.org/abs/2504.11343