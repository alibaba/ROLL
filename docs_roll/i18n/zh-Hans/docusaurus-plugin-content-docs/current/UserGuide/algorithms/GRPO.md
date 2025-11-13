# Group Relative Policy Optimization (GRPO)

## 简介

Group Relative Policy Optimization (GRPO) 是一种强化学习算法，它通过消除对价值函数（critic）模型的需求来简化训练过程。GRPO通过以下方式工作：

1. **组采样**：对于给定的问题，模型生成多个可能的解决方案，形成一个"组"的输出。
2. **奖励分配**：每个解决方案都会被评估并分配一个基于其正确性或质量的奖励。
3. **基线计算**：组的平均奖励作为基线。
4. **策略更新**：模型通过将每个解决方案的奖励与组基线进行比较来更新其参数，强化优于平均的解决方案，抑制劣于平均的解决方案。

这种方法通过避免训练单独的价值估计模型来减少计算开销，使学习过程更加高效。

## GRPO 配置参数

在 ROLL 中，使用GRPO算法特有的配置参数如下(`roll.pipeline.rlvr.rlvr_config.RLVRConfig`)：

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

### 核心参数说明

- `rollout_batch_size`: 每个rollout_batch_size prompt的数量
- `num_return_sequences_in_group`: 每个prompt生成的response数量（组大小），每个pipeline step训练的总样本数是(rollout_batch_size * num_return_sequences_in_group)
- `prompt_length`: prompt的最大长度
- `response_length`: response的最大长度
- `adv_estimator`: 优势估计器类型，设置为 "grpo"
- `ppo_epochs`: 每个批次样本的优化轮数
- `use_kl_loss`: 是否使用 KL 散度损失
- `kl_loss_coef`: KL-loss系数
- `loss_agg_mode`: 损失聚合模式，默认值是"seq-mean-token-sum", Literal["token-mean", "seq-mean-token-sum", "seq-mean-token-mean", "seq-mean-token-sum-norm"]

### PPO 相关参数

以下参数是PPO里常见的参数，但在 GRPO 中同样适用：
- `whiten_advantages`: 是否对优势值进行白化处理
- `advantage_clip`: 优势值裁剪范围
- `dual_clip_loss`: 是否使用双重裁剪损失
- `reward_clip`: 奖励值裁剪范围
- `norm_mean_type`: 奖励归一化均值类型，可选值为 "batch", "group", "running", None，默认值为None
- `norm_std_type`: 奖励归一化标准差类型，可选值为 "batch", "group", "running", None，默认值为None
- `add_token_level_kl`: 是否添加 token 级别的 KL 惩罚

## GRPO 与 PPO 的区别

GRPO 与传统的 PPO 算法的主要区别在于：

1. **无需 Critic 模型**：GRPO 不需要训练单独的价值网络（critic）
2. **组采样**：GRPO 为每个提示生成多个完成（响应），而不是为每个输入评估一个 rollout
3. **相对奖励**：在每个组内，完成度会根据组内情况进行评分和归一化
4. **KL 损失**：GRPO 通过直接在损失函数中添加训练策略与参考策略之间的 KL 散度来进行正则化

## 参考示例

可以参考以下配置文件来设置 GRPO 训练：
- `./examples/docs_examples/example_grpo.yaml`
