# Proximal Policy Optimization (PPO)

## 简介

Proximal Policy Optimization (PPO) 是由 OpenAI 在 2017 年提出的一类策略梯度方法，用于强化学习。PPO 在简洁性、稳定性和性能之间取得了平衡，使其成为现代 RL 应用中最广泛使用的算法之一，包括大规模语言模型的微调。

传统的策略梯度方法（如 REINFORCE 或 Vanilla Policy Gradient）存在以下问题：

1. 高方差和样本效率低下
2. 由于策略更新过大而导致的不稳定性

PPO 通过使用裁剪的替代目标函数来解决这些问题，该函数可以避免过大的更新，而无需计算二阶导数。

## PPO 配置参数

在ROLL中，PPO 算法的配置参数如下(`roll.pipeline.rlvr.rlvr_config.RLVRConfig`)：

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

### PPO相关参数说明

| 参数                              | 默认值                  | 可选项                                                                                  | 说明                                 |
|---------------------------------|----------------------|--------------------------------------------------------------------------------------|------------------------------------|
| `rollout_batch_size`            | 512                  | 正整数                                                                                  | 每批次提示的数量                           |
| `prompt_length`                 | 2048                 | 正整数                                                                                  | 提示的最大长度                            |
| `response_length`               | 4096                 | 正整数                                                                                  | 响应的最大长度                            |
| `adv_estimator`                 | "gae"                | "gae", "reinforce", "grpo"                                                           | 优势估计器类型                            |
| `num_return_sequences_in_group` | 1                    | 正整数                                                                                  | 每个提示生成的响应数量                        |
| `ppo_epochs`                    | 1                    | 正整数                                                                                  | 每个批次样本的优化轮数                        |
| `use_kl_loss`                   | true                 | true, false                                                                          | 是否使用 KL 散度损失                       |
| `kl_loss_coef`                  | 0.001                | 浮点数                                                                                  | KL 散度损失系数                          |
| `loss_agg_mode`                 | "seq-mean-token-sum" | "token-mean", "seq-mean-token-sum", "seq-mean-token-mean", "seq-mean-token-sum-norm" | 损失聚合模式                             |
| `whiten_advantages`             | true                 | true, false                                                                          | 是否对优势值进行白化处理                       |
| `advantage_clip`                | 2.0                  | 浮点数, ~ (表示不设置)                                                                       | 优势值裁剪范围                            |
| `reward_clip`                   | ~                    | 浮点数, ~ (表示不设置)                                                                       | 奖励值裁剪范围                            |
| `dual_clip_loss`                | true                 | true, false                                                                          | 是否使用双重裁剪损失                         |
| `lambd`                         | 0.95                 | [0, 1] 区间内的浮点数                                                                       | GAE 估计器中的 lambda 参数，用于在偏差和方差之间进行权衡 |
| `gamma`                         | 1                    | [0, 1] 区间内的浮点数                                                                       | 折扣因子                               |
| `pg_clip`                       | 0.2                  | 浮点数                                                                                  | PPO 裁剪范围                           |
| `value_clip`                    | ~                    | 浮点数, ~ (表示不设置)                                                                       | 价值函数裁剪范围                           |
| `kl_penalty`                    | "kl"                 | "kl", "abs", "mse", "full"                                                           | KL 惩罚选项                            |
| `target_kl`                     | ~                    | 浮点数, ~ (表示不设置)                                                                       | 自适应 KL 控制的目标 KL 值                  |
| `init_kl_coef`                  | 0.2                  | 浮点数                                                                                  | 初始 KL 惩罚系数                         |
| `kl_horizon`                    | 10000                | 正整数                                                                                  | 自适应 KL 控制的范围                       |
| `add_token_level_kl`            | false                | true, false                                                                          | 是否添加 token 级别的 KL 惩罚               |
| `norm_mean_type`                | None                 | "batch", "group", "running", None                                                    | 奖励归一化中均值的类型                            |
| `norm_std_type`                 | None                 | "batch", "group", "running", None                                                    | 奖励归一化中标准差的类型                            |


## PPO 的关键组件

1. **Actor-Critic 架构**：PPO 需要一个 actor 模型（策略）和一个 critic 模型（价值函数）。这与不需要 critic 模型的 GRPO 和 RLOO 等算法不同。

2. **广义优势估计 (GAE)**：PPO 使用 GAE 来计算优势值，这有助于减少策略梯度估计中的方差，同时保持低偏差。

3. **裁剪替代目标函数**：PPO 的核心是通过裁剪的替代目标函数实现的，该函数限制了策略更新。

## KL 散度控制

PPO 提供了两种机制来防止策略偏离参考策略太远：

1. **KL 损失**(GRPO中的做法，可选)：
   - `use_kl_loss`: 是否在 actor 中使用 KL 损失
   - `kl_loss_coef`: KL 损失的系数
   - `kl_penalty`: KL 惩罚选项

2. **奖励中的 KL 惩罚**：
   - 可以在奖励函数中添加 KL 惩罚项来控制策略更新

## Dual-clip PPO

Dual-Clip PPO 通过在优势小于零时对策略比率应用下界来引入一种方法，当乘以一个大比率时，不会超过指定的下界。

## 使用建议

1. **批处理大小**：根据 GPU 内存调整 `rollout_batch_size` 和相关参数
2. **KL 控制**：建议启用 `use_kl_loss` 并设置合适的 `kl_loss_coef` 值（如 0.001）
3. **裁剪参数**：`pg_clip` 通常设置为 0.2，可以根据具体任务进行调整
4. **优势估计**：`whiten_advantages` 通常设置为 true 以提高训练稳定性
5. **损失聚合模式**：可以尝试不同的 `loss_agg_mode` 选项来优化训练效果

## 参考示例

可以参考以下配置文件来设置 PPO 训练：

- `/examples/docs_examples/example_ppo.yaml`

这个示例展示了如何配置和运行 PPO 训练。