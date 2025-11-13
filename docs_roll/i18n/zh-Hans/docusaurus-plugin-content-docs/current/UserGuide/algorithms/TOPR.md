# TOPR (Tapered Off-Policy REINFORCE)

## 简介

TOPR (Tapered Off-Policy REINFORCE) 是一种为大语言模型设计的稳定且高效的强化学习算法。TOPR 通过结合 off-policy 机制和 tapering 技术来提高训练的稳定性和效率。TOPR 通过以下方式工作：

1. **Off-policy 机制**：利用历史数据进行训练，提高样本效率。
2. **Tapering 技术**：通过逐渐减少对旧策略的依赖来稳定训练过程。
3. **策略更新**：结合正负样本的损失函数来更新策略参数。

## TOPR 配置参数

在 ROLL 中，使用 TOPR 算法特有的配置参数如下(`roll.pipeline.rlvr.rlvr_config.RLVRConfig`)：

```yaml
# TOPR core config
# TOPR
rl_loss_coef: 0.0
positive_loss_coef: x_1 # x_1 > 0.0
use_topr_neg_loss_coef: x_2 # x_2 > 0.0

# ppo related，其他部分可以和GRPO/PPO等设置兼容
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
norm_mean_type: ~
norm_std_type: ~
```

### 核心参数说明

- `rl_loss_coef`: 强化学习损失项的系数，默认值为 0.0
- `positive_loss_coef`: 正样本损失项的系数，需要设置为大于 0.0 的值
- `use_topr_neg_loss_coef`: 负样本损失项的系数，需要设置为大于 0.0 的值

### PPO 相关参数

以下参数是PPO类算法通用的配置项：

- `rollout_batch_size`: 每个rollout_batch_size prompt的数量，默认值为 512
- `prompt_length`: prompt的最大长度，默认值为 2048
- `response_length`: response的最大长度，默认值为 4096
- `adv_estimator`: 优势估计器类型，可选值为 "gae", "reinforce", "grpo"，默认值为 "gae"
- `num_return_sequences_in_group`: 每个prompt生成的response数量（组大小），默认值为 1
- `ppo_epochs`: 每个批次样本的优化轮数，默认值为 1
- `use_kl_loss`: 是否使用 KL 散度损失，默认值为 true
- `kl_loss_coef`: KL-loss系数，默认值为 0.001
- `loss_agg_mode`: 损失聚合模式，默认值是"seq-mean-token-sum", 可选值为 "token-mean", "seq-mean-token-sum", "seq-mean-token-mean", "seq-mean-token-sum-norm"
- `whiten_advantages`: 是否对优势值进行白化处理，默认值为 true
- `advantage_clip`: 优势值裁剪范围，默认值为 2.0
- `reward_clip`: 奖励值裁剪范围，默认值为 ~ (表示不设置)
- `dual_clip_loss`: 是否使用双重裁剪损失，默认值为 true
- `lambd`: GAE 估计器中的 lambda 参数，用于在偏差和方差之间进行权衡，默认值为 0.95
- `gamma`: 折扣因子，默认值为 1
- `pg_clip`: PPO 裁剪范围，默认值为 0.2
- `value_clip`: 价值函数裁剪范围，默认值为 ~ (表示不设置)
- `kl_penalty`: KL 惩罚选项，可选值为 "kl", "abs", "mse", "full"，默认值为 "kl"
- `target_kl`: 自适应 KL 控制的目标 KL 值，默认值为 ~ (表示不设置)
- `init_kl_coef`: 初始 KL 惩罚系数，默认值为 0.2
- `kl_horizon`: 自适应 KL 控制的范围，默认值为 10000
- `add_token_level_kl`: 是否添加 token 级别的 KL 惩罚，默认值为 false
- `norm_mean_type`: 奖励归一化均值类型，可选值为 "batch", "group", "running", None，默认值为None
- `norm_std_type`: 奖励归一化标准差类型，可选值为 "batch", "group", "running", None，默认值为None

## 参考示例

可以参考以下配置文件来设置 TOPR 训练：
- `./examples/docs_examples/example_topr.yaml`

## 参考文献
[1] https://arxiv.org/abs/2503.14286