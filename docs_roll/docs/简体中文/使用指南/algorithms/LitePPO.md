# Lite PPO

## 简介

LitePPO是一种轻量级的近端策略优化算法，专为大语言模型的高效训练而设计。LitePPO 仅通过 token 级别的损失计算和"组内均值+批标准差归一化"来提高训练效率和稳定性。LitePPO 通过以下方式工作：

1. **Token 级别损失计算**：在 token 级别计算损失，提高训练的细粒度和效率。
2. **组级奖励归一化**：使用"组内均值+批标准差归一化"来稳定训练过程。
3. 去冗余设计：移除overlong filtering等非必要组件，保留PPO原始目标函数。

## LitePPO 配置参数

在 ROLL 中，使用 LitePPO 算法特有的配置参数如下(`roll.pipeline.rlvr.rlvr_config.RLVRConfig`)：

```yaml
# LitePPO core config
## normalization
norm_mean_type: group
norm_std_type: batch

## token-level loss 
token_level_loss: true

# ppo related，其他部分可以和GRPO/PPO等设置兼容
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
reward_shift: false
reward_scale: false
```

### 核心参数说明

- `norm_mean_type`: 奖励归一化均值类型，可选值为 "batch", "group", "running", None，默认值为None
- `norm_std_type`: 奖励归一化标准差类型，可选值为 "batch", "group", "running", None，默认值为None
- `token_level_loss`: 是否启用 token 级别的损失计算，默认值为 true

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
- `reward_shift`: 是否在奖励归一化中仅减去均值，默认值为 false
- `reward_scale`: 是否在奖励归一化中仅除以标准差，默认值为 false

## 参考文献
[1] Liu, Z.; Liu, J.; He, Y.; Wang, W.; Liu, J.; Pan, L.; Hu, X.; Xiong, S.; Huang, J.; Hu, J.; Huang, S.; Yang, S.; Wang, J.; Su, W.; Zheng, B. Part I: Tricks or Traps? A Deep Dive into RL for LLM Reasoning. arXiv August 11, 2025. https://doi.org/10.48550/arXiv.2508.08221.
