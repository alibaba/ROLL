# Group Sequence Policy Optimization (GSPO)

## 简介

Group Sequence Policy Optimization (GSPO) 是阿里巴巴Qwen团队提出的一种强化学习算法，用于训练大语言模型[^1]。GSPO通过以下方式工作：

1. **序列级优化**：与GRPO等算法不同，GSPO在序列级别而非token级别进行重要性比率计算、奖励分配和优化。
2. **组采样**：对于给定的问题，模型生成多个可能的解决方案，形成一个"组"的输出。
3. **奖励分配**：每个解决方案都会被评估并分配一个基于其正确性或质量的奖励。
4. **基线计算**：组的平均奖励作为基线。
5. **策略更新**：模型通过将每个解决方案的奖励与组基线进行比较来更新其参数。

## GSPO 配置参数

在 ROLL 中，使用GSPO算法特有的配置参数如下：

```yaml
# GSPO related
adv_estimator: "grpo"
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
whiten_advantages: false
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

- `adv_estimator`: 优势估计器类型，设置为 "grpo"
- `importance_sampling`: 重要性采样方式，设置为 "seq" 表示序列级采样
- `rollout_batch_size`: 每个rollout_batch_size prompt的数量
- `num_return_sequences_in_group`: 每个prompt生成的response数量（组大小），每个pipeline step训练的总样本数是(rollout_batch_size * num_return_sequences_in_group)
- `prompt_length`: prompt的最大长度
- `response_length`: response的最大长度

### PPO 相关参数

以下参数是PPO里常见的参数，但在 GSPO 中同样适用：
- `ppo_epochs`: 每个批次样本的优化轮数
- `use_kl_loss`: 是否使用 KL 散度损失
- `kl_loss_coef`: KL-loss系数
- `loss_agg_mode`: 损失聚合模式，默认值是"seq-mean-token-sum", Literal["token-mean", "seq-mean-token-sum", "seq-mean-token-mean", "seq-mean-token-sum-norm"]
- `whiten_advantages`: 是否对优势值进行白化处理
- `advantage_clip`: 优势值裁剪范围
- `dual_clip_loss`: 是否使用双重裁剪损失
- `reward_clip`: 奖励值裁剪范围
- `norm_mean_type`: 奖励归一化均值类型，可选值为 "batch", "group", "running", None，默认值为None
- `norm_std_type`: 奖励归一化标准差类型，可选值为 "batch", "group", "running", None，默认值为None
- `add_token_level_kl`: 是否添加 token 级别的 KL 惩罚

## GSPO 与 GRPO 的区别

GSPO 与 GRPO 算法的主要区别：

| 对比维度 | GRPO (Group Relative Policy Optimization) | GSPO (Group Sequence Policy Optimization) |
|---------|------------------------------------------|------------------------------------------|
| **优化粒度** | Token级别优化 | 序列级别优化，与奖励计算的粒度保持一致 |
| **重要性比率计算** | 基于token级别的概率比计算，每个token独立计算重要性权重 | 基于序列级别的概率比计算，通过几何平均平滑处理，计算整个序列的联合概率比 |
| **专家混合模型(MoE)支持** | 在MoE模型中训练不稳定，需要额外技巧来维持专家激活的一致性 | 天然支持MoE模型训练，无需额外技巧，因为只关注序列级别的似然 |
| **方差控制** | 由于逐token计算重要性权重，容易引入高方差噪声 | 通过序列级重要性采样和长度归一化，显著降低了方差 |
| **裁剪机制** | 在token级别进行裁剪，可能导致不一致的梯度更新 | 在序列级别进行裁剪，提供更一致和稳定的梯度更新 |

## 参考示例

可以参考以下配置文件来设置 GSPO 训练：
- `./examples/docs_examples/example_gspo.yaml`

## 参考文献
[1]: Qwen Team. "Group Sequence Policy Optimization." arXiv preprint arXiv:2507.18071 (2025). https://arxiv.org/abs/2507.18071
