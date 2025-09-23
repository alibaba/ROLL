# TrajWiseLearning——StarPO (State-Thinking-Actions-Reward Policy Optimization)

## 简介

StarPO (State-Thinking-Actions-Reward Policy Optimization) 是一种用于LLM智能体训练的强化学习算法。它通过将整个多轮交互轨迹（包括观察、推理轨迹、动作和反馈）视为一个连贯的单元来进行优化，而不是像传统方法那样独立处理每个动作。

StarPO的核心思想是轨迹级别的优化，它交替进行两个阶段：
1. **Rollout阶段**：生成推理-交互轨迹
2. **Update阶段**：基于完整轨迹进行模型优化

## StarPO 配置参数

在 ROLL 中，StarPO实现核心代码位于`roll/pipeline/agentic/utils.py`，使用StarPO算法特有的配置参数如下(`roll.pipeline.agentic.agentic_config.AgenticConfig`)：

```yaml
# StarPO core config
# StarPO related
adv_estimator: "reinforce"

# rollout_batch_size是轨迹的条数
rollout_batch_size: 1024
val_batch_size: 1024
sequence_length: 1024

advantage_clip: 0.2
ppo_epochs: 1

# pg_clip: 0.1
#dual_clip_loss: True
init_kl_coef: 0.0
whiten_advantages: true
entropy_loss_coef: 0
max_grad_norm: 1.0

reward_normalization:
  grouping: traj_group_id # 可以tags(env_type)/traj_group_id(group)/batch(rollout_batch)... group_by计算reward/adv
  method: mean # asym_clip / identity / mean_std / mean

train_env_manager:
  max_env_num_per_worker: 16
  num_env_groups: 128
  # under the same group, the env config and env seed are ensured to be equal
  group_size: 8 # grpo的grpo
  tags: [FrozenLake]
  num_groups_partition: [128] # If not set, all env names divide nums equally. Under the same group, the env config and env seed (prompt) are equal in each generation

env_manager_cls: roll.pipeline.agentic.env_manager.traj_env_manager.TrajEnvManager
```

### 核心参数说明

- `adv_estimator`: 优势估计器类型，设置为 "reinforce"，这是StarPO算法的核心配置
- `env_manager_cls`: 轨迹环境管理器类，StarPO需要使用 `roll.pipeline.agentic.env_manager.traj_env_manager.TrajEnvManager`

### PPO 相关参数

以下参数是PPO类算法通用的配置项：

- `rollout_batch_size`: 每个rollout批次的轨迹数量，默认值为 1024
- `val_batch_size`: 验证批次大小，默认值为 1024
- `sequence_length`: 序列最大长度，默认值为 1024
- `advantage_clip`: 优势值裁剪范围，默认值为 0.2
- `ppo_epochs`: 每个批次样本的优化轮数，默认值为 1
- `init_kl_coef`: KL惩罚的初始系数，默认值为 0.0
- `whiten_advantages`: 是否对优势值进行白化处理，默认值为 true
- `entropy_loss_coef`: 熵损失系数，默认值为 0
- `max_grad_norm`: 梯度裁剪的最大范数，默认值为 1.0

### 环境管理器参数

- `train_env_manager.max_env_num_per_worker`: 每个工作进程的最大环境数，默认值为 16
- `train_env_manager.num_env_groups`: 训练环境组数量，默认值为 128
- `train_env_manager.group_size`: 每组环境数量，默认值为 8
- `train_env_manager.tags`: 环境标签列表，默认值为 [FrozenLake]
- `train_env_manager.num_groups_partition`: 各环境类型的组数分配，默认值为 [128]

## 参考示例

可以参考以下配置文件来设置 StarPO 训练：
- `./examples/qwen2.5-0.5B-agentic/agent_val_frozen_lake.yaml`

## 参考文献
[1] Liu, T.; Feng, L.; An, B. StarPO: State-Regularized Policy Optimization for LLM Agent Training. arXiv 2025, 2504.20073.