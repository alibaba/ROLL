# StepWise Learning——GiGPO (Group-in-Group Policy Optimization)

## 简介

GiGPO (Group-in-Group Policy Optimization) 是一种用于LLM智能体训练的新型强化学习算法。它在保持基于组的RL的吸引属性（无评论家、低内存和稳定收敛）的同时，实现了对LLM智能体的细粒度信用分配。

GiGPO引入了一个两层结构来估计相对优势：
1. 在情节级别，GiGPO基于完整轨迹组计算宏观相对优势
2. 在步骤级别，GiGPO引入锚定状态分组机制，通过识别跨轨迹的重复环境状态来追溯构建步骤级组

这种分层结构有效地捕捉了全局轨迹质量和局部步骤效果，而无需依赖辅助模型或额外的rollout。

## GiGPO 配置参数

在 ROLL 中，GiGPO实现核心代码位于`roll/pipeline/agentic/utils.py`，使用GiGPO算法特有的配置参数如下(`roll.pipeline.agentic.agentic_config.AgenticConfig`)：

```yaml
# GiGPO core config
adv_estimator: "gigpo"
batch_adjust_mode: "copy"
step_reward_weight: 1.0
episode_reward_weight: 1.0
step_reward_gamma: 0.95

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
  group_size: 8
  tags: [FrozenLake]
  num_groups_partition: [128] # If not set, all env names divide nums equally. Under the same group, the env config and env seed (prompt) are equal in each generation

env_manager_cls: roll.pipeline.agentic.env_manager.step_env_manager.StepEnvManager
```

### 核心参数说明

- `adv_estimator`: 优势估计器类型，设置为 "gigpo"，这是GiGPO算法的核心配置
- `batch_adjust_mode`: 批次调整模式，可选值为 "copy", "delete", "auto"，默认值为 "copy"
- `step_reward_weight`: 步骤奖励权重，用于GiGPO算法，默认值为 1.0
- `episode_reward_weight`: 情节奖励权重，用于GiGPO算法，默认值为 1.0
- `step_reward_gamma`: 步骤奖励计算的折扣因子，默认值为 0.95
- `env_manager_cls`: 环境管理器类，GiGPO需要使用 `roll.pipeline.agentic.env_manager.step_env_manager.StepEnvManager`

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

可以参考以下配置文件来设置 GiGPO 训练：
- `./examples/docs_examples/example_gigpo.yaml`

## 参考文献
[1] Feng, L.; Xue, Z.; Liu, T.; An, B. Group-in-Group Policy Optimization for LLM Agent Training. arXiv 2025, 2505.10978.