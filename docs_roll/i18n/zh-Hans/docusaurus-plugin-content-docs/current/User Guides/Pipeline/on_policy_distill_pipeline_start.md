# On-Policy Distillation 流水线

**目录**

- [On-Policy Distillation 流水线](#on-policy-distillation-流水线)
  - [概述](#️概述)
  - [核心原理](#️核心原理)
    - [什么是 On-Policy Distillation？](#什么是-on-policy-distillation)
    - [与 Off-Policy Distillation 的区别](#与-off-policy-distillation-的区别)
    - [与 RLVR 的区别](#与-rlvr-的区别)
    - [损失函数：Reverse KL](#损失函数reverse-kl)
  - [核心组件](#️核心组件)
    - [主模块](#主模块)
    - [配置文件](#配置文件)
    - [Worker 角色](#worker-角色)
  - [数据准备](#️数据准备)
    - [数据格式](#数据格式)
    - [纯 OPD 模式与混合模式的数据差异](#纯-opd-模式与混合模式的数据差异)
  - [运行流水线](#️运行流水线)
    - [方法1：使用Python启动脚本](#方法1使用python启动脚本)
    - [方法2：使用辅助Shell脚本](#方法2使用辅助shell脚本)
  - [配置详解](#️配置详解)
    - [核心配置参数](#核心配置参数)
  - [逐步示例](#️逐步示例)
    - [步骤1：配置设置](#步骤1配置设置)
    - [步骤2：准备环境和依赖](#步骤2准备环境和依赖)
    - [步骤3：启动流水线](#步骤3启动流水线)
    - [步骤4：监控](#步骤4监控)
    - [步骤5：输出和结果](#步骤5输出和结果)
  - [常见问题](#️常见问题)
  - [参考资料](#参考资料)

---

## ✨️概述

On-Policy Distillation（在线蒸馏，简称 OPD）是一种结合了**在线学习**和**知识蒸馏**的训练方法，通过让学生模型在自己生成的轨迹上学习教师模型的行为，实现高效的模型压缩和能力迁移。

此流水线提供以下核心优势：

* **高效的训练方式**：相比强化学习（RL），OPD 提供密集的奖励信号，可以实现更高效的训练
* **Teacher 即 Reward Model**：直接使用教师模型的 log probabilities 计算奖励，无需单独训练 Reward Model
* **在线学习优势**：学生模型在自己的状态分布上学习，避免分布偏移问题
* **完全复用 RLVR Pipeline**：基于 RLVR 架构实现，配置简单，易于使用
* **支持混合模式**：可以同时使用 OPD 奖励和外部奖励（如数学验证、代码执行等）

---

## ✨️核心原理

### 什么是 On-Policy Distillation？

On-Policy Distillation 的核心思想是：从**学生模型**采样轨迹，然后使用高性能的**教师模型**对轨迹中的**每个 token** 进行评分。

```
┌─────────────────────────────────────────────────────────────────┐
│                    On-Policy Distillation 流程                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   1. Sample Trajectories                                         │
│   ┌──────────┐     ┌──────────────────────────────────┐         │
│   │  Prompt  │ ──▶ │  Student Model (rollout)         │         │
│   └──────────┘     │  生成轨迹 + student_log_probs    │         │
│                    └──────────────────────────────────┘         │
│                              │                                   │
│                              ▼                                   │
│   2. Compute Teacher Log Probs                                   │
│                    ┌──────────────────────────────────┐         │
│                    │  Teacher Model (forward)         │         │
│                    │  计算 teacher_log_probs          │         │
│                    └──────────────────────────────────┘         │
│                              │                                   │
│                              ▼                                   │
│   3. Compute Advantage                                           │
│                    advantage = teacher_log_prob - student_log_prob│
│                              │                                   │
│                              ▼                                   │
│   4. Train with Importance Sampling                              │
│                    ┌──────────────────────────────────┐         │
│                    │  Student Model (train)           │         │
│                    │  使用 advantage 进行策略更新      │         │
│                    └──────────────────────────────────┘         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 与 Off-Policy Distillation 的区别

| 特性 | Off-Policy Distillation | On-Policy Distillation |
|------|--------------------|------------------------|
| **数据来源** | 预先生成的数据 | 学生模型实时生成的数据 |
| **状态分布** | 教师模型的状态分布 | 学生模型的状态分布 |
| **奖励信号** | 密集（每步都有） | 密集（每步都有） |
| **分布偏移** | 存在（学生可能进入教师未见过的状态） | 不存在（在自己的分布上学习） |
| **适用场景** | 大规模离线蒸馏 | 需要在线适应的场景 |

### 与 RLVR 的区别

| 特性 | RLVR | On-Policy Distillation |
|------|------|------------------------|
| **奖励来源** | 外部奖励模型（如数学验证、代码执行） | 教师模型的 log probabilities |
| **奖励密度** | 稀疏（通常只有最终答案有奖励） | 密集（每个 token 都有奖励） |
| **训练效率** | 相对较低 | 更高（密集信号） |
| **奖励可黑箱化** | 不可（教师模型无法被"欺骗"） | 可（低 KL = 高质量行为） |

### 损失函数：Reverse KL

On-Policy Distillation 使用 **Reverse KL** 作为核心损失函数：

$$\text{KL}(\pi_\theta || \pi_\text{teacher}) = \mathbb{E}_{x \sim \pi_\theta} \left[ \log \pi_\theta(x_{t+1} | x_{1..t}) - \log \pi_\text{teacher}(x_{t+1} | x_{1..t}) \right]$$

**优势**：
1. **Mode Seeking**：学习教师模型的特定行为，而不是在多个次优选项间分散
2. **不可欺骗**：低 KL 始终对应教师模型认可的高质量行为
3. **减少暴露偏差**：在学生自己的状态分布上学习

**实现**：
```python
# 伪代码
reverse_kl = sampled_logprobs - teacher_logprobs
advantages = -reverse_kl  # 负号：最小化 KL = 最大化 advantage
```

---

## ✨️核心组件

### 主模块

纯 OPD 模式复用现有的 Pipeline，根据 `pure_opd_pipeline_type` 配置选择：

- **RLVR 模式**（默认）：使用 `RLVRConfig` + `RLVRPipeline`
- **Agentic 模式**：使用 `AgenticConfig` + `AgenticPipeline`

主要区别在于：

* **奖励计算方式**：使用 Teacher Model 的 log probabilities 替代外部奖励模型
* **Advantage 计算**：`advantage = teacher_log_prob - student_log_prob`
* **Worker 映射**：`student_train` → `actor_train`，`student_infer` → `actor_infer`，`teacher` → `reference`

**源代码**：
- 启动脚本：`examples/start_onpolicy_distill_pipeline.py`
- Pipeline：`roll/pipeline/rlvr/rlvr_pipeline.py` 或 `roll/pipeline/agentic/agentic_pipeline.py`
- 配置处理：`roll/configs/base_config.py` 中的 `_handle_opd_mapping()` 方法

---

### 配置文件

ROLL 支持两种 On-Policy Distillation 模式，均基于 `RLVRConfig`（或 `AgenticConfig`）配置类实现：

#### 模式一：纯 OPD 模式 (`is_pure_opd=True`)

适用于**只需要蒸馏信号**的场景，奖励完全来自 Teacher Model 的 KL 散度。

**启动方式**：使用 `start_onpolicy_distill_pipeline.py` 脚本，该脚本会自动设置 `is_pure_opd=True`。

```yaml
# 配置 student_train, student_infer, teacher 三个角色
student_train:
  model_args:
    model_name_or_path: Qwen/Qwen3-8B
  # ... 训练配置

student_infer:
  model_args:
    model_name_or_path: Qwen/Qwen3-8B
  # ... 推理配置

teacher:
  model_args:
    model_name_or_path: Qwen/Qwen3-32B  # 可以与 student 不同
  # ... 推理配置
```

**内部映射**：
- `student_train` → `actor_train`
- `student_infer` → `actor_infer`
- `teacher` → `reference`

**计算公式**：
```
token_level_rewards = -reverse_kl  # 纯 KL 信号，无外部奖励
```

**支持的 Pipeline 类型**：通过 `pure_opd_pipeline_type` 配置：
- `"rlvr"`（默认）：使用 RLVRConfig + RLVRPipeline
- `"agentic"`：使用 AgenticConfig + AgenticPipeline


#### 模式二：混合模式 (`use_opd=True`)

适用于**同时使用外部奖励和蒸馏信号**的场景，例如数学推理任务中结合规则验证和 Teacher KL。

```yaml
# 使用标准 RLVRConfig 配置，启用 use_opd
use_opd: true
opd_kl_coef: 1.0  # OPD KL 系数，控制蒸馏信号权重

# 配置 teacher（会自动映射到 reference）
teacher:
  model_args:
    model_name_or_path: Qwen/Qwen3-32B

# actor_train 和 actor_infer 正常配置
actor_train:
  model_args:
    model_name_or_path: Qwen/Qwen3-8B
  # ...

actor_infer:
  model_args:
    model_name_or_path: Qwen/Qwen3-8B
  # ...
```

**计算公式**：
```
token_level_rewards = external_reward - opd_kl_coef * reverse_kl
```

#### 两种模式对比

| 特性 | 纯 OPD 模式 | 混合模式 |
|------|------------|---------|
| **配置类** | `RLVRConfig` / `AgenticConfig` | `RLVRConfig` / `AgenticConfig` |
| **标识参数** | `is_pure_opd=True`（脚本自动设置） | `use_opd=True`（用户配置） |
| **启动脚本** | `start_onpolicy_distill_pipeline.py` | `start_rlvr_pipeline.py` |
| **Worker 配置** | `student_train`, `student_infer`, `teacher` | `actor_train`, `actor_infer`, `teacher` |
| **奖励来源** | 仅 Teacher KL | 外部奖励 + Teacher KL |
| **Reward Workers** | 用于验证和统计 | 用于奖励计算 |
| **适用场景** | 纯蒸馏训练 | RL + 蒸馏联合训练 |

---

### Worker 角色

On-Policy Distillation 的 Worker 角色根据模式有所不同：

#### 纯 OPD 模式

配置三个角色，自动映射到内部 Worker：

| 配置名称 | 内部映射 | 职责 |
|----------|----------|------|
| `student_train` | `actor_train` | 训练学生模型，使用 Teacher KL 计算损失 |
| `student_infer` | `actor_infer` | 生成轨迹，计算 student log_probs |
| `teacher` | `reference` | 计算 teacher log_probs |

**注意**：配置文件中使用 `student_train`、`student_infer`、`teacher` 名称，系统会自动映射。

#### 混合模式

使用标准 RLVR Worker 名称：

| Worker | 职责 |
|--------|------|
| `actor_train` | 结合外部奖励和 Teacher KL 进行训练 |
| `actor_infer` | 生成轨迹，计算 student log_probs |
| `teacher` | 计算 teacher log_probs（自动映射到 reference） |
| Reward Workers | **参与训练**（计算外部奖励）|

---

## ✨️数据准备

On-Policy Distillation 的数据格式与 RLVR 完全相同，**不包含 response**（由模型生成），只需提供 prompt 和奖励相关字段。

### 数据格式

```json
{
    "id": "0",
    "source": "math_dataset",
    "difficulty": 0,
    "prompt": "解决以下数学问题：计算 3x + 5 = 14 中 x 的值",
    "messages": "[{\"role\": \"system\", \"content\": \"你是一个数学助手。\"}, {\"role\": \"user\", \"content\": \"解决以下数学问题：计算 3x + 5 = 14 中 x 的值\"}]",
    "tag": "math_rule"
}
```

### 纯 OPD 模式与混合模式的数据差异

| 字段 | 纯 OPD 模式 | 混合模式 |
|------|------------|---------|
| `ground_truth` | **需要**（用于验证和监控） | **需要**（用于奖励计算） |
| `test_cases` | **需要**（代码领域，用于验证和监控） | **需要**（代码领域，用于奖励计算） |
| `prompt` / `messages` | 需要 | 需要 |

**说明**：
- **纯 OPD 模式**：奖励由 Teacher Model 的 KL 散度提供，但 `ground_truth` 等字段用于验证阶段评估和训练过程监控
- **混合模式**：需要 `ground_truth` 或 `test_cases` 等字段，外部奖励是训练信号的一部分

---

## ✨️运行流水线

### 方法1：使用Python启动脚本

```bash
# 确保在项目根目录
python examples/start_onpolicy_distill_pipeline.py \
    --config_path examples/qwen3-8B-onpolicy-distill-megatron \
    --config_name onpolicy_distill_config
```

### 方法2：使用辅助Shell脚本

```bash
bash examples/qwen3-8B-onpolicy-distill-megatron/run_onpolicy_distill_pipeline.sh
```

---

## ✨️配置详解

### 核心配置参数

#### 纯 OPD 模式

通过 `start_onpolicy_distill_pipeline.py` 脚本启动，自动设置 `is_pure_opd=True`。

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `pure_opd_pipeline_type` | Pipeline 类型，可选 `"rlvr"` 或 `"agentic"` | `"rlvr"` |
| `student_train` | 学生模型训练配置（映射到 actor_train） | 必须配置 |
| `student_infer` | 学生模型推理配置（映射到 actor_infer） | 必须配置 |
| `teacher` | 教师模型配置（映射到 reference） | 必须配置 |

#### 混合模式

通过 `start_rlvr_pipeline.py` 脚本启动，需要手动配置 `use_opd=True`。

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `use_opd` | 启用混合模式 OPD（将 Teacher KL 添加到奖励中） | `false` |
| `opd_kl_coef` | OPD KL 系数，控制蒸馏信号相对于外部奖励的权重 | `1.0` |
| `teacher` | 教师模型配置（自动映射到 reference） | 必须配置 |


---

## ✨️逐步示例

### 步骤1：配置设置

* 文件：`examples/qwen3-8B-onpolicy-distill-megatron/onpolicy_distill_config.yaml`
* 关键部分包括 `exp_name`、`seed`、`output_dir`、模型路径、`student_train`、`student_infer`、`teacher` 和奖励配置。

* 特别注意这些配置部分：
  * **数据配置**：`student_train.data_args.file_name`
  * **模型配置**：`pretrain`（学生模型）和 Teacher 模型路径
  * **分布式策略**：每个 Worker 的 `strategy_args` 和 `device_mapping`
  * **奖励配置**：`rewards` 部分中配置 Reward Workers

### 步骤2：准备环境和依赖

* 确保安装了所有必要的依赖：

  ```bash
  pip install -r requirements.txt
  ```

* 验证配置中的所有模型路径是否可访问。

* 准备训练和验证数据集，确保它们符合数据格式要求（包含 `id`、`messages`/`prompt`、`tag`、`ground_truth` 等字段）。

### 步骤3：启动流水线

```bash
python examples/start_onpolicy_distill_pipeline.py \
       --config_path examples/qwen3-8B-onpolicy-distill-megatron \
       --config_name onpolicy_distill_config
```

### 步骤4：监控

* **控制台输出** – 观察 Hydra、Ray 和流水线日志
* **日志文件** – 检查 YAML 中指定的 `logging_dir`
* **TensorBoard**

  ```bash
  tensorboard --logdir <your_log_dir>
  ```

### 步骤5：输出和结果

* **训练模型** – 检查点保存在 `output_dir` 中
* **评估指标** – 记录在 TensorBoard 和控制台中
* **生成示例** – 流水线定期输出生成示例，以便您可以直观地评估模型改进。

---

## ✨️常见问题

### Q1: 混合模式如何配置？

使用 `RLVRConfig`（或 `AgenticConfig`），设置 `use_opd: true`：

```yaml
# 混合模式配置
use_opd: true
opd_kl_coef: 0.5  # 根据 reward 量级调整

# 必须配置外部奖励
rewards:
  math_rule:
    worker_cls: roll.pipeline.rlvr.rewards.math_rule_reward_worker.MathRuleRewardWorker
    tag_included: [math]

# Teacher 配置（自动映射到 reference）
teacher:
  model_args:
    model_name_or_path: Qwen/Qwen3-32B

# actor_train 和 actor_infer 正常配置
actor_train:
  model_args:
    model_name_or_path: Qwen/Qwen3-8B

actor_infer:
  model_args:
    model_name_or_path: Qwen/Qwen3-8B
```

### Q2: 纯 OPD 模式如何配置？

使用 `start_onpolicy_distill_pipeline.py` 脚本启动：

```yaml
# 配置三个角色
student_train:
  model_args:
    model_name_or_path: Qwen/Qwen3-8B
  # ... 训练配置

student_infer:
  model_args:
    model_name_or_path: Qwen/Qwen3-8B
  # ... 推理配置

teacher:
  model_args:
    model_name_or_path: Qwen/Qwen3-32B  # Teacher 可以与 Student 不同
  # ... 推理配置
```

启动命令：
```bash
python examples/start_onpolicy_distill_pipeline.py \
    --config_path examples/qwen3-8B-onpolicy-distill-megatron \
    --config_name onpolicy_distill_config
```

### Q3: 为什么需要配置 Reward Workers？

无论是纯 OPD 模式还是混合模式，都必须配置 Reward Workers：

1. **验证评估**：Validation 阶段需要 Reward Workers 评估模型性能
2. **训练监控**：观察奖励统计量，监控训练质量
3. **混合模式额外作用**：外部奖励是训练信号的一部分

### Q4: 两种模式如何选择？

- **纯 OPD 模式**：适合纯蒸馏训练，只需要 Teacher KL 信号，使用 `start_onpolicy_distill_pipeline.py`
- **混合模式**：适合 RL + 蒸馏联合训练，使用 `start_rlvr_pipeline.py` 并配置 `use_opd: true`

---

## 参考资料

- [On-Policy Distillation Blog](https://thinkingmachines.ai/blog/on-policy-distillation/)

---

*祝您实验愉快！*
