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
  - [Multi-Teacher OPD](#️multi-teacher-opd多教师蒸馏)
    - [配置示例](#配置示例)
    - [核心机制](#核心机制)
  - [OPSD（On-Policy Self-Distillation）](#️opsdon-policy-self-distillation)
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
* **Worker 映射**：`student_train` → `actor_train`，`student_infer` → `actor_infer`，`teacher` 和/或 `reference` → `reference`（合并到统一的 `_reference_configs`）

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
| `teacher` 和/或 `reference` | `reference` / `references` | 计算 teacher log_probs（支持单个 WorkerConfig 或多 teacher Dict） |

**注意**：配置文件中使用 `student_train`、`student_infer`、`teacher` 名称，系统会自动映射。`reference` 可与 `teacher` 同时或单独配置——两者合并到统一的 `_reference_configs` 字典（`reference` → `"reference"`，单个 `teacher` → `"default"`，字典 `teacher` → 按字典键）。多 teacher 时 `teacher` 为 `Dict[str, WorkerConfig]`，内部归一化为 `self.references: Dict[str, Cluster]`。

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
| `teacher` 或 `reference` | 教师/参考模型配置（合并到 _reference_configs） | 必须配置 |

#### 混合模式

通过 `start_rlvr_pipeline.py` 脚本启动，需要手动配置 `use_opd=True`。

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `use_opd` | 启用混合模式 OPD（将 Teacher KL 添加到奖励中） | `false` |
| `teacher` 或 `reference` | 教师/参考模型配置（合并到 _reference_configs） | 必须配置 |

#### Multi-Teacher 模式参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `teacher` | Dict[str, WorkerConfig] 多教师配置 | — |
| `teacher.{name}.opd_kl_coef` | 该教师的 KL 系数 | `1.0` |
| `teacher.{name}.tag_included` | 该教师负责的 tag 列表，空表示全量 | `[]` |
| `tag_to_template` | 按 tag 选择不同的 chat template | `{}` |

#### OPSD 模式参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `opsd_mode` | 启用 OPSD（teacher prompt 注入参考解 y\*）。需配合 `is_pure_opd=True` 或 `use_opd=True` | `false` |
| `opsd_solution_key` | 数据集中参考解字段的列名 | `"reference_solution"` |
| `opsd_teacher_template` | Teacher prompt 格式化模板，占位符 `{problem}` 和 `{solution}` | 内置默认模板 |
| `global_template` | Chat 模板名称，用于 teacher prompt 格式化 | — |
| `sequence_length` | Batch 张量总长度。OPSD teacher prompt（problem + y\*）比 student prompt 更长，需设为大于 `prompt_length + response_length` 的值留出 buffer | `prompt_length + response_length` |
| `opsd_max_solution_length` | 参考解（y\*）的最大 token 长度（可选硬上限）。设置后超过此长度的 solution 会在构建 teacher prompt 前被截断。不设置则自动按 `sequence_length` 截断（动态测量模板 overhead） | `None` |


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

## ✨️Multi-Teacher OPD（多教师蒸馏）

### 概述

Multi-Teacher OPD 允许多个专长不同的教师模型同时指导一个学生模型，按数据领域（tag）将数据路由到对应的教师，避免无效计算并实现更精准的蒸馏。

```
┌──────────────────────────────────────────────────────────────────┐
│              Multi-Teacher OPD 数据流                              │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│   Student Infer rollout → batch (含 tag/domain 字段)              │
│          │                                                        │
│          ├── [math_dapo 数据] ──▶  Teacher-32B (数学专长)         │
│          │                          计算 ref_log_probs_teacher_32B │
│          │                                                        │
│          └── [KodCode 数据]  ──▶  Teacher-14B (代码专长)          │
│                                     计算 ref_log_probs_teacher_14B │
│          │                                                        │
│          ▼                                                        │
│   Compute Advantage:                                              │
│     对每条数据，只累加被路由到的 teacher 的 KL:                     │
│     advantage = -Σ(opd_kl_coef_i * KL_i) (仅路由到的 teacher)      │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### 配置示例

#### 多教师纯 OPD 模式

```yaml
is_pure_opd: true
global_template: qwen3

# 按 tag 选择不同的 chat template（可选）
tag_to_template:
  math_dapo: qwen3        # 数学数据使用 qwen3 template（带 thinking）
  KodCode: qwen3_nothink  # 代码数据使用 qwen3_nothink template

student_train:
  model_args:
    model_name_or_path: Qwen/Qwen3-8B
  data_args:
    file_name:
      - data/dapo_math_17k_simple_boxed.jsonl
      - data/code_KodCode_data.jsonl
    domain_interleave_probs:
      math_rule: 0.6
      code_rule: 0.4
  device_mapping: list(range(0,8))
  # ...

student_infer:
  model_args:
    model_name_or_path: Qwen/Qwen3-8B
  device_mapping: list(range(0,8))
  # ...

# teacher 配置为 Dict[str, WorkerConfig]
teacher:
  teacher_32B:
    model_args:
      model_name_or_path: Qwen/Qwen3-32B  # 数学专长教师
    opd_kl_coef: 1.0
    tag_included: [math_dapo]  # 只处理数学数据
    device_mapping: list(range(8,16))
    strategy_args:
      strategy_name: megatron_infer
      strategy_config:
        tensor_model_parallel_size: 2
        pipeline_model_parallel_size: 4

  teacher_14B:
    model_args:
      model_name_or_path: Qwen/Qwen3-14B  # 代码专长教师
    opd_kl_coef: 1.0
    tag_included: [KodCode]  # 只处理代码数据
    device_mapping: list(range(16,24))
    strategy_args:
      strategy_name: megatron_infer
      strategy_config:
        tensor_model_parallel_size: 2
        pipeline_model_parallel_size: 2

rewards:
  math_rule:
    worker_cls: roll.pipeline.rlvr.rewards.math_rule_reward_worker.MathRuleRewardWorker
    tag_included: [math_dapo]
  code_rule:
    worker_cls: roll.pipeline.rlvr.rewards.code_sandbox_reward_worker.CodeSandboxRewardWorker
    tag_included: [KodCode]
```

#### 混合路由配置（通用教师 + 专长教师）

```yaml
teacher:
  teacher_general:
    model_args:
      model_name_or_path: Qwen/Qwen3-72B
    opd_kl_coef: 0.3
    tag_included: []  # 空 = 负责所有 tag（通用教师）

  teacher_math_specialist:
    model_args:
      model_name_or_path: DeepSeek-Math-67B
    opd_kl_coef: 0.7
    tag_included: [math_dapo, aime]  # 仅负责数学
```

此配置下，数学数据会同时被 `teacher_general`（KL 系数 0.3）和 `teacher_math_specialist`（KL 系数 0.7）计算 KL，两者的加权 KL 都参与 advantage。非数学数据只有 `teacher_general` 参与。

### 核心机制

#### 1. Tag 路由

每条训练数据带有 `tag` 字段（如 `math_dapo`、`KodCode`）。每个 teacher 通过 `tag_included` 声明自己负责的 tag：

- `tag_included: [math_dapo]` — 只处理 tag 为 `math_dapo` 的数据
- `tag_included: []`（空列表）— 处理所有数据（通用教师）

路由发生在 ref_log_probs 计算阶段（pipeline 层），teacher 只对被路由到的数据做 forward，避免无效推理开销。

#### 2. Per-Teacher KL 系数

每个 teacher 有独立的 `opd_kl_coef`，用于控制该教师蒸馏信号的权重：

```
advantage = -Σ(opd_kl_coef_i * KL(student || teacher_i))
```

只有被路由到的 teacher 参与该样本的 KL 累加。

#### 3. 并行推理优化

当多个 teacher 使用不同的 GPU（`device_mapping` 不重叠）时，系统会自动使用多线程并行执行各 teacher 的 forward pass，减少总推理时间。

#### 4. tag_to_template

不同领域的数据可能需要不同的 chat template 编码方式。通过 `tag_to_template` 配置，可以为特定 tag 的数据使用不同的 tokenization template：

```yaml
tag_to_template:
  math_dapo: qwen3         # 带 thinking token
  KodCode: qwen3_nothink   # 不带 thinking token
```

未在 `tag_to_template` 中配置的 tag 会 fallback 到 `global_template`。

### 单 Teacher 向后兼容

单 teacher 配置（`teacher` 为 WorkerConfig 而非 Dict）与原有行为完全一致：

```yaml
# 以下配置与多 teacher 之前的版本行为完全一样
teacher:
  model_args:
    model_name_or_path: Qwen/Qwen3-32B
  device_mapping: list(range(0,16))
```

内部会自动归一化为 `{"default": WorkerConfig}`，循环只执行一次。

---

## ✨️OPSD（On-Policy Self-Distillation）

### 概述

OPSD 是 OPD 的扩展模式：Teacher 在评估 student 生成的 response 时，prompt 中包含**参考解 y\***（特权信息），而非仅原始问题。这让 teacher "知道答案"，从而对通向正确答案的推理路径赋予更高概率，提供更精准的蒸馏信号。

```
┌──────────────────────────────────────────────────────────────────┐
│                    OPSD 数据流                                     │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│   Student Infer:                                                 │
│     prompt = 原始问题（不含 y*）                                   │
│     → 生成 response                                              │
│                                                                   │
│   Teacher Forward（OPSD 变身）:                                    │
│     prompt = 原始问题 + y* + 指令（特权信息）                      │
│     + 同一 response tokens                                        │
│     → 计算 ref_log_probs                                         │
│     → 对齐回 student 布局                                         │
│                                                                   │
│   Advantage = -KL(student || teacher)                            │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

**与标准 OPD 的区别**：

| 特性 | 标准 OPD | OPSD |
|------|---------|------|
| Teacher prompt | 仅原始问题 | 原始问题 + 参考解 y\* |
| 特权信息 | 无 | 有（y* 注入 teacher prompt） |
| 蒸馏信号 | 通用行为对齐 | 引导推理路径走向正确答案 |
| 配置 | `is_pure_opd=True` 或 `use_opd=True` | 额外开启 `opsd_mode=True` |

### 配置参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `opsd_mode` | 启用 OPSD 模式（teacher prompt 注入 y\*）。需配合 `is_pure_opd=True` 或 `use_opd=True` | `false` |
| `opsd_solution_key` | 数据集中参考解字段的列名 | `"reference_solution"` |
| `opsd_teacher_template` | Teacher prompt 的格式化字符串模板，支持 `{problem}` 和 `{solution}` 占位符 | 内置默认模板 |
| `global_template` | Chat 模板名称（如 `qwen3`），用于 teacher prompt 格式化 | — |

### 数据要求

数据集 JSONL 中必须包含 `reference_solution` 字段（或 `opsd_solution_key` 指定的字段名），内容为**完整 CoT 推理过程**（而非仅最终答案）：

```json
{
    "id": "0",
    "prompt": "证明对于所有正整数 n，n^3 - n 能被 6 整除",
    "messages": "[{\"role\": \"user\", \"content\": \"证明对于所有正整数 n，n^3 - n 能被 6 整除\"}]",
    "ground_truth": "证明完成",
    "reference_solution": "n^3 - n = n(n-1)(n+1) = (n-1)n(n+1)...\n因此 n^3 - n 是三个连续整数之积，必被 6 整除。",
    "tag": "math_opsd"
}
```

### Teacher Prompt 构建

Teacher prompt 由 `opsd_teacher_template` 格式化后，通过 `global_template` 指定的 chat 模板包装：

```
opsd_teacher_template.format(problem=..., solution=...)
  → user_content（问题 + y* + 指令）
  → [{"role": "user", "content": user_content}]
  → get_chat_template(global_template, tokenizer)(..., add_generation_prompt=True)
  → teacher_prompt_text（含 chat 格式 token）
```

默认模板引导学生独立推理而非复制参考解。自定义模板时，用 `{problem}` 和 `{solution}` 作为占位符。

### 配置示例

```yaml
# OPSD 配置
is_pure_opd: true  # 由 start_onpolicy_distill_pipeline.py 自动设置
opsd_mode: true
opsd_solution_key: "reference_solution"
global_template: qwen3  # teacher prompt 使用 qwen3 chat 模板（含 thinking）
# 可选：solution 长度硬上限。不设置则自动按 sequence_length 截断。
opsd_max_solution_length: 2048

# OPSD teacher prompt（problem + y*）比 student prompt 更长。
# 设 sequence_length > prompt_length + response_length 留出 buffer。
prompt_length: 2048
response_length: 4096
sequence_length: 6656  # 2048 + 4096 + 512 buffer for teacher prompt

student_train:
  model_args:
    model_name_or_path: Qwen/Qwen3-8B
  data_args:
    file_name:
      - data/openthoughts_math_opsd.jsonl  # 必须含 reference_solution 字段
    domain_interleave_probs:
      math_rule: 1.0
  # ...

student_infer:
  model_args:
    model_name_or_path: Qwen/Qwen3-8B
  # ...

teacher:
  model_args:
    model_name_or_path: Qwen/Qwen3-8B  # 自蒸馏：teacher = student 初始权重
  # ...
```

### 兼容性

- **多 Teacher**：OPSD 兼容多 teacher 路由。每个 teacher 的 `ref_log_probs_{name}` 都会被自动对齐回 student 布局
- **Reference + Teacher**：`reference` 和 `teacher` 可同时配置，合并到统一的 `_reference_configs` 字典。每个条目的 KL 按 `opd_kl_coef` 加权后累加到 advantage
- **混合模式**：OPSD 可与 `use_opd=True` 混合使用，advantage = `rl_advantages - total_weighted_kld`

### 注意事项

- OPSD teacher prompt（problem + y\*）比 student prompt 更长。需设置 `sequence_length` 大于 `prompt_length + response_length` 留出 buffer。Solution 超限时自动截断：系统动态测量模板 overhead（构建空 solution 的 prompt），计算 solution 可用空间，截断 solution 文本——保留 OPSD 模板结构完整。可选设置 `opsd_max_solution_length` 作为 solution 长度硬上限
- `opsd_teacher_template` 中的字面花括号需用 `{{` 和 `}}` 转义（Python `.format()` 标准）
- OPSD 模式不支持同时配置 `reference` 和 `teacher`
- OPSD 同时支持 LoRA 和非 LoRA 分支：
  - **非 LoRA 分支**：teacher 是独立 cluster
  - **LoRA 分支**：teacher 是 actor 模型自身（adapter disabled），无需配置 `teacher`

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

# Teacher 或 reference 配置（自动映射到 reference）
# 两者可同时配置——合并到 _reference_configs
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

### Q5: Multi-Teacher 模式下，如果某条数据没有任何 teacher 被路由到怎么办？

该条数据的 `total_weighted_kld = 0`：
- 纯 OPD 模式下 `advantage = 0`（该条数据不产生梯度）
- 混合模式下 `advantage = rl_advantages`（仅 RL 信号，没有蒸馏信号）

### Q6: 多个 teacher 的 device_mapping 可以重叠吗？

可以，但不推荐：
- **不重叠**（推荐）：系统自动并行执行各 teacher 的 forward pass，显著减少推理时间
- **重叠**：系统会串行执行，不会冲突但总耗时为所有 teacher 之和

---

## 参考资料

- [On-Policy Distillation Blog](https://thinkingmachines.ai/blog/on-policy-distillation/)

---

*祝您实验愉快！*
