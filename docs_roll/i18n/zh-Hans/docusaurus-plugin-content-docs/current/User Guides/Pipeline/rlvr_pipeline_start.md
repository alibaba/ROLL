# RLVR 流水线

**目录**

- [RLVR 流水线](#rlvr-流水线)
  - [✨️概述](#️概述)
  - [✨️核心组件](#️核心组件)
    - [主模块 (`RLVRPipeline`)](#主模块-rlvrpipeline)
    - [配置文件 (`RLVRConfig`)](#配置文件-rlvrconfig)
      - [配置文件结构和组织](#配置文件结构和组织)
    - [奖励工作器](#奖励工作器)
  - [✨️数据准备](#️数据准备)
    - [数据格式](#数据格式)
      - [通用数据字段](#通用数据字段)
      - [领域特定字段](#领域特定字段)
  - [✨️运行流水线](#️运行流水线)
    - [方法1：使用Python启动脚本](#方法1使用python启动脚本)
    - [方法2：使用辅助Shell脚本](#方法2使用辅助shell脚本)
  - [✨️逐步示例](#️逐步示例)
    - [步骤1：配置设置](#步骤1配置设置)
    - [步骤2：准备环境和依赖](#步骤2准备环境和依赖)
    - [步骤3：启动流水线](#步骤3启动流水线)
    - [步骤4：监控](#步骤4监控)
    - [步骤5：输出和结果](#步骤5输出和结果)

---



## ✨️概述

此流水线提供以下核心优势：

* **多样化任务支持**：内置支持各种任务类型，包括数学推理、代码生成、LLM作为评判器评估和指令遵循，每种类型都配备了专门的奖励评估机制和灵活的扩展接口，以适应新任务类型。

* **多任务联合训练**：能够同时优化模型在数学、编程和通用推理等多个领域的能力，灵活控制每个领域的数据采样比例和奖励权重配置。

* **算法友好的强化学习框架**：提供丰富的强化学习策略选项（超过20种），包括但不限于奖励归一化、奖励裁剪、各种优势估计方法等。不限于单一算法实现，支持多种强化学习算法，如PPO、GRPO、Reinforce++、TOPR和RAFT++。

* **全面的性能监控**：细粒度的指标跟踪系统，同时监控组级别和批次级别的性能指标，为模型训练过程提供全面的可视化和分析能力。

* **高效的分布式计算**：利用[Ray](https://www.ray.io/)框架在大规模GPU集群上实现高效的分布式训练，显著提高训练速度和资源利用率。

---



## ✨️核心组件

### 主模块 (`RLVRPipeline`)

`RLVRPipeline`（位于`roll/pipeline/rlvr/rlvr_pipeline.py`）是整个强化学习过程的主要协调器。它管理完整的训练工作流，包括：

* 初始化和管理分布式工作器（actor、critic、reference和各种奖励工作器）。
* 协调数据收集和处理。
* 执行模型训练步骤（例如，actor和critic的PPO更新）。
* 处理模型同步和检查点保存。
* 验证集评估。
* 记录指标和实验跟踪。

**源代码**：`roll/pipeline/rlvr/rlvr_pipeline.py`

---

### 配置文件 (`RLVRConfig`)

`RLVRConfig`（定义于`roll/pipeline/rlvr/rlvr_config.py`）是一个基于Pydantic/dataclass的配置对象，用于指定运行rlvr流水线的所有参数。此配置系统设计灵活，支持通过YAML文件进行配置，并使用Hydra框架进行管理。

#### 配置文件结构和组织

配置文件（如`examples/qwen2.5-7B-rlvr_megatron/rlvr_config.yaml`）按功能模块组织，包含以下主要部分：

1. **实验基本设置**
   * `exp_name`：实验名称，用于标识特定的训练运行
   * `logging_dir`：保存日志文件的路径
   * `output_dir`：保存模型检查点和输出文件的路径

2. **训练控制参数**
   * `max_steps`：最大训练步数
   * `save_steps`：保存模型检查点的频率
   * `logging_steps`：记录训练指标的频率
   * `eval_steps`：执行验证评估的频率
   * `resume_from_checkpoint`：是否从检查点继续训练

3. **模型配置**
   * `pretrain`：Actor和Reference模型的预训练权重路径
   * `reward_pretrain`：Critic模型的预训练权重路径

4. **强化学习算法参数**
   * `ppo_epochs`：每批数据的PPO更新次数
   * `init_kl_coef`：KL散度的初始系数
   * `target_kl`：KL散度的目标值
   * `adv_estimator`：优势估计方法（例如，`gae`）
   * `gamma`：折扣因子
   * `lambd`：GAE lambda参数
   * `reward_normalize`：是否归一化奖励
   * `reward_clip`：奖励裁剪范围
   * `value_clip`：值裁剪范围
   * ...

5. **工作器配置**
   每个工作器（`actor_train`、`actor_infer`、`critic`、`reference`）配置包含：

   * **模型参数** (`model_args`)
     * `model_type`：模型类型（例如，`causal_lm`）
     * `dtype`：计算精度（例如，`bf16`、`fp16`）
     * ...
   * **训练参数** (`training_args`)
     * `learning_rate`：学习率
     * `per_device_train_batch_size`：每个设备的训练批次大小
     * `gradient_accumulation_steps`：梯度累积步数
     * `weight_decay`：权重衰减系数
     * `max_grad_norm`：梯度裁剪阈值
     * ...
   * **生成参数** (`generating_args`)
     * `max_new_tokens`：生成的新token最大数量
     * `top_p`：核采样参数
     * `temperature`：采样温度
     * `do_sample`：是否使用采样进行生成
     * ...
   * **分布式策略** (`strategy_args`)
     * `strategy_name`：要使用的分布式策略（例如，`megatron_train`、`vllm`、`sglang`、`hf_infer`）
     * 策略特定参数：例如，`tp_size`（张量并行大小）、`pp_size`（流水线并行大小）
     * `gpu_memory_utilization`：GPU内存利用率（vLLM特定）
   * **设备映射** (`device_mapping`)
     * 指定工作器应使用的GPU设备

6. **奖励设置**
   `rewards`部分包含不同领域的奖励工作器配置：

   * **数学** (`math_rule`)
     * `worker_cls`：工作器类名（例如，`MathRuleRewardWorker`）
     * `tag_included`：这些标签使用奖励领域进行计算
     * `model_args`：奖励模型参数
     * `world_size`：工作器数量

   * **代码** (`code_sandbox`)
     * 类似配置，但用于代码评估

   * **通用推理** (`llm_judge`)
     * 使用LLM作为评判器的配置

7. **验证和评估设置**
   `validation`部分配置验证数据集和评估方法：

   * `file_name`：验证数据集文件路径
   * `batch_size`：验证批次大小
   * `metrics`：要计算的评估指标

---

### 奖励工作器

rlvr流水线支持各种rlvr领域的奖励机制：

* **数学规则奖励** (`MathRuleRewardWorker`) – 评估数学推理的正确性和步骤。
* **代码沙盒奖励** (`CodeSandboxRewardWorker`) – 通过执行代码并验证其输出来评估代码生成。
* **LLM评判奖励** (`LLMJudgeRewardWorker`) – 使用另一个LLM作为评判器来评估生成答案的质量。

---



## ✨️数据准备

### 数据格式

rlvr流水线使用JSON格式的数据文件。不同领域需要特定字段：

#### 通用数据字段

所有领域都需要以下字段：
* `id`：数据点的唯一标识符 **(必需)**
* `messages` 或 `prompt`：输入提示，可以是消息列表（JSON字符串）或单个提示字符串 **(必需)**
* `tag`：用于更细粒度的分类（例如，`gsm8k`、`olympiads`等） **(必需)**
* `difficulty`：问题难度级别 **(可选)**

#### 领域特定字段

根据领域不同，数据点需要包含以下特定字段：

1. **数学** (`math_rule`)
   * `ground_truth`：正确答案或解题步骤 **(必需)**
2. **代码** (`code_sandbox`)
   * `test_cases`：用于验证代码正确性的测试用例 **(必需)**
   * `case_type`：测试用例类型（例如，`pytest`） **(必需)**
   * `test_case_function`：测试函数定义 **(可选)**
   * `ground_truth`：参考答案 **(可选)**
3. **通用推理** (`llm_judge`)
   * `ground_truth`：标准答案或参考响应 **(必需)**

示例数据格式（MATH）：
```json
{
    "id": "0",
    "source": "gsm8k",
    "difficulty": 0,
    "prompt": "解方程 3x + 5 = 14",
    "messages": "[{\"role\": \"system\", \"content\": \"你是一个擅长解决复杂数学问题的数学助手。\"}, {\"role\": \"user\", \"content\": \"解方程 3x + 5 = 14\"}]",
    "ground_truth": "204",
    "case_type": "",
    "test_case_function": "",
    "test_cases": "",
    "tag": "math_rule"
 }
```

示例数据格式（代码领域）：
```json
{
  "id": "5ea1ab",
  "source": "codeforeces",
  "difficulty": "0",
  "prompt": "你是一位专业的Python程序员。你将收到一个问题（问题描述）并生成一个正确的Python程序，该程序符合描述并能通过所有测试。\\n\\n### 问题：编写一个函数，接收一个不同整数的数组并返回所有可能的排列（任意顺序）。每个排列应表示为一个整数数组。该函数应能高效处理不同长度的数组。\\n\\n### 格式：你将使用以下起始代码编写问题的解决方案，并将代码包含在分隔符内。\\n```python\\ndef permute(nums):\\n```\\n\\n### 答案：（使用提供的格式和反引号）",
  "messages": "[{\"role\": \"user\", \"content\": \"你是一位专业的Python程序员。你将收到一个问题（问题描述）并生成一个正确的Python程序，该程序符合描述并能通过所有测试。 \\n\\n### 问题：编写一个函数，接收一个不同整数的数组并返回所有可能的排列（任意顺序）。每个排列应表示为一个整数数组。该函数应能高效处理不同长度的数组。\\n\\n### 格式：你将使用以下起始代码编写问题的解决方案，并将代码包含在分隔符内。\\n```python\\ndef permute(nums):\\n```\\n\\n### 答案：（使用提供的格式和反引号）\"}]",
  "ground_truth": "[\"def permute(nums):\\n    \\\"\\\"\\\"\\n    给定一个不同整数的数组，返回所有可能的排列。\\n    每个排列是一个整数数组。\\n    \\\"\\\"\\\"\\n    def backtrack(start, end):\\n        if start == end:\\n            permutations.append(nums[:])\\n        for i in range(start, end):\\n            nums[start], nums[i] = nums[i], nums[start]\\n            backtrack(start + 1, end)\\n            nums[start], nums[i] = nums[i], nums[start]\\n\\n    permutations = []\\n    backtrack(0, len(nums))\\n    return permutations\"]",
  "case_type": "pytest",
  "test_case_function": " ",
  "test_cases": "[{\"assert_code\": \"\\n\\n\\ndef test_permute_single_element():\\n    assert permute([1]) == [[1]]\\n\\ndef test_permute_two_elements():\\n    result = permute([1, 2])\\n    expected = [[1, 2], [2, 1]]\\n    assert sorted(result) == sorted(expected)\\n\\ndef test_permute_three_elements():\\n    result = permute([1, 2, 3])\\n    expected = [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]\\n    assert sorted(result) == sorted(expected)\\n\\ndef test_permute_four_elements():\\n    result = permute([1, 2, 3, 4])\\n    expected = [\\n        [1, 2, 3, 4], [1, 2, 4, 3], [1, 3, 2, 4], [1, 3, 4, 2], [1, 4, 2, 3], [1, 4, 3, 2],\\n        [2, 1, 3, 4], [2, 1, 4, 3], [2, 3, 1, 4], [2, 3, 4, 1], [2, 4, 1, 3], [2, 4, 3, 1],\\n        [3, 1, 2, 4], [3, 1, 4, 2], [3, 2, 1, 4], [3, 2, 4, 1], [3, 4, 1, 2], [3, 4, 2, 1],\\n        [4, 1, 2, 3], [4, 1, 3, 2], [4, 2, 1, 3], [4, 2, 3, 1], [4, 3, 1, 2], [4, 3, 2, 1]\\n    ]\\n    assert sorted(result) == sorted(expected)\"}]",
  "tag": "KodCode"
}
```

在配置文件中，您可以使用`domain_interleave_probs`设置不同领域的采样比例，例如：
```yaml
domain_interleave_probs:
  math_rule: 0.6
  code_sandbox: 0.3
  llm_judge: 0.1
```

---



## ✨️运行流水线



### 方法1：使用Python启动脚本

主要方法是使用`examples/start_rlvr_pipeline.py`脚本。此脚本使用Hydra加载和管理配置。

1. **选择或创建配置文件**  
   从示例YAML（例如，`examples/qwen2.5-7B-rlvr_megatron/rlvr_config.yaml`）开始或创建您自己的配置。

2. **执行Python启动脚本**

   ```bash
   # 确保您在ROLL (ScaleAligner)项目的根目录中
   # export PYTHONPATH=$(pwd):$PYTHONPATH
   
   python examples/start_rlvr_pipeline.py \
          --config_path examples/qwen2.5-7B-rlvr_megatron \
          --config_name rlvr_config
   ```

   * `--config_path` – 包含YAML配置的目录。
   * `--config_name` – 文件名（不带`.yaml`）。



### 方法2：使用辅助Shell脚本

`examples`目录通常包含包装Python启动器的shell脚本（例如，`start_ppo_pipeline_math_hz.sh`）。

示例结构：

```bash
#!/bin/bash
# 示例: examples/qwen2.5-7B-rlvr_megatron/run_rlvr_pipeline.sh

CONFIG_NAME="rlvr_config"                         # rlvr_config.yaml
CONFIG_PATH="examples/qwen2.5-7B-rlvr_megatron"

# 设置环境变量和其他配置

python examples/start_rlvr_pipeline.py \
       --config_path $CONFIG_PATH \
       --config_name $CONFIG_NAME \
       "$@"   # 传递任何附加参数
```

使用以下命令运行：

```bash
bash examples/qwen2.5-7B-rlvr_megatron/run_rlvr_pipeline.sh
```

---



## ✨️逐步示例



### 步骤1：配置设置

* 文件：`examples/qwen2.5-7B-rlvr_megatron/rlvr_config.yaml`  
  关键部分包括`exp_name`、`seed`、`output_dir`、模型路径、`actor_train`、`actor_infer`、`reference`、PPO参数和奖励配置。

* 特别注意这些配置部分：
  * 数据配置：`actor_train.data_args.file_name`和`domain_interleave_probs`
  * 模型配置：`pretrain`和`reward_pretrain`路径
  * 分布式策略：每个工作器的`strategy_args`和`device_mapping`
  * 奖励配置：`rewards`部分中不同领域的奖励工作器

### 步骤2：准备环境和依赖

* 确保安装了所有必要的依赖：

  ```bash
  pip install -r requirements.txt
  ```

* 验证配置中的所有模型路径是否可访问。

* 准备训练和验证数据集，确保它们符合上述数据格式要求。

### 步骤3：启动流水线

```bash
python examples/start_rlvr_pipeline.py \
       --config_path examples/qwen2.5-7B-rlvr_megatron_hz \
       --config_name ppo
```

### 步骤4：监控

* **控制台输出** – 观察Hydra、Ray和流水线日志。
* **日志文件** – 检查YAML中指定的`logging_dir`。
* **TensorBoard**

  ```bash
  tensorboard --logdir <your_log_dir>
  ```

### 步骤5：输出和结果

* **训练模型** – 检查点保存在`output_dir`中。
* **评估指标** – 记录在TensorBoard和控制台中。
* **生成示例** – 流水线定期输出生成示例，以便您可以直观地评估模型改进。

---

*祝您实验愉快！*