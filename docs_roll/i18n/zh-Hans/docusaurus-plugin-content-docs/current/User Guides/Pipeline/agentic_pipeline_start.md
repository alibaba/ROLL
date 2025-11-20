# Agentic Pipeline

**目录**

- [Agentic Pipeline](#agentic-pipeline)
  - [✨️ 概述](#️-概述)
  - [✨️ 核心组件](#️-核心组件)
    - [主模块（`AgenticPipeline`）](#主模块agenticpipeline)
    - [配置文件（`AgenticConfig`）](#配置文件agenticconfig)
      - [配置文件结构与组织](#配置文件结构与组织)
  - [✨️ 环境准备](#️-环境准备)
    - [环境类型](#环境类型)
    - [环境配置](#环境配置)
  - [✨️ 运行Pipeline](#️-运行pipeline)
    - [方法 1：使用 Python 启动脚本](#方法-1使用-python-启动脚本)
    - [方法 2：使用辅助 Shell 脚本](#方法-2使用辅助-shell-脚本)
  - [✨️ 逐步示例](#️-逐步示例)
    - [步骤 1：配置设置](#步骤-1配置设置)
    - [步骤 2：准备环境与依赖](#步骤-2准备环境与依赖)
    - [步骤 3：启动Pipeline](#步骤-3启动pipeline)
    - [步骤 4：监控](#步骤-4监控)
    - [步骤 5：输出与结果](#步骤-5输出与结果)

---

## ✨️ 概述

Agentic Pipeline 是ROLL提供的智能体训练核心Pipeline，支持多种算法如PPO、GRPO等等。它提供以下核心优势：
* **gym-like环境定义**: 支持多种环境类型，包括 FrozenLake、Sokoban 等，可以轻松按gym-like接口扩展自定义环境。
* **丰富的学习粒度**: 支持TrajectoryWise形式(StarPO)和StepWise(GiGPO)训练形式
* **环境粒度的异步并行rollout**: 各环境独立采样轨迹，提高采样效率
* **异步训练**: rollout/training解耦，支持异步训练
* **多轮交互支持本地调试**: 多轮交互rollout支持本地调试，提高多轮交互业务开发效率
* **灵活的策略配置**：支持多种分布式训练策略，如 Megatron、DeepSpeed、vLLM 等，可以根据硬件资源进行灵活配置。
---

## ✨️ 核心组件

### 主模块（`AgenticPipeline`）

`AgenticPipeline`（位于 `roll/pipeline/agentic/agentic_pipeline.py`）是整个智能体训练的主流程。它管理完整的训练工作流，包括：

* 初始化并管理分布式工作进程（Actor、Critic、Reference 等工作进程）。
* 协调环境交互和数据收集。
* 执行模型训练步骤。
* 处理检查点保存。
* 记录指标和实验跟踪。

**源码**：`roll/pipeline/agentic/agentic_pipeline.py`

---

### 配置文件（`AgenticConfig`）

`AgenticConfig`（定义于 `roll/pipeline/agentic/agentic_config.py`）是一个基于 Pydantic/dataclass 的配置对象，用于指定运行 AgenticPipeline 的全部参数。该配置系统支持通过 YAML 文件配置，并使用 Hydra 框架进行管理。
配置系统描述参见[config_system](../Configuration/config_system.md)
#### 配置文件结构与组织

配置文件（如 `examples/qwen2.5-0.5B-agentic/agent_val_frozen_lake.yaml`）按功能模块组织，主要包含以下部分：

1. **实验基本设置**
   * `exp_name`：实验名称，用于标识一次具体训练任务
   * `seed`：随机种子，确保实验可复现
   * `logging_dir`：日志文件保存路径
   * `output_dir`：模型检查点和输出文件保存路径
   * `render_save_dir`：渲染帧保存路径（用于可视化环境）

2. **训练控制参数**
   * `max_steps`：最大训练步数
   * `save_steps`：保存模型检查点的频率
   * `logging_steps`：记录训练指标的频率
   * `eval_steps`：执行验证评估的频率
   * `resume_from_checkpoint`：是否从检查点继续训练。若想继续训练，请设为其路径；否则设为 `False`。

3. **模型配置**
   * `pretrain`：预训练模型路径
   * `reward_pretrain`：奖励模型预训练权重路径

4. **算法参数**
   * `adv_estimator`：优势估计器类型（如 `gae`、`grpo`、`reinforce`）
   * `ppo_epochs`：每个样本批次的优化轮数
   * `gamma`：折扣因子，用于计算回报
   * `lambd`：GAE 中的 lambda 参数
   * `pg_clip`：PPO 策略梯度损失的裁剪范围
   * `init_kl_coef`：KL 惩罚的初始系数
   * `target_kl`：自适应 KL 控制的目标 KL 值
   * `whiten_advantages`：是否对优势进行白化处理
   * `entropy_loss_coef`：熵损失的系数

5. **工作进程配置**
   每个工作进程（`actor_train`、`actor_infer`、`critic`、`reference`）配置包含：

   * **模型参数**（`model_args`）
     * `model_type`：模型类型（如 `causal_lm`）
     * `dtype`：计算精度（如 `bf16`、`fp16`）
     * `attn_implementation`：注意力实现方式（如 `fa2`）
     * `disable_gradient_checkpointing`：是否禁用梯度检查点
   * **训练参数**（`training_args`）
     * `learning_rate`：学习率
     * `per_device_train_batch_size`：每个设备的训练批次大小
     * `gradient_accumulation_steps`：梯度累积步数
     * `weight_decay`：权重衰减系数
     * `warmup_steps`：学习率预热步数
     * `lr_scheduler_type`：学习率调度器类型
   * **生成参数**（`generating_args`）
     * `max_new_tokens`：生成的最大新 token 数
     * `top_p`： nucleus sampling 参数
     * `temperature`：温度参数
     * `num_return_sequences`：返回序列数
   * **分布式策略**（`strategy_args`）
     * `strategy_name`：使用的分布式策略（如 `megatron_train`、`vllm`、`hf_infer`）
     * 策略特定参数：如 `tp_size`（张量并行规模）、`pp_size`（Pipeline并行规模）
     * `gpu_memory_utilization`：GPU 内存利用率（特定于 vLLM）
   * **设备映射**（`device_mapping`）
     * 指定该工作进程应使用哪些 GPU 设备

6. **环境管理器配置**
   * `train_env_manager`：训练环境管理器配置
   * `val_env_manager`：验证环境管理器配置
   * 环境相关参数：
     * `num_env_groups`：环境组数量
     * `group_size`：每组环境数量
     * `tags`：环境标签列表
     * `num_groups_partition`：各环境类型的组数分配
     * `max_env_num_per_worker`：每个工作进程的最大环境数

---

## ✨️ 环境准备

### 环境类型

Agentic Pipeline 支持多种环境类型，包括但不限于：

* **FrozenLake**：经典的强化学习环境，智能体需要在冰面上找到通往目标的路径。
* **Sokoban**：推箱子游戏环境，智能体需要将箱子推到指定位置。
* **WebShop**：模拟在线购物环境，智能体需要根据用户需求找到合适的商品。
* more env support...

### 环境配置

在配置文件中，通过 `custom_envs` 字段定义自定义环境。每个环境配置包含：

* `env_type`：环境类型
* `env_config`：环境具体配置参数
* `max_tokens_per_step`：每步最大 token 数

---

## ✨️ 运行Pipeline

### 方法 1：使用 Python 启动脚本

主要方法是使用 `examples/start_agentic_pipeline.py` 脚本。该脚本利用 Hydra 加载并管理配置。

1. **选择或创建配置文件**  
   从示例 YAML（如 `examples/qwen2.5-0.5B-agentic/agent_val_frozen_lake.yaml`）开始，或创建自己的配置。

2. **执行 Python 启动脚本**

   ```bash
   # 确保你在 ROLL 项目根目录
   # export PYTHONPATH=$(pwd):$PYTHONPATH
   
   python examples/start_agentic_pipeline.py \
          --config_path examples/qwen2.5-0.5B-agentic \
          --config_name agent_val_frozen_lake
   ```

   * `--config_path` – 包含 YAML 配置的目录。
   * `--config_name` – 文件名（不含 `.yaml`）。

### 方法 2：使用辅助 Shell 脚本

`examples` 目录通常包含包装了 Python 启动器的 shell 脚本。

示例结构：

```bash
#!/bin/bash
# 示例：examples/qwen2.5-0.5B-agentic/run_agentic_pipeline_frozen_lake.sh

CONFIG_PATH=$(basename $(dirname $0))
python examples/start_agentic_pipeline.py \
       --config_path $CONFIG_PATH \
       --config_name agent_val_frozen_lake
```

运行方式：

```bash
bash examples/qwen2.5-0.5B-agentic/run_agentic_pipeline_frozen_lake.sh
```

---

## ✨️ 逐步示例

### 步骤 1：配置设置

* 文件：`examples/qwen2.5-0.5B-agentic/agent_val_frozen_lake.yaml`  
  关键部分包括 `exp_name`、`seed`、`output_dir`、模型路径、各工作进程配置。

* 特别注意这些配置段：
  * 模型配置：`pretrain` 路径
  * 算法参数：`adv_estimator`、`ppo_epochs` 等
  * 分布式策略：每个工作进程的 `strategy_args` 和 `device_mapping`
  * 环境配置：`train_env_manager` 和 `val_env_manager`

### 步骤 2：准备环境与依赖

* 确保已安装所有必要依赖，建议从[镜像启动](../../Getting%20Started/Installation/installation.md)：

  ```bash
  pip install -r requirements.txt
  ```

* 确认配置中所有模型路径均可访问。

* 准备训练环境，确保支持所选的环境类型。

### 步骤 3：启动Pipeline

```bash
python examples/start_agentic_pipeline.py \
       --config_path examples/qwen2.5-0.5B-agentic \
       --config_name agent_val_frozen_lake
```

### 步骤 4：监控

* **控制台输出** – 观察 Hydra、Ray 和Pipeline日志。
* **日志文件** – 检查 YAML 中指定的 `logging_dir`。
* **TensorBoard**

  ```bash
  tensorboard --logdir <your_log_dir>
  ```

### 步骤 5：输出与结果

* **已训练模型** – 检查点保存在 `checkpoint_config`中，具体参考文档[checkpoint_and_resume](../Advanced%20Features/checkpoint_and_resume.md)。
* **评估指标** – 记录在 TensorBoard 和终端中。
* **渲染帧** – 如果配置了 `render_save_dir`，会在该目录保存环境渲染帧，方便可视化观察交互过程。
