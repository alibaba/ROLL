# DPOPipeline Quick Start

**目录**

- [DPO Pipeline](#DPO Pipeline)
  - [✨️ 概述](#️-概述)
  - [✨️ 核心组件](#️-核心组件)
    - [主模块（`DPOPipeline`）](#主模块DPOPipeline)
    - [配置文件（`DPOConfig`）](#配置文件DPOConfig)
      - [配置文件结构与组织](#配置文件结构与组织)
  - [✨️ 数据准备](#️-数据准备)
    - [数据格式](#数据格式)
      - [通用数据字段](#通用数据字段)
  - [✨️ 运行Pipeline](#️-运行Pipeline)
    - [方法 1：使用 Python 启动脚本](#方法-1使用-python-启动脚本)
    - [方法 2：使用辅助 Shell 脚本](#方法-2使用辅助-shell-脚本)
  - [✨️ 逐步示例](#️-逐步示例)
    - [步骤 1：配置设置](#步骤-1配置设置)
    - [步骤 2：准备环境与依赖](#步骤-2准备环境与依赖)
    - [步骤 3：启动Pipeline](#步骤-3启动Pipeline)
    - [步骤 4：监控](#步骤-4监控)
    - [步骤 5：输出与结果](#步骤-5输出与结果)

---

## ✨️ 概述

本Pipeline提供以下核心优势：

* **多种DPO损失**：支持用不同DPO损失训练模型，并可通过相应参数进行更精细的配置。

* **全面的性能监控**：细粒度度量跟踪系统，监控性能指标，为模型训练过程提供全面的可视化和分析能力。

* **高效分布式计算**：利用 [Ray](https://www.ray.io/) 框架，在大型 GPU 集群上实现高效的分布式训练，显著提升训练速度和资源利用率。

---

## ✨️ 核心组件

### 主模块（`DPOPipeline`）

`DPOPipeline`（位于 `roll/pipeline/dpo/dpo_pipeline.py`）是整个DPO训练的主流程。它管理完整的训练工作流，包括：

* 初始化并管理分布式工作进程（Actor 和 Reference 工作进程）。
* 协调数据收集与处理。
* 执行模型训练步骤。
* 处理检查点保存。
* 记录指标和实验跟踪。

**源码**：`roll/pipeline/dpo/dpo_pipeline.py`

---

### 配置文件（`DPOConfig`）

`DPOConfig`（定义于 `roll/pipeline/dpo/dpo_config.py`）是一个基于 Pydantic/dataclass 的配置对象，用于指定运行DPOPipeline的全部参数。该配置系统支持通过 YAML 文件配置，并使用 Hydra 框架进行管理。

#### 配置文件结构与组织

配置文件（如 `examples/qwen2.5-3B-dpo_megatron/dpo_config.yaml`）按功能模块组织，主要包含以下部分：

1. **实验基本设置**
   * `exp_name`：实验名称，用于标识一次具体训练任务
   * `logging_dir`：日志文件保存路径
   * `output_dir`：模型检查点和输出文件保存路径

2. **训练控制参数**
   * `max_steps`：最大训练步数
   * `save_steps`：保存模型检查点的频率
   * `logging_steps`：记录训练指标的频率
   * `resume_from_checkpoint`：是否从检查点继续训练。若想继续训练，请设为其路径；否则设为 `False`

3. **DPO算法参数**
   * `ipo`：使用ipo的loss函数
   * `beta`：用于调节模型对人类偏好数据的敏感度
   * `label_smoothing`：一种正则化技术，通过软化模型对标签的绝对置信度，减少过拟合风险

4. **工作进程配置**
   每个工作进程（`actor_train`、`reference`）配置包含：

   * **模型参数**（`model_args`）
     * `model_type`：模型类型（如 `causal_lm`）
     * `dtype`：计算精度（如 `bf16`、`fp16`）
     * ...
   * **训练参数**（`training_args`）
     * `learning_rate`：学习率
     * `per_device_train_batch_size`：每个设备的训练批次大小
     * `gradient_accumulation_steps`：梯度累积步数
     * `weight_decay`：权重衰减系数
     * `max_grad_norm`：梯度裁剪阈值
     * ...
   * **分布式策略**（`strategy_args`）
     * `strategy_name`：使用的分布式策略（如 `megatron_train`、`deepspeed_infer`）
     * 策略特定参数：如 `tp_size`（张量并行规模）、`pp_size`（Pipeline并行规模）
     * `gpu_memory_utilization`：GPU 内存利用率（特定于 vLLM）
   * **设备映射**（`device_mapping`）
     * 指定该工作进程应使用哪些 GPU 设备

---

## ✨️ 数据准备

### 数据格式

DPOPipeline要求训练数据以 **JSON** 文件形式存储。

### 必需字段

每条数据样本必须包含一个问题以及chosen answer和rejected answer
在 YAML 文件中，请使用 `chosen_key` 和 `rejected_key` 来指定这两个数据在数据集中对应的字段名称。
例如：
```
"instruction": "Select a color and provide some adjectives to describe it.",
"input": "",
"chosen": "The color is blue. Adjectives to describe it include calming, soothing, serene, and peaceful.",
"rejected": "Red"
```
---

## ✨️ 运行Pipeline

### 方法 1：使用 Python 启动脚本

主要方法是使用 `examples/start_dpo_pipeline.py` 脚本。该脚本利用 Hydra 加载并管理配置。

1. **选择或创建配置文件**  
   从示例 YAML（如 `examples/qwen2.5-3B-dpo_megatron/dpo_config.yaml`）开始，或创建自己的配置。

2. **执行 Python 启动脚本**

   ```bash
   # 确保你在 ROLL 项目根目录
   # export PYTHONPATH=$(pwd):$PYTHONPATH
   
   python examples/start_dpo_pipeline.py \
          --config_path examples/qwen2.5-3B-dpo_megatron \
          --config_name dpo_config
   ```

   * `--config_path` – 包含 YAML 配置的目录。
   * `--config_name` – 文件名（不含 `.yaml`）。

### 方法 2：使用辅助 Shell 脚本

`examples` 目录通常包含包装了 Python 启动器的 shell 脚本。

示例结构：

```bash
#!/bin/bash
# 示例：examples/qwen2.5-3B-dpo_megatron/run_dpo_pipeline.sh

CONFIG_NAME="dpo_config"
CONFIG_PATH="examples/qwen2.5-3B-dpo_megatron"

# 设置环境变量及其他配置

python examples/start_dpo_pipeline.py \
       --config_path $CONFIG_PATH \
       --config_name $CONFIG_NAME \
       "$@"   # 传递任何额外参数
```

运行方式：

```bash
bash examples/qwen2.5-3B-dpo_megatron/run_dpo_pipeline.sh
```

---

## ✨️ 逐步示例

### 步骤 1：配置设置

* 文件：`examples/qwen2.5-3B-dpo_megatron/dpo_config.yaml`  
  关键部分包括 `exp_name`、`seed`、`output_dir`、模型路径、`actor_train` 和 `reference` 配置。

* 特别注意这些配置段：
  * 数据配置：`actor_train.data_args.file_name`
  * 模型配置：`pretrain` 路径
  * 分布式策略：每个工作进程的 `strategy_args` 和 `device_mapping`

### 步骤 2：准备环境与依赖

* 确保已安装所有必要依赖：

  ```bash
  pip install -r requirements.txt
  ```

* 确认配置中所有模型路径均可访问。

* 准备训练数据集，确保符合上述数据格式要求。

### 步骤 3：启动Pipeline

```bash
   python examples/start_dpo_pipeline.py \
          --config_path examples/qwen2.5-3B-dpo_megatron \
          --config_name dpo_config
```

### 步骤 4：监控

* **控制台输出** – 观察 Hydra、Ray 和Pipeline日志。
* **日志文件** – 检查 YAML 中指定的 `logging_dir`。
* **TensorBoard**

  ```bash
  tensorboard --logdir <your_log_dir>
  ```

### 步骤 5：输出与结果

* **已训练模型** – 检查点保存在 `output_dir`。
* **评估指标** – 记录在 TensorBoard 和终端中。

---

*祝实验愉快！*