# SFT 流水线

**目录**

- [SFT 流水线](#sft-流水线)
  - [✨️概述](#️概述)
  - [✨️核心组件](#️核心组件)
    - [主模块（`SFTPipeline`）](#主模块sftpipeline)
    - [工作器（`SFTWorker`）](#工作器sftworker)
    - [配置文件（`SFTConfig`）](#配置文件sftconfig)
      - [配置文件结构和组织](#配置文件结构和组织)
  - [✨️数据准备](#️数据准备)
    - [数据格式](#数据格式)
      - [必需字段与字段映射](#必需字段与字段映射)
      - [对话模板与标签（labels）规则](#对话模板与标签labels规则)
    - [验证集（`validation`）](#验证集validation)
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

此流水线用于监督微调（SFT），提供：

* **统一的数据编码与对话模板**：支持 system/user/assistant 对话格式拼接，并自动构造 `labels`（仅对回答部分计 loss）。
* **高效分布式训练**：使用 [Ray](https://www.ray.io/) + Cluster/Worker 抽象启动分布式训练。
* **全面的性能监控**：细粒度度量跟踪系统，监控性能指标，为模型训练过程提供全面的可视化和分析能力。
* **高效训练优化**：支持 **Sequence Packing**（将多条短样本拼接成连续序列，减少 padding）。配置方法和实现原理详见`sequence packing`对应文档。

---

## ✨️核心组件

### 主模块（`SFTPipeline`）

`SFTPipeline`（位于 `roll/pipeline/sft/sft_pipeline.py`）是 SFT 训练的主流程，负责：

* 加载 tokenizer。
* 加载训练数据集 与（可选）验证数据集。
* 按模板编码数据：生成 `input_ids` / `attention_mask` / `labels`。
* 初始化分布式训练集群（`Cluster` + `SFTWorker`）。
* 训练循环：按 step 训练、按 `eval_steps` 验证、按保存策略写 checkpoint、记录指标并上报 tracker。

---

### 工作器（`SFTWorker`）

`SFTWorker`（位于 `roll/pipeline/sft/sft_worker.py`）负责执行训练、验证与保存：

* `initialize()`：创建并初始化分布式策略（`create_strategy`），并加载模型。
* `train_step()`：执行一次训练 step，返回训练 metrics。
* `val_step()`：执行一次验证 step（前向 + loss），返回验证 metrics。
* `do_checkpoint()`：保存 checkpoint，并返回保存耗时等 metrics。

---

### 配置文件（`SFTConfig`）

`SFTConfig`（定义于 `roll/pipeline/sft/sft_config.py`）是 SFT 流水线的配置对象（dataclass 风格），支持通过 YAML + Hydra 管理。

#### 配置文件结构和组织

示例配置文件：`examples/qwen2.5-7B-sft_megatron/sft_config.yaml`

配置通常包含以下部分：

1. **实验基本设置**
   * `exp_name`：实验名称
   * `seed`：随机种子
   * `logging_dir`：日志目录
   * `output_dir`：checkpoint/输出目录

2. **训练控制参数**
   * `save_steps`：保存 checkpoint 的频率
   * `logging_steps`：记录训练指标的频率
   * `eval_steps`：验证频率（启用验证集时生效）
   * `resume_from_checkpoint`：断点续训配置

3. **模型配置**
   * `pretrain`：预训练模型路径  

4. **数据字段映射（关键）**
   * `system_key`：system prompt 字段（可选）
   * `prompt_key`：prompt 字段名（默认 `instruction`）
   * `query_key`：query 字段名（可选）
   * `response_key`：response 字段名（默认 `output`）
   * `global_template`：全局模板名（可选；否则使用 `sft_train.data_args.template`）

5. **工作器配置（`sft_train`）**
   `sft_train` 是一个 `WorkerConfig`，包含：

   * **数据参数**（`data_args`）
     * `file_name`：训练数据 JSON 路径（字符串或列表）
     * `template`：对话模板名（当未设置 `global_template` 时使用）
     * `preprocessing_num_workers`：数据预处理并行数
   * **训练参数**（`training_args`）
     * `num_train_epochs`
     * `learning_rate`
     * `per_device_train_batch_size`
     * `gradient_accumulation_steps`
     * `dataloader_num_workers`
     * ...
   * **策略参数**（`strategy_args`）
     * `strategy_name`：如 `megatron_train` / `deepspeed_train` 等
     * 并行相关参数（tensor/pipeline 并行大小等）
   * **设备映射**（`device_mapping`）
     * 指定该 worker 使用哪些 GPU
   * **验证 batch**（推理 batch）
     * `infer_batch_size`：验证阶段使用

6. **验证配置（可选）**
   * `validation.data_args.file_name`：验证集 JSON 路径（配置后才会启用验证）

---

## ✨️数据准备

### 数据格式

SFT 流水线使用 **JSON** 文件，并通过 HuggingFace Datasets 加载。

#### 必需字段与字段映射

每条样本至少需要能映射出：

* Prompt：由 `prompt_key` 指定（默认 `instruction`）
* Response：由 `response_key` 指定（默认 `output`）

可选字段：

* `system_key`：system prompt（可选）
* `query_key`：附加输入（可选，会拼到 user 内容中）

#### 对话模板与标签（labels）规则

对话结构：

- system（可选）
- user（prompt + query）
- assistant（response）

labels 构造：

* prompt 部分全部置为 `IGNORE_INDEX`（不参与 loss）
* response 部分使用真实 token id（参与 loss）

即：只监督模型“回答部分”。

---

### 验证集（`validation`）

验证集是可选项：

* 仅当配置了 `validation.data_args.file_name` 才加载验证集。
* 训练时按 `eval_steps` 触发验证。
* 验证由 `sft_train.val_step` 执行（不会额外启动一个 validation worker）。

---

## ✨️运行流水线

### 方法1：使用Python启动脚本

使用 `examples/start_sft_pipeline.py` 启动，Hydra 负责加载配置：

```bash
# 确保您在 ROLL 项目根目录
# export PYTHONPATH=$(pwd):$PYTHONPATH

python examples/start_sft_pipeline.py \
       --config_path examples/qwen2.5-7B-sft_megatron \
       --config_name sft_config
```

* `--config_path` – 配置目录：`examples/qwen2.5-7B-sft_megatron`
* `--config_name` – 配置文件名：`sft_config`（对应 `sft_config.yaml`）

---

### 方法2：使用辅助Shell脚本

示例：

```bash
#!/bin/bash
# 示例：examples/qwen2.5-7B-sft_megatron/run_sft_pipeline.sh

CONFIG_NAME="sft_config"
CONFIG_PATH="examples/qwen2.5-7B-sft_megatron"

python examples/start_sft_pipeline.py \
       --config_path $CONFIG_PATH \
       --config_name $CONFIG_NAME \
       "$@"
```

运行：

```bash
bash examples/qwen2.5-7B-sft_megatron/run_sft_pipeline.sh
```

---

## ✨️逐步示例

### 步骤1：配置设置

配置文件：`examples/qwen2.5-7B-sft_megatron/sft_config.yaml`

重点检查：

* **数据配置**：`sft_train.data_args.file_name`
* **字段映射**：`prompt_key/query_key/response_key/system_key`
* **模型配置**：`pretrain`
* **分布式策略**：`sft_train.strategy_args` 与 `sft_train.device_mapping`
* **验证配置（可选）**：`validation.data_args.file_name` 与 `eval_steps`
* **模板选择**：`global_template` 或 `sft_train.data_args.template`

### 步骤2：准备环境和依赖

```bash
pip install -r requirements.txt
```

并确保：

* `pretrain` 路径可访问
* 训练/验证 JSON 的字段与 `prompt_key/response_key/...` 对齐

### 步骤3：启动流水线

```bash
python examples/start_sft_pipeline.py \
       --config_path examples/qwen2.5-7B-sft_megatron \
       --config_name sft_config
```

### 步骤4：监控

* **控制台输出** – 观察 Hydra、Ray 与流水线日志
* **日志文件** – 检查 `logging_dir`
* **TensorBoard**
  ```bash
  tensorboard --logdir <your_log_dir>
  ```

### 步骤5：输出和结果

* **训练模型** – checkpoint 保存在 `output_dir` 下，默认目录结构为：

  ```
  <output_dir>/sft_train/checkpoint-<global_step>/<cluster_name>/
  ```

  其中：
  * `<global_step>`：当前训练步数（例如 `checkpoint-200`）
  * `<cluster_name>`：分布式集群名称（由 Cluster/Ray 运行时决定）

* **训练/验证指标** – 记录在终端与 tracker/TensorBoard（取决于 tracker 配置）

---

*祝您实验愉快！*