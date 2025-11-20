
# VLM RLVR 流水线


- [VLM RLVR 流水线](#vlm-rlvr-流水线)
  - [✨️概述](#️概述)
  - [✨️核心组件](#️核心组件)
    - [主模块 (`RLVRPipeline`)](#主模块-rlvrvlmpipeline)
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

 该 VLM RLVR 流水线与 LLM 流水线具有相同的优势，而在该流水线中内置支持了视觉推理和视觉感知任务，目前包括数学（推理）和检测（感知），每个任务都配备专门的奖励评估机制，使得能够同时优化模型在多个领域（如数学和检测）的能力。

---

## ✨️核心组件

### 主模块 (`RLVRVLMPipeline`)

`RLVRVLMPipeline` (位于`roll/pipeline/rlvr/rlvr_vlm_pipeline.py`) 是整个强化学习过程的主要协调器。它管理完整的训练工作流，包括：

* 初始化和管理分布式工作器（actor、critic、reference和各种奖励工作器）。
* 协调数据收集和处理。
* 执行模型训练步骤（例如，actor和critic的PPO更新）。
* 处理模型同步和检查点保存。
* 验证集评估。
* 记录指标和实验跟踪。

**源代码**：`roll/pipeline/rlvr/rlvr_vlm_pipeline.py`, 其中直接支持了 Qwen2.5-VL 模型

---

### 配置文件 (`RLVRConfig`)

VLM RLVR 流水线和 LLM 共享相同的配置文件（`RLVRConfig`），有关配置详情请参考[LLM RLVR Pipeline文档](./rlvr_pipeline_start.md)。VLM RLVR 流水线配置示例可在 `examples/qwen2.5-vl-7B-rlvr/rlvr_megatron.yaml` 中找到，和 LLM RLVR 流水线配置最大差异在于奖励设置，其中包含了视觉特有的奖励并将在后面介绍。

   **奖励设置**
   `rewards`部分包含不同领域的奖励工作器配置：

   * **数学**
     * `worker_cls`：工作器类名（例如，`MathRuleRewardWorker`）
     * `tag_included`：这些标签使用奖励领域进行计算
     * `model_args`：奖励模型参数
     * `world_size`：工作器数量

   * **cv_detection**
     * 类似配置，但用于评估检测任务

   注意：此处提供的领域（math 和 cv_detection）应与 `domain_interleave_probs` 中列出的相同。

---

### 奖励工作器

VLM RLVR 流水线支持不同领域的奖励机制，示例中使用的奖励包括：

* **数学规则奖励** (`MathRuleRewardWorker`) – 评估数学推理的正确性和步骤。
* **检测奖励** (`DetectionRewardWorker`) – 评估检测的结果。
  - 此处使用的检测验证器参考了 [MiniMax-AI/One-RL-to-See-Them-All](https://github.com/MiniMax-AI/One-RL-to-See-Them-All/blob/main/reward_server/verify.py)，它使用指定的加权系数结合 IoU 和 mAP 分数来计算奖励（还额外结合了格式奖励）
  - 对于 IoU 分数：
    - IoU 阈值由环境变量 DET_IOU_THRESHOLD 定义，可设置为 `[0.5, 0.55, 0.6,0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99] `中的一个，或设置为 `average` 或 `dynamic`
    - `average` 是默认值，对所有阈值下的 IoU 分数取平均
    - `dynamic` 是一种引入的动态 IoU 奖励机制，提供自适应、渐进式反馈以提高稳定性和性能。它对不同的训练阶段使用不同的阈值。具体来说，前 10% 的训练步骤使用 0.85 的 IoU 阈值，接下来的 15% 使用 0.95，剩余训练步骤使用 0.99
    - 此外，每个阈值对应的 IoU 分数包括不同评估策略（`greedy_match_by_iou_max_iou_first` 和 `greedy_match_by_iou_max_label_first`）下的两项，每项得分也根据系数将 completeness （计算方式为 `1.0 - (假阳性率+假阴性率)/2.0`）加权计入在内

  更多详情请参考[论文](https://arxiv.org/pdf/2505.18129)

---

## ✨️数据准备

### 数据格式

对于多域 RLVR，我们使用来自 [One-RL-to-See-Them-All/Orsta-Data-47k](https://huggingface.co/datasets/One-RL-to-See-Them-All/Orsta-Data-47k) 的 parquet 格式数据文件作为示例输入，因此也遵循其数据模式。以下是数学领域的示例数据格式：

```python
{
    "data_source": "mm_math",
    "images": [<PIL.PngImagePlugin.PngImageFile image mode=RGB size=252x56 at 0x15EEEA390>],
    "prompt": [
        {
            "content": "<image>As shown in the figure, point D is the midpoint of line segment AC, $BC = \\frac{1}{2}AB$, and $BD = 1$cm. What is the length of AB?",
            "role": "user",
        }
    ],
    "ability": "math",
    "reward_model": {
        "answer": "4",
        "ground_truth": "\\boxed{4}",
        "accuracy_ratio": 1.0,
        "format_ratio": 0.0,
        "verifier": "mathverify",
        "verifier_parm": {
            "det_verifier_normalized": None,
            "det_reward_ratio": {
                "iou_max_label_first": None,
                "iou_max_iou_first": None,
                "iou_completeness": None,
                "map": None,
                "map50": None,
                "map75": None,
            },
        },
    },
    "extra_info": {"id": None, "image_path": "images/51284809.png"},
}
```

底层数据模式为：

```python
{
    'data_source': Value(dtype='string', id=None),
    'images': Sequence(feature=Image(mode=None, decode=True, id=None), length=-1, id=None),
    'prompt': [{'content': Value(dtype='string', id=None), 'role': Value(dtype='string', id=None)}],
    'ability': Value(dtype='string', id=None),
    'reward_model': {
        'answer': Value(dtype='string', id=None),
        'ground_truth': Value(dtype='string', id=None),
        'accuracy_ratio': Value(dtype='float32', id=None),
        'format_ratio': Value(dtype='float32', id=None),
        'verifier': Value(dtype='string', id=None),
        'verifier_parm': {
            'det_verifier_normalized': Value(dtype='bool', id=None),
            'det_reward_ratio': {
                'iou_max_label_first': Value(dtype='float32', id=None),
                'iou_max_iou_first': Value(dtype='float32', id=None),
                'iou_completeness': Value(dtype='float32', id=None),
                'map': Value(dtype='float32', id=None),
                'map50': Value(dtype='float32', id=None),
                'map75': Value(dtype='float32', id=None)
            }
        }
    },
    'extra_info': {'id': Value(dtype='string', id=None), 'image_path': Value(dtype='string', id=None)}
}
```

其中各字段描述如下：

- `data_source` (必需): 数据来源，例如 `object365_train`。注意：它应是配置文件中 rewards 部分下某个领域 `tag_included` 中包含的值，以指示此样本属于该领域并使用相应的奖励工作器。此外，在验证中用于为不同数据源提供分离的指标
- `images` (必需): PIL 图像列表
- `prompt` (必需): 使用 chat templates 的 prompt。 注意：使用 `<image>` 作为图像标记，且 prompt 中的图像标记数量应与 `images` 相同
- `ability` (可选): 指示数据领域，例如 cv_detection。同一领域的数据应具有相同的 ability 值。目前尚未使用
- `reward_model` (必需): 奖励相关信息，用于计算奖励  
  - `answer` (可选): 真实值，目前未使用，实际使用的是 `ground_truth`
  - `ground_truth` (必需): 格式化的真实值，格式与奖励工作器中的答案提取方法对应。例如，检测的 `ground_truth` 应包含在 `"<answer>"` 和 `"</answer>"` 中，以与 `detection_reward_worker.py` 保持一致，后者使用 `"<answer>"` 和 `"</answer>"` 提取答案
  - `accuracy_ratio` (必需): 用于分数/奖励计算的准确率系数
  - `format_ratio` (必需): 用于分数/奖励计算的格式系数
  - `verifier` (可选): 验证器名称，例如 detection，目前未使用 
  - `verifier_parm` (Optional): 验证器参数，所有子字段均为可选，仅检测需要
    - `det_verifier_normalized`: 是否将边界框 IoU 归一化为高度/宽度尺寸 1000，当真实值已归一化时需要
    - `det_reward_ratio`: 检测分数加权组合的各种系数
      - `iou_max_label_first`: 使用 `greedy_match_by_iou_max_iou_first` 策略的 IoU 分数系数
      - `iou_max_iou_first`: 使用 `greedy_match_by_iou_max_label_first` 策略的 IoU 分数系数
      - `iou_completeness`: IoU completeness 加权系数 
      - `map`: mAP 加权系数
      - `map50`: mAP50 加权系数
      - `map75`: mAP75 加权系数
- `extra_info` (Optional): 额外信息，目前未使用


来自检测领域的包含所有字段的更复杂示例如下：

```python
{
    "data_source": "v3det_train",
    "images": [<PIL.PngImagePlugin.PngImageFile image mode=RGB size=799x533 at 0x15EF1DF90>],
    "prompt": [
        {
            "content": "<image>\nLocate all objects of the designated class present in the image:\n- dog collar\n- shiba dog\n\nBegin by clearly explaining your thought process enclosed within <think> and </think> tags. Afterward, present your final detection results enclosed within <answer> and </answer> tags.\nFor example:\n<think>\nYour detailed reasoning process here.\n</think>\n<answer>\n[{'bbox_2d': [x1,y1,x2,y2],'label': label_name}]\n</answer>",
            "role": "user",
        }
    ],
    "ability": "cv_detection",
    "reward_model": {
        "answer": "[{'bbox_2d': [484, 227, 818, 998], 'label': 'shiba dog'}, {'bbox_2d': [106, 142, 473, 998], 'label': 'shiba dog'}, {'bbox_2d': [274, 468, 427, 652], 'label': 'dog collar'}, {'bbox_2d': [490, 522, 609, 611], 'label': 'dog collar'}]",
        "ground_truth": "<answer>\n[{'bbox_2d': [484, 227, 818, 998], 'label': 'shiba dog'}, {'bbox_2d': [106, 142, 473, 998], 'label': 'shiba dog'}, {'bbox_2d': [274, 468, 427, 652], 'label': 'dog collar'}, {'bbox_2d': [490, 522, 609, 611], 'label': 'dog collar'}]\n</answer>",
        "accuracy_ratio": 1.0,
        "format_ratio": 0.10000000149011612,
        "verifier": "detection",
        "verifier_parm": {
            "det_verifier_normalized": True,
            "det_reward_ratio": {
                "iou_max_label_first": 1.0,
                "iou_max_iou_first": 0.0,
                "iou_completeness": 0.30000001192092896,
                "map": 0.0,
                "map50": 0.0,
                "map75": 0.0,
            },
        },
    },
    "extra_info": {"id": None, "image_path": "images/a00004438/19_1169_36968567106_1209f085a7_c.jpg"},
}
```


可以按照上面提供的格式来自定义数据并在流水线中直接使用。

---


## ✨️运行流水线


### 方法1：使用Python启动脚本

主要方法是使用`examples/start_rlvr_vl_pipeline.py`脚本。此脚本使用Hydra加载和管理配置。

1. **选择或创建配置文件**
   从示例YAML（例如，`examples/qwen2.5-vl-7B-rlvr/rlvr_megatron.yaml`）开始或创建您自己的配置。

2. **执行Python启动脚本**

   ```bash
   # 确保您在ROLL (ScaleAligner)项目的根目录中
   # export PYTHONPATH=$(pwd):$PYTHONPATH
   
   python examples/start_rlvr_vl_pipeline.py \
          --config_path examples/qwen2.5-vl-7B-rlvr \
          --config_name rlvr_megatron
   ```

   * `--config_path` – 包含YAML配置的目录。
   * `--config_name` – 文件名（不带`.yaml`）



### 方法2：使用辅助Shell脚本

`examples`目录通常包含包装Python启动器的shell脚本 (e.g., `run_rlvr_pipeline.sh`).

示例结构：

```bash
#!/bin/bash
# 示例: examples/qwen2.5-vl-7B-rlvr/run_rlvr_pipeline.sh

CONFIG_NAME="rlvr_megatron"            # rlvr_megatron.yaml
CONFIG_PATH="examples/qwen2.5-vl-7B-rlvr"

# 设置环境变量和其他配置

python examples/start_rlvr_vl_pipeline.py \
       --config_path $CONFIG_PATH \
       --config_name $CONFIG_NAME \
       "$@"   # Pass any additional parameters
```

Run using:

```bash
bash examples/qwen2.5-vl-7B-rlvr/run_rlvr_pipeline.sh
```

---


## ✨️逐步示例

### 步骤1：配置设置

* 文件：`examples/qwen2.5-vl-7B-rlvr/rlvr_megatron.yaml`  
  关键部分包括`exp_name`、`seed`、`output_dir`、模型路径、`actor_train`、`actor_infer`、`reference`、PPO参数和奖励配置。


* 特别注意这些配置部分：
  * 数据配置：`actor_train.data_args.file_name`和`domain_interleave_probs`
  * 模型配置：`pretrain`路径
  * 分布式策略：每个工作器的`strategy_args`和`device_mapping`
  * 奖励配置：`rewards`部分中不同领域的奖励工作器

### 步骤2：准备环境和依赖

* 确保安装了所有必要的依赖。注意：VLM 流水线当前只支持使用 VLLM 作为推理引擎，因而需要选择使用对应的requirement文件：

  ```bash
  pip install -r requirements_torch260_vllm.txt
  ```

* 验证配置中的所有模型路径是否可访问。

* 准备训练和验证数据集，确保它们符合上述数据格式要求。

### 步骤3：启动流水线

```bash
python examples/start_rlvr_vl_pipeline.py \
       --config_path examples/qwen2.5-vl-7B-rlvr \
       --config_name rlvr_megatron
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