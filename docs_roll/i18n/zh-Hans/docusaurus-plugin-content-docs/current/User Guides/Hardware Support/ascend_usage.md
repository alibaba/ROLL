# ROLL x Ascend

最后更新：2026/03/13。

我们在 ROLL 上增加对华为昇腾设备的支持。

## 硬件支持

Atlas 900 A3 PODc

## 安装

### 基础环境准备

| 软件 | 版本 |
| ---- | ---- |
| Python | 3.11 |
| CANN | 8.5.1 |

### 创建 conda 环境

使用以下命令在 Miniconda 中创建新的 conda 环境：

```
conda create --name roll python=3.11
conda activate roll
```

### 安装 torch & torch_npu

为了能在 ROLL 中正常使用 torch 和 torch_npu，需使用以下命令安装 torch 和 torch_npu：

```
# 安装 torch 的 CPU 版本
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cpu

# 安装 torch_npu 2.9.0
pip install torch_npu==2.9.0
```

### 安装 vllm & vllm-ascend

为了能够在 ROLL 中正常使用 vllm，需使用以下命令编译安装 vllm 和 vllm-ascend：

```
# vllm
git clone -b v0.13.0 --depth 1 https://github.com/vllm-project/vllm.git
cd vllm
pip install -r requirements/build.txt

VLLM_TARGET_DEVICE=empty pip install -v -e .
cd ..

# vllm-ascend
git clone -b v0.13.0 --depth 1 https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend

pip install -e .
cd ..
```

或者可以从预编译的 wheel 包安装 `vllm` 和 `vllm-ascend`：

```
# 安装 vllm-project/vllm，最新支持版本为 v0.13.0
pip install vllm==0.13.0

# 从 pypi 安装 vllm-project/vllm-ascend
pip install vllm-ascend==0.13.0
```

### 安装 ROLL

```
git clone https://github.com/alibaba/ROLL.git
cd ROLL
pip install -r requirements_common.txt
pip install deepspeed==0.16.4
cd ..
```

### 其他三方库说明

| 软件 | 说明 |
| ---- | ---- |
| transformers | >= v4.57.6 |
| flash_attn | 不支持 |
| transformer-engine[pytorch] | 不支持 |

1. `transformers` v4.57.6 支持启用 `--flash_attention_2`。
2. 目前不支持 `flash_attn` 加速。
3. 目前不支持 `transformer-engine[pytorch]`。

```
pip install transformers==4.57.6
```

## 快速开始：单节点部署指引

正式使用前，建议您通过对单节点流水线的训练尝试以检验环境准备和安装的正确性。
由于目前暂不支持 Megatron-LM 训练，请首先将对应文件中 `strategy_args` 参数修改为 `deepspeed` 选项。

**注意：** 目前 NPU 上不支持 colocated 模式。你需要修改 `device_mapping`，确保训练和推理在不同的卡上执行。

1. 使用 shell 执行单节点流水线：

```
bash examples/agentic_demo/run_agentic_pipeline_frozen_lake_single_node_demo.sh  
```

2. 使用配置文件执行 agentic pipeline：

```
# 确保当前位于 ROLL 项目目录的根目录下

python examples/start_agentic_pipeline.py \
        --config_path qwen2.5-0.5B-agentic \
        --config_name agentic_val_sokoban
```

- `--config_path` – 包含您的 YAML 配置文件的目录。
- `--config_name` – 文件名（不含 `.yaml` 后缀）。

## 支持现状

| 功能 | 示例 | 训练后端 | 推理后端 | 硬件 |
| ---- | ---- | -------- | -------- | ---- |
| Agentic | examples/qwen2.5-0.5B-agentic/run_agentic_pipeline_sokoban.sh | DeepSpeed | vLLM | Atlas 900 A3 PODc |
| Agentic-Rollout | examples/qwen2.5-0.5B-agentic/run_agentic_rollout_sokoban.sh | DeepSpeed | vLLM | Atlas 900 A3 PODc |
| DPO | examples/qwen2.5-3B-dpo_megatron/run_dpo_pipeline.sh | DeepSpeed | vLLM | Atlas 900 A3 PODc |
| RLVR | examples/qwen2.5-7B-rlvr_megatron/run_rlvr_pipeline.sh | DeepSpeed | vLLM | Atlas 900 A3 PODc |

## 声明

ROLL 中提供的 Ascend 支持代码皆为参考样例，生产环境使用请通过官方正式途径沟通。
