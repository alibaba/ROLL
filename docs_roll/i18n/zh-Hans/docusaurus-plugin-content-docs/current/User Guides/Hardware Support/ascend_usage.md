# ROLL x Ascend

最后更新：2026/06/23。

我们在 ROLL 上增加对华为昇腾设备的支持。

## 硬件配套和支持的操作系统

ROLL 昇腾适配当前覆盖以下训练系列硬件：

| 产品 | 是否支持 | 说明 |
| ---- | -------- | ---- |
| Atlas 900 A2 PODc（Ascend 910B1）/ Atlas A2 训练系列产品 | √ | 使用 `docker/Dockerfile.A2` 或 `roll:ascend-a2` 镜像。 |
| Atlas 900 A3 PODc（Ascend 910_9391）/ Atlas A3 训练系列产品 | √ | 使用 `docker/Dockerfile.A3` 或 `roll:ascend-a3` 镜像。 |
| Atlas A5 训练系列产品 | √ | 使用 A5 安装配置：torch 2.10、vLLM v0.20.2、vLLM-Ascend `main`，并在构建 vLLM-Ascend 时设置 `COMPILE_CUSTOM_KERNELS=1`。 |
| Atlas A2/A3 推理系列产品、Atlas 200I/500 A2 推理产品 | x | 当前 ROLL NPU 镜像和示例面向训练系列设备。 |
| 其他昇腾训练或推理产品 | 未验证 | 使用前请确认驱动、固件、CANN、`torch_npu` 与 vLLM-Ascend 版本配套。 |

> 本节表格中 `√` 代表当前 ROLL 昇腾 Dockerfile、示例或 A5 手动安装配置已支持，`x` 代表当前 ROLL NPU 配套不支持。

支持的操作系统：

| 部署场景 | 支持的操作系统 | 说明 |
| -------- | -------------- | ---- |
| 物理机宿主机 | Ubuntu 22.04 | 当前 ROLL 昇腾文档推荐并验证的宿主机操作系统。 |
| ROLL 昇腾容器 | Ubuntu 22.04 | A2/A3 Dockerfile 基于 `quay.io/ascend/cann:9.0.0-*-ubuntu22.04-py3.11`。 |
| Atlas A5 手动安装 | Ubuntu 22.04 | 使用下文 A5 专用 torch/vLLM 版本组合。驱动、固件、CANN 和 `torch_npu` 版本需要与目标 A5 环境匹配。 |
| 其他宿主机 OS 上的虚拟机或容器部署 | 以昇腾/CANN 兼容性说明为准 | 请结合目标硬件查询昇腾兼容性查询助手，以及 CANN 软件安装文档中的操作系统兼容性说明。 |

## 安装

### 基础环境准备

| 软件 | 版本 |
|-----------|-------------|
| Python    |  3.11       |
| CANN      |  9.0.0      |

Atlas A5 请保持 Python 3.11，并使用下文 [A5 安装配置](#a5-安装配置) 中的专用 torch/vLLM 版本组合。

### 创建 conda 环境

使用以下命令在 Miniconda 中创建新的 conda 环境：

```
conda create --name roll python=3.11
conda activate roll
```

### 安装 torch & torch_npu

为了能在 ROLL 中正常使用 torch 和 torch_npu，需使用以下命令安装 torch 和 torch_npu：

```
# 在预构建镜像外手动安装时，使用 CPU 版 torch
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cpu

# 安装与 torch/CANN 匹配的 torch_npu
pip install torch_npu==2.9.0
```

### 安装 vllm & vllm-ascend

为了能够在 ROLL 中正常使用 vllm，需使用以下命令编译安装 vllm 和 vllm-ascend：

```
# vllm
git clone -b v0.18.0 --depth 1 https://github.com/vllm-project/vllm.git
cd vllm
pip install -r requirements/build.txt

VLLM_TARGET_DEVICE=empty pip install -v -e .
cd ..

# vllm-ascend
git clone -b v0.18.0 --depth 1 https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend

pip install -e .
cd ..
```

或者可以从预编译的 wheel 包安装 `vllm` 和 `vllm-ascend`：

```
# 安装 vllm-project/vllm，最新支持版本为 v0.18.0
pip install vllm==0.18.0

# 从 pypi 安装 vllm-project/vllm-ascend
pip install vllm-ascend==0.18
```

### A5 安装配置

Atlas A5 上使用 torch 2.10、vLLM v0.20.2，并从 `main` 分支安装 vLLM-Ascend。安装 vLLM-Ascend 前需要设置 `COMPILE_CUSTOM_KERNELS=1`，以便编译自定义 kernel：

```
# 安装 torch 2.10
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cpu

# 安装与 torch 2.10 和 CANN 版本匹配的 torch_npu
pip install torch_npu==2.10.0

# vLLM v0.20.2
git clone -b v0.20.2 --depth 1 https://github.com/vllm-project/vllm.git
cd vllm
pip install -r requirements/build.txt
VLLM_TARGET_DEVICE=empty pip install -v -e .
cd ..

# vLLM-Ascend main
git clone -b main --depth 1 https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
export COMPILE_CUSTOM_KERNELS=1
pip install -v -e .
cd ..
```

### 安装 ROLL

```
git clone https://github.com/alibaba/ROLL.git
cd ROLL
pip install -r requirements_common.txt
pip install -e .
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
由于 NPU 上不支持 Megatron-LM 训练，请首先将对应文件中 `strategy_args` 参数修改为 `fsdp2` 选项。


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
| Agentic | examples/qwen2.5-0.5B-agentic/run_agentic_pipeline_sokoban.sh | FSDP2 | vLLM | Atlas 900 A2/A3 PODc |
| Agentic-Rollout | examples/qwen2.5-0.5B-agentic/run_agentic_rollout_sokoban.sh | FSDP2 | vLLM | Atlas 900 A2/A3 PODc |
| RLVR | examples/ascend_examples/run_rlvr_pipeline.sh | FSDP2 | vLLM | Atlas 900 A2/A3/A5 训练系列 |

## 声明

ROLL 中提供的 Ascend 支持代码皆为参考样例，生产环境使用请通过官方正式途径沟通。
