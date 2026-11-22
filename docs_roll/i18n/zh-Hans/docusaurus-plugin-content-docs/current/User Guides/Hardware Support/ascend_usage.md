# ROLL x Ascend

最后更新：2026/11/18。

我们在 ROLL 上增加对华为昇腾设备的支持。

## 硬件配套和支持的操作系统

ROLL 昇腾适配当前覆盖以下训练系列硬件：

| 产品 | 是否支持 | 说明 |
| ---- | -------- | ---- |
| Atlas A2 训练系列产品 | √ | 使用 `docker/Dockerfile.A2` 或 `roll:v0.3-cann9.1.0-torch_npu2.10.0.post4-910b-ubuntu22.04-py3.12` 镜像。 |
| Atlas A3 训练系列产品 | √ | 使用 `docker/Dockerfile.A3` 或 `roll:v0.3-cann9.1.0-torch_npu2.10.0.post4-a3-ubuntu22.04-py3.12` 镜像。 |
| Ascend 950 训练系列产品 | √ | 使用下文 CANN 9.2.0 安装配置：Python 3.12、torch 2.10.0，以及源码安装的 vLLM 0.30.0 和 vLLM-Ascend 0.30.0rc2。 |
| Atlas A2/A3 推理系列产品、Atlas 200I/500 A2 推理产品 | x | 当前 ROLL NPU 镜像和示例面向训练系列设备。 |
| 其他昇腾训练或推理产品 | 未验证 | 使用前请确认驱动、固件、CANN、`torch_npu` 与 vLLM-Ascend 版本配套。 |

> 本节表格中 `√` 代表当前 ROLL 昇腾 Dockerfile、示例或 Ascend 950 手动安装配置已支持，`x` 代表当前 ROLL NPU 配套不支持。

支持的操作系统：

| 部署场景 | 支持的操作系统 | 说明 |
| -------- | -------------- | ---- |
| 物理机宿主机 | Ubuntu 22.04 | 当前 ROLL 昇腾文档推荐并验证的宿主机操作系统。 |
| ROLL 昇腾容器 | Ubuntu 22.04 | A2/A3 Dockerfile 基于 `quay.io/ascend/cann:9.1.0-*-ubuntu22.04-py3.12-devel`。 |
| Ascend 950  手动安装 | Ubuntu 22.04 | 使用下文 Ascend 950 专用 torch/vLLM 版本组合。驱动、固件、CANN 和 `torch_npu` 版本需要与目标 Ascend 950 环境匹配。 |
| 其他宿主机 OS 上的虚拟机或容器部署 | 以昇腾/CANN 兼容性说明为准 | 请结合目标硬件查询昇腾兼容性查询助手，以及 CANN 软件安装文档中的操作系统兼容性说明。 |

## 安装

### 基础环境准备

| 软件 | 版本 |
|-----------|-------------|
| Python    |  3.12       |
| CANN      |  9.2.0      |

以下手动安装步骤使用 CANN 9.2.0、vLLM 0.30.0 和 vLLM-Ascend 0.30.0rc2。Ascend 950 使用下文 [Ascend 950 安装配置](#ascend-950-安装配置) 中的版本组合。

### 创建 conda 环境

使用以下命令在 Miniconda 中创建新的 conda 环境：

```
conda create --name roll python=3.12
conda activate roll
```

### 安装 torch & torch_npu

为了能在 ROLL 中正常使用 torch 和 torch_npu，需使用以下命令安装 torch 和 torch_npu：

```
# 在预构建镜像外手动安装时，使用 CPU 版 torch
pip install \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://mirrors.huaweicloud.com/repository/pypi/simple \
    torch==2.10.0+cpu torchvision==0.25.0 torchaudio==2.10.0

# 安装与 torch/CANN 匹配的 torch_npu
python -m pip install \
    --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
    --no-deps torch-npu==2.10.0.post4
```

### 安装 vllm & vllm-ascend

vLLM 和 vLLM-Ascend 均固定发布版本并从源码安装，其他依赖可使用预编译包。以下命令适用于 A2/A3，Ascend 950 使用下节的源码构建步骤。先按发布版本的 requirements 准备运行依赖，以下 `--no-deps` 命令复用已准备的依赖，不能单独用于空环境。编译前加载 CANN 9.2.0 开发环境。vLLM 使用 `empty` 构建目标，避免编译 CUDA kernel。

```bash
git clone -b v0.30.0 --depth 1 https://github.com/vllm-project/vllm.git
cd vllm
python -m pip install "cmake>=3.26.1" ninja "setuptools>=77.0.3,<81" "setuptools-scm>=8" "setuptools-rust>=1.9.0" wheel packaging jinja2
VLLM_TARGET_DEVICE=empty VLLM_VERSION_OVERRIDE=0.30.0 \
    python -m pip install -v -e . --no-deps --no-build-isolation
cd ..

git clone -b v0.30.0rc2 --depth 1 https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
git submodule update --init --recursive
python -m pip install "cmake>=3.26" "setuptools>=64" "setuptools-scm>=8" pybind11 wheel
COMPILE_CUSTOM_KERNELS=1 \
    python -m pip install -v -e . --no-deps --no-build-isolation
cd ..

pip install triton-ascend==3.2.2 --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi
```

### Ascend 950 安装配置

Ascend 950DT（A5）配合 CANN 9.2.0 使用以下版本组合。vLLM 和 vLLM-Ascend 均从发布标签安装，并为目标芯片编译昇腾自定义算子。

| 组件 | 版本 |
| ---- | ---------- |
| Python / 架构 | 3.12 / aarch64 |
| CANN | 9.2.0 |
| torch / torch-npu | 2.10.0 / 2.10.0.post4 |
| vLLM / vLLM-Ascend | 0.30.0 / 0.30.0rc2（源码安装） |
| triton-ascend | 3.2.2 |
| Transformers | 5.14.1 |
| FLA | flash-linear-attention-npu-a5 26.9.1 |
| Ray / NumPy / FastAPI | 2.48.0 / 1.26.4 / 0.136.3 |

使用已配置好配套驱动、CANN 9.2.0 toolkit、对应芯片 ops 包、C/C++ 编译器和 torch 软件栈的容器，并在执行下列命令前准备运行依赖。这里的 `--no-deps` 用于复用已准备的运行环境，不是空环境下的完整依赖安装命令。两个发布版本的 FastAPI 要求存在冲突，上表是运行环境参考，不能视为通过所有 `pip check` 检查的依赖锁定文件。

```bash
# vLLM：只构建 Python 包，不编译 CUDA kernel。
git clone -b v0.30.0 --depth 1 https://github.com/vllm-project/vllm.git
cd vllm
python -m pip install "cmake>=3.26.1" ninja "setuptools>=77.0.3,<81" "setuptools-scm>=8" "setuptools-rust>=1.9.0" wheel packaging jinja2
VLLM_TARGET_DEVICE=empty VLLM_VERSION_OVERRIDE=0.30.0 \
    python -m pip install -v -e . --no-deps --no-build-isolation
cd ..

# vLLM-Ascend：为 A5 编译扩展和自定义算子。
git clone -b v0.30.0rc2 --depth 1 https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
git submodule update --init --recursive
python -m pip install "cmake>=3.26" "setuptools>=64" "setuptools-scm>=8" pybind11 wheel
SOC_VERSION=ascend950dt_9582 COMPILE_CUSTOM_KERNELS=1 \
    python -m pip install -v -e . --no-deps --no-build-isolation
cd ..

# Qwen3.5 GDN 算子使用 A5 包，不使用 A2 变体。
python -m pip install --no-deps \
    https://mirrors.huaweicloud.com/ascend/repos/pypi/flash-linear-attention-npu-a5/flash_linear_attention_npu_a5-26.9.1-py3-none-manylinux_2_34_aarch64.whl
```

加载对应的 CANN 环境后，在**启动 Python/Ray 前**设置自定义算子路径：

```bash
vllm_ascend_dir="$(python -c 'import importlib.util; print(importlib.util.find_spec("vllm_ascend").submodule_search_locations[0])')"
fla_npu_dir="$(python -c 'import importlib.util; print(importlib.util.find_spec("fla_npu").submodule_search_locations[0])')"
export ASCEND_CUSTOM_OPP_PATH="$vllm_ascend_dir/_cann_ops_custom/vendors/custom_transformer:$fla_npu_dir/opp/vendors/fla_npu_transformer:$fla_npu_dir/opp"
export LD_LIBRARY_PATH="$vllm_ascend_dir/_cann_ops_custom/vendors/custom_transformer/op_impl/ai_core/tbe/op_tiling:${LD_LIBRARY_PATH}"
export HCCL_NPU_SOCKET_PORT_RANGE=auto
export VLLM_ASCEND_ENABLE_NZ=0
```

算子路径按实际 Python 包位置解析，以兼容源码可编辑安装。`op_tiling` 库路径用于解析 `liboptiling.so`；缺少这一路径时，此前实测出现过 `AddRmsNormBias` 初始化失败。A5 测试还为 driver 和 worker 设置了 `ATEN_CPU_CAPABILITY=default`，用于避开该主机上观测到的较慢 CPU BF16 转换路径。示例 SOC 对应 Ascend 950DT，其他 A5 型号需结合 `npu-smi info` 和上游构建说明调整。

训练和推理共卡时，在配置的 `system_envs` 中设置 `ROLL_ASCEND_COLOCATED_IPC: '1'`，并在 vLLM strategy 中使用 `sleep_level: 2` 和 `additional_config.rl_config.enabled: true`。ROLL 的 NPU 适配调用官方分层权重重载 API，需要安装提供这些接口的 vLLM/vLLM-Ascend 配套版本，无需修改 vLLM 或 vLLM-Ascend 源码。

此前同版本环境已完成完整层数的 Qwen3.5-9B、Qwen3.5-35B-A3B 文本 RLVR 5 步测试，训练、推理和 reference 共用 0–7 卡。此前使用预编译的 vLLM-Ascend 包，本节源码构建方式尚未重新进行端到端测试。这些小批次、短序列测试不代表收敛、多模态或 checkpoint 导出恢复验证。流水线配置参见 [RLVR 指南](ascend_npu_rlvr.md)。

### 安装 ROLL

```
git clone https://github.com/alibaba/ROLL.git
cd ROLL
pip install -r requirements_common.txt
pip install -e .
cd ..
```

### 其他三方库说明

以下依赖与本文 CANN 9.2.0 / vLLM 0.30.0 配置保持一致。安装 ROLL 后检查版本，避免其他依赖安装过程覆盖这些版本。

| 软件 | 说明 |
| ---- | ---- |
| transformers | 5.14.1 |
| torch-npu | 2.10.0.post4 |
| triton-ascend | 3.2.2（替代 `triton`） |
| flash_attn | 不支持 |
| transformer-engine[pytorch] | 不支持 |

1. 该环境固定使用 `transformers==5.14.1`；注意力配置需与模型和 NPU 后端匹配。
2. 目前不支持 `flash_attn` 加速。
3. 目前不支持 `transformer-engine[pytorch]`。

```
pip install transformers==5.14.1
pip uninstall -y triton triton-ascend
pip install triton-ascend==3.2.2 --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi
```

A2 实测：以官方 main（`7c0ded3`）加本分支的最小 NPU 适配，在 CANN 9.2.0 / vLLM 0.30.0 / vLLM-Ascend 0.30.0rc2 环境完成 Qwen3.5-9B 文本 RLVR 5 步训练。配置为 8 卡 FSDP2、推理 TP2、每步 4 个 prompt × 2 条回答、response 上限 512 token，关闭 KL/reference；5 步梯度均有限且非零，两个推理 worker 各完成 5 次官方分层权重重载。本次验证了短序列训练流程，未验证收敛或 checkpoint 导出恢复。

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
| Agentic | examples/qwen2.5-0.5B-agentic/run_agentic_pipeline_sokoban.sh | FSDP2 | vLLM | Atlas 900 A2/A3/Ascend 950 训练系列 |
| Agentic-Rollout | examples/qwen2.5-0.5B-agentic/run_agentic_rollout_sokoban.sh | FSDP2 | vLLM | Atlas 900 A2/A3/Ascend 950 训练系列 |
| RLVR | examples/ascend_examples/run_rlvr_pipeline.sh | FSDP2 | vLLM | Atlas 900 A2/A3/Ascend 950 训练系列 |

## 声明

ROLL 中提供的 Ascend 支持代码皆为参考样例，生产环境使用请通过官方正式途径沟通。
