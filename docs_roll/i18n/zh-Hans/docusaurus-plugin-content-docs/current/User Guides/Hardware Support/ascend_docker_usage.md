# 使用 Docker 在昇腾 NPU 上运行 ROLL

最后更新：2026/04/27。

本指南介绍如何使用 `Dockerfile.A2` 和 `Dockerfile.A3` 在**华为昇腾 NPU** 上构建并运行 ROLL。

## 硬件与软件要求

| 项目 | Dockerfile.A2 | Dockerfile.A3 |
| ---- | ------------- | ------------- |
| 硬件 | Atlas 900 A2 PODc（Ascend 910B1） | Atlas 900 A3 PODc（Ascend 910_9391） |
| 宿主机操作系统 | Ubuntu 22.04 | Ubuntu 22.04 |
| CANN | 8.5.1 | 8.5.1 |
| Python | 3.11 | 3.11 |
| Docker | >= 20.10 | >= 20.10 |
| 昇腾 NPU 驱动 | 已安装在宿主机上 | 已安装在宿主机上 |

## 主要组件

两个 Dockerfile 安装的核心依赖版本相同：

| 组件 | 版本 |
| ---- | ---- |
| PyTorch | 2.8.0+cpu |
| vLLM | 0.13.0 |
| vLLM-Ascend | 0.13.0 |
| DeepSpeed | 0.16.4 |
| Transformers | 4.57.6 |
| triton-ascend | 3.2.0 |

主要区别在于基础镜像和 SOC 版本：

| 项目 | Dockerfile.A2 | Dockerfile.A3 |
| ---- | ------------- | ------------- |
| 基础镜像 | `quay.io/ascend/cann:8.5.1-910b-ubuntu22.04-py3.11` | `quay.io/ascend/cann:8.5.1-a3-ubuntu22.04-py3.11` |
| SOC_VERSION | `ascend910b1` | `ascend910_9391` |

## 构建 Docker 镜像

### 1. 克隆 ROLL 仓库

```bash
git clone https://github.com/alibaba/ROLL.git
cd ROLL
```

### 2. 构建镜像

根据你的硬件选择对应的 Dockerfile：

**Atlas 900 A2 PODc（Ascend 910B1）：**

```bash
docker build -f docker/Dockerfile.A2 -t roll:ascend-a2 .
```

**Atlas 900 A3 PODc（Ascend 910_9391）：**

```bash
docker build -f docker/Dockerfile.A3 -t roll:ascend-a3 .
```

> **注意：** 构建过程会从源码编译 vLLM 和 vLLM-Ascend，耗时较长，请确保有足够的磁盘空间（至少 50GB）和网络访问。

你也可以在构建时自定义 SOC 版本：

```bash
# A2 自定义 SOC 版本
docker build -f docker/Dockerfile.A2 --build-arg SOC_VERSION=ascend910b1 -t roll:ascend-a2 .

# A3 自定义 SOC 版本
docker build -f docker/Dockerfile.A3 --build-arg SOC_VERSION=ascend910_9391 -t roll:ascend-a3 .
```

## 运行容器

### 基本启动

**A2：**

```bash
docker run -dit \
    --name roll_a2 \
    --device /dev/davinci0 \
    --device /dev/davinci1 \
    --device /dev/davinci2 \
    --device /dev/davinci3 \
    --device /dev/davinci4 \
    --device /dev/davinci5 \
    --device /dev/davinci6 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/Ascend/add-ons:/usr/local/Ascend/add-ons \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /home/$USER:/home/$USER \
    --ipc=host \
    --net=host \
    roll:ascend-a2 \
    /bin/bash
```

**A3：**

```bash
docker run -dit \
    --name roll_a3 \
    --device /dev/davinci0 \
    --device /dev/davinci1 \
    --device /dev/davinci2 \
    --device /dev/davinci3 \
    --device /dev/davinci4 \
    --device /dev/davinci5 \
    --device /dev/davinci6 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/Ascend/add-ons:/usr/local/Ascend/add-ons \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /home/$USER:/home/$USER \
    --ipc=host \
    --net=host \
    roll:ascend-a3 \
    /bin/bash
```

### 多卡启动（训练推荐）

多 NPU 训练时，需要挂载所有可用的 NPU 设备。根据节点上的 NPU 数量调整 `--device /dev/davinciX` 的数量：

```bash
docker run -dit \
    --name roll_ascend \
    --device /dev/davinci0 \
    --device /dev/davinci1 \
    --device /dev/davinci2 \
    --device /dev/davinci3 \
    --device /dev/davinci4 \
    --device /dev/davinci5 \
    --device /dev/davinci6 \
    --device /dev/davinci7 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/Ascend/add-ons:/usr/local/Ascend/add-ons \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /home/$USER:/home/$USER \
    -v /path/to/models:/path/to/models \
    -v /path/to/data:/path/to/data \
    --ipc=host \
    --net=host \
    roll:ascend-a3 \
    /bin/bash
```

> **注意：**
> - `--device /dev/davinciX`：挂载 NPU 设备，根据可用 NPU 数量增减。
> - `--device /dev/davinci_manager`、`--device /dev/devmm_svm`、`--device /dev/hisi_hdc`：昇腾 NPU 必需的管理设备。
> - `-v /usr/local/Ascend/driver`：挂载宿主机昇腾驱动。
> - `-v /path/to/models` 和 `-v /path/to/data`：根据需要挂载模型权重和训练数据目录。

### 进入容器

```bash
# A2
docker exec -it roll_a2 /bin/bash

# A3
docker exec -it roll_a3 /bin/bash
```

## 验证环境

进入容器后，验证昇腾环境是否正确配置：

```bash
# 验证 NPU 可见性
npu-smi info

# 验证 CANN 环境已加载
env | grep -E "ASCEND|LD_LIBRARY_PATH|PATH"

# 验证 Python 包
python -c "import torch; import torch_npu; print(torch_npu.npu.is_available())"
python -c "import vllm; print(f'vllm: {vllm.__version__}')"
python -c "import vllm_ascend; print(f'vllm_ascend available')"
```

## 运行 ROLL 流水线

### 重要配置说明

由于昇腾 NPU 上暂不支持 Megatron-LM 训练，需要使用 **DeepSpeed** 作为训练后端。请确保配置文件中使用以下设置：

1. 将 `strategy_args` 设置为使用 DeepSpeed
2. 设置 `device_mapping`，确保训练和推理在不同的 NPU 卡上执行


### 示例：RLVR 流水线

```bash
python examples/start_rlvr_pipeline.py \
    --config_path qwen2.5-7B-rlvr_megatron \
    --config_name rlvr_config_amd
```

> **注意：** `rlvr_config_amd` 配置专为非 NVIDIA 硬件设计，使用 DeepSpeed 作为训练后端。请根据你的 NPU 拓扑调整配置文件中的 `device_mapping`。

## 常见问题

### 容器内 NPU 不可见

确保所有必需的设备和管理路径已正确挂载。在容器内使用 `npu-smi info` 检查。

### vLLM-Ascend 导入错误

验证 CANN 环境是否正确加载：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
```

这些命令在镜像构建时已自动添加到 `/root/.bashrc`。如果切换到非 root 用户，可能需要手动执行。

### 显存不足

在配置文件中减小 `rollout_batch_size` 或 `num_return_sequences_in_group` 以降低 NPU 显存占用。

## 声明

ROLL 中提供的 Ascend 支持代码皆为参考样例，生产环境使用请通过官方正式途径沟通。
