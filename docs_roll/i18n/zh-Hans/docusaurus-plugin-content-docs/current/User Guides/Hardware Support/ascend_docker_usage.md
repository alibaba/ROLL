# 使用 Docker 在昇腾 NPU 上运行 ROLL

最后更新：2026 年 8 月 18 日。

## 快速参考

- ROLL 仓库：[alibaba/ROLL](https://github.com/alibaba/ROLL)
- 昇腾使用指南：[ROLL x Ascend](ascend_usage.md)
- 昇腾 NPU 示例：[昇腾 NPU 示例](ascend_npu_examples.md)
- 可用的预构建镜像：[Quay 上的 ROLL 镜像](https://quay.io/repository/ascend/roll?tab=tags)
- Issue 跟踪器：[GitHub Issues](https://github.com/alibaba/ROLL/issues)

---

## ROLL NPU 镜像

ROLL 昇腾 NPU 镜像提供在华为昇腾训练系列 NPU 上运行 ROLL 所需的运行时和 Python 依赖。建议优先使用预构建镜像；如果需要自定义镜像或在本地重新构建，请使用 `Dockerfile.A2` 或 `Dockerfile.A3`。

当前 Dockerfile 覆盖 Atlas 900 A2 和 A3 训练系列设备。

---

## 支持的标签和 Dockerfile 链接

| 硬件 | 本地标签 | 预构建镜像 | Dockerfile | 基础镜像 |
|---|---|---|---|---|
| Atlas 900 A2 PODc / Ascend 910B1 | `roll:v0.3-cann9.1.0-torch_npu2.10.0.post4-910b-ubuntu22.04-py3.12` | `quay.io/ascend/roll:main-a2` | [`docker/Dockerfile.A2`](https://github.com/alibaba/ROLL/blob/main/docker/Dockerfile.A2) | `quay.io/ascend/cann:9.1.0-910b-ubuntu22.04-py3.12-devel` |
| Atlas 900 A3 PODc / Ascend 910_9391 | `roll:v0.3-cann9.1.0-torch_npu2.10.0.post4-a3-ubuntu22.04-py3.12` | `quay.io/ascend/roll:main-a3` | [`docker/Dockerfile.A3`](https://github.com/alibaba/ROLL/blob/main/docker/Dockerfile.A3) | `quay.io/ascend/cann:9.1.0-a3-ubuntu22.04-py3.12-devel` |

---

## 镜像内容

| 组件 | 版本 |
|---|---|
| CANN | 9.1.0 |
| Python | 3.12 |
| PyTorch | 2.10.0 |
| torch-npu | 2.10.0.post4 |
| vLLM | 0.23.0 |
| vLLM-Ascend | 0.23.0rc1 |
| Transformers | 4.57.6 |
| triton-ascend | 3.2.1 |

---

## 支持的硬件

| 硬件 | SOC_VERSION | Docker 支持 |
|---|---|---|
| Atlas 900 A2 PODc / Ascend 910B1 | `ascend910b1` | `Dockerfile.A2` 和 `roll:v0.3-cann9.1.0-torch_npu2.10.0.post4-910b-ubuntu22.04-py3.12` 支持 |
| Atlas 900 A3 PODc / Ascend 910_9391 | `ascend910_9391` | `Dockerfile.A3` 和 `roll:v0.3-cann9.1.0-torch_npu2.10.0.post4-a3-ubuntu22.04-py3.12` 支持 |

A2/A3 Docker 镜像的宿主机要求：

| 项目 | 要求 |
|---|---|
| 宿主机操作系统 | Ubuntu 22.04 |
| Docker | >= 20.10 |
| 昇腾 NPU 驱动 | 安装在宿主机上 |

---

## 快速开始

### 获取镜像

拉取与目标硬件匹配的镜像，并设置后续命令使用的本地标签：

Atlas 900 A2 PODc：

```bash
docker pull quay.io/ascend/roll:main-a2
docker tag quay.io/ascend/roll:main-a2 roll:v0.3-cann9.1.0-torch_npu2.10.0.post4-910b-ubuntu22.04-py3.12
```

Atlas 900 A3 PODc：

```bash
docker pull quay.io/ascend/roll:main-a3
docker tag quay.io/ascend/roll:main-a3 roll:v0.3-cann9.1.0-torch_npu2.10.0.post4-a3-ubuntu22.04-py3.12
```

### 构建镜像

从 Dockerfile 构建前，先克隆仓库：

```bash
git clone https://github.com/alibaba/ROLL.git
cd ROLL
```

构建 Atlas 900 A2 PODc 镜像：

```bash
docker build -f docker/Dockerfile.A2 -t roll:v0.3-cann9.1.0-torch_npu2.10.0.post4-910b-ubuntu22.04-py3.12 .
```

构建 Atlas 900 A3 PODc 镜像：

```bash
docker build -f docker/Dockerfile.A3 -t roll:v0.3-cann9.1.0-torch_npu2.10.0.post4-a3-ubuntu22.04-py3.12 .
```

构建过程会从源码编译 vLLM 和 vLLM-Ascend。请预留至少 50 GB 磁盘空间，并确保构建主机可以访问所需的软件包和源码仓库。

也可以显式覆盖 SOC 版本：

```bash
# A2
docker build -f docker/Dockerfile.A2 \
    --build-arg SOC_VERSION=ascend910b1 \
    -t roll:v0.3-cann9.1.0-torch_npu2.10.0.post4-910b-ubuntu22.04-py3.12 .

# A3
docker build -f docker/Dockerfile.A3 \
    --build-arg SOC_VERSION=ascend910_9391 \
    -t roll:v0.3-cann9.1.0-torch_npu2.10.0.post4-a3-ubuntu22.04-py3.12 .
```

---

## 运行容器

宿主机必须向容器暴露昇腾设备文件和驱动目录。下面的示例启动一个挂载 8 个 NPU 的 A2 容器：

```bash
docker run -dit \
    --name roll_a2 \
    --ulimit nofile=65536:65536 \
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
    roll:v0.3-cann9.1.0-torch_npu2.10.0.post4-910b-ubuntu22.04-py3.12 \
    /bin/bash
```

A3 使用 `roll:v0.3-cann9.1.0-torch_npu2.10.0.post4-a3-ubuntu22.04-py3.12`，并修改容器名称、增加可用的设备文件。完整的 16 卡 A3 节点还需要挂载 `/dev/davinci8` 到 `/dev/davinci15`：

```bash
docker run -dit \
    --name roll_a3 \
    --ulimit nofile=65536:65536 \
    --device /dev/davinci0 \
    --device /dev/davinci1 \
    --device /dev/davinci2 \
    --device /dev/davinci3 \
    --device /dev/davinci4 \
    --device /dev/davinci5 \
    --device /dev/davinci6 \
    --device /dev/davinci7 \
    --device /dev/davinci8 \
    --device /dev/davinci9 \
    --device /dev/davinci10 \
    --device /dev/davinci11 \
    --device /dev/davinci12 \
    --device /dev/davinci13 \
    --device /dev/davinci14 \
    --device /dev/davinci15 \
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
    roll:v0.3-cann9.1.0-torch_npu2.10.0.post4-a3-ubuntu22.04-py3.12 \
    /bin/bash
```

根据宿主机上的 NPU 数量调整 `/dev/davinciX` 条目。进行多 NPU 训练时，请挂载训练拓扑所需的全部设备。

进入运行中的容器：

```bash
# A2
docker exec -it roll_a2 /bin/bash

# A3
docker exec -it roll_a3 /bin/bash
```

---

## 验证环境

在容器内执行以下检查：

```bash
# 检查 NPU 是否可见
npu-smi info

# 检查 CANN 环境变量
env | grep -E "ASCEND|LD_LIBRARY_PATH|PATH"

# 检查 Python 软件包和 NPU 可用性
python -c "import torch; import torch_npu; print(torch.npu.is_available())"
python -c "import vllm; print(f'vllm: {vllm.__version__}')"
python -c "import vllm_ascend; print('vllm_ascend available')"
```

---

## 在昇腾 NPU 上运行 ROLL 流水线

仓库内置的 RLVR 示例采用 **FSDP2**，使用昇腾基础镜像中的依赖即可运行。A2/A3 上兼容的 Megatron 配置需要按照[在昇腾上安装 Megatron](ascend_usage.md#在昇腾上安装-megatron)补充可选依赖；A2/A3 Dockerfile 默认不会安装这些依赖。

运行内置 FSDP2 RLVR 示例时，请将 `strategy_args` 设置为 FSDP2。


### 示例：RLVR 流水线

```bash
python examples/start_rlvr_pipeline.py \
    --config_path examples/ascend_examples \
    --config_name qwen3_8b_rlvr_fsdp2
```

仓库还在 `examples/ascend_examples` 下提供了 `qwen3_30b_rlvr_fsdp2.yaml` 和 `run_rlvr_pipeline.sh`。请根据可用 NPU 的显存和拓扑选择配置。

---

## 注意事项

- 启动容器前，请在宿主机上安装兼容的昇腾 NPU 驱动。
- A2/A3 Docker 镜像基于 Ubuntu 22.04 和 Python 3.12。
- 如果无法导入 `vLLM-Ascend`，请在容器内重新加载昇腾环境：

  ```bash
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  source /usr/local/Ascend/nnal/atb/set_env.sh
  ```

- 镜像构建时会将这些命令写入 `/root/.bashrc`。如果切换到非 root 用户，必要时请手动执行这些命令。
- 如果 NPU 不可见，请检查挂载的设备文件、驱动路径和 `npu-smi info` 输出。

---

## 许可证

ROLL 基于 [Apache License 2.0](https://github.com/alibaba/ROLL/blob/main/LICENSE) 发布。

CANN、昇腾驱动组件、Python 软件包、系统库和其他预安装依赖可能分别受其各自许可证约束。
