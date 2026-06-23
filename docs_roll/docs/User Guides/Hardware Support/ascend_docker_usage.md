# Running ROLL on Ascend NPU with Docker

Last updated: 06/23/2026.

This guide explains how to get, build, and run ROLL images on **Huawei Ascend NPU**. Prefer the pre-built image when possible; use `Dockerfile.A2` or `Dockerfile.A3` when you need to customize dependencies. Atlas A5 currently follows the manual installation profile in [ROLL x Ascend](ascend_usage.md).

## Hardware & Software Requirements

| Item | Dockerfile.A2 | Dockerfile.A3 |
| ---- | ------------- | ------------- |
| Hardware | Atlas 900 A2 PODc (Ascend 910B1) | Atlas 900 A3 PODc (Ascend 910_9391) |
| Host OS | Ubuntu 22.04 | Ubuntu 22.04 |
| CANN | 9.0.0 | 9.0.0 |
| Python | 3.11 | 3.11 |
| Docker | >= 20.10 | >= 20.10 |
| Ascend NPU Driver | Installed on host | Installed on host |

This Docker guide covers the A2/A3 Dockerfiles. For Atlas A5, use the manual installation profile: torch 2.10, vLLM v0.20.2, vLLM-Ascend `main`, and `COMPILE_CUSTOM_KERNELS=1` when building vLLM-Ascend.

## Key Components

Both Dockerfiles install the same versions of core dependencies:

| Component | Version |
| --------- | ------- |
| PyTorch | 2.9.0+cpu |
| vLLM | 0.18.0 |
| vLLM-Ascend | 0.18 |
| Transformers | 4.57.6 |
| triton-ascend | 3.2.1 |

Atlas A5 uses a newer manual installation stack:

| Component | Atlas A5 Version / Setting |
| --------- | -------------------------- |
| PyTorch | 2.10 |
| vLLM | v0.20.2 |
| vLLM-Ascend | `main` branch |
| Required build variable | `COMPILE_CUSTOM_KERNELS=1` |

The primary difference is the base image and SOC version:

| Item | Dockerfile.A2 | Dockerfile.A3 |
| ---- | ------------- | ------------- |
| Base Image | `quay.io/ascend/cann:9.0.0-910b-ubuntu22.04-py3.11` | `quay.io/ascend/cann:9.0.0-a3-ubuntu22.04-py3.11` |
| SOC_VERSION | `ascend910b1` | `ascend910_9391` |

## Get the Docker Image

### Option A: Use the Pre-built Image (Recommended)

Pull the image that matches your hardware, then tag it with the local name used by the commands below:

**For Atlas 900 A2 PODc (Ascend 910B1):**

```bash
docker pull quay.io/ascend/roll:main-a2
docker tag quay.io/ascend/roll:main-a2 roll:ascend-a2
```

**For Atlas 900 A3 PODc (Ascend 910_9391):**

```bash
docker pull quay.io/ascend/roll:main-a3
docker tag quay.io/ascend/roll:main-a3 roll:ascend-a3
```

Check https://quay.io/repository/ascend/roll?tab=tags for available image tags. If you use a pre-built image, continue with [Run the Container](#run-the-container).

### Option B: Build from Dockerfile

### 1. Clone the ROLL Repository

```bash
git clone https://github.com/alibaba/ROLL.git
cd ROLL
```

### 2. Build the Image

Choose the Dockerfile that matches your hardware:

**For Atlas 900 A2 PODc (Ascend 910B1):**

```bash
docker build -f docker/Dockerfile.A2 -t roll:ascend-a2 .
```

**For Atlas 900 A3 PODc (Ascend 910_9391):**

```bash
docker build -f docker/Dockerfile.A3 -t roll:ascend-a3 .
```

> **Note:** The build process compiles vLLM and vLLM-Ascend from source, which may take a considerable amount of time. Please ensure sufficient disk space (at least 50GB) and network access.

You can also customize the SOC version at build time:

```bash
# A2 with custom SOC version
docker build -f docker/Dockerfile.A2 --build-arg SOC_VERSION=ascend910b1 -t roll:ascend-a2 .

# A3 with custom SOC version
docker build -f docker/Dockerfile.A3 --build-arg SOC_VERSION=ascend910_9391 -t roll:ascend-a3 .
```

## Run the Container

### Basic Startup

**For A2:**

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
    --ipc=host \
    --net=host \
    roll:ascend-a2 \
    /bin/bash
```

**For A3:**

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
    --ipc=host \
    --net=host \
    roll:ascend-a3 \
    /bin/bash
```

### Multi-NPU Startup (Recommended for Training)

For multi-NPU training, mount all available NPU devices. Adjust the number of `--device /dev/davinciX` entries according to the NPU count on your node:

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

> **Note:**
> - `--device /dev/davinciX`: Mounts NPU devices. Add or remove entries based on available NPU count.
> - `--device /dev/davinci_manager`, `--device /dev/devmm_svm`, `--device /dev/hisi_hdc`: Required management devices for Ascend NPU.
> - `-v /usr/local/Ascend/driver`: Mounts the host Ascend driver.
> - `-v /path/to/models` and `-v /path/to/data`: Mount model weights and training data directories as needed.

### Enter the Container

```bash
# For A2
docker exec -it roll_a2 /bin/bash

# For A3
docker exec -it roll_a3 /bin/bash
```

## Verify the Environment

After entering the container, verify that the Ascend environment is properly configured:

```bash
# Verify NPU visibility
npu-smi info

# Verify CANN environment is loaded
env | grep -E "ASCEND|LD_LIBRARY_PATH|PATH"

# Verify Python packages
python -c "import torch; import torch_npu; print(torch.npu.is_available())"
python -c "import vllm; print(f'vllm: {vllm.__version__}')"
python -c "import vllm_ascend; print(f'vllm_ascend available')"
```

## Run ROLL Pipelines

### Important Configuration Notes

Since Megatron-LM is not supported on Ascend NPU, you need to use **FSDP2** as the training backend. Make sure your configuration files use the following settings:

1. Set `strategy_args` to use FSDP2

### Example: RLVR Pipeline

```bash
# After modifying model paths and adjusting device_mapping
python examples/start_rlvr_pipeline.py \
    --config_path ascend_examples \
    --config_name qwen3_30b_rlvr_fsdp2
```

> **Note:** The `qwen3_30b_rlvr_fsdp2` configuration is specifically designed for Ascend NPU with FSDP2 as the training backend. Adjust `device_mapping` in the configuration file according to your NPU topology.

## Troubleshooting

### NPU Not Visible Inside Container

Ensure all required devices and driver paths are mounted correctly. Check with `npu-smi info` inside the container.

### vLLM-Ascend Import Error

Verify that the CANN environment is properly sourced:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
```

These commands are automatically added to `/root/.bashrc` during the image build. If you switch to a non-root user, you may need to source them manually.

### Out of Memory

Reduce `rollout_batch_size` or `num_return_sequences_in_group` in your configuration file to lower NPU memory usage.

## Disclaimer

The Ascend support provided in ROLL is intended as a reference example. For production use, please consult official channels.
