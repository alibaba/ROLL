# Running ROLL on Ascend NPU with Docker

Last updated: 04/27/2026.

This guide explains how to build and run ROLL on **Huawei Ascend NPU** using `Dockerfile.A2` and `Dockerfile.A3`.

## Hardware & Software Requirements

| Item | Dockerfile.A2 | Dockerfile.A3 |
| ---- | ------------- | ------------- |
| Hardware | Atlas 900 A2 PODc (Ascend 910B1) | Atlas 900 A3 PODc (Ascend 910_9391) |
| Host OS | Ubuntu 22.04 | Ubuntu 22.04 |
| CANN | 8.5.1 | 8.5.1 |
| Python | 3.11 | 3.11 |
| Docker | >= 20.10 | >= 20.10 |
| Ascend NPU Driver | Installed on host | Installed on host |

## Key Components

Both Dockerfiles install the same versions of core dependencies:

| Component | Version |
| --------- | ------- |
| PyTorch | 2.8.0+cpu |
| vLLM | 0.13.0 |
| vLLM-Ascend | 0.13.0 |
| DeepSpeed | 0.16.4 |
| Transformers | 4.57.6 |
| triton-ascend | 3.2.0 |

The primary difference is the base image and SOC version:

| Item | Dockerfile.A2 | Dockerfile.A3 |
| ---- | ------------- | ------------- |
| Base Image | `quay.io/ascend/cann:8.5.1-910b-ubuntu22.04-py3.11` | `quay.io/ascend/cann:8.5.1-a3-ubuntu22.04-py3.11` |
| SOC_VERSION | `ascend910b1` | `ascend910_9391` |

## Build the Docker Image

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
python -c "import torch; import torch_npu; print(torch_npu.npu.is_available())"
python -c "import vllm; print(f'vllm: {vllm.__version__}')"
python -c "import vllm_ascend; print(f'vllm_ascend available')"
```

## Run ROLL Pipelines

### Important Configuration Notes

Since Megatron-LM training is not yet supported on Ascend NPU, you need to use **DeepSpeed** as the training backend. Make sure your configuration files use the following settings:

1. Set `strategy_args` to use DeepSpeed
2. Set `device_mapping` to ensure training and inference are performed on different NPUs

### Example: RLVR Pipeline

```bash
# After modifying model paths and adjusting device_mapping
python examples/start_rlvr_pipeline.py \
    --config_path ascend_examples \
    --config_name qwen3_8b_rlvr_deepspeed
```

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
