# ROLL x Ascend

Last updated: 06/23/2026.

We have added support for Huawei Ascend devices in ROLL.

## Hardware Compatibility and Supported Operating Systems

ROLL's Ascend support is currently validated on training-series Ascend hardware:

| Product | Support status | Notes |
| ------- | -------------- | ----- |
| Atlas 900 A2 PODc (Ascend 910B1) / Atlas A2 training series | √ | Use `docker/Dockerfile.A2` or the `roll:ascend-a2` image. |
| Atlas 900 A3 PODc (Ascend 910_9391) / Atlas A3 training series | √ | Use `docker/Dockerfile.A3` or the `roll:ascend-a3` image. |
| Atlas A5 training series | √ | Use the A5 installation profile: torch 2.10, vLLM v0.20.2, vLLM-Ascend `main`, and `COMPILE_CUSTOM_KERNELS=1` when building vLLM-Ascend. |
| Atlas A2/A3 inference series and Atlas 200I/500 A2 inference products | x | Current ROLL NPU images and examples target training-series devices. |
| Other Ascend training or inference products | Not validated | Validate the driver, firmware, CANN, `torch_npu`, and vLLM-Ascend versions before use. |

> In this table, `√` means supported by the current ROLL Ascend Dockerfiles/examples or the manual A5 installation profile, and `x` means not supported in the current ROLL NPU setup.

Supported operating systems:

| Deployment scenario | Supported OS | Notes |
| ------------------- | ------------ | ----- |
| Physical host | Ubuntu 22.04 | Recommended and validated by the current ROLL Ascend guides. |
| ROLL Ascend container | Ubuntu 22.04 | The A2/A3 Dockerfiles are based on `quay.io/ascend/cann:9.0.0-*-ubuntu22.04-py3.11`. |
| Atlas A5 manual installation | Ubuntu 22.04 | Use the A5-specific torch/vLLM stack below. Keep the driver, firmware, CANN, and `torch_npu` versions aligned with the target A5 environment. |
| VM/container deployments on other host OS versions | Follow Ascend/CANN compatibility guidance | Check the Ascend compatibility query assistant and the CANN Software Installation OS compatibility notes for the target hardware. |

## Installation

### Basic Environment Setup

| Software | Version |
| -------- |---------|
| Python   | 3.11    |
| CANN     | 9.0.0   |

For Atlas A5, keep Python 3.11 and use the A5-specific torch/vLLM stack described in [A5 Installation Profile](#a5-installation-profile).

### Create Conda Environment

Use the following commands to create a new conda environment in Miniconda:

```
conda create --name roll python=3.11
conda activate roll
```

### Install torch & torch_npu

To use torch and torch_npu in ROLL, install them using the commands below:

```
# Use CPU-only torch when installing outside the pre-built image
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cpu

# Install the torch_npu version matching torch/CANN
pip install torch_npu==2.9.0
```

### Install vllm & vllm-ascend

To use vllm in ROLL, compile and install vllm and vllm-ascend as follows:

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

Or you could install `vllm` and `vllm-ascend` from pre-built wheel:
```
# Install vllm-project/vllm. The newest supported version is v0.18.0.
pip install vllm==0.18.0

# Install vllm-project/vllm-ascend from pypi.
pip install vllm-ascend==0.18
```

### A5 Installation Profile

For Atlas A5, use torch 2.10, vLLM v0.20.2, and vLLM-Ascend from the `main` branch. Set `COMPILE_CUSTOM_KERNELS=1` before installing vLLM-Ascend so its custom kernels are built:

```
# Install torch 2.10
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cpu

# Install the torch_npu package matching torch 2.10 and your CANN release
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

### Install ROLL

```
git clone https://github.com/alibaba/ROLL.git
cd ROLL
pip install -r requirements_common.txt
pip install -e .
cd ..
```

### Additional Third-Party Libraries

| Software                    | Description   |
| --------------------------- | ------------- |
| transformers                | >= v4.57.6    |
| flash_attn                  | not supported |
| transformer-engine[pytorch] | not supported |

1. `transformers` v4.57.6 supports enabling `--flash_attention_2`.
2. `flash_attn` acceleration is not supported currently.
3. `transformer-engine[pytorch]` is currently not supported.

```
pip install transformers==4.57.6
```

## Quick Start: Single-Node Deployment

Before full usage, we recommend testing the single-node pipeline to verify your environment and installation.
Since Megatron-LM is not supported on NPU, first change `strategy_args` in the relevant files to use the `fsdp2` option.


1. Run the single-node pipeline via shell:

```
bash examples/agentic_demo/run_agentic_pipeline_frozen_lake_single_node_demo.sh  
```

2. Run the agentic pipeline using a config file:

```
# Make sure you are in the root directory of the ROLL project

python examples/start_agentic_pipeline.py \
        --config_path qwen2.5-0.5B-agentic \
        --config_name agentic_val_sokoban
```

- `--config_path` – Directory containing your YAML configuration files.
- `--config_name` – Filename (without the `.yaml` extension).

## Current Support Status

| Feature         | Example                                                      | Training Backend | Inference Backend | Hardware          |
| --------------- | ------------------------------------------------------------ | ---------------- | ----------------- | ----------------- |
| Agentic         | examples/qwen2.5-0.5B-agentic/run_agentic_pipeline_sokoban.sh | FSDP2            | vLLM              | Atlas 900 A2/A3 PODc |
| Agentic-Rollout | examples/qwen2.5-0.5B-agentic/run_agentic_rollout_sokoban.sh | FSDP2            | vLLM              | Atlas 900 A2/A3 PODc |
| RLVR            | examples/ascend_examples/run_rlvr_pipeline.sh                | FSDP2            | vLLM              | Atlas 900 A2/A3/A5 training series |

## Disclaimer

The Ascend support provided in ROLL is intended as a reference example. For production use, please consult official channels.
