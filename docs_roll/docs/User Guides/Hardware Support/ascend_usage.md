# ROLL x Ascend

Last updated: 11/25/2025.

We have added support for Huawei Ascend devices in ROLL.

## Hardware Support 

Atlas 900 A2 PODc

## Installation

### Basic Environment Setup

| Software | Version |
| -------- | ------- |
| Python   | 3.11    |
| CANN     | 8.3.RC1 |

### Create Conda Environment

Use the following commands to create a new conda environment in Miniconda:

```
conda create --name roll python=3.11
conda activate roll
```

### Install torch & torch_npu

To use torch and torch_npu in ROLL, install them using the commands below:

```
# Use CPU only torch
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cpu

# Install torch_npu 2.7.1
pip install torch_npu==2.7.1
```

### Install vllm & vllm-ascend

To use vllm in ROLL, compile and install vllm and vllm-ascend as follows:

```
# vllm
git clone -b v0.11.0 --depth 1 https://github.com/vllm-project/vllm.git
cd vllm
pip install -r requirements/build.txt

VLLM_TARGET_DEVICE=empty pip install -v -e .
cd ..

# vllm-ascend
git clone -b v0.11.0rc1 --depth 1 https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend

pip install -e .
cd ..
```

Or you could install `vllm` and `vllm-ascend` from pre-built wheel:
```
# Install vllm-project/vllm. The newest supported version is v0.11.0.
pip install vllm==0.11.0

# Install vllm-project/vllm-ascend from pypi.
pip install vllm-ascend==0.11.0rc1
```

### Install ROLL

```
git clone https://github.com/alibaba/ROLL.git
cd ROLL
pip install -r requirements_common.txt
pip install deepspeed==0.16.4
cd ..
```

### Additional Third-Party Libraries

| Software                    | Description   |
| --------------------------- | ------------- |
| transformers                | >= v4.57.1    |
| flash_attn                  | not supported |
| transformer-engine[pytorch] | not supported |

1. `transformers` v4.57.1 supports enabling `--flash_attention_2`.
2. `flash_attn` acceleration is not supported currently.
3. `transformer-engine[pytorch]` is currently not supported.

```
pip install transformers==4.57.1
```

## Quick Start: Single-Node Deployment

Before full usage, we recommend testing the single-node pipeline to verify your environment and installation.
Since Megatron-LM training is not yet supported, first change `strategy_args` in the relevant files to use the `deepspeed` option.

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
| Agentic         | examples/qwen2.5-0.5B-agentic/run_agentic_pipeline_sokoban.sh | DeepSpeed        | vLLM              | Atlas 900 A2 PODc |
| Agentic-Rollout | examples/qwen2.5-0.5B-agentic/run_agentic_rollout_sokoban.sh | DeepSpeed        | vLLM              | Atlas 900 A2 PODc |
| DPO             | examples/qwen2.5-3B-dpo_megatron/run_dpo_pipeline.sh         | DeepSpeed        | vLLM              | Atlas 900 A2 PODc |
| RLVR            | examples/qwen2.5-7B-rlvr_megatron/run_rlvr_pipeline.sh       | DeepSpeed        | vLLM              | Atlas 900 A2 PODc |

## Disclaimer

The Ascend support provided in ROLL is intended as a reference example. For production use, please consult official channels.