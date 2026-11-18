# ROLL x Ascend

Last updated: 11/18/2026.

We have added support for Huawei Ascend devices in ROLL.

## Hardware Compatibility and Supported Operating Systems

ROLL's Ascend support is currently validated on training-series Ascend hardware:

| Product | Support status | Notes |
| ------- | -------------- | ----- |
| Atlas A2 training series | √ | Use `docker/Dockerfile.A2` or the `roll:v0.3-cann9.1.0-torch_npu2.10.0.post4-910b-ubuntu22.04-py3.12` image. |
| Atlas A3 training series | √ | Use `docker/Dockerfile.A3` or the `roll:v0.3-cann9.1.0-torch_npu2.10.0.post4-a3-ubuntu22.04-py3.12` image. |
| Ascend 950 training series | √ | Use the CANN 9.2.0 profile below: Python 3.12, torch 2.10.0, and source installations of vLLM 0.30.0 and vLLM-Ascend 0.30.0rc2. |
| Atlas A2/A3 inference series and Atlas 200I/500 A2 inference products | x | Current ROLL NPU images and examples target training-series devices. |
| Other Ascend training or inference products | Not validated | Validate the driver, firmware, CANN, `torch_npu`, and vLLM-Ascend versions before use. |

> In this table, `√` means supported by the current ROLL Ascend Dockerfiles/examples or the manual Ascend 950 installation profile, and `x` means not supported in the current ROLL NPU setup.

Supported operating systems:

| Deployment scenario | Supported OS | Notes |
| ------------------- | ------------ | ----- |
| Physical host | Ubuntu 22.04 | Recommended and validated by the current ROLL Ascend guides. |
| ROLL Ascend container | Ubuntu 22.04 | The A2/A3 Dockerfiles are based on `quay.io/ascend/cann:9.1.0-*-ubuntu22.04-py3.12-devel`. |
| Ascend 950 manual installation | Ubuntu 22.04 | Use the Ascend 950-specific torch/vLLM stack below. Keep the driver, firmware, CANN, and `torch_npu` versions aligned with the target Ascend 950 environment. |
| VM/container deployments on other host OS versions | Follow Ascend/CANN compatibility guidance | Check the Ascend compatibility query assistant and the CANN Software Installation OS compatibility notes for the target hardware. |

## Installation

### Basic Environment Setup

| Software | Version |
| -------- |---------|
| Python   | 3.12    |
| CANN     | 9.2.0   |

The manual installation steps below use CANN 9.2.0, vLLM 0.30.0 and vLLM-Ascend 0.30.0rc2. For Ascend 950, use the versions in [Ascend 950 Installation Profile](#ascend-950-installation-profile).

### Create Conda Environment

Use the following commands to create a new conda environment in Miniconda:

```
conda create --name roll python=3.12
conda activate roll
```

### Install torch & torch_npu

To use torch and torch_npu in ROLL, install them using the commands below:

```
# Use CPU-only torch when installing outside the pre-built image
pip install \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://mirrors.huaweicloud.com/repository/pypi/simple \
    torch==2.10.0+cpu torchvision==0.25.0 torchaudio==2.10.0

# Install the torch_npu version matching torch/CANN
python -m pip install \
    --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
    --no-deps torch-npu==2.10.0.post4
```

### Install vllm & vllm-ascend

Install both vLLM and vLLM-Ascend from the pinned source releases. Other dependencies can use pre-built packages. The following commands target A2/A3; for Ascend 950, use the source build in the next section. Prepare runtime dependencies from the release requirements first: these `--no-deps` commands reuse prepared dependencies and are not a complete installation into an empty environment. Load the CANN 9.2.0 development environment before compiling. Build vLLM with the `empty` target to avoid compiling CUDA kernels.

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

### Ascend 950 Installation Profile

Use the following version combination for Ascend 950DT (A5) with CANN 9.2.0. Build vLLM and vLLM-Ascend from their release tags, with the Ascend custom operators compiled for the target chip.

| Component | Version |
| --------- | ----------------- |
| Python / architecture | 3.12 / aarch64 |
| CANN | 9.2.0 |
| torch / torch-npu | 2.10.0 / 2.10.0.post4 |
| vLLM / vLLM-Ascend | 0.30.0 / 0.30.0rc2 (source installation) |
| triton-ascend | 3.2.2 |
| Transformers | 5.14.1 |
| FLA | flash-linear-attention-npu-a5 26.9.1 |
| Ray / NumPy / FastAPI | 2.48.0 / 1.26.4 / 0.136.3 |

Use a container with the matching driver, CANN 9.2.0 toolkit, chip-specific ops package, C/C++ compiler and torch stack already configured. Prepare runtime dependencies before the commands below. The `--no-deps` installs reuse that prepared runtime; they do not install all dependencies in an empty environment. The releases have conflicting FastAPI requirements, so the table is a runtime reference rather than a dependency lock that passes every `pip check`.

```bash
# vLLM: build the Python package without CUDA kernels.
git clone -b v0.30.0 --depth 1 https://github.com/vllm-project/vllm.git
cd vllm
python -m pip install "cmake>=3.26.1" ninja "setuptools>=77.0.3,<81" "setuptools-scm>=8" "setuptools-rust>=1.9.0" wheel packaging jinja2
VLLM_TARGET_DEVICE=empty VLLM_VERSION_OVERRIDE=0.30.0 \
    python -m pip install -v -e . --no-deps --no-build-isolation
cd ..

# vLLM-Ascend: build the extension and custom operators for A5.
git clone -b v0.30.0rc2 --depth 1 https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
git submodule update --init --recursive
python -m pip install "cmake>=3.26" "setuptools>=64" "setuptools-scm>=8" pybind11 wheel
SOC_VERSION=ascend950dt_9582 COMPILE_CUSTOM_KERNELS=1 \
    python -m pip install -v -e . --no-deps --no-build-isolation
cd ..

# Qwen3.5 GDN kernels: use the A5 package, not the A2 variant.
python -m pip install --no-deps \
    https://mirrors.huaweicloud.com/ascend/repos/pypi/flash-linear-attention-npu-a5/flash_linear_attention_npu_a5-26.9.1-py3-none-manylinux_2_34_aarch64.whl
```

After loading the matching CANN environment, set the custom operator paths **before starting Python/Ray**:

```bash
vllm_ascend_dir="$(python -c 'import importlib.util; print(importlib.util.find_spec("vllm_ascend").submodule_search_locations[0])')"
fla_npu_dir="$(python -c 'import importlib.util; print(importlib.util.find_spec("fla_npu").submodule_search_locations[0])')"
export ASCEND_CUSTOM_OPP_PATH="$vllm_ascend_dir/_cann_ops_custom/vendors/custom_transformer:$fla_npu_dir/opp/vendors/fla_npu_transformer:$fla_npu_dir/opp"
export LD_LIBRARY_PATH="$vllm_ascend_dir/_cann_ops_custom/vendors/custom_transformer/op_impl/ai_core/tbe/op_tiling:${LD_LIBRARY_PATH}"
export HCCL_NPU_SOCKET_PORT_RANGE=auto
export VLLM_ASCEND_ENABLE_NZ=0
```

Resolve operator paths from the installed package locations so editable source installations work as well. The `op_tiling` library path resolves `liboptiling.so`; omitting it previously caused `AddRmsNormBias` initialization to fail. The A5 tests also set `ATEN_CPU_CAPABILITY=default` for the driver and workers to avoid a slow CPU BF16 conversion path observed on that host. The example SOC is Ascend 950DT; adjust it for other A5 SKUs using `npu-smi info` and the upstream build guidance.

For colocated training and inference, set `ROLL_ASCEND_COLOCATED_IPC: '1'` in the configuration's `system_envs`, and use `sleep_level: 2` with `additional_config.rl_config.enabled: true` in the vLLM strategy. ROLL's NPU adapter uses the official layerwise weight-reload APIs; install a vLLM/vLLM-Ascend release pair that provides those APIs. No vLLM or vLLM-Ascend source edits are required.

Previous tests with these versions completed five-step text RLVR runs with full-layer Qwen3.5-9B and Qwen3.5-35B-A3B, with training, inference and reference sharing devices 0–7. Those tests used a pre-built vLLM-Ascend package; the source build documented here has not yet been retested end to end. These small-batch, short-sequence tests do not establish convergence, multimodal support or checkpoint export/resume. See the [RLVR guide](ascend_npu_rlvr.md) for the pipeline configuration.

### Install ROLL

```
git clone https://github.com/alibaba/ROLL.git
cd ROLL
pip install -r requirements_common.txt
pip install -e .
cd ..
```

### Additional Third-Party Libraries

Keep the following dependencies aligned with the CANN 9.2.0 / vLLM 0.30.0 profile. Check these versions after installing ROLL in case another dependency installation replaced them.

| Software                    | Description   |
| --------------------------- | ------------- |
| transformers                | 5.14.1    |
| torch-npu                   | 2.10.0.post4  |
| triton-ascend               | 3.2.2 (required instead of `triton`) |
| flash_attn                  | not supported |
| transformer-engine[pytorch] | not supported |

1. Pin `transformers==5.14.1` for this environment and select attention settings compatible with the model and NPU backend.
2. `flash_attn` acceleration is not supported currently.
3. `transformer-engine[pytorch]` is currently not supported.

```
pip install transformers==5.14.1
pip uninstall -y triton triton-ascend
pip install triton-ascend==3.2.2 --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi
```

A2 validation: official main (`7c0ded3`) plus this branch's minimal NPU adapter completed five text RLVR steps with Qwen3.5-9B on CANN 9.2.0, vLLM 0.30.0 and vLLM-Ascend 0.30.0rc2. The run used eight-card FSDP2, TP2 inference, four prompts with two responses each per step, a 512-token response limit, and disabled KL/reference. All five gradient norms were finite and nonzero; each inference worker completed five official layerwise reloads. This validates the short-sequence training flow; convergence and checkpoint export/resume were not tested.

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
| RLVR            | examples/ascend_examples/run_rlvr_pipeline.sh                | FSDP2            | vLLM              | Atlas 900 A2/A3/Ascend 950 training series |

## Disclaimer

The Ascend support provided in ROLL is intended as a reference example. For production use, please consult official channels.
