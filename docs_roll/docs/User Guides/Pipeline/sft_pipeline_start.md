# SFT Pipeline

**Table of Contents**

- [SFT Pipeline](#sft-pipeline)
  - [✨️ Overview](#️-overview)
  - [✨️ Core Components](#️-core-components)
    - [Main Module (`SFTPipeline`)](#main-module-sftpipeline)
    - [Worker (`SFTWorker`)](#worker-sftworker)
    - [Configuration (`SFTConfig`)](#configuration-sftconfig)
      - [Config Structure and Organization](#config-structure-and-organization)
  - [✨️ Data Preparation](#️-data-preparation)
    - [Data Format](#data-format)
      - [Required Fields and Field Mapping](#required-fields-and-field-mapping)
      - [Chat Template and Labels Rules](#chat-template-and-labels-rules)
    - [Validation Set (`validation`)](#validation-set-validation)
  - [✨️ Running the Pipeline](#️-running-the-pipeline)
    - [Method 1: Start with a Python Script](#method-1-start-with-a-python-script)
    - [Method 2: Use a Helper Shell Script](#method-2-use-a-helper-shell-script)
  - [✨️ Step-by-step Example](#️-step-by-step-example)
    - [Step 1: Configuration](#step-1-configuration)
    - [Step 2: Prepare Environment and Dependencies](#step-2-prepare-environment-and-dependencies)
    - [Step 3: Launch the Pipeline](#step-3-launch-the-pipeline)
    - [Step 4: Monitoring](#step-4-monitoring)
    - [Step 5: Outputs and Results](#step-5-outputs-and-results)

---

## ✨️ Overview

This pipeline is designed for Supervised Fine-Tuning (SFT) and provides:

- **Unified data encoding and chat templates**: Supports concatenating system/user/assistant chat formats and automatically constructs `labels` (loss is computed only on the answer portion).
- **Efficient distributed training**: Uses [Ray](https://www.ray.io/) plus a Cluster/Worker abstraction to launch distributed training.
- **Comprehensive performance monitoring**: A fine-grained metrics tracking system that monitors performance indicators and provides full visualization and analysis of the training process.
- **Efficient Training Optimization**: Supports **Sequence Packing** (concatenating multiple short samples into a continuous sequence to reduce padding). For configuration methods and implementation details, please refer to the dedicated documentation for `sequence packing`.
---

## ✨️ Core Components

### Main Module (`SFTPipeline`)

`SFTPipeline` (located at `roll/pipeline/sft/sft_pipeline.py`) is the main SFT training flow and is responsible for:

- Loading the tokenizer.
- Loading the training dataset and the (optional) validation dataset.
- Encoding data with templates to generate `input_ids` / `attention_mask` / `labels`.
- Initializing the distributed training cluster (`Cluster` + `SFTWorker`).
- Training loop: trains by step, evaluates every `eval_steps`, saves checkpoints according to the save policy, records metrics, and reports them to the tracker.

---

### Worker (`SFTWorker`)

`SFTWorker` (located at `roll/pipeline/sft/sft_worker.py`) executes training, evaluation, and checkpoint saving:

- `initialize()`: Creates and initializes the distributed strategy (`create_strategy`) and loads the model.
- `train_step()`: Runs one training step and returns training metrics.
- `val_step()`: Runs one validation step (forward + loss) and returns validation metrics.
- `do_checkpoint()`: Saves a checkpoint and returns metrics such as save time.

---

### Configuration (`SFTConfig`)

`SFTConfig` (defined in `roll/pipeline/sft/sft_config.py`) is the configuration object (dataclass-style) for the SFT pipeline, and supports YAML + Hydra management.

#### Config Structure and Organization

Example config file: `examples/qwen2.5-7B-sft_megatron/sft_config.yaml`

A typical config includes:

1. **Experiment basics**
   - `exp_name`: experiment name
   - `seed`: random seed
   - `logging_dir`: log directory
   - `output_dir`: checkpoint/output directory

2. **Training control parameters**
   - `save_steps`: checkpoint saving frequency
   - `logging_steps`: training metrics logging frequency
   - `eval_steps`: evaluation frequency (effective when a validation set is enabled)
   - `resume_from_checkpoint`: settings for resuming from a checkpoint

3. **Model configuration**
   - `pretrain`: path to the pretrained model

4. **Data field mapping (critical)**
   - `system_key`: system prompt field (optional)
   - `prompt_key`: prompt field name (default: `instruction`)
   - `query_key`: query field name (optional)
   - `response_key`: response field name (default: `output`)
   - `global_template`: global template name (optional; otherwise use `sft_train.data_args.template`)

5. **Worker configuration (`sft_train`)**  
   `sft_train` is a `WorkerConfig` and includes:

   - **Data args** (`data_args`)
     - `file_name`: training data JSON path (string or list)
     - `template`: template name (used when `global_template` is not set)
     - `preprocessing_num_workers`: number of preprocessing workers
   - **Training args** (`training_args`)
     - `num_train_epochs`
     - `learning_rate`
     - `per_device_train_batch_size`
     - `gradient_accumulation_steps`
     - `dataloader_num_workers`
     - ...
   - **Strategy args** (`strategy_args`)
     - `strategy_name`: e.g., `megatron_train` / `deepspeed_train`, etc.
     - Parallelism-related parameters (tensor/pipeline parallel sizes, etc.)
   - **Device mapping** (`device_mapping`)
     - Specifies which GPUs the worker uses
   - **Inference batch** (used in validation)
     - `infer_batch_size`: used during validation

6. **Validation configuration (optional)**
   - `validation.data_args.file_name`: validation data JSON path (validation is enabled only if set)

---

## ✨️ Data Preparation

### Data Format

The SFT pipeline uses **JSON** files loaded via HuggingFace Datasets.

#### Required Fields and Field Mapping

Each sample must be mappable to at least:

- Prompt: specified by `prompt_key` (default: `instruction`)
- Response: specified by `response_key` (default: `output`)

Optional fields:

- `system_key`: system prompt (optional)
- `query_key`: additional input (optional; appended to the user content)

#### Chat Template and Labels Rules

Chat structure:

- system (optional)
- user (prompt + query)
- assistant (response)

Labels construction:

- All tokens in the prompt portion are set to `IGNORE_INDEX` (not included in loss).
- Tokens in the response portion use real token ids (included in loss).

In other words: supervision is applied only to the model’s “answer portion”.

---

### Validation Set (`validation`)

The validation set is optional:

- It is loaded only if `validation.data_args.file_name` is configured.
- During training, validation is triggered according to `eval_steps`.
- Validation is executed by `sft_train.val_step` (no separate validation worker is launched).

---

## ✨️ Running the Pipeline

### Method 1: Start with a Python Script

Start with `examples/start_sft_pipeline.py`; Hydra loads the configuration:

```bash
# Make sure you are in the ROLL project root directory
# export PYTHONPATH=$(pwd):$PYTHONPATH

python examples/start_sft_pipeline.py \
       --config_path examples/qwen2.5-7B-sft_megatron \
       --config_name sft_config
```

- `--config_path` – config directory: `examples/qwen2.5-7B-sft_megatron`
- `--config_name` – config file name: `sft_config` (corresponds to `sft_config.yaml`)

---

### Method 2: Use a Helper Shell Script

Example:

```bash
#!/bin/bash
# Example: examples/qwen2.5-7B-sft_megatron/run_sft_pipeline.sh

CONFIG_NAME="sft_config"
CONFIG_PATH="examples/qwen2.5-7B-sft_megatron"

python examples/start_sft_pipeline.py \
       --config_path $CONFIG_PATH \
       --config_name $CONFIG_NAME \
       "$@"
```

Run:

```bash
bash examples/qwen2.5-7B-sft_megatron/run_sft_pipeline.sh
```

---

## ✨️ Step-by-step Example

### Step 1: Configuration

Config file: `examples/qwen2.5-7B-sft_megatron/sft_config.yaml`

Key items to check:

- **Data config**: `sft_train.data_args.file_name`
- **Field mapping**: `prompt_key/query_key/response_key/system_key`
- **Model config**: `pretrain`
- **Distributed strategy**: `sft_train.strategy_args` and `sft_train.device_mapping`
- **Validation config (optional)**: `validation.data_args.file_name` and `eval_steps`
- **Template selection**: `global_template` or `sft_train.data_args.template`

### Step 2: Prepare Environment and Dependencies

```bash
pip install -r requirements.txt
```

Also ensure:

- The `pretrain` path is accessible
- The fields in training/validation JSON match `prompt_key/response_key/...`

### Step 3: Launch the Pipeline

```bash
python examples/start_sft_pipeline.py \
       --config_path examples/qwen2.5-7B-sft_megatron \
       --config_name sft_config
```

### Step 4: Monitoring

- **Console output** – watch Hydra, Ray, and pipeline logs
- **Log files** – check `logging_dir`
- **TensorBoard**
  ```bash
  tensorboard --logdir <your_log_dir>
  ```

### Step 5: Outputs and Results

- **Trained model** – checkpoints are saved under `output_dir` with the default structure:

  ```
  <output_dir>/sft_train/checkpoint-<global_step>/<cluster_name>/
  ```

  Where:
  - `<global_step>`: current training step (e.g., `checkpoint-200`)
  - `<cluster_name>`: distributed cluster name (determined by Cluster/Ray runtime)

- **Training/validation metrics** – recorded in the terminal and tracker/TensorBoard (depending on tracker configuration)

---

*Happy experimenting!*