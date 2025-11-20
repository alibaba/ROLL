# DPO Pipeline

**Table of Contents**

- [DPO Pipeline](#DPO-pipeline)
  - [✨️Overview](#️overview)
  - [✨️Core Components](#️core-components)
    - [Main Module (`DPOPipeline`)](#main-module-DPOPipeline)
    - [Configuration File (`DPOConfig`)](#configuration-file-DPOConfig)
      - [Configuration File Structure and Organization](#configuration-file-structure-and-organization)
  - [✨️Data Preparation](#️data-preparation)
    - [Data Format](#data-format)
      - [Common Data Fields](#common-data-fields)
  - [✨️Running the Pipeline](#️running-the-pipeline)
    - [Method 1: Using Python Launcher Script](#method-1-using-python-launcher-script)
    - [Method 2: Using Helper Shell Scripts](#method-2-using-helper-shell-scripts)
  - [✨️Step-by-Step Example](#️step-by-step-example)
    - [Step 1: Configure Settings](#step-1-configure-settings)
    - [Step 2: Prepare Environment and Dependencies](#step-2-prepare-environment-and-dependencies)
    - [Step 3: Launch the Pipeline](#step-3-launch-the-pipeline)
    - [Step 4: Monitoring](#step-4-monitoring)
    - [Step 5: Outputs and Results](#step-5-outputs-and-results)

---



## ✨️Overview

 This pipeline offers the following core advantages:

* **Various DPO losses**: Support for training the model with different DPO losses and finer-grained configuration via the corresponding parameters.

* **Comprehensive Performance Monitoring**: Fine-grained metric tracking system that monitors performance metrics, providing comprehensive visualization and analysis capabilities for the model training process.

* **Efficient Distributed Computing**: Leverages the [Ray](https://www.ray.io/) framework to implement efficient distributed training on large-scale GPU clusters, significantly improving training speed and resource utilization.

---



## ✨️Core Components

### Main Module (`DPOPipeline`)

`DPOPipeline` (located in `roll/pipeline/dpo/dpo_pipeline.py`) is the primary coordinator for the entire DPO training process. It manages the complete training workflow, including:

* Initializing and managing distributed workers (Actor and Reference workers).
* Coordinating data collection and processing.
* Executing model training steps.
* Handling checkpoint saving.
* Recording metrics and experiment tracking.

**Source code**: `roll/pipeline/dpo/dpo_pipeline.py`

---

### Configuration File (`DPOConfig`)

`DPOConfig` (defined in `roll/pipeline/dpo/dpo_config.py`) is a Pydantic/dataclass-based configuration object used to specify all parameters for running the DPOPipeline. This configuration system is flexibly designed, supporting configuration via YAML files and managed using the Hydra framework.

#### Configuration File Structure and Organization

Configuration files (such as `examples/qwen2.5-3B-dpo_megatron/dpo_config.yaml`) are organized by functional modules, containing the following main sections:

1. **Experiment Basic Settings**
   * `exp_name`: Experiment name, used to identify a specific training run
   * `logging_dir`: Path for saving log files
   * `output_dir`: Path for saving model checkpoints and output files

2. **Training Control Parameters**
   * `max_steps`: Maximum number of training steps
   * `save_steps`: Frequency for saving model checkpoints
   * `logging_steps`: Frequency for recording training metrics
   * `resume_from_checkpoint`: Whether to continue training from a checkpoint. Set it to the checkpoint path if you want to resume; otherwise, set it to `False`.

3. **DPO Algorithm Parameters**
   * `ipo`: Use IPO loss function
   * `beta`: Regulates the model's sensitivity to human preference data
   * `label_smoothing`: A regularization technique that reduces overfitting risk by softening the model's absolute confidence in labels

5. **Worker Configuration**
   Each worker (`actor_train`, `reference`) configuration contains:

   * **Model Parameters** (`model_args`)
     * `model_type`: Model type (e.g., `causal_lm`)
     * `dtype`: Computation precision (e.g., `bf16`, `fp16`)
     * ...
   * **Training Parameters** (`training_args`)
     * `learning_rate`: Learning rate
     * `per_device_train_batch_size`: Training batch size per device
     * `gradient_accumulation_steps`: Gradient accumulation steps
     * `weight_decay`: Weight decay coefficient
     * `max_grad_norm`: Gradient clipping threshold
     * ...
   * **Distributed Strategy** (`strategy_args`)
     * `strategy_name`: Distributed strategy to use (e.g., `megatron_train`, `deepspeed_infer`)
     * Strategy-specific parameters: e.g., `tp_size` (tensor parallelism size), `pp_size` (pipeline parallelism size)
     * `gpu_memory_utilization`: GPU memory utilization (vLLM-specific)
   * **Device Mapping** (`device_mapping`)
     * Specifies which GPU devices the worker should use

---


## ✨️Data Preparation

### Data Format

The DPO pipeline expects the training data to be stored in **JSON** files.

### Required Columns

Each data sample must contain a question, a chosen answer, and a rejected answer.
In the YAML file, use chosen_key and rejected_key to specify the corresponding field names in the dataset.

For example:
```
"instruction": "Select a color and provide some adjectives to describe it.",
"input": "",
"chosen": "The color is blue. Adjectives to describe it include calming, soothing, serene, and peaceful.",
"rejected": "Red"
```

---


## ✨️Running the Pipeline

### Method 1: Using Python Launcher Script

The primary method uses the examples/start_dpo_pipeline.py script. This script leverages Hydra to load and manage configurations.

1. **Select or Create a Configuration File**  
   Start with an example YAML (e.g., `examples/qwen2.5-3B-dpo_megatron/dpo_config.yaml`) or create your own configuration.

2. **Execute the Python Launcher Script**

  ```bash
   # Make sure you are in the root directory of the ROLL project
   # export PYTHONPATH=$(pwd):$PYTHONPATH
   
   python examples/start_dpo_pipeline.py \
          --config_path examples/qwen2.5-3B-dpo_megatron \
          --config_name dpo_config
   ```

   * `--config_path` – Directory containing your YAML configuration.
   * `--config_name` – Filename (without `.yaml`).



### Method 2: Using Helper Shell Scripts

The `examples` directory typically contains shell scripts that wrap the Python launcher.

Example structure:

```bash
#!/bin/bash
# Example: examples/qwen2.5-3B-dpo_megatron/run_dpo_pipeline.sh

CONFIG_NAME="dpo_config"
CONFIG_PATH="examples/qwen2.5-3B-dpo_megatron"

# Set environment variables and other configurations

python examples/start_dpo_pipeline.py \
       --config_path $CONFIG_PATH \
       --config_name $CONFIG_NAME \
       "$@"   # Pass any additional arguments
```

Run using:

```bash
bash bash examples/qwen2.5-3B-dpo_megatron/run_dpo_pipeline.sh
```

---



## ✨️Step-by-Step Example



### Step 1: Configure Settings

* File: `examples/qwen2.5-3B-dpo_megatron/dpo_config.yaml`  
  Key sections include exp_name, seed, output_dir, model paths, and configurations for actor_train and reference.

* Pay special attention to these configuration sections:
  * Data configuration: `actor_train.data_args.file_name`
  * Model configuration: `pretrain` path
  * Distributed strategies: `strategy_args` and `device_mapping` for each worker

### Step 2: Prepare Environment and Dependencies

* Ensure all necessary dependencies are installed:

  ```bash
  pip install -r requirements.txt
  ```

* Verify that all model paths in the configuration are accessible.

* Prepare training datasets, ensuring they conform to the data format requirements described above.

### Step 3: Launch the Pipeline

```bash
python examples/start_dpo_pipeline.py \
      --config_path examples/qwen2.5-3B-dpo_megatron \
      --config_name dpo_config
```

### Step 4: Monitoring

* **Console Output** – Observe Hydra, Ray, and pipeline logs.
* **Log Files** – Check the `logging_dir` specified in the YAML.
* **TensorBoard**

  ```bash
  tensorboard --logdir <your_log_dir>
  ```

### Step 5: Outputs and Results

* **Trained Models** – Checkpoints are saved in the `output_dir`.
* **Evaluation Metrics** – Recorded in TensorBoard and the console.

---

*Happy experimenting!*
