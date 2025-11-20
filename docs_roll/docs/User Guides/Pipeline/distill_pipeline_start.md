# Distill Pipeline

**Table of Contents**

- [Distill Pipeline](#distill-pipeline)
  - [✨️Overview](#️overview)
  - [✨️Core Components](#️core-components)
    - [Main Module (`DistillPipeline`)](#main-module-distillpipeline)
    - [Configuration File (`DistillConfig`)](#configuration-file-distillconfig)
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

* **Various distillation losses**: Support for training the model with different distillation losses and finer-grained configuration via the corresponding parameters.

* **Comprehensive Performance Monitoring**: Fine-grained metric tracking system that monitors performance metrics, providing comprehensive visualization and analysis capabilities for the model training process.

* **Efficient Distributed Computing**: Leverages the [Ray](https://www.ray.io/) framework to implement efficient distributed training on large-scale GPU clusters, significantly improving training speed and resource utilization.

---



## ✨️Core Components

### Main Module (`DistillPipeline`)

`DistillPipeline` (located in `roll/pipeline/distill/distill_pipeline.py`) is the primary coordinator for the entire distill training process. It manages the complete training workflow, including:

* Initializing and managing distributed workers (Student and Teacher workers).
* Coordinating data collection and processing.
* Executing model training steps.
* Handling checkpoint saving.
* Recording metrics and experiment tracking.

**Source code**: `roll/pipeline/distill/distill_pipeline.py`

---

### Configuration File (`DistillConfig`)

`DistillConfig` (defined in `roll/pipeline/distill/distill_config.py`) is a Pydantic/dataclass-based configuration object used to specify all parameters for running the distill pipeline. This configuration system is flexibly designed, supporting configuration via YAML files and managed using the Hydra framework.

#### Configuration File Structure and Organization

Configuration files (such as `examples/qwen2.5-7B-distill_megatron/distill_megatron.yaml`) are organized by functional modules, containing the following main sections:

1. **Experiment Basic Settings**
   * `exp_name`: Experiment name, used to identify a specific training run
   * `logging_dir`: Path for saving log files
   * `output_dir`: Path for saving model checkpoints and output files

2. **Training Control Parameters**
   * `save_steps`: Frequency for saving model checkpoints
   * `logging_steps`: Frequency for recording training metrics
   * `resume_from_checkpoint`: Whether to continue training from a checkpoint. Set it to the checkpoint path if you want to resume; otherwise, set it to `False`.

3. **Model Configuration**
   * `student_pretrain`: Path to pre-trained weights for Student model
   * `teacher_pretrain`: Path to pre-trained weights for Teacher model

4. **Distill Algorithm Parameters**
   * `distill_loss_weight`: Fraction of the total loss assigned to the distillation term (SFT loss weight is 1 − this value).  
   * `kd_temperature`: Soft-max temperature applied to the student logits during knowledge distillation.  
   * `teacher_temperature`: Temperature applied to the teacher logits to control their softness.  
   * `kd_objective`: Divergence measure used to compare student and teacher distributions (e.g., `forward_kl`, `reverse_kl`).  
   * `adaptive_kl_alpha`: Weighting factor that blends forward and reverse KL when `kd_objective` is `adaptive_kl`.  
   * `skew_lambda`: Skewing coefficient applied in `skewed_forward_kl` or `skewed_reverse_kl` objectives.

5. **Logits Transfer Configuration**
   * `logits_transfer_backend`: Backend used for logits transfer. Supports three modes: 'ipc+nccl', 'nccl-only', and 'ray'.  
'ipc+nccl' uses CUDA IPC to transfer logits directly via shared GPU memory when running on the same device.  
If the device does not support IPC, use either 'nccl-only' or 'ray' mode instead.

6. **Worker Configuration**
   Each worker (`student`, `teacher`) configuration contains:

   * **Model Parameters** (`model_args`)
     * `model_type`: Model type (e.g., `causal_lm`)
     * `dtype`: Computation precision (e.g., `bf16`, `fp16`)
     * ...
   * **Training Parameters** (`training_args`)
     * `num_train_epochs`: Num of training epochs
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

The distill pipeline expects the training data to be stored in **JSON** files.

### Required Columns

Each data sample must contain a question and its corresponding answer.  
In the YAML file, use the keys `question_key` and `answer_key` to specify the field names that hold these two pieces of data.

---


## ✨️Running the Pipeline

### Method 1: Using Python Launcher Script

The primary method is to use the `examples/start_distill_pipeline.py` script. This script uses Hydra to load and manage configurations.

1. **Select or Create a Configuration File**  
   Start with an example YAML (e.g., `examples/qwen2.5-7B-distill_megatron/distill_megatron.yaml`) or create your own configuration.

2. **Execute the Python Launcher Script**

   ```bash
   # Make sure you are in the root directory of the ROLL project
   # export PYTHONPATH=$(pwd):$PYTHONPATH
   
   python examples/start_distill_pipeline.py \
          --config_path examples/qwen2.5-7B-distill_megatron \
          --config_name distill_megatron
   ```

   * `--config_path` – Directory containing your YAML configuration.
   * `--config_name` – Filename (without `.yaml`).



### Method 2: Using Helper Shell Scripts

The `examples` directory typically contains shell scripts that wrap the Python launcher.

Example structure:

```bash
#!/bin/bash
# Example: examples/qwen2.5-7B-distill_megatron/run_distill_pipeline.sh

CONFIG_NAME="distill_megatron"                         # distill_megatron.yaml
CONFIG_PATH="examples/qwen2.5-7B-distill_megatron"

# Set environment variables and other configurations

python examples/start_distill_pipeline.py \
       --config_path $CONFIG_PATH \
       --config_name $CONFIG_NAME \
       "$@"   # Pass any additional parameters
```

Run using:

```bash
bash examples/qwen2.5-7B-distill_megatron/run_distill_pipeline.sh
```

---



## ✨️Step-by-Step Example



### Step 1: Configure Settings

* File: `examples/qwen2.5-7B-distill_megatron/distill_megatron.yaml`  
  Key sections include `exp_name`, `seed`, `output_dir`, model paths, `student` and `teacher` configurations.

* Pay special attention to these configuration sections:
  * Data configuration: `student.data_args.file_name`
  * Model configuration: `student_pretrain` and `teacher_pretrain` paths (The distill pipeline currently only supports student and teacher models of the same type, for example, both the student and teacher models are Qwen.)
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
python examples/start_distill_pipeline.py \
       --config_path examples/qwen2.5-7B-distill_megatron \
       --config_name distill_megatron
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
