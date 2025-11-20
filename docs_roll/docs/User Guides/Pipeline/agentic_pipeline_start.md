# Agentic Pipeline

**Table of Contents**

- [Agentic Pipeline](#agentic-pipeline)
  - [✨️ Overview](#️-overview)
  - [✨️ Core Components](#️-core-components)
    - [Main Module (`AgenticPipeline`)](#main-module-agenticpipeline)
    - [Configuration File (`AgenticConfig`)](#configuration-file-agenticconfig)
      - [Configuration Structure and Organization](#configuration-structure-and-organization)
  - [✨️ Environment Preparation](#️-environment-preparation)
    - [Environment Types](#environment-types)
    - [Environment Configuration](#environment-configuration)
  - [✨️ Running the Pipeline](#️-running-the-pipeline)
    - [Method 1: Using Python Startup Script](#method-1-using-python-startup-script)
    - [Method 2: Using Helper Shell Script](#method-2-using-helper-shell-script)
  - [✨️ Step-by-Step Example](#️-step-by-step-example)
    - [Step 1: Configuration Setup](#step-1-configuration-setup)
    - [Step 2: Environment and Dependency Preparation](#step-2-environment-and-dependency-preparation)
    - [Step 3: Starting the Pipeline](#step-3-starting-the-pipeline)
    - [Step 4: Monitoring](#step-4-monitoring)
    - [Step 5: Output and Results](#step-5-output-and-results)

---

## ✨️ Overview

Agentic Pipeline is ROLL's core pipeline for agent training, supporting multiple algorithms such as PPO, GRPO, and more. It provides the following core advantages:

* **Gym-like Environment Definition**: Supports various environment types, including FrozenLake, Sokoban, etc., and can easily extend custom environments through gym-like interfaces.
* **Rich Learning Granularity**: Supports TrajectoryWise form (StarPO) and StepWise (GiGPO) training forms.
* **Asynchronous Parallel Rollout at Environment Granularity**: Independent trajectory sampling across environments improves sampling efficiency.
* **Asynchronous Training**: Decoupling of rollout/training supports asynchronous training.
* **Multi-turn Interaction Support for Local Debugging**: Multi-turn interaction rollout supports local debugging, improving development efficiency for multi-turn interaction business.
* **Flexible Policy Configuration**: Supports multiple distributed training strategies such as Megatron, DeepSpeed, vLLM, etc., allowing flexible configuration based on hardware resources.

---

## ✨️ Core Components

### Main Module (`AgenticPipeline`)

`AgenticPipeline` (located at `roll/pipeline/agentic/agentic_pipeline.py`) is the main process for the entire agent training. It manages the complete training workflow, including:

* Initializing and managing distributed worker processes (Actor, Critic, Reference, etc.).
* Coordinating environment interaction and data collection.
* Executing model training steps.
* Handling checkpoint saving.
* Recording metrics and experiment tracking.

**Source Code**: `roll/pipeline/agentic/agentic_pipeline.py`

---

### Configuration File (`AgenticConfig`)

`AgenticConfig` (defined in `roll/pipeline/agentic/agentic_config.py`) is a configuration object based on Pydantic/dataclass used to specify all parameters for running AgenticPipeline. This configuration system supports YAML file configuration and uses the Hydra framework for management.

For configuration system description, see [config_system](../Configuration/config_system.md)

#### Configuration Structure and Organization

Configuration files (such as `examples/qwen2.5-0.5B-agentic/agent_val_frozen_lake.yaml`) are organized by functional modules and mainly include the following sections:

1. **Basic Experiment Settings**
   * `exp_name`: Experiment name, used to identify a specific training task
   * `seed`: Random seed to ensure reproducible experiments
   * `logging_dir`: Path to save log files
   * `output_dir`: Path to save model checkpoints and output files
   * `render_save_dir`: Path to save rendered frames (for environment visualization)

2. **Training Control Parameters**
   * `max_steps`: Maximum training steps
   * `save_steps`: Frequency of saving model checkpoints
   * `logging_steps`: Frequency of recording training metrics
   * `eval_steps`: Frequency of performing validation evaluation
   * `resume_from_checkpoint`: Whether to resume training from a checkpoint. To continue training, set to its path; otherwise, set to `False`.

3. **Model Configuration**
   * `pretrain`: Pretrained model path
   * `reward_pretrain`: Reward model pretrained weights path

4. **Algorithm Parameters**
   * `adv_estimator`: Advantage estimator type (such as `gae`, `grpo`, `reinforce`)
   * `ppo_epochs`: Number of optimization epochs per sample batch
   * `gamma`: Discount factor for calculating returns
   * `lambd`: Lambda parameter in GAE
   * `pg_clip`: Clipping range for PPO policy gradient loss
   * `init_kl_coef`: Initial coefficient for KL penalty
   * `target_kl`: Target KL value for adaptive KL control
   * `whiten_advantages`: Whether to whiten advantages
   * `entropy_loss_coef`: Coefficient for entropy loss

5. **Worker Process Configuration**
   Each worker process (`actor_train`, `actor_infer`, `critic`, `reference`) configuration includes:

   * **Model Parameters** (`model_args`)
     * `model_type`: Model type (such as `causal_lm`)
     * `dtype`: Computation precision (such as `bf16`, `fp16`)
     * `attn_implementation`: Attention implementation (such as `fa2`)
     * `disable_gradient_checkpointing`: Whether to disable gradient checkpointing
   * **Training Parameters** (`training_args`)
     * `learning_rate`: Learning rate
     * `per_device_train_batch_size`: Training batch size per device
     * `gradient_accumulation_steps`: Gradient accumulation steps
     * `weight_decay`: Weight decay coefficient
     * `warmup_steps`: Learning rate warmup steps
     * `lr_scheduler_type`: Learning rate scheduler type
   * **Generation Parameters** (`generating_args`)
     * `max_new_tokens`: Maximum new tokens to generate
     * `top_p`: Nucleus sampling parameter
     * `temperature`: Temperature parameter
     * `num_return_sequences`: Number of return sequences
   * **Distributed Strategy** (`strategy_args`)
     * `strategy_name`: Distributed strategy used (such as `megatron_train`, `vllm`, `hf_infer`)
     * Strategy-specific parameters: such as `tp_size` (tensor parallel size), `pp_size` (pipeline parallel size)
     * `gpu_memory_utilization`: GPU memory utilization (specific to vLLM)
   * **Device Mapping** (`device_mapping`)
     * Specifies which GPU devices the worker process should use

6. **Environment Manager Configuration**
   * `train_env_manager`: Training environment manager configuration
   * `val_env_manager`: Validation environment manager configuration
   * Environment-related parameters:
     * `num_env_groups`: Number of environment groups
     * `group_size`: Number of environments per group
     * `tags`: List of environment tags
     * `num_groups_partition`: Group allocation for each environment type
     * `max_env_num_per_worker`: Maximum number of environments per worker

---

## ✨️ Environment Preparation

### Environment Types

Agentic Pipeline supports various environment types, including but not limited to:

* **FrozenLake**: Classic reinforcement learning environment where the agent needs to find a path to the goal on ice.
* **Sokoban**: Box-pushing game environment where the agent needs to push boxes to designated positions.
* **WebShop**: Simulated online shopping environment where the agent needs to find suitable products based on user requirements.
* More environment support...

### Environment Configuration

In the configuration file, custom environments are defined through the `custom_envs` field. Each environment configuration includes:

* `env_type`: Environment type
* `env_config`: Specific environment configuration parameters
* `max_tokens_per_step`: Maximum tokens per step

---

## ✨️ Running the Pipeline

### Method 1: Using Python Startup Script

The main method is to use the `examples/start_agentic_pipeline.py` script. This script uses Hydra to load and manage configurations.

1. **Select or Create a Configuration File**  
   Start with example YAML (such as `examples/qwen2.5-0.5B-agentic/agent_val_frozen_lake.yaml`) or create your own configuration.

2. **Execute the Python Startup Script**

   ```bash
   # Make sure you are in the ROLL project root directory
   # export PYTHONPATH=$(pwd):$PYTHONPATH
   
   python examples/start_agentic_pipeline.py \
          --config_path examples/qwen2.5-0.5B-agentic \
          --config_name agent_val_frozen_lake
   ```

   * `--config_path` – Directory containing the YAML configuration.
   * `--config_name` – File name (without `.yaml`).

### Method 2: Using Helper Shell Script

The `examples` directory typically contains shell scripts that wrap the Python launcher.

Example structure:

```bash
#!/bin/bash
# Example: examples/qwen2.5-0.5B-agentic/run_agentic_pipeline_frozen_lake.sh

CONFIG_PATH=$(basename $(dirname $0))
python examples/start_agentic_pipeline.py \
       --config_path $CONFIG_PATH \
       --config_name agent_val_frozen_lake
```

Running method:

```bash
bash examples/qwen2.5-0.5B-agentic/run_agentic_pipeline_frozen_lake.sh
```

---

## ✨️ Step-by-Step Example

### Step 1: Configuration Setup

* File: `examples/qwen2.5-0.5B-agentic/agent_val_frozen_lake.yaml`  
  Key sections include `exp_name`, `seed`, `output_dir`, model paths, and worker process configurations.

* Pay special attention to these configuration sections:
  * Model configuration: `pretrain` path
  * Algorithm parameters: `adv_estimator`, `ppo_epochs`, etc.
  * Distributed strategy: `strategy_args` and `device_mapping` for each worker process
  * Environment configuration: `train_env_manager` and `val_env_manager`

### Step 2: Environment and Dependency Preparation

* Ensure all necessary dependencies are installed, it's recommended to start from [image launch](../../Getting%20Started/Installation/installation.md):

  ```bash
  pip install -r requirements.txt
  ```

* Confirm all model paths in the configuration are accessible.

* Prepare the training environment and ensure support for the selected environment types.

### Step 3: Starting the Pipeline

```bash
python examples/start_agentic_pipeline.py \
       --config_path examples/qwen2.5-0.5B-agentic \
       --config_name agent_val_frozen_lake
```

### Step 4: Monitoring

* **Console Output** – Observe Hydra, Ray, and Pipeline logs.
* **Log Files** – Check the `logging_dir` specified in the YAML.
* **TensorBoard**

  ```bash
  tensorboard --logdir <your_log_dir>
  ```

### Step 5: Output and Results

* **Trained Model** – Checkpoints are saved in `checkpoint_config`, refer to documentation [checkpoint_and_resume](../Advanced%20Features/checkpoint_and_resume.md) for details.
* **Evaluation Metrics** – Recorded in TensorBoard and terminal.
* **Rendered Frames** – If `render_save_dir` is configured, environment rendered frames will be saved in that directory, facilitating visualization of the interaction process.

---

*Happy experimenting!*