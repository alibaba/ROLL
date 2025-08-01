# Comprehensive Guide: Using the Agentic Part of ROLL

**Table of Contents**

- [Overview](#overview)
- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Core Components](#core-components)
- [Running the Agentic Pipeline](#running-the-agentic-pipeline)
- [Step-by-Step Example](#step-by-step-example)
- [Troubleshooting](#troubleshooting)
- [Further Information](#further-information)

---

## Overview

The ROLL (Reinforcement Learning Optimization for Large-Scale Learning) agentic pipeline empowers you to:

* Define, configure, and simulate interactions between AI agents (often Large Language Models) and diverse environments.  
* Train these agents using reinforcement learning algorithms like Proximal Policy Optimization (PPO), GRPO, and **reinforce++**.  
* Evaluate agent performance on specific tasks and complex reasoning scenarios.  
* Leverage [Ray](https://www.ray.io/) for efficient, distributed computation across large-scale GPU setups.

This guide provides a step-by-step walkthrough for utilizing these agentic capabilities.

---

## Introduction

This tutorial guides you through setting up, configuring, and running the agentic components of the ROLL library. The agentic part of ROLL is specifically designed for building and training AI agents that learn through interaction. This is crucial for tasks like aligning LLMs with human preferences, teaching them complex reasoning, or improving their performance in multi-turn interaction scenarios.

We will cover the general workflow, which you can adapt for various environments and agent configurations based on your specific research or application needs.

---

## Prerequisites

Before you begin, ensure you have the following:

1. **ROLL Project Installed** – The ROLL project should be cloned and properly set up. Please refer to the main `README.md` in the project root for detailed installation instructions. Key requirements often include:  
   * CUDA Version >= 12.4  
   * cuDNN Version >= 9.1.0  
   * PyTorch >= 2.5.1  
   * vLLM >= 0.7.3  
   * SGlang >= 0.4.3  
   ROLL also provides Docker images for a quick start.

2. **Python Dependencies** – Install all necessary Python dependencies, typically via the requirements file:

   ```bash
   pip install -r requirements.txt   # Or a specific file like requirements_torch260.txt
   ```

   Ensure any specific dependencies for your chosen agentic environments or models are also met.

3. **Python Environment** – A working Python environment (python=3.10, or use our docker_images).

4. **Ray Setup (Recommended for Distributed Execution)**  
   * Install Ray:

     ```bash
     pip install ray
     ```

   * The project may contain example scripts for starting a Ray cluster, such as `examples/scripts/start_ray_cluster.sh`.

5. **Environment Variables** – Some scripts might require specific environment variables (e.g., `PYTHONPATH`). The provided execution scripts (like those in `examples`) often handle setting these.

---

## Core Components

### Agentic Pipeline (`AgenticPipeline`)

The main orchestrator for the agentic RL process. It manages the entire workflow, including:

* Initializing and managing distributed workers (actors, critics, environment managers).  
* Coordinating data collection (rollouts) where the agent interacts with the environment.  
* Executing model training steps (e.g., PPO updates for actor and critic).  
* Handling model synchronization and checkpointing.  
* Running evaluation loops.  
* Logging metrics and experiment tracking.

**Source Code**: `roll/pipeline/agentic/agentic_pipeline.py`

---

### Agentic Configuration (`AgenticConfig`)

The `AgenticConfig` (defined in `roll/pipeline/agentic/agentic_config.py`) is a Pydantic/dataclass-based configuration object that specifies all parameters for an agentic pipeline run. It is typically defined and populated from a YAML file using Hydra.

**Role**: Holds settings for environments, agent models (actor, critic, reference), reward mechanisms, training algorithms (e.g., PPO hyperparameters), rollout procedures, logging, and distributed execution strategies.

**Key Nested Configurations**

* `actor_train`, `actor_infer`, `critic`, `reference` – Instances of `WorkerConfig` defining the models, hardware, and strategies for each role.  
* `train_env_manager`, `val_env_manager` – Instances of `EnvManagerConfig` specifying how environments are instantiated and managed for training and validation rollouts.  
* `reward_normalization` – Configuration for normalizing rewards.

**Example Configurations** – See example YAML files in directories like `examples/qwen2.5-0.5B-agentic_ds` (e.g., `agent_val_frozen_lake.yaml`). These YAMLs use Hydra to set the fields within `AgenticConfig`.

**Key parameters often set via YAML for `AgenticConfig` include**

* `exp_name`, `seed`, `logging_dir`, `output_dir` – General experiment settings.  
* `max_steps`, `save_steps`, `logging_steps`, `eval_steps` – Pipeline control for duration and frequencies.  
* `pretrain`, `reward_pretrain` – Paths to base models for actor/reference and critic respectively.  
* `ppo_epochs`, `init_kl_coef`, `adv_estimator`, `gamma`, `lambd` – PPO and RL algorithm parameters.  
* `custom_envs` – A dictionary defining custom environments available to the pipeline, including their type, instructions, and specific configurations.  
* `train_env_manager` / `val_env_manager` settings:  
  * `num_env_groups`, `group_size` – For managing parallel environments.  
  * `tags`, `num_groups_partition` – To select and apportion environments from `custom_envs`.  
* Worker-specific settings under `actor_train`, `actor_infer`, etc.:  
  * `model_args` – Model architecture, dtype (e.g., bf16), attention type (e.g., flash_attn).  
  * `training_args` – Learning rate, batch size, gradient accumulation.  
  * `generating_args` – `max_new_tokens`, `top_p`, `temperature` for inference.  
  * `strategy_args` – Distributed strategy (e.g., `deepspeed_train`, `vllm`, `sglang`, `hf_infer`) and its specific configuration.  
  * `device_mapping` – GPU allocation for the worker.

---

### Environments (`BaseEnv` and implementations)

ROLL supports various environments for agent training, typically located in subdirectories under `roll/agentic/env/`.

* **Base Class** – `roll/agentic/env/base.py` defines the interface for all environments, including methods like `reset` and `step`.  
* **Examples** – `FrozenLakeEnv`, `SokobanEnv`. Specific configuration (e.g., map size, game rules) is defined within the `custom_envs` section of the main YAML config.  
* Each environment type might also have its own specific config dataclass (e.g., `roll/agentic/env/frozen_lake/config.py`) that is instantiated based on the YAML.

---

### Models & Workers

The agentic pipeline typically involves multiple model roles, managed by dedicated workers:

* **Actor Model (`ActorWorker`)** – The policy that generates actions based on observations. It's trained to maximize expected rewards.  
* **Critic Model (`CriticWorker`)** – Estimates the value function (e.g., expected return from a state). Used in algorithms like PPO with GAE to reduce variance and guide actor training.  
* **Reference Model (`ActorWorker`)** – Often a snapshot of the initial policy or a fixed reference policy, used for calculating KL divergence to regularize policy updates.  
* **Environment Manager (`EnvironmentWorker`, managed by `RolloutScheduler`)** – Handles stepping through multiple environment instances in parallel, collecting trajectories.

Model paths, types, and distributed strategies are specified in their respective worker configuration blocks within the main YAML (e.g., `actor_train`, `critic`).

---

## Running the Agentic Pipeline

### Method&nbsp;1: Using the Python Launcher Script with Hydra (Recommended)

The primary method is to use the `examples/start_agentic_pipeline.py` script. This script uses Hydra to load and manage configurations.

1. **Choose or Create a Configuration File**  
   Start with an example YAML (e.g., `examples/qwen2.5-0.5B-agentic_ds/agent_val_frozen_lake.yaml`) or create your own.

2. **Execute the Python Launcher Script**

   ```bash
   # Ensure you are in the root of the ROLL project directory
   # export PYTHONPATH=$(pwd):$PYTHONPATH

   python examples/start_agentic_pipeline.py \
          --config-path examples/qwen2.5-0.5B-agentic_ds \
          --config-name agent_val_frozen_lake
   ```

   * `--config-path` – Directory containing your YAML configuration.  
   * `--config-name` – Filename (without `.yaml`).  
   * You can add Hydra overrides, e.g. `exp_name=my_new_experiment seed=123`.

The `start_agentic_pipeline.py` script:

* Initializes Hydra.  
* Composes the configuration.  
* Converts the OmegaConf object into an `AgenticConfig` dataclass instance.  
* Initializes Ray.  
* Instantiates the `AgenticPipeline` and calls `run()`.

---

### Method&nbsp;2: Using Helper Shell Scripts

The `examples` directory often contains shell scripts wrapping the Python launcher (e.g., `run_agentic_pipeline_frozen_lake.sh`).

Example structure:

```bash
#!/bin/bash
# Example: examples/qwen2.5-0.5B-agentic_ds/run_agentic_pipeline_frozen_lake.sh

CONFIG_NAME="agent_val_frozen_lake"                         # agent_val_frozen_lake.yaml
CONFIG_PATH_DIR="examples/qwen2.5-0.5B-agentic_ds"

SCRIPT_DIR=$(dirname "$(realpath "$0")")
PROJECT_ROOT=$(dirname "$(dirname "$SCRIPT_DIR")")

export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
export RAY_LOG_WITHOUT_ANSI_CODES=1

echo "PYTHONPATH: $PYTHONPATH"
echo "Using Hydra config directory: $CONFIG_PATH_DIR and config name: $CONFIG_NAME"

python $PROJECT_ROOT/examples/start_agentic_pipeline.py \
       --config-path $CONFIG_PATH_DIR \
       --config-name $CONFIG_NAME \
       "$@"   # Pass through any extra args
```

Run it with:

```bash
bash examples/qwen2.5-0.5B-agentic_ds/run_agentic_pipeline_frozen_lake.sh
```

---

## Step-by-Step Example

### Step&nbsp;1: Locate and Review Configuration

* File: `examples/qwen2.5-0.5B-agentic_ds/agent_val_frozen_lake.yaml`  
  Key sections include `exp_name`, `seed`, `output_dir`, model paths, `actor_train`, `actor_infer`, `reference`, PPO parameters, and `custom_envs`.

### Step&nbsp;2: Prepare the Environment and Dependencies

* Ensure `gymnasium` (or the relevant env dependency) is installed:

  ```bash
  pip install gymnasium
  ```

* Verify all model paths in the YAML are accessible.

### Step&nbsp;3: Launch the Pipeline

```bash
python examples/start_agentic_pipeline.py \
       --config-path examples/qwen2.5-0.5B-agentic_ds \
       --config-name agent_val_frozen_lake
```

### Step&nbsp;4: Monitoring

* **Console Output** – Observe Hydra, Ray, and pipeline logs.  
* **Log Files** – Check the `logging_dir` specified in YAML.  
* **Experiment Tracking** – If configured, metrics appear in WandB.  
* **TensorBoard**

  ```bash
  tensorboard --logdir <your_log_dir>
  ```

### Step&nbsp;5: Outputs and Results

* **Trained Models** – Checkpoints saved under `output_dir`.  
* **Evaluation Metrics** – Logged in WandB/TensorBoard and console.

---

## Troubleshooting

* **Import Errors** – Ensure `PYTHONPATH` includes the ROLL project root.  
* **Configuration Errors**  
  * YAML syntax – lint your YAML.  
  * Hydra path/name – verify `--config-path` and `--config-name`.  
  * Pydantic validation – check `roll/pipeline/agentic/agentic_config.py` field definitions.  
* **Model Loading Issues** – Confirm paths, model types, and strategies (vLLM, SGLang, DeepSpeed, Megatron-Core).  
* **CUDA/GPU Issues** – Adjust `CUDA_VISIBLE_DEVICES`, batch sizes, or `gpu_memory_utilization`.  
* **Ray Issues** – Ensure Ray is started and resource requests match hardware.  
* **Environment Registration** – Verify `env_type` values correspond to registered env classes.

---

## Further Information

* Full configuration definitions: `roll/pipeline/agentic/agentic_config.py` and related dataclasses.  
* Environment implementations: `roll/agentic/env/` (e.g., `frozen_lake/env.py`).  
* Core pipeline logic: `roll/pipeline/agentic/agentic_pipeline.py`.  
* The project `README.md` provides additional high-level details on features like GRPO, Reasoning Pipeline, and integrations with Ray, DeepSpeed, Megatron-Core, vLLM, and SGlang.

---

*Happy experimenting!*