# RLVR Pipeline

**Table of Contents**

- [RLVR Pipeline](#rlvr-pipeline)
  - [✨️Overview](#️overview)
  - [✨️Core Components](#️core-components)
    - [Main Module (`RLVRPipeline`)](#main-module-rlvrpipeline)
    - [Configuration File (`RLVRConfig`)](#configuration-file-rlvrconfig)
      - [Configuration File Structure and Organization](#configuration-file-structure-and-organization)
    - [Reward Worker](#reward-worker)
  - [✨️Data Preparation](#️data-preparation)
    - [Data Format](#data-format)
      - [Common Data Fields](#common-data-fields)
      - [Domain-Specific Fields](#domain-specific-fields)
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

* **Diverse Task Support**: Built-in support for various task types including mathematical reasoning, code generation, LLM-as-judge evaluation, and instruction following, each equipped with specialized reward evaluation mechanisms and flexible extension interfaces to accommodate new task types.

* **Multi-Task Joint Training**: Enables simultaneous optimization of model capabilities across multiple domains such as math, programming, and general reasoning, with flexible control over data sampling ratios and reward weight configurations for each domain.

* **Algorithm-Friendly Reinforcement Learning Framework**: Provides a rich set of reinforcement learning strategy options (over 20 types), including but not limited to reward normalization, reward clipping, various advantage estimation methods, and more. Not limited to a single algorithm implementation, it supports multiple reinforcement learning algorithms such as PPO, GRPO, Reinforce++, TOPR and RAFT++.

* **Comprehensive Performance Monitoring**: Fine-grained metric tracking system that simultaneously monitors group-level and batch-level performance metrics, providing comprehensive visualization and analysis capabilities for the model training process.

* **Efficient Distributed Computing**: Leverages the [Ray](https://www.ray.io/) framework to implement efficient distributed training on large-scale GPU clusters, significantly improving training speed and resource utilization.

---



## ✨️Core Components

### Main Module (`RLVRPipeline`)

`RLVRPipeline` (located in `roll/pipeline/rlvr/rlvr_pipeline.py`) is the primary coordinator for the entire reinforcement learning process. It manages the complete training workflow, including:

* Initializing and managing distributed workers (actor, critic, reference, and various reward workers).
* Coordinating data collection and processing.
* Executing model training steps (e.g., PPO updates for actor and critic).
* Handling model synchronization and checkpoint saving.
* Validation set evaluation.
* Recording metrics and experiment tracking.

**Source code**: `roll/pipeline/rlvr/rlvr_pipeline.py`

---

### Configuration File (`RLVRConfig`)

`RLVRConfig` (defined in `roll/pipeline/rlvr/rlvr_config.py`) is a Pydantic/dataclass-based configuration object used to specify all parameters for running the rlvr pipeline. This configuration system is flexibly designed, supporting configuration via YAML files and managed using the Hydra framework.

#### Configuration File Structure and Organization

Configuration files (such as `examples/qwen2.5-7B-rlvr_megatron/rlvr_config.yaml`) are organized by functional modules, containing the following main sections:

1. **Experiment Basic Settings**
   * `exp_name`: Experiment name, used to identify a specific training run
   * `logging_dir`: Path for saving log files
   * `output_dir`: Path for saving model checkpoints and output files

2. **Training Control Parameters**
   * `max_steps`: Maximum number of training steps
   * `save_steps`: Frequency for saving model checkpoints
   * `logging_steps`: Frequency for recording training metrics
   * `eval_steps`: Frequency for performing validation evaluations
   * `resume_from_checkpoint`: Whether to continue training from a checkpoint

3. **Model Configuration**
   * `pretrain`: Path to pre-trained weights for Actor and Reference models
   * `reward_pretrain`: Path to pre-trained weights for Critic model

4. **Reinforcement Learning Algorithm Parameters**
   * `ppo_epochs`: Number of PPO updates per batch of data
   * `init_kl_coef`: Initial coefficient for KL divergence
   * `target_kl`: Target value for KL divergence
   * `adv_estimator`: Advantage estimation method (e.g., `gae`)
   * `gamma`: Discount factor
   * `lambd`: GAE lambda parameter
   * `reward_normalize`: Whether to normalize rewards
   * `reward_clip`: Reward clipping range
   * `value_clip`: Value clipping range
   * ...

5. **Worker Configuration**
   Each worker (`actor_train`, `actor_infer`, `critic`, `reference`) configuration contains:

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
   * **Generation Parameters** (`generating_args`)
     * `max_new_tokens`: Maximum number of new tokens to generate
     * `top_p`: Nucleus sampling parameter
     * `temperature`: Sampling temperature
     * `do_sample`: Whether to use sampling for generation
     * ...
   * **Distributed Strategy** (`strategy_args`)
     * `strategy_name`: Distributed strategy to use (e.g., `megatron_train`, `vllm`, `sglang`, `hf_infer`)
     * Strategy-specific parameters: e.g., `tp_size` (tensor parallelism size), `pp_size` (pipeline parallelism size)
     * `gpu_memory_utilization`: GPU memory utilization (vLLM-specific)
   * **Device Mapping** (`device_mapping`)
     * Specifies which GPU devices the worker should use

6. **Reward Settings**
   The `rewards` section contains reward worker configurations for different domains:

   * **Math** (`math_rule`)
     * `worker_cls`: Worker class name (e.g., `MathRuleRewardWorker`)
     * `tag_included`: These tags use the reward domain for calculation
     * `model_args`: Reward model parameters
     * `world_size`: Number of workers

   * **Code** (`code_sandbox`)
     * Similar configuration, but for code evaluation

   * **General Reasoning** (`llm_judge`)
     * Configuration for using an LLM as a judge

7. **Validation and Evaluation Settings**
   The `validation` section configures validation datasets and evaluation methods:

   * `file_name`: Path to validation dataset file
   * `batch_size`: Validation batch size
   * `metrics`: Evaluation metrics to calculate

---

### Reward Worker

The rlvr pipeline supports various reward mechanisms for different rlvr domains:

* **Mathematical Rule Reward (`MathRuleRewardWorker`)** – Evaluates the correctness and steps of mathematical reasoning.
* **Code Sandbox Reward (`CodeSandboxRewardWorker`)** – Evaluates code generation by executing the code and verifying its output.
* **LLM Judge Reward (`LLMJudgeRewardWorker`)** – Uses another LLM as a judge to evaluate the quality of generated answers.

---



## ✨️Data Preparation

### Data Format

The rlvr pipeline uses data files in JSON format. Different domains require specific fields:

#### Common Data Fields

All domains require the following fields:
* `id`: Unique identifier for the data point **(required)**
* `messages` or `prompt`: Input prompt, can be a list of messages (JSON string) or a single prompt string **(required)**
* `tag`: For more fine-grained classification (e.g., `gsm8k`, `olympiads`, etc.) **(required)**
* `difficulty`: Problem difficulty level **(optional)**

#### Domain-Specific Fields

Depending on the domain, data points need to include the following specific fields:

1. **Math (`math_rule`)**
   * `ground_truth`: Correct answer or solution steps **(required)**
2. **Code (`code_sandbox`)**
   * `test_cases`: Test cases for verifying code correctness **(required)**
   * `case_type`: Test case type (e.g., `pytest`) **(required)**
   * `test_case_function`: Test function definition **(optional)**
   * `ground_truth`: Reference answer **(optional)**
3. **General Reasoning (`llm_judge`)**
   * `ground_truth`: Standard answer or reference response **(required)**

Example data format (MATH):
```json
{
    "id": "0",
    "source": "gsm8k",
    "difficulty": 0,
    "prompt": "Solve the equation 3x + 5 = 14",
    "messages": "[{\"role\": \"system\", \"content\": \"You are a math assistant skilled at solving complex mathematical problems.\"}, {\"role\": \"user\", \"content\": \"Solve the equation 3x + 5 = 14\"}]",
    "ground_truth": "204",
    "case_type": "",
    "test_case_function": "",
    "test_cases": "",
    "tag": "math_rule"
 }
```

Example data format (Code domain):
```json
{
  "id": "5ea1ab",
  "source": "codeforeces",
  "difficulty": "0",
  "prompt": "You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. \n\n### Question: Write a function that takes an array of distinct integers and returns all possible permutations (in any order). Each permutation should be represented as an array of integers. The function should handle arrays of different lengths efficiently.\n\n### Format: You will use the following starter code to write the solution to the problem and enclose your code within delimiters.\n```python\ndef permute(nums):\n```\n\n### Answer: (use the provided format with backticks)",
  "messages": "[{\"role\": \"user\", \"content\": \"You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. \\n\\n### Question: Write a function that takes an array of distinct integers and returns all possible permutations (in any order). Each permutation should be represented as an array of integers. The function should handle arrays of different lengths efficiently.\\n\\n### Format: You will use the following starter code to write the solution to the problem and enclose your code within delimiters.\\n```python\\ndef permute(nums):\\n```\\n\\n### Answer: (use the provided format with backticks)\"}]",
  "ground_truth": "[\"def permute(nums):\\n    \\\"\\\"\\\"\\n    Given an array of distinct integers, return all possible permutations.\\n    Each permutation is an array of integers.\\n    \\\"\\\"\\\"\\n    def backtrack(start, end):\\n        if start == end:\\n            permutations.append(nums[:])\\n        for i in range(start, end):\\n            nums[start], nums[i] = nums[i], nums[start]\\n            backtrack(start + 1, end)\\n            nums[start], nums[i] = nums[i], nums[start]\\n\\n    permutations = []\\n    backtrack(0, len(nums))\\n    return permutations\"]",
  "case_type": "pytest",
  "test_case_function": " ",
  "test_cases": "[{\"assert_code\": \"\\n\\n\\ndef test_permute_single_element():\\n    assert permute([1]) == [[1]]\\n\\ndef test_permute_two_elements():\\n    result = permute([1, 2])\\n    expected = [[1, 2], [2, 1]]\\n    assert sorted(result) == sorted(expected)\\n\\ndef test_permute_three_elements():\\n    result = permute([1, 2, 3])\\n    expected = [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]\\n    assert sorted(result) == sorted(expected)\\n\\ndef test_permute_four_elements():\\n    result = permute([1, 2, 3, 4])\\n    expected = [\\n        [1, 2, 3, 4], [1, 2, 4, 3], [1, 3, 2, 4], [1, 3, 4, 2], [1, 4, 2, 3], [1, 4, 3, 2],\\n        [2, 1, 3, 4], [2, 1, 4, 3], [2, 3, 1, 4], [2, 3, 4, 1], [2, 4, 1, 3], [2, 4, 3, 1],\\n        [3, 1, 2, 4], [3, 1, 4, 2], [3, 2, 1, 4], [3, 2, 4, 1], [3, 4, 1, 2], [3, 4, 2, 1],\\n        [4, 1, 2, 3], [4, 1, 3, 2], [4, 2, 1, 3], [4, 2, 3, 1], [4, 3, 1, 2], [4, 3, 2, 1]\\n    ]\\n    assert sorted(result) == sorted(expected)\"}]",
  "tag": "KodCode"
}
```

In the configuration file, you can set the sampling ratio for different domains using `domain_interleave_probs`, for example:
```yaml
domain_interleave_probs:
  math_rule: 0.6
  code_sandbox: 0.3
  llm_judge: 0.1
```

---



## ✨️Running the Pipeline



### Method 1: Using Python Launcher Script

The primary method is to use the `examples/start_rlvr_pipeline.py` script. This script uses Hydra to load and manage configurations.

1. **Select or Create a Configuration File**  
   Start with an example YAML (e.g., `examples/qwen2.5-7B-rlvr_megatron/rlvr_config.yaml`) or create your own configuration.

2. **Execute the Python Launcher Script**

   ```bash
   # Make sure you are in the root directory of the ROLL (ScaleAligner) project
   # export PYTHONPATH=$(pwd):$PYTHONPATH
   
   python examples/start_rlvr_pipeline.py \
          --config_path examples/qwen2.5-7B-rlvr_megatron \
          --config_name rlvr_config
   ```

   * `--config_path` – Directory containing your YAML configuration.
   * `--config_name` – Filename (without `.yaml`).



### Method 2: Using Helper Shell Scripts

The `examples` directory typically contains shell scripts that wrap the Python launcher (e.g., `start_ppo_pipeline_math_hz.sh`).

Example structure:

```bash
#!/bin/bash
# Example: examples/qwen2.5-7B-rlvr_megatron/run_rlvr_pipeline.sh

CONFIG_NAME="rlvr_config"                         # rlvr_config.yaml
CONFIG_PATH="examples/qwen2.5-7B-rlvr_megatron"

# Set environment variables and other configurations

python examples/start_rlvr_pipeline.py \
       --config_path $CONFIG_PATH \
       --config_name $CONFIG_NAME \
       "$@"   # Pass any additional parameters
```

Run using:

```bash
bash examples/qwen2.5-7B-rlvr_megatron/run_rlvr_pipeline.sh
```

---



## ✨️Step-by-Step Example



### Step 1: Configure Settings

* File: `examples/qwen2.5-7B-rlvr_megatron/rlvr_config.yaml`  
  Key sections include `exp_name`, `seed`, `output_dir`, model paths, `actor_train`, `actor_infer`, `reference`, PPO parameters, and reward configurations.

* Pay special attention to these configuration sections:
  * Data configuration: `actor_train.data_args.file_name` and `domain_interleave_probs`
  * Model configuration: `pretrain` and `reward_pretrain` paths
  * Distributed strategies: `strategy_args` and `device_mapping` for each worker
  * Reward configuration: Reward workers for different domains in the `rewards` section

### Step 2: Prepare Environment and Dependencies

* Ensure all necessary dependencies are installed:

  ```bash
  pip install -r requirements.txt
  ```

* Verify that all model paths in the configuration are accessible.

* Prepare training and validation datasets, ensuring they conform to the data format requirements described above.

### Step 3: Launch the Pipeline

```bash
python examples/start_rlvr_pipeline.py \
       --config_path examples/qwen2.5-7B-rlvr_megatron_hz \
       --config_name ppo
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
* **Generated Examples** – The pipeline periodically outputs generated examples so you can visually assess model improvements.

---

*Happy experimenting!*
