# On-Policy Distillation Pipeline

**Table of Contents**

- [On-Policy Distillation Pipeline](#on-policy-distillation-pipeline)
  - [Overview](#overview)
  - [Core Principles](#core-principles)
    - [What is On-Policy Distillation?](#what-is-on-policy-distillation)
    - [Difference from Off-Policy Distillation](#difference-from-off-policy-distillation)
    - [Difference from RLVR](#difference-from-rlvr)
    - [Loss Function: Reverse KL](#loss-function-reverse-kl)
  - [Core Components](#core-components)
    - [Main Module (`OnPolicyDistillPipeline`)](#main-module-onpolicydistillpipeline)
    - [Configuration (`OnPolicyDistillConfig`)](#configuration-onpolicydistillconfig)
    - [Worker Roles](#worker-roles)
  - [Data Preparation](#data-preparation)
    - [Data Format](#data-format)
    - [Data Differences: Pure OPD vs Mixed Mode](#data-differences-pure-opd-vs-mixed-mode)
  - [Running the Pipeline](#running-the-pipeline)
    - [Method 1: Using Python Launch Script](#method-1-using-python-launch-script)
    - [Method 2: Using Helper Shell Script](#method-2-using-helper-shell-script)
  - [Configuration Details](#configuration-details)
    - [Core Configuration Parameters](#core-configuration-parameters)
  - [Step-by-Step Example](#step-by-step-example)
    - [Step 1: Configuration Setup](#step-1-configuration-setup)
    - [Step 2: Prepare Environment and Dependencies](#step-2-prepare-environment-and-dependencies)
    - [Step 3: Launch the Pipeline](#step-3-launch-the-pipeline)
    - [Step 4: Monitoring](#step-4-monitoring)
    - [Step 5: Outputs and Results](#step-5-outputs-and-results)
  - [Multi-Teacher OPD](#multi-teacher-opd)
    - [Configuration Examples](#configuration-examples)
    - [Core Mechanisms](#core-mechanisms)
  - [OPSD (On-Policy Self-Distillation)](#opsd-on-policy-self-distillation)
  - [FAQ](#faq)
  - [References](#references)

---

## Overview

On-Policy Distillation (OPD) is a training method that combines **online learning** and **knowledge distillation**. By having the student model learn the teacher model's behavior on its own generated trajectories, OPD achieves efficient model compression and capability transfer.

This pipeline provides the following core advantages:

* **Efficient Training**: Compared to reinforcement learning (RL), OPD provides dense reward signals, enabling more efficient training
* **Teacher as Reward Model**: Directly uses the teacher model's log probabilities to compute rewards, eliminating the need to train a separate Reward Model
* **Online Learning Advantage**: The student model learns on its own state distribution, avoiding distribution shift issues
* **Full Reuse of RLVR Pipeline**: Built on the RLVR architecture, simple configuration, easy to use
* **Support for Mixed Mode**: Can simultaneously use OPD rewards and external rewards (e.g., math verification, code execution)

---

## Core Principles

### What is On-Policy Distillation?

The core idea of On-Policy Distillation is: sample trajectories from the **student model**, then use a high-performance **teacher model** to score **each token** in the trajectory.

```
┌─────────────────────────────────────────────────────────────────┐
│                    On-Policy Distillation Flow                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   1. Sample Trajectories                                         │
│   ┌──────────┐     ┌──────────────────────────────────┐         │
│   │  Prompt  │ ──▶ │  Student Model (rollout)         │         │
│   └──────────┘     │  Generate trajectories +          │         │
│                    │  student_log_probs               │         │
│                    └──────────────────────────────────┘         │
│                              │                                   │
│                              ▼                                   │
│   2. Compute Teacher Log Probs                                   │
│                    ┌──────────────────────────────────┐         │
│                    │  Teacher Model (forward)         │         │
│                    │  Compute teacher_log_probs       │         │
│                    └──────────────────────────────────┘         │
│                              │                                   │
│                              ▼                                   │
│   3. Compute Advantage                                           │
│                    advantage = teacher_log_prob - student_log_prob│
│                              │                                   │
│                              ▼                                   │
│   4. Train with Importance Sampling                              │
│                    ┌──────────────────────────────────┐         │
│                    │  Student Model (train)           │         │
│                    │  Policy update using advantage   │         │
│                    └──────────────────────────────────┘         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Difference from Off-Policy Distillation

| Feature | Off-Policy Distillation | On-Policy Distillation |
|---------|------------------------|------------------------|
| **Data Source** | Pre-generated data | Data generated in real-time by student model |
| **State Distribution** | Teacher model's state distribution | Student model's state distribution |
| **Reward Signal** | Dense (at each step) | Dense (at each step) |
| **Distribution Shift** | Exists (student may enter states unseen by teacher) | None (learns on own distribution) |
| **Use Case** | Large-scale offline distillation | Scenarios requiring online adaptation |

### Difference from RLVR

| Feature | RLVR | On-Policy Distillation |
|---------|------|------------------------|
| **Reward Source** | External reward models (e.g., math verification, code execution) | Teacher model's log probabilities |
| **Reward Density** | Sparse (usually only final answer has reward) | Dense (every token has reward) |
| **Training Efficiency** | Relatively lower | Higher (dense signals) |
| **Reward Gaming** | Possible (teacher model cannot be "gamed") | Not possible (low KL = high quality behavior) |

### Loss Function: Reverse KL

On-Policy Distillation uses **Reverse KL** as the core loss function:

$$\text{KL}(\pi_\theta || \pi_\text{teacher}) = \mathbb{E}_{x \sim \pi_\theta} \left[ \log \pi_\theta(x_{t+1} | x_{1..t}) - \log \pi_\text{teacher}(x_{t+1} | x_{1..t}) \right]$$

**Advantages**:
1. **Mode Seeking**: Learns specific behaviors from the teacher model rather than spreading across multiple suboptimal options
2. **Cannot Be Gamed**: Low KL always corresponds to high-quality behavior recognized by the teacher model
3. **Reduced Exposure Bias**: Learns on the student's own state distribution

**Implementation**:
```python
# Pseudocode
reverse_kl = sampled_logprobs - teacher_logprobs
advantages = -reverse_kl  # Negative sign: minimize KL = maximize advantage
```

---

## Core Components

### Main Module

Pure OPD mode reuses existing Pipelines, selected by `pure_opd_pipeline_type` config:

- **RLVR Mode** (default): Uses `RLVRConfig` + `RLVRPipeline`
- **Agentic Mode**: Uses `AgenticConfig` + `AgenticPipeline`

The main differences from standard RLVR/Agentic training are:

* **Reward Computation**: Uses Teacher Model's log probabilities instead of external reward models
* **Advantage Computation**: `advantage = teacher_log_prob - student_log_prob`
* **Worker Mapping**: `student_train` → `actor_train`, `student_infer` → `actor_infer`, `teacher` and/or `reference` → `reference` (merged into unified `_reference_configs`)

**Source Code**:
- Launcher script: `examples/start_onpolicy_distill_pipeline.py`
- Pipeline: `roll/pipeline/rlvr/rlvr_pipeline.py` or `roll/pipeline/agentic/agentic_pipeline.py`
- Config handling: `roll/configs/base_config.py` (`_handle_opd_mapping()` method)

---

### Configuration

ROLL supports two On-Policy Distillation modes, both based on `RLVRConfig` (or `AgenticConfig`) config class:

#### Mode 1: Pure OPD Mode (`is_pure_opd=True`)

Suitable for scenarios that **only need distillation signals**, where rewards come entirely from the Teacher Model's KL divergence.

**Launch Method**: Use `start_onpolicy_distill_pipeline.py` script, which automatically sets `is_pure_opd=True`.

```yaml
# Configure student_train, student_infer, teacher roles
student_train:
  model_args:
    model_name_or_path: Qwen/Qwen3-8B
  # ... training config

student_infer:
  model_args:
    model_name_or_path: Qwen/Qwen3-8B
  # ... inference config

teacher:
  model_args:
    model_name_or_path: Qwen/Qwen3-32B  # Can be different from student
  # ... inference config
```

**Internal Mapping**:
- `student_train` → `actor_train`
- `student_infer` → `actor_infer`
- `teacher` → `reference`

**Computation Formula**:
```
token_level_rewards = -reverse_kl  # Pure KL signal, no external rewards
```

**Supported Pipeline Types**: Configured via `pure_opd_pipeline_type`:
- `"rlvr"` (default): Uses RLVRConfig + RLVRPipeline
- `"agentic"`: Uses AgenticConfig + AgenticPipeline


#### Mode 2: Mixed Mode (`use_opd=True`)

Suitable for scenarios that **use both external rewards and distillation signals**, for example, combining rule verification and Teacher KL in math reasoning tasks.

```yaml
# Use standard RLVRConfig config, enable use_opd
use_opd: true
opd_kl_coef: 1.0  # OPD KL coefficient, controls distillation signal weight

# Configure teacher (will be auto-mapped to reference)
teacher:
  model_args:
    model_name_or_path: Qwen/Qwen3-32B

# actor_train and actor_infer configured normally
actor_train:
  model_args:
    model_name_or_path: Qwen/Qwen3-8B
  # ...

actor_infer:
  model_args:
    model_name_or_path: Qwen/Qwen3-8B
  # ...
```

**Computation Formula**:
```
token_level_rewards = external_reward - opd_kl_coef * reverse_kl
```

#### Comparison of Two Modes

| Feature | Pure OPD Mode | Mixed Mode |
|---------|--------------|------------|
| **Config Class** | `RLVRConfig` / `AgenticConfig` | `RLVRConfig` / `AgenticConfig` |
| **Identifier Parameter** | `is_pure_opd=True` (set by script) | `use_opd=True` (user config) |
| **Launch Script** | `start_onpolicy_distill_pipeline.py` | `start_rlvr_pipeline.py` |
| **Worker Config** | `student_train`, `student_infer`, `teacher` | `actor_train`, `actor_infer`, `teacher` |
| **Reward Source** | Teacher KL only | External reward + Teacher KL |
| **Reward Workers** | For validation and statistics | For reward computation |
| **Use Case** | Pure distillation training | RL + distillation joint training |

---

### Worker Roles

On-Policy Distillation's Worker roles differ by mode:

#### Pure OPD Mode

Configure three roles, automatically mapped to internal Workers:

| Config Name | Internal Mapping | Responsibility |
|----------|----------|------|
| `student_train` | `actor_train` | Train student model, compute loss using Teacher KL |
| `student_infer` | `actor_infer` | Generate trajectories, compute student log_probs |
| `teacher` and/or `reference` | `reference` / `references` | Compute teacher log_probs (supports single WorkerConfig or multi-teacher Dict) |

**Note**: Config file uses `student_train`, `student_infer`, `teacher` names, system will automatically map them. `reference` can be used alongside or instead of `teacher` — both are merged into a unified `_reference_configs` dict (`reference` → `"reference"`, single `teacher` → `"default"`, dict `teacher` → by keys). For multi-teacher, `teacher` is `Dict[str, WorkerConfig]`, internally normalized to `self.references: Dict[str, Cluster]`.

#### Mixed Mode

Uses standard RLVR Worker names:

| Worker | Responsibility |
|--------|------|
| `actor_train` | Train with external rewards combined with Teacher KL |
| `actor_infer` | Generate trajectories, compute student log_probs |
| `teacher` | Compute teacher log_probs (auto-mapped to reference) |
| Reward Workers | **Participate in training** (compute external rewards) |

---

## Data Preparation

On-Policy Distillation's data format is identical to RLVR, **does not include response** (generated by the model), only needs to provide prompt and reward-related fields.

### Data Format

```json
{
    "id": "0",
    "source": "math_dataset",
    "difficulty": 0,
    "prompt": "Solve the following math problem: Calculate the value of x in 3x + 5 = 14",
    "messages": "[{\"role\": \"system\", \"content\": \"You are a math assistant.\"}, {\"role\": \"user\", \"content\": \"Solve the following math problem: Calculate the value of x in 3x + 5 = 14\"}]",
    "tag": "math_rule"
}
```

### Data Differences: Pure OPD vs Mixed Mode

| Field | Pure OPD Mode | Mixed Mode |
|-------|--------------|------------|
| `ground_truth` | **Required** (for validation and monitoring) | **Required** (for reward computation) |
| `test_cases` | **Required** (code domain, for validation and monitoring) | **Required** (code domain, for reward computation) |
| `prompt` / `messages` | Required | Required |

**Notes**:
- **Pure OPD Mode**: Rewards are provided by Teacher Model's KL divergence, but `ground_truth` and other fields are used for validation phase evaluation and training process monitoring
- **Mixed Mode**: Requires `ground_truth` or `test_cases` fields, external rewards are part of the training signal

---

## Running the Pipeline

### Method 1: Using Python Launch Script

```bash
# Make sure you're in the project root directory
python examples/start_onpolicy_distill_pipeline.py \
    --config_path examples/qwen3-8B-onpolicy-distill-megatron \
    --config_name onpolicy_distill_config
```

### Method 2: Using Helper Shell Script

```bash
bash examples/qwen3-8B-onpolicy-distill-megatron/run_onpolicy_distill_pipeline.sh
```

---

## Configuration Details

### Core Configuration Parameters

#### Pure OPD Mode

Launched via `start_onpolicy_distill_pipeline.py`, which automatically sets `is_pure_opd=True`.

| Parameter | Description | Default |
|-----------|-------------|---------|
| `pure_opd_pipeline_type` | Pipeline type, one of `"rlvr"` or `"agentic"` | `"rlvr"` |
| `student_train` | Student model training config (mapped to actor_train) | Required |
| `student_infer` | Student model inference config (mapped to actor_infer) | Required |
| `teacher` or `reference` | Teacher/reference model config (merged into _reference_configs) | Required |

#### Mixed Mode (`PPOConfig` / `RLVRConfig`)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `use_opd` | Enable mixed mode OPD (add Teacher KL to rewards) | `false` |
| `teacher` or `reference` | Teacher/reference model config (merged into _reference_configs) | Required |

#### Multi-Teacher Mode Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `teacher` | `Dict[str, WorkerConfig]` multi-teacher config | — |
| `teacher.{name}.opd_kl_coef` | Per-teacher KL coefficient | `1.0` |
| `teacher.{name}.tag_included` | Tags this teacher handles; empty means all | `[]` |
| `tag_to_template` | Select different chat templates by tag | `{}` |

#### OPSD Mode Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `opsd_mode` | Enable OPSD (inject reference solution y\* into teacher prompt). Requires `is_pure_opd=True` or `use_opd=True` | `false` |
| `opsd_solution_key` | Dataset column name for the reference solution | `"reference_solution"` |
| `opsd_teacher_template` | Format string for teacher prompt, placeholders `{problem}` and `{solution}` | Built-in default template |
| `global_template` | Chat template name for teacher prompt formatting | — |
| `sequence_length` | Total batch tensor length. OPSD teacher prompt (problem + y\*) is longer than student prompt — set larger than `prompt_length + response_length` to give buffer | `prompt_length + response_length` |
| `opsd_max_solution_length` | Optional hard cap on reference solution (y\*) token length. If set, solutions exceeding this are truncated before building teacher prompt. If not set, solutions are auto-truncated to fit `sequence_length` (template overhead measured dynamically) | `None` |


---

## Step-by-Step Example

### Step 1: Configuration Setup

* File: `examples/qwen3-8B-onpolicy-distill-megatron/onpolicy_distill_config.yaml`
* Key sections include `exp_name`, `seed`, `output_dir`, model paths, `student_train`, `student_infer`, `teacher`, and reward configuration.

* Pay special attention to these configuration sections:
  * **Data Configuration**: `student_train.data_args.file_name`
  * **Model Configuration**: `pretrain` (student model) and Teacher model path
  * **Distributed Strategy**: `strategy_args` and `device_mapping` for each Worker
  * **Reward Configuration**: Configure Reward Workers in the `rewards` section

### Step 2: Prepare Environment and Dependencies

* Ensure all necessary dependencies are installed:

  ```bash
  pip install -r requirements.txt
  ```

* Verify that all model paths in the configuration are accessible.

* Prepare training and validation datasets, ensuring they conform to the data format requirements (containing `id`, `messages`/`prompt`, `tag`, `ground_truth`, etc. fields).

### Step 3: Launch the Pipeline

```bash
python examples/start_onpolicy_distill_pipeline.py \
       --config_path examples/qwen3-8B-onpolicy-distill-megatron \
       --config_name onpolicy_distill_config
```

### Step 4: Monitoring

* **Console Output** – Observe Hydra, Ray, and pipeline logs
* **Log Files** – Check `logging_dir` specified in YAML
* **TensorBoard**

  ```bash
  tensorboard --logdir <your_log_dir>
  ```

### Step 5: Outputs and Results

* **Trained Model** – Checkpoints saved in `output_dir`
* **Evaluation Metrics** – Logged in TensorBoard and console
* **Generation Examples** – The pipeline periodically outputs generation examples for you to visually evaluate model improvements.

---

## Multi-Teacher OPD

### Overview

Multi-Teacher OPD allows multiple specialized teacher models to simultaneously guide a single student model. Data is routed to the appropriate teacher by domain/tag, avoiding unnecessary computation and enabling more precise distillation.

```
┌──────────────────────────────────────────────────────────────────┐
│              Multi-Teacher OPD Data Flow                           │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│   Student Infer rollout → batch (with tag/domain field)           │
│          │                                                        │
│          ├── [math_dapo data] ──▶  Teacher-32B (math specialist)  │
│          │                          compute ref_log_probs_32B     │
│          │                                                        │
│          └── [KodCode data]  ──▶  Teacher-14B (code specialist)   │
│                                     compute ref_log_probs_14B     │
│          │                                                        │
│          ▼                                                        │
│   Compute Advantage:                                              │
│     For each sample, only accumulate KL from routed teachers:     │
│     advantage = -Σ(opd_kl_coef_i * KL_i) (routed teachers only)  │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### Configuration Examples

#### Multi-Teacher Pure OPD Mode

```yaml
is_pure_opd: true
global_template: qwen3

# Select different chat templates by tag (optional)
tag_to_template:
  math_dapo: qwen3        # Math data uses qwen3 template (with thinking)
  KodCode: qwen3_nothink  # Code data uses qwen3_nothink template

student_train:
  model_args:
    model_name_or_path: Qwen/Qwen3-8B
  data_args:
    file_name:
      - data/dapo_math_17k_simple_boxed.jsonl
      - data/code_KodCode_data.jsonl
    domain_interleave_probs:
      math_rule: 0.6
      code_rule: 0.4
  device_mapping: list(range(0,8))
  # ...

student_infer:
  model_args:
    model_name_or_path: Qwen/Qwen3-8B
  device_mapping: list(range(0,8))
  # ...

# teacher configured as Dict[str, WorkerConfig]
teacher:
  teacher_32B:
    model_args:
      model_name_or_path: Qwen/Qwen3-32B  # Math specialist teacher
    opd_kl_coef: 1.0
    tag_included: [math_dapo]  # Only processes math data
    device_mapping: list(range(8,16))
    strategy_args:
      strategy_name: megatron_infer
      strategy_config:
        tensor_model_parallel_size: 2
        pipeline_model_parallel_size: 4

  teacher_14B:
    model_args:
      model_name_or_path: Qwen/Qwen3-14B  # Code specialist teacher
    opd_kl_coef: 1.0
    tag_included: [KodCode]  # Only processes code data
    device_mapping: list(range(16,24))
    strategy_args:
      strategy_name: megatron_infer
      strategy_config:
        tensor_model_parallel_size: 2
        pipeline_model_parallel_size: 2

rewards:
  math_rule:
    worker_cls: roll.pipeline.rlvr.rewards.math_rule_reward_worker.MathRuleRewardWorker
    tag_included: [math_dapo]
  code_rule:
    worker_cls: roll.pipeline.rlvr.rewards.code_sandbox_reward_worker.CodeSandboxRewardWorker
    tag_included: [KodCode]
```

#### Mixed Routing Config (General Teacher + Specialist Teacher)

```yaml
teacher:
  teacher_general:
    model_args:
      model_name_or_path: Qwen/Qwen3-72B
    opd_kl_coef: 0.3
    tag_included: []  # Empty = handles all tags (general teacher)

  teacher_math_specialist:
    model_args:
      model_name_or_path: DeepSeek-Math-67B
    opd_kl_coef: 0.7
    tag_included: [math_dapo, aime]  # Only handles math
```

In this configuration, math data will have KL computed by both `teacher_general` (coef 0.3) and `teacher_math_specialist` (coef 0.7), with both weighted KL values contributing to the advantage. Non-math data only has `teacher_general` participating.

### Core Mechanisms

#### 1. Tag Routing

Each training sample has a `tag` field (e.g., `math_dapo`, `KodCode`). Each teacher declares the tags it handles via `tag_included`:

- `tag_included: [math_dapo]` — only processes samples with tag `math_dapo`
- `tag_included: []` (empty list) — processes all data (general teacher)

Routing happens at the ref_log_probs computation stage (pipeline layer). Teachers only run forward on their routed data, avoiding unnecessary inference cost.

#### 2. Per-Teacher KL Coefficient

Each teacher has its own `opd_kl_coef`, controlling the weight of that teacher's distillation signal:

```
advantage = -Σ(opd_kl_coef_i * KL(student || teacher_i))
```

Only routed teachers participate in the KL accumulation for each sample.

#### 3. Parallel Inference Optimization

When multiple teachers use different GPUs (non-overlapping `device_mapping`), the system automatically uses multi-threaded parallel execution for each teacher's forward pass, reducing total inference time.

#### 4. tag_to_template

Different domains may require different chat template encoding. With `tag_to_template`, you can use different tokenization templates for specific tags:

```yaml
tag_to_template:
  math_dapo: qwen3         # With thinking token
  KodCode: qwen3_nothink   # Without thinking token
```

Tags not configured in `tag_to_template` fall back to `global_template`.

### Single Teacher Backward Compatibility

Single teacher configuration (`teacher` as WorkerConfig rather than Dict) maintains identical behavior to before:

```yaml
# This config behaves exactly the same as before multi-teacher support
teacher:
  model_args:
    model_name_or_path: Qwen/Qwen3-32B
  device_mapping: list(range(0,16))
```

Internally normalized to `{"default": WorkerConfig}`, the loop executes only once.

---

## OPSD (On-Policy Self-Distillation)

### Overview

OPSD extends OPD: when the teacher evaluates the student's response, its prompt includes the **reference solution y\*** (privileged information) in addition to the original problem. This makes the teacher "know the answer," assigning higher probability to reasoning paths that lead to the correct answer, providing a more precise distillation signal.

```
┌──────────────────────────────────────────────────────────────────┐
│                    OPSD Data Flow                                 │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│   Student Infer:                                                  │
│     prompt = original problem (no y*)                             │
│     → generate response                                           │
│                                                                   │
│   Teacher Forward (OPSD transform):                                │
│     prompt = original problem + y* + instruction (privileged)     │
│     + same response tokens                                        │
│     → compute ref_log_probs                                       │
│     → align back to student layout                                │
│                                                                   │
│   Advantage = -KL(student || teacher)                            │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

**Difference from standard OPD**:

| Feature | Standard OPD | OPSD |
|---------|-------------|------|
| Teacher prompt | Original problem only | Original problem + reference solution y\* |
| Privileged info | None | y\* injected into teacher prompt |
| Distillation signal | General behavior alignment | Guides reasoning toward correct answer |
| Config | `is_pure_opd=True` or `use_opd=True` | Additionally enable `opsd_mode=True` |

### Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `opsd_mode` | Enable OPSD mode (inject y\* into teacher prompt). Requires `is_pure_opd=True` or `use_opd=True` | `false` |
| `opsd_solution_key` | Dataset column name for the reference solution | `"reference_solution"` |
| `opsd_teacher_template` | Format string for teacher prompt, supports `{problem}` and `{solution}` placeholders | Built-in default template |
| `global_template` | Chat template name (e.g., `qwen3`) for teacher prompt formatting | — |

### Data Requirements

The dataset JSONL must contain a `reference_solution` field (or the field specified by `opsd_solution_key`), containing the **full CoT reasoning process** (not just the final answer):

```json
{
    "id": "0",
    "prompt": "Prove that for all positive integers n, n^3 - n is divisible by 6",
    "messages": "[{\"role\": \"user\", \"content\": \"Prove that for all positive integers n, n^3 - n is divisible by 6\"}]",
    "ground_truth": "Proof complete",
    "reference_solution": "n^3 - n = n(n-1)(n+1) = (n-1)n(n+1)...\nThus n^3 - n is a product of three consecutive integers, divisible by 6.",
    "tag": "math_opsd"
}
```

### Teacher Prompt Construction

The teacher prompt is built by formatting `opsd_teacher_template`, then wrapping with the chat template specified by `global_template`:

```
opsd_teacher_template.format(problem=..., solution=...)
  → user_content (problem + y* + instruction)
  → [{"role": "user", "content": user_content}]
  → get_chat_template(global_template, tokenizer)(..., add_generation_prompt=True)
  → teacher_prompt_text (with chat format tokens)
```

The default template instructs the student to reason independently rather than copy the reference solution. When customizing, use `{problem}` and `{solution}` as placeholders.

### Configuration Example

```yaml
# OPSD configuration
is_pure_opd: true  # Set automatically by start_onpolicy_distill_pipeline.py
opsd_mode: true
opsd_solution_key: "reference_solution"
global_template: qwen3  # teacher prompt uses qwen3 chat template (with thinking)
# Optional: hard cap on solution length. If not set, auto-truncates to fit sequence_length.
opsd_max_solution_length: 2048

# OPSD teacher prompt (problem + y*) is longer than student prompt.
# Set sequence_length > prompt_length + response_length to give buffer.
prompt_length: 2048
response_length: 4096
sequence_length: 6656  # 2048 + 4096 + 512 buffer for teacher prompt

student_train:
  model_args:
    model_name_or_path: Qwen/Qwen3-8B
  data_args:
    file_name:
      - data/openthoughts_math_opsd.jsonl  # Must contain reference_solution field
    domain_interleave_probs:
      math_rule: 1.0
  # ...

student_infer:
  model_args:
    model_name_or_path: Qwen/Qwen3-8B
  # ...

teacher:
  model_args:
    model_name_or_path: Qwen/Qwen3-8B  # Self-distillation: teacher = student initial weights
  # ...
```

### Compatibility

- **Multi-Teacher**: OPSD is compatible with multi-teacher routing. Each teacher's `ref_log_probs_{name}` is automatically aligned back to student layout
- **Reference + Teacher**: Both `reference` and `teacher` can be configured simultaneously. They are merged into a unified `_reference_configs` dict. Each entry's KL contributes to the advantage weighted by `opd_kl_coef`
- **Mixed Mode**: OPSD can be combined with `use_opd=True`, advantage = `rl_advantages - total_weighted_kld`

### Caveats

- OPSD teacher prompt (problem + y\*) is longer than student prompt. Set `sequence_length` larger than `prompt_length + response_length` to give buffer. Solutions are auto-truncated to fit: the system measures template overhead dynamically (by building an empty-solution prompt), calculates available space for the solution, and truncates the solution text — preserving the OPSD template structure. Optionally set `opsd_max_solution_length` for a hard cap on solution length.
- Literal curly braces in `opsd_teacher_template` must be escaped as `{{` and `}}` (standard Python `.format()`)
- OPSD mode does not support configuring both `reference` and `teacher` simultaneously
- OPSD supports both LoRA and non-LoRA branches:
  - **Non-LoRA branch**: teacher is a separate cluster
  - **LoRA branch**: teacher is the actor model itself (adapter disabled), no `teacher` config needed

---

## FAQ

### Q1: How to configure mixed mode?

Use `RLVRConfig` (or `AgenticConfig`), set `use_opd: true`:

```yaml
# Mixed mode configuration
use_opd: true
opd_kl_coef: 0.5  # Adjust based on reward magnitude

# Must configure external rewards
rewards:
  math_rule:
    worker_cls: roll.pipeline.rlvr.rewards.math_rule_reward_worker.MathRuleRewardWorker
    tag_included: [math]

# Teacher or reference configuration (automatically mapped to reference)
# Both can be configured simultaneously — they are merged into _reference_configs
teacher:
  model_args:
    model_name_or_path: Qwen/Qwen3-32B

# actor_train and actor_infer configured normally
actor_train:
  model_args:
    model_name_or_path: Qwen/Qwen3-8B
  # ... training config

actor_infer:
  model_args:
    model_name_or_path: Qwen/Qwen3-8B
  # ... inference config
```

### Q2: How to configure pure OPD mode?

Use `start_onpolicy_distill_pipeline.py` script to launch:

```yaml
# Configure three roles
student_train:
  model_args:
    model_name_or_path: Qwen/Qwen3-8B
  # ... training config

student_infer:
  model_args:
    model_name_or_path: Qwen/Qwen3-8B
  # ... inference config

teacher:
  model_args:
    model_name_or_path: Qwen/Qwen3-32B  # Teacher can be different from Student
  # ... inference config
```

Launch command:
```bash
python examples/start_onpolicy_distill_pipeline.py \
    --config_path examples/qwen3-8B-onpolicy-distill-megatron \
    --config_name onpolicy_distill_config
```

### Q3: Why do I need to configure Reward Workers?

Whether in pure OPD mode or mixed mode, Reward Workers must be configured:

1. **Validation Evaluation**: Validation phase needs Reward Workers to evaluate model performance
2. **Training Monitoring**: Observe reward statistics to monitor training quality
3. **Mixed Mode Additional Role**: External rewards are part of the training signal

### Q4: How to choose between modes?

- **Pure OPD Mode**: Best for pure distillation training, only needs Teacher KL signal, use `start_onpolicy_distill_pipeline.py`
- **Mixed Mode**: Best for RL + distillation joint training, use `start_rlvr_pipeline.py` with `use_opd: true`

### Q5: In Multi-Teacher mode, what happens if no teacher is routed to a sample?

That sample's `total_weighted_kld = 0`:
- In pure OPD mode: `advantage = 0` (sample produces no gradient)
- In mixed mode: `advantage = rl_advantages` (RL signal only, no distillation signal)

### Q6: Can multiple teachers' device_mapping overlap?

Yes, but not recommended:
- **Non-overlapping** (recommended): System automatically parallelizes each teacher's forward pass, significantly reducing inference time
- **Overlapping**: System will execute sequentially, no conflicts but total time equals sum of all teachers

---

## References

- [On-Policy Distillation Blog](https://thinkingmachines.ai/blog/on-policy-distillation/)

---

*Happy experimenting!*
