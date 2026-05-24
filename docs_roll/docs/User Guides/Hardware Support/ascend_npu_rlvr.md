# Running RLVR Pipeline on Ascend NPU

Last updated: 04/28/2026.

This guide provides a complete end-to-end walkthrough for running the RLVR (Reinforcement Learning with Verifiable Rewards) pipeline on Huawei Ascend NPU, covering environment setup, data preparation, model download, configuration, training launch, monitoring & evaluation, and checkpoint resumption.

## Workflow Overview

Running an RLVR task on NPU from scratch involves the following steps:

```
1. Environment Setup → 2. Data Preparation → 3. Model Preparation → 4. Write Config → 5. Launch Training → 6. Monitor & Evaluate → 7. Resume from Checkpoint
```

## Step 1: Environment Setup

### 1.1 Hardware & Driver Prerequisites

Ensure your hardware and host drivers are ready:

| Item | Requirement |
| ---- | ----------- |
| Hardware | Atlas 900 A2 PODc (Ascend 910B1) or Atlas 900 A3 PODc (Ascend 910_9391) |
| Host OS | Ubuntu 22.04 |
| CANN | 8.5.1 |
| Ascend NPU Driver | Installed on host (`npu-smi info` shows devices) |
| Docker | >= 20.10 |

### 1.2 Get the Docker Image

Use the pre-built Ascend image that matches your hardware. Official ROLL NPU image tags are available at https://quay.io/repository/ascend/roll?tab=tags. For container launch details, see the [Ascend NPU Docker Usage Guide](ascend_docker_usage.md).

```bash
# For A2 hardware
docker pull roll-registry.cn-hangzhou.cr.aliyuncs.com/roll/pytorch:cann851-910b-py311-torch280-vllm0130
docker tag roll-registry.cn-hangzhou.cr.aliyuncs.com/roll/pytorch:cann851-910b-py311-torch280-vllm0130 roll:ascend-a2

# For A3 hardware
docker pull roll-registry.cn-hangzhou.cr.aliyuncs.com/roll/pytorch:cann851-a3-py311-torch280-vllm0130
docker tag roll-registry.cn-hangzhou.cr.aliyuncs.com/roll/pytorch:cann851-a3-py311-torch280-vllm0130 roll:ascend-a3
```

The current repository does not include `Dockerfile.A2` or `Dockerfile.A3`. If you maintain a custom image, keep the dependency versions aligned with the pre-built image.

### 1.3 Start the Container

```bash
docker run -dit \
    --name roll_npu \
    --ulimit nofile=65536:65536 \
    --device /dev/davinci0 \
    --device /dev/davinci1 \
    --device /dev/davinci2 \
    --device /dev/davinci3 \
    --device /dev/davinci4 \
    --device /dev/davinci5 \
    --device /dev/davinci6 \
    --device /dev/davinci7 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/Ascend/add-ons:/usr/local/Ascend/add-ons \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /path/to/models:/data/models \
    -v /path/to/data:/data \
    --ipc=host \
    --net=host \
    roll:ascend-a3 \
    /bin/bash
```

> **Note:** `-v /path/to/models:/data/models` and `-v /path/to/data:/data` mount model weights and training data directories respectively. Adjust paths to your setup.

### 1.4 Verify the Environment

After entering the container, run:

```bash
# Verify NPU visibility
npu-smi info

# Verify CANN environment is loaded
env | grep -E "ASCEND|LD_LIBRARY_PATH|PATH"

# Verify Python packages
python -c "import torch; import torch_npu; print(torch.npu.is_available())"
python -c "import vllm; print(f'vllm: {vllm.__version__}')"
python -c "import vllm_ascend; print(f'vllm_ascend available')"
```

If all verifications pass, the environment is ready. For detailed environment variable descriptions, see the [NPU Environment Configuration Guide](ascend_npu_env_config.md).

## Step 2: Data Preparation

The RLVR pipeline uses JSONL format data files. Different reward domains require different data fields.

### 2.1 Data Format

#### Common Fields (required for all domains)

| Field | Type | Required | Description |
| ----- | ---- | -------- | ----------- |
| `id` | string/int | Yes | Unique identifier for the data point |
| `messages` or `prompt` | string | Yes | Input prompt; `messages` is a JSON string of message list |
| `tag` | string | Yes | Reward domain label, determines which Reward Worker to use |

#### Domain-Specific Fields

| Domain | tag value | Required fields | Description |
| ------ | --------- | --------------- | ----------- |
| Math rule | `math_rule` | `ground_truth` | Correct answer |
| Code sandbox | `code_sandbox` (e.g., `KodCode`) | `test_cases`, `case_type` | Test cases and type (e.g., `pytest`) |
| LLM judge | `llm_judge` (e.g., `RLVR`) | `ground_truth` | Reference answer or response |
| IFEval | `ifeval` | No extra fields | Rule-based instruction following evaluation |
| CrossThinkQA | `crossthinkqa` | `ground_truth` | Cross-disciplinary reasoning answer |

#### Data Examples

**Math domain (math_rule):**

```json
{
    "id": "0",
    "source": "gsm8k",
    "difficulty": 0,
    "prompt": "Solve the equation 3x + 5 = 14",
    "messages": "[{\"role\": \"system\", \"content\": \"You are a math assistant.\"}, {\"role\": \"user\", \"content\": \"Solve the equation 3x + 5 = 14\"}]",
    "ground_truth": "3",
    "tag": "math_rule"
}
```

**Code domain (code_sandbox):**

```json
{
    "id": "5ea1ab",
    "source": "codeforces",
    "difficulty": "0",
    "prompt": "Write a function that takes an array of distinct integers and returns all possible permutations.",
    "messages": "[{\"role\": \"user\", \"content\": \"Write a function...\"}]",
    "ground_truth": "[\"def permute(nums): ...\"]",
    "case_type": "pytest",
    "test_case_function": "",
    "test_cases": "[{\"assert_code\": \"def test_permute(): ...\"}]",
    "tag": "KodCode"
}
```

### 2.2 Data Placement

Place data files in a directory inside the container (e.g., `/data/`) and specify the paths in `actor_train.data_args`:

```yaml
actor_train:
  data_args:
    file_name:
      - data/math_deepmath_deal.jsonl
      - data/code_KodCode_data.jsonl
    dataset_dir: data
```

### 2.3 Validation Data

Validation data is used for periodic evaluation during training. Specify it in the `validation` config:

```yaml
validation:
  data_args:
    template: qwen2_5
    file_name:
      - data/math_benchmarks.jsonl
  generating_args:
    max_new_tokens: ${response_length}
    top_p: 0.6
    temperature: 0.6
    num_return_sequences: 1
```

The `tag` field in validation data should match the tags in training data so that accuracy can be reported per domain.

## Step 3: Model Preparation

### 3.1 Download Model Weights

The RLVR pipeline requires the following models:

| Model | Config key | Description |
| ----- | ---------- | ----------- |
| Actor / Reference model | `pretrain` | Policy model for training and inference |
| Reward model | `reward_pretrain` | Model used in Reward Workers (e.g., for answer extraction in math rule rewards) |

Example with Qwen2.5-7B:

```bash
# Download using ModelScope (recommended for users in China)
pip install modelscope
modelscope download --model Qwen/Qwen2.5-7B --local_dir /data/models/Qwen2.5-7B

# Or download using HuggingFace
huggingface-cli download Qwen/Qwen2.5-7B --local-dir /data/models/Qwen2.5-7B
```

### 3.2 Specify Model Path in Config

```yaml
pretrain: Qwen/Qwen2.5-7B           # Auto-downloads from ModelScope/HuggingFace
# Or use a local path
# pretrain: /data/models/Qwen2.5-7B

reward_pretrain: Qwen/Qwen2.5-7B
```

> **Tip:** If network access is limited inside the container, download models to the host machine in advance, mount them via `-v`, and use local paths in the config.

## Step 4: Write the NPU Configuration

### Key Differences from GPU

When adapting the GPU RLVR configuration for NPU, the following changes are **required**:

| Item | GPU | NPU |
| ---- | --- | --- |
| Training backend | Megatron or DeepSpeed | DeepSpeed only (Megatron not supported) |
| Inference backend | vLLM | vLLM-Ascend |
| Reference model strategy | `megatron_infer` | `hf_infer` |
| Device placement | Colocated mode supported | Colocated mode **not** supported; training and inference must use separate NPUs |
| Attention implementation | `flash_attn` or `fa2` | `fa2` via `transformers` (not `flash_attn` package) |
| Communication backend | NCCL | HCCL |
| Device visibility | `CUDA_VISIBLE_DEVICES` | `ASCEND_RT_VISIBLE_DEVICES` |
| DeepSpeed config | ZeRO-2 or ZeRO-3 | ZeRO-3 + CPU offloading recommended for 7B+ models |

### Complete NPU Configuration Example

Create a YAML config file based on an existing GPU config (such as `examples/qwen2.5-7B-rlvr_megatron/rlvr_config_amd.yaml`). Below is a complete NPU-adapted configuration with key differences marked with `# NPU` comments:

```yaml
defaults:
  - ../config/deepspeed_zero@_here_
  - ../config/deepspeed_zero2@_here_
  - ../config/deepspeed_zero3@_here_
  - ../config/deepspeed_zero3_cpuoffload@_here_

hydra:
  run:
    dir: .
  output_subdir: null

exp_name: "qwen2.5-7B-rlvr-npu"
seed: 42
logging_dir: ./output/logs
output_dir: ./output

checkpoint_config:
  type: file_system
  output_dir: /data/models/${exp_name}

num_gpus_per_node: 8

max_steps: 500
save_steps: 100
logging_steps: 1
eval_steps: 10
resume_from_checkpoint: false

rollout_batch_size: 64
prompt_length: 2048
response_length: 4096
num_return_sequences_in_group: 8

ppo_epochs: 1
adv_estimator: "reinforce"

value_clip: 0.5
reward_clip: 10
advantage_clip: 2.0
dual_clip_loss: true

norm_mean_type: ~
norm_std_type: ~

max_len_mask: true
difficulty_mask: true
difficulty_low_threshold: 0.1
difficulty_high_threshold: 0.95
error_max_len_clip: false

difficulty_loss_weight: false
length_loss_weight: false

add_token_level_kl: false
whiten_advantages: true

pretrain: Qwen/Qwen2.5-7B
reward_pretrain: Qwen/Qwen2.5-7B

track_with: tensorboard
tracker_kwargs:
  log_dir: ./output/tensorboard/rlvr_npu

validation:
  data_args:
    template: qwen2_5
    file_name:
      - data/math_benchmarks.jsonl
  generating_args:
    max_new_tokens: ${response_length}
    top_p: 0.6
    top_k: 50
    num_beams: 1
    temperature: 0.6
    num_return_sequences: 1

actor_train:
  model_args:
    attn_implementation: fa2            # NPU: Use fa2 via transformers, NOT flash_attn
    disable_gradient_checkpointing: false
    dtype: bf16
    model_type: ~
  training_args:
    learning_rate: 1.0e-6
    weight_decay: 0
    per_device_train_batch_size: 1
    gradient_accumulation_steps: 32
    warmup_steps: 20
  data_args:
    template: qwen2_5
    file_name:
      - data/math_deepmath_deal.jsonl
      - data/code_KodCode_data.jsonl
    domain_interleave_probs:
      math_rule: 0.5
      code_sandbox: 0.5
    dataset_dir: data
    messages: messages
    interleave_probs: "1.0"
  strategy_args:
    strategy_name: deepspeed_train      # NPU: Must use DeepSpeed, NOT megatron_train
    strategy_config: ${deepspeed_zero3_cpuoffload}  # NPU: Use ZeRO-3 + CPU offloading for 7B
  device_mapping: list(range(0,4))      # NPU: Training on NPUs 0-3
  infer_batch_size: 4

actor_infer:
  model_args:
    disable_gradient_checkpointing: true
    dtype: bf16
  generating_args:
    max_new_tokens: ${response_length}
    top_p: 0.99
    top_k: 100
    num_beams: 1
    temperature: 0.99
    num_return_sequences: ${num_return_sequences_in_group}
  data_args:
    template: qwen2_5
  strategy_args:
    strategy_name: vllm                 # NPU: vLLM-Ascend for inference
    strategy_config:
      gpu_memory_utilization: 0.8
      block_size: 16
      max_model_len: 8000
  device_mapping: list(range(4,8))      # NPU: Inference on NPUs 4-7 (separate from training)
  infer_batch_size: 1

reference:
  model_args:
    attn_implementation: fa2            # NPU: Use fa2 via transformers
    disable_gradient_checkpointing: true
    dtype: bf16
    model_type: ~
  data_args:
    template: qwen2_5
  strategy_args:
    strategy_name: hf_infer             # NPU: Use hf_infer, NOT megatron_infer
    strategy_config: ~
  device_mapping: list(range(4,8))      # NPU: Share inference NPUs with actor_infer
  infer_batch_size: 8

rewards:
  math_rule:
    worker_cls: roll.pipeline.rlvr.rewards.math_rule_reward_worker.MathRuleRewardWorker
    model_args:
      model_name_or_path: ${reward_pretrain}
    data_args:
      template: qwen2_5
    tag_included: [deepmath_103k, aime]
    world_size: 4
    infer_batch_size: 1
  code_sandbox:
    use_local: true
    worker_cls: roll.pipeline.rlvr.rewards.code_sandbox_reward_worker.CodeSandboxRewardWorker
    tag_included: [KodCode]
    model_args:
      model_name_or_path: ${reward_pretrain}
    data_args:
      template: qwen2_5
    world_size: 4
    infer_batch_size: 1
```

### Key Configuration Changes Explained

#### 1. Training Strategy: DeepSpeed instead of Megatron

```yaml
# GPU (original)
actor_train:
  strategy_args:
    strategy_name: megatron_train
    strategy_config:
      tensor_model_parallel_size: 1
      pipeline_model_parallel_size: 1

# NPU (adapted)
actor_train:
  strategy_args:
    strategy_name: deepspeed_train
    strategy_config: ${deepspeed_zero3_cpuoffload}
```

For 7B models on 4 NPUs, use `deepspeed_zero3_cpuoffload` to avoid OOM. For smaller models (e.g., 0.5B), `deepspeed_zero3` may be sufficient.

#### 2. Reference Model: hf_infer instead of megatron_infer

```yaml
# GPU
reference:
  strategy_args:
    strategy_name: megatron_infer

# NPU
reference:
  strategy_args:
    strategy_name: hf_infer
    strategy_config: ~
```

#### 3. Device Mapping: Separate Training and Inference NPUs

NPU does **not** support colocated mode. Training and inference must run on different NPUs:

```yaml
actor_train:
  device_mapping: list(range(0,4))    # Training: NPUs 0-3
actor_infer:
  device_mapping: list(range(4,8))    # Inference: NPUs 4-7
reference:
  device_mapping: list(range(4,8))    # Shares inference NPUs
```

See [Device Mapping Reference](#device-mapping-reference) for more allocation patterns.

#### 4. Attention Implementation

Use `fa2` through the `transformers` library instead of the `flash_attn` package:

```yaml
actor_train:
  model_args:
    attn_implementation: fa2    # NOT flash_attn
```

#### 5. System Environment Variables

ROLL injects device visibility and Ray runtime variables for workers, but production runs should still set HCCL, memory, vLLM-Ascend, cache, and logging variables explicitly. See the [NPU Environment Configuration Guide](ascend_npu_env_config.md) for the recommended single-node and multi-node settings.

## Step 5: Launch Training

### Single Node

Run the checked-in Ascend RLVR example:

```bash
cd /workspace/ROLL
export PYTHONPATH="/workspace/ROLL:$PYTHONPATH"

python examples/start_rlvr_pipeline.py \
    --config_path ascend_examples \
    --config_name qwen3_8b_rlvr_deepspeed
```

If you save the custom configuration above as `<config_dir>/rlvr_npu.yaml`, use `--config_path <config_dir> --config_name rlvr_npu` instead.

### Multi-Node

For multi-node training across multiple Ascend NPU nodes, ROLL provides automatic Ray cluster management via environment variables.

#### Setup

On **every** node, set the following environment variables before launching. Replace placeholders with actual values:

**Head node (RANK=0):**

```bash
# Ray cluster
export RANK=0
export WORLD_SIZE=2
export MASTER_ADDR=10.0.0.1            # Head node IP
export MASTER_PORT=6379
export DASHBOARD_PORT=8265

# HCCL multi-node
export HCCL_CONNECT_TIMEOUT=3600
export HCCL_EXEC_TIMEOUT=3600
export HCCL_DETERMINISTIC=false
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_IF_IP=10.0.0.1             # This node's IP
export HCCL_SOCKET_IFNAME="enp194s0f0" # HCCL network interface
export HCCL_IF_BASE_PORT=23456

# NPU memory, CPU, vLLM, cache, logging... (same as single-node)
# See the NPU Environment Configuration Guide for the full list
```

**Worker node (RANK=1):**

```bash
# Ray cluster
export RANK=1
export WORLD_SIZE=2
export MASTER_ADDR=10.0.0.1            # Head node IP (same as above)
export MASTER_PORT=6379
export DASHBOARD_PORT=8265

# HCCL multi-node
export HCCL_CONNECT_TIMEOUT=3600
export HCCL_EXEC_TIMEOUT=3600
export HCCL_DETERMINISTIC=false
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_IF_IP=10.0.0.2             # This node's IP
export HCCL_SOCKET_IFNAME="enp194s0f0"
export HCCL_IF_BASE_PORT=23456

# NPU memory, CPU, vLLM, cache, logging... (same as single-node)
```

#### Launch

Run the **same** command on all nodes. ROLL reads `RANK` to decide whether to start as head or worker:

Before running these commands, save your multi-node config as `<config_dir>/rlvr_npu_multinode.yaml`.

**On the head node:**

```bash
cd /workspace/ROLL
export PYTHONPATH="/workspace/ROLL:$PYTHONPATH"

python examples/start_rlvr_pipeline.py \
    --config_path <config_dir> \
    --config_name rlvr_npu_multinode
```

**On each worker node:**

```bash
cd /workspace/ROLL
export PYTHONPATH="/workspace/ROLL:$PYTHONPATH"

python examples/start_rlvr_pipeline.py \
    --config_path <config_dir> \
    --config_name rlvr_npu_multinode
```

Worker nodes will output logs indicating they've joined the cluster, then exit (`sys.exit(0)`). Their Ray processes stay alive to serve training tasks. The head node continues executing the full training pipeline.

:::tip
You can also pre-start Ray manually (`ray start --head` on head, `ray start --address=...` on workers) before running the pipeline. ROLL detects the existing cluster and skips auto-start.
:::

#### Verify the Cluster

From the head node, check that all nodes have joined:

```bash
ray status
```

The output should show NPU resources from all nodes. For example, with 2 nodes × 8 NPUs:

```
Resources
---------------------------------------------------------------
Total: 128.0 CPU, 16.0 NPU, ...
```

#### Multi-Node Config

For multi-node configs, adjust `device_mapping` to cover NPUs across nodes. For example, with 2 nodes × 8 NPUs:

```yaml
num_gpus_per_node: 8

# Training on Node0 NPUs 0-7
actor_train:
  strategy_args:
    strategy_name: deepspeed_train
    strategy_config: ${deepspeed_zero3_cpuoffload}
  device_mapping: list(range(0,8))

# Inference on Node1 NPUs 0-7
actor_infer:
  strategy_args:
    strategy_name: vllm
    strategy_config:
      gpu_memory_utilization: 0.8
      max_model_len: 8000
  device_mapping: list(range(8,16))

# Reference model shares inference NPUs
reference:
  strategy_args:
    strategy_name: hf_infer
    strategy_config: ~
  device_mapping: list(range(8,16))
```

See [NPU End-to-End Configuration Examples](ascend_npu_examples.md#example-3-multi-node-distributed-training) for a complete multi-node configuration example with data preparation and reward workers.

#### Important Multi-Node Notes

- **Shared storage is required:** Model weights, training data, and checkpoints must be accessible from all nodes at the same paths. Mount NFS or other shared filesystems into each container.
- **Network requirements:** All nodes must be on the same Layer 2 network. The head node's port 6379 must be reachable from all worker nodes.
- **HCCL network interface:** `HCCL_SOCKET_IFNAME` must be the same on all nodes and correspond to the high-speed interconnect (e.g., RoCE). Use `ip addr` or `hccn_tool` to identify the correct interface.

## Step 6: Monitor & Evaluate

### 6.1 Training Monitoring

ROLL has built-in TensorBoard support. Specify the log directory in the config:

```yaml
track_with: tensorboard
tracker_kwargs:
  log_dir: ./output/tensorboard/rlvr_npu
```

Start TensorBoard:

```bash
tensorboard --logdir ./output/tensorboard/rlvr_npu --port 6006
```

Key metrics to monitor:

| Metric | Description |
| ------ | ----------- |
| `time/step_total` | Total time per step |
| `time/step_generate` | Inference generation time |
| `time/step_train` | Training update time |
| `train/loss` | Training loss |
| `train/lr` | Current learning rate |
| `reward/mean` | Average reward |
| `response_length/mean` | Average generation length |

### 6.2 Validation Evaluation

The pipeline automatically runs validation evaluation at `eval_steps` intervals. Validation results include:

| Metric | Description |
| ------ | ----------- |
| `val_correct/all/mean` | Accuracy across all validation samples |
| `val_correct/<tag>/mean` | Accuracy per tag group (e.g., `val_correct/math_rule/mean`) |

Validation accuracy is the core metric for measuring RLVR training effectiveness. It should gradually increase as training progresses.

### 6.3 Generation Examples

During training, generated examples are printed to the log every `logging_steps` steps, allowing you to visually assess model output quality.

## Step 7: Resume from Checkpoint

### 7.1 Checkpoint Saving

The pipeline automatically saves checkpoints to `checkpoint_config.output_dir` at `save_steps` intervals:

```yaml
checkpoint_config:
  type: file_system
  output_dir: /data/models/${exp_name}

save_steps: 100
```

### 7.2 Resume from Checkpoint

Set `resume_from_checkpoint` to the checkpoint path to resume training:

```yaml
resume_from_checkpoint: /data/models/qwen2.5-7B-rlvr-npu/checkpoint-100
```

Or override via the launch command:

```bash
python examples/start_rlvr_pipeline.py \
    --config_path <config_dir> \
    --config_name rlvr_npu \
    resume_from_checkpoint=/data/models/qwen2.5-7B-rlvr-npu/checkpoint-100
```

## Device Mapping Reference

Since NPU does not support colocated mode, you must allocate separate NPUs for training and inference. Below are common allocation patterns for RLVR:

### 8-NPU Single Node (7B Model)

| Component | NPUs | Count | Notes |
| --------- | ---- | ----- | ----- |
| actor_train | 0-3 | 4 | DeepSpeed ZeRO-3 + CPU offloading |
| actor_infer | 4-7 | 4 | vLLM-Ascend |
| reference | 4-7 (shared) | - | hf_infer, shares with actor_infer |
| reward workers | CPU | - | Math rule & code sandbox run on CPU |

### 16-NPU Single Node (A3, 7B Model)

| Component | NPUs | Count | Notes |
| --------- | ---- | ----- | ----- |
| actor_train | 0-7 | 8 | DeepSpeed ZeRO-3 |
| actor_infer | 8-15 | 8 | vLLM-Ascend |
| reference | 8-15 (shared) | - | hf_infer, shares with actor_infer |
| reward workers | CPU | - | Math rule & code sandbox run on CPU |

### 2×8-NPU Multi-Node (7B Model)

| Component | NPUs | Count | Notes |
| --------- | ---- | ----- | ----- |
| actor_train | Node0: 0-7 | 8 | DeepSpeed ZeRO-3 + CPU offloading |
| actor_infer | Node1: 0-7 | 8 | vLLM-Ascend |
| reference | Node1: 0-7 (shared) | - | hf_infer, shares with actor_infer |
| reward workers | CPU | - | Math rule & code sandbox run on CPU |

## Supported Reward Workers on NPU

The following RLVR reward workers are supported on NPU:

| Reward Worker | Class | NPU Compatibility | Notes |
| ------------- | ----- | ----------------- | ----- |
| Math Rule Reward | `MathRuleRewardWorker` | ✅ Supported | Rule-based evaluation, runs on CPU |
| Code Sandbox Reward | `CodeSandboxRewardWorker` | ✅ Supported | Executes code in sandbox, runs on CPU |
| LLM Judge Reward | `LLMJudgeRewardWorker` | ✅ Supported | Requires additional NPU for judge model inference |
| IFEval Rule Reward | `GeneralRuleRewardWorker` | ✅ Supported | Rule-based evaluation, runs on CPU |
| CrossThinkQA Reward | `CrossThinkQARuleRewardWorker` | ✅ Supported | Rule-based evaluation, runs on CPU |

:::caution
When using `LLMJudgeRewardWorker`, the judge model requires its own NPU devices for inference. Ensure you allocate separate NPUs in `device_mapping` for the judge model, and do not share them with `actor_infer` or `actor_train`.
:::

## GPU-to-NPU Configuration Migration Checklist

Use this checklist when migrating an existing GPU RLVR configuration to NPU:

- [ ] Change `actor_train.strategy_args.strategy_name` from `megatron_train` to `deepspeed_train`
- [ ] Change `actor_train.strategy_args.strategy_config` to `${deepspeed_zero3_cpuoffload}` or `${deepspeed_zero3}`
- [ ] Change `reference.strategy_args.strategy_name` from `megatron_infer` to `hf_infer`
- [ ] Set `reference.strategy_args.strategy_config` to `~` (null)
- [ ] Add `attn_implementation: fa2` to `actor_train.model_args` and `reference.model_args`
- [ ] Ensure `device_mapping` separates training and inference NPUs (no colocated mode)
- [ ] Remove any `flash_attn` references
- [ ] Remove any Megatron-specific config (e.g., `tensor_model_parallel_size`, `pipeline_model_parallel_size`)
- [ ] Verify `llm_judge` reward worker has separate NPU allocation (if used)

## Troubleshooting

### First Inference Request Is Very Slow

The first inference request after model loading triggers operator compilation, which can take several minutes. This is a one-time cost. To mitigate:

1. Enable operator compilation cache (see `ACL_OP_COMPILER_CACHE_MODE` in the [NPU Environment Configuration Guide](ascend_npu_env_config.md)).
2. Run a warmup request before starting the actual training loop.

### OOM on 7B Model with 4 NPUs

If you encounter OOM with a 7B model on 4 NPUs:

1. Switch to `deepspeed_zero3_cpuoffload` strategy.
2. Reduce `per_device_train_batch_size` to 1.
3. Increase `gradient_accumulation_steps` accordingly.
4. Reduce `max_model_len` in vLLM config (e.g., from 8192 to 4096).

### HCCL Communication Timeout

See [HCCL Communication Timeout or Failure](ascend_npu_faq.md#hccl-communication-timeout-or-failure) in the FAQ.

### vLLM-Ascend Import Error

Verify that the CANN environment is properly sourced:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
```

### triton Conflict

The `triton` package conflicts with `triton-ascend` on NPU. Fix with:

```bash
pip uninstall -y triton triton-ascend
pip install triton-ascend==3.2.0
```

For more troubleshooting tips, see the [Ascend NPU FAQ](ascend_npu_faq.md).

## Disclaimer

The Ascend support provided in ROLL is intended as a reference example. For production use, please consult official channels.
