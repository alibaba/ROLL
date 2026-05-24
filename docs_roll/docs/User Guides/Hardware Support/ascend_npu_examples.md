# Ascend NPU End-to-End Configuration Examples

Last updated: 04/27/2026.

This document provides end-to-end configuration examples for running ROLL on Huawei Ascend NPU, including environment setup, resource allocation, and launch commands for both single-node and multi-node scenarios.

## Prerequisites

Before running these examples, ensure you have:

1. Pulled the pre-built Ascend image that matches your hardware (see [Docker Usage Guide](ascend_docker_usage.md)).
2. Verified the environment inside the container (see [Verify the Environment](ascend_docker_usage.md#verify-the-environment)).
3. Downloaded the model weights to a directory accessible from inside the container.

The repository currently includes runnable Ascend examples in `examples/ascend_examples`, including `qwen3_8b_rlvr_deepspeed.yaml`, `qwen3_4B_dpo_deepspeed.yaml`, `run_rlvr_pipeline.sh`, and `run_dpo_pipeline.sh`.

## Key Differences from GPU

When adapting GPU configurations for NPU, the following changes are **required**:

| Item | GPU | NPU |
| ---- | --- | --- |
| Training backend | Megatron or DeepSpeed | DeepSpeed only (Megatron not supported) |
| Device placement | Colocated mode supported | Colocated mode **not** supported; training and inference must use separate NPUs |
| Attention implementation | `flash_attn` or `fa2` | `fa2` via `transformers` (not `flash_attn` package) |
| Communication backend | NCCL | HCCL |
| Device visibility | `CUDA_VISIBLE_DEVICES` | `ASCEND_RT_VISIBLE_DEVICES` |

## Example 1: Single-Node Agentic Pipeline (Qwen2.5-0.5B)

This example runs the FrozenLake agentic pipeline on a single 8-NPU node using DeepSpeed ZeRO-3.

### Step 1: Start the Container

```bash
docker run -dit \
    --name roll_npu_single \
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

### Step 2: Set Environment Variables

```bash
# HCCL communication
export HCCL_CONNECT_TIMEOUT=3600
export HCCL_DETERMINISTIC=false
export HCCL_OP_EXPANSION_MODE="AIV"

# NPU memory
export NPU_MEMORY_FRACTION=0.96
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export MULTI_STREAM_MEMORY_REUSE=1
export TASK_QUEUE_ENABLE=2
export COMBINED_ENABLE=1

# CPU scheduling
export CPU_AFFINITY_CONF=2
export OMP_NUM_THREADS=1

# vLLM-Ascend inference
export VLLM_USE_V1=1
export VLLM_ASCEND_ENABLE_FLASHCOMM=1
export VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE=1
export VLLM_ASCEND_ENABLE_PREFETCH_MLP=1

# Operator compilation cache
export ACL_OP_COMPILER_CACHE_MODE=enable
export ACL_OP_COMPILER_CACHE_DIR=/tmp/npu_cache
export ASCEND_MAX_OP_CACHE_SIZE=5000

# Logging (production)
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASDOPS_LOG_LEVEL=ERROR
export ATB_LOG_LEVEL=ERROR
```

### Step 3: Create NPU Configuration File

Create a YAML config file (e.g., `agentic_frozen_lake_npu.yaml`) with the following NPU-specific settings. Key differences from the GPU config are marked with `# NPU` comments:

```yaml
defaults:
  - ../config/traj_envs@_here_
  - ../config/deepspeed_zero@_here_
  - ../config/deepspeed_zero2@_here_
  - ../config/deepspeed_zero3@_here_
  - ../config/deepspeed_zero3_cpuoffload@_here_

hydra:
  run:
    dir: .
  output_subdir: null

exp_name: "agentic_frozen_lake_npu"
seed: 42
logging_dir: ./output/logs
output_dir: ./output
render_save_dir: ./output/render
system_envs:
  USE_MODELSCOPE: '1'
  HCCL_CONNECT_TIMEOUT: "3600"
  HCCL_DETERMINISTIC: "false"
  HCCL_OP_EXPANSION_MODE: "AIV"
  NPU_MEMORY_FRACTION: "0.96"
  CPU_AFFINITY_CONF: "2"
  OMP_NUM_THREADS: "1"
  VLLM_USE_V1: "1"

checkpoint_config:
  type: file_system
  output_dir: /data/models/${exp_name}

num_gpus_per_node: 8

max_steps: 1024
save_steps: 10000
logging_steps: 1
eval_steps: 10
resume_from_checkpoint: false

rollout_batch_size: 1024
val_batch_size: 1024
sequence_length: 8192

advantage_clip: 0.2
ppo_epochs: 1
adv_estimator: "grpo"
init_kl_coef: 0.0
whiten_advantages: true
entropy_loss_coef: 0
max_grad_norm: 1.0

pretrain: Qwen/Qwen2.5-0.5B-Instruct
reward_pretrain: Qwen/Qwen2.5-0.5B-Instruct

actor_train:
  model_args:
    attn_implementation: fa2          # Use fa2 via transformers, NOT flash_attn
    disable_gradient_checkpointing: false
    dtype: bf16
    model_type: ~
  training_args:
    learning_rate: 1.0e-6
    weight_decay: 0
    per_device_train_batch_size: 2
    gradient_accumulation_steps: 64
    warmup_steps: 10
    lr_scheduler_type: cosine
  data_args:
    template: qwen2_5
  strategy_args:
    strategy_name: deepspeed_train    # NPU: Must use DeepSpeed, NOT megatron_train
    strategy_config: ${deepspeed_zero3}
  device_mapping: list(range(0,4))    # NPU: Training on NPUs 0-3
  infer_batch_size: 2

actor_infer:
  model_args:
    disable_gradient_checkpointing: true
    dtype: bf16
  generating_args:
    max_new_tokens: 128
    top_p: 0.99
    top_k: 100
    num_beams: 1
    temperature: 0.99
    num_return_sequences: 1
  data_args:
    template: qwen2_5
  strategy_args:
    strategy_name: vllm
    strategy_config:
      gpu_memory_utilization: 0.8
      block_size: 16
      load_format: auto
  device_mapping: list(range(4,8))    # NPU: Inference on NPUs 4-7 (separate from training)
  infer_batch_size: 2

reference:
  model_args:
    attn_implementation: fa2
    disable_gradient_checkpointing: true
    dtype: bf16
    model_type: ~
  data_args:
    template: qwen2_5
  strategy_args:
    strategy_name: hf_infer
    strategy_config: ~
  device_mapping: list(range(4,8))    # NPU: Share inference NPUs with actor_infer
  infer_batch_size: 2

reward_normalization:
  grouping: traj_group_id
  method: mean_std

train_env_manager:
  max_env_num_per_worker: 16
  num_env_groups: 128
  group_size: 8
  tags: [FrozenLake]
  num_groups_partition: [128]

val_env_manager:
  max_env_num_per_worker: 32
  num_env_groups: 1024
  group_size: 1
  tags: [SimpleSokoban, LargerSokoban, SokobanDifferentGridVocab, FrozenLake]
  num_groups_partition: [256, 256, 256, 256]

max_tokens_per_step: 64

custom_envs:
  SimpleSokoban:
    ${custom_env.SimpleSokoban}
  LargerSokoban:
    ${custom_env.LargerSokoban}
  SokobanDifferentGridVocab:
    ${custom_env.SokobanDifferentGridVocab}
  FrozenLake:
    ${custom_env.FrozenLake}
  FrozenLakeThink:
    ${custom_env.FrozenLakeThink}
```

### Step 4: Launch

```bash
cd /workspace/ROLL
export PYTHONPATH="/workspace/ROLL:$PYTHONPATH"

python examples/start_agentic_pipeline.py \
    --config_path <config_dir> \
    --config_name agentic_frozen_lake_npu
```

## Example 2: Single-Node RLVR Pipeline (Qwen3-8B)

This example runs the RLVR pipeline on Ascend NPU using the repository config `examples/ascend_examples/qwen3_8b_rlvr_deepspeed.yaml`.

### Key Configuration Changes

```yaml
system_envs:
  USE_MODELSCOPE: '1'
  HCCL_CONNECT_TIMEOUT: "3600"
  HCCL_DETERMINISTIC: "false"
  HCCL_OP_EXPANSION_MODE: "AIV"
  NPU_MEMORY_FRACTION: "0.96"
  CPU_AFFINITY_CONF: "2"
  OMP_NUM_THREADS: "1"
  VLLM_USE_V1: "1"
  PYTORCH_NPU_ALLOC_CONF: "expandable_segments:True"

rollout_batch_size: 32
prompt_length: 2048
response_length: 8192
num_return_sequences_in_group: 8

pretrain: Qwen/Qwen3-8B-Base
reward_pretrain: Qwen/Qwen3-8B-Base

actor_train:
  model_args:
    attn_implementation: fa2          # NPU: Use fa2 via transformers, NOT flash_attn
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
    template: qwen3
    file_name:
      - data/math_deepmath_deal.jsonl
    domain_interleave_probs:
      math_rule: 1
    dataset_dir: data
    messages: messages
    interleave_probs: "1.0"
  strategy_args:
    strategy_name: deepspeed_train    # NPU: Must use DeepSpeed
    strategy_config: ${deepspeed_zero3}
  device_mapping: list(range(0,8))    # NPU: Training on NPUs 0-7
  infer_batch_size: 2

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
    template: qwen3
  strategy_args:
    strategy_name: vllm
    strategy_config:
      gpu_memory_utilization: 0.8
      block_size: 16
      max_model_len: 8000
  device_mapping: list(range(8,12))   # NPU: Inference on NPUs 8-11
  infer_batch_size: 4

reference:
  model_args:
    disable_gradient_checkpointing: true
    dtype: bf16
    model_type: ~
  data_args:
    template: qwen3
  strategy_args:
    strategy_name: hf_infer
    strategy_config: ~
  device_mapping: list(range(12,16))  # NPU: Reference on NPUs 12-15
  infer_batch_size: 1

rewards:
  math_rule:
    worker_cls: roll.pipeline.rlvr.rewards.math_rule_reward_worker.MathRuleRewardWorker
    model_args:
      model_name_or_path: ${reward_pretrain}
    data_args:
      template: qwen3
    tag_included: [deepmath_103k, MATH-500, OlympiadBench, minervamath, aime2025, gsm8k, aime, amc23, math_rule]
    world_size: 8
    infer_batch_size: 1
```

### Launch

```bash
cd /workspace/ROLL
export PYTHONPATH="/workspace/ROLL:$PYTHONPATH"

python examples/start_rlvr_pipeline.py \
    --config_path ascend_examples \
    --config_name qwen3_8b_rlvr_deepspeed
```

## Example 3: Multi-Node Distributed Training

This example shows how to run ROLL across multiple Ascend NPU nodes. ROLL supports two methods for multi-node setup:

- **Method A (Recommended):** Auto-launch via environment variables — set `RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT` on each node, and ROLL automatically starts and manages the Ray cluster.
- **Method B:** Manual Ray cluster — pre-start Ray on each node manually before running ROLL.

### Architecture Overview

```
┌──────────────────────────────────────────────────────┐
│  Head Node (RANK=0)                                   │
│  ┌────────────────────────────────────────────────┐   │
│  │ Docker Container (--net=host)                   │   │
│  │  ├─ Ray Head (port 6379)                       │   │
│  │  ├─ Ray Dashboard (port 8265)                  │   │
│  │  └─ Training Driver (python start_xxx.py)      │   │
│  └────────────────────────────────────────────────┘   │
└──────────────────────┬───────────────────────────────┘
                       │ HCCL (tcp)
         ┌─────────────┼─────────────┐
         ▼                           ▼
┌─────────────────────┐    ┌─────────────────────┐
│ Worker Node 1       │    │ Worker Node 2       │
│ (RANK=1)            │    │ (RANK=2)            │
│ ┌─────────────────┐ │    │ ┌─────────────────┐ │
│ │ Docker Container │ │    │ │ Docker Container │ │
│ │ Ray Worker      │ │    │ │ Ray Worker      │ │
│ │ ray start       │ │    │ │ ray start       │ │
│ │ --address=...   │ │    │ │ --address=...   │ │
│ └─────────────────┘ │    │ └─────────────────┘ │
└─────────────────────┘    └─────────────────────┘
```

### Prerequisites for Multi-Node

- All nodes must be on the same Layer 2 network.
- The head node's `MASTER_PORT` (default 6379) and `DASHBOARD_PORT` (default 8265) must be accessible from all worker nodes (disable firewalls or open these ports).
- A shared storage volume (NFS or similar) mounted at the same path on all nodes is required for model weights, data, and checkpoints.
- All nodes must use the same Docker image and CANN version.

### Network Interface Identification

Before starting, identify the correct HCCL network interface on each node:

```bash
# List available network interfaces
ip addr

# Or use the NPU tool to check HCCL interfaces
for i in {0..7}; do hccn_tool -i $i -ip -g; done

# The NPU device IPs are typically on a high-speed interconnect (e.g., 192.168.x.x).
# Use the corresponding ethernet interface name (e.g., enp194s0f0, eth0) for HCCL_SOCKET_IFNAME.
```

### Step 1: Start Containers on All Nodes

On **each** node, start the Docker container with `--net=host` and mount shared storage:

```bash
docker run -dit \
    --name roll_npu_multi \
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
    -v /shared/storage:/data \
    --ipc=host \
    --net=host \
    roll:ascend-a3 \
    /bin/bash
```

> **Important:** `-v /shared/storage:/data` mounts shared storage for model weights, training data, and checkpoints. This directory must be accessible from all nodes at the same path. Use NFS, HDFS, or other shared filesystem solutions.

### Step 2: Verify NPU Network Connectivity

On **each** node, verify that NPU devices can communicate:

```bash
# Check link status (all should show "up")
for i in {0..7}; do hccn_tool -i $i -link -g; done

# Check TLS consistency (all should show the same switch value)
for i in {0..7}; do hccn_tool -i $i -tls -g; done | grep switch

# If TLS is inconsistent, disable it on all cards on all nodes:
for i in {0..7}; do hccn_tool -i $i -tls -s enable 0; done

# Check NPU device IPs
for i in {0..7}; do hccn_tool -i $i -ip -g; done

# Test cross-node connectivity (run on node B, replace with node A's device IP)
hccn_tool -i 0 -ping -g address <node_a_device_ip>
```

### Step 3: Set Environment Variables

On **each** node, set all environment variables. Replace `<NODE_IP>`, `<HEAD_IP>`, and `<interface>` accordingly:

```bash
# === Ray cluster variables (multi-node) ===
export RANK=<0_for_head_or_1_2_3_for_worker>
export WORLD_SIZE=2                  # Total number of nodes
export MASTER_ADDR=<HEAD_IP>         # IP address of the head node
export MASTER_PORT=6379              # Ray communication port
export DASHBOARD_PORT=8265           # Ray dashboard port

# === HCCL multi-node communication ===
export HCCL_CONNECT_TIMEOUT=3600
export HCCL_EXEC_TIMEOUT=3600
export HCCL_DETERMINISTIC=false
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_IF_IP=<NODE_IP>          # Current node's IP address
export HCCL_SOCKET_IFNAME=<interface> # e.g., enp194s0f0
export HCCL_IF_BASE_PORT=23456

# === NPU memory ===
export NPU_MEMORY_FRACTION=0.96
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export MULTI_STREAM_MEMORY_REUSE=1
export TASK_QUEUE_ENABLE=2
export COMBINED_ENABLE=1

# === CPU scheduling ===
export CPU_AFFINITY_CONF=2
export OMP_NUM_THREADS=1

# === vLLM-Ascend inference ===
export VLLM_USE_V1=1
export VLLM_ASCEND_ENABLE_FLASHCOMM=1
export VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE=1
export VLLM_ASCEND_ENABLE_PREFETCH_MLP=1

# === Operator compilation cache ===
export ACL_OP_COMPILER_CACHE_MODE=enable
export ACL_OP_COMPILER_CACHE_DIR=/tmp/npu_cache
export ASCEND_MAX_OP_CACHE_SIZE=5000

# === Logging (production) ===
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASDOPS_LOG_LEVEL=ERROR
export ATB_LOG_LEVEL=ERROR
```

### Step 4: Launch (Method A — Auto-Launch, Recommended)

Simply run the ROLL pipeline on **all** nodes simultaneously. ROLL automatically detects the `RANK` and starts or joins the Ray cluster:

Before running the commands below, save the multi-node configuration in this section as `<config_dir>/rlvr_npu_multinode.yaml`.

On the **head** node (RANK=0):

```bash
cd /workspace/ROLL
export PYTHONPATH="/workspace/ROLL:$PYTHONPATH"

# Run the training script — ROLL will auto-start Ray head and wait for workers
python examples/start_rlvr_pipeline.py \
    --config_path <config_dir> \
    --config_name rlvr_npu_multinode
```

On all **worker** nodes (RANK=1,2,3...):

```bash
cd /workspace/ROLL
export PYTHONPATH="/workspace/ROLL:$PYTHONPATH"

# Run the same script — ROLL will auto-join the Ray cluster, then sys.exit(0)
python examples/start_rlvr_pipeline.py \
    --config_path <config_dir> \
    --config_name rlvr_npu_multinode
```

After the Ray cluster is established, worker nodes will exit automatically. The head node continues to execute the training pipeline. You should see log messages like:

```
Starting ray cluster: ray start --head --port=6379 ...
1 nodes have joined so far, waiting for 1.
Current ray cluster resources: {'NPU': 16, 'CPU': ...}
```

### Step 4 (Alternative): Launch (Method B — Manual Ray Cluster)

If you prefer to manage the Ray cluster manually:

On the **head** node:

```bash
ray start --head --port=6379 --dashboard-port=8265
```

On all **worker** nodes (replace `<HEAD_IP>` with the head node's IP):

```bash
ray start --address=<HEAD_IP>:6379
```

Verify the cluster:

```bash
ray status
```

You should see all NPU resources from all nodes. Then launch the pipeline only on the **head** node:

```bash
cd /workspace/ROLL
export PYTHONPATH="/workspace/ROLL:$PYTHONPATH"

python examples/start_rlvr_pipeline.py \
    --config_path <config_dir> \
    --config_name rlvr_npu_multinode
```

### Step 5: Monitor the Cluster

From any node, you can monitor the Ray cluster:

```bash
# Check cluster status
ray status

# View the Ray dashboard (open in browser)
# http://<HEAD_IP>:8265
```

### Multi-Node Configuration

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
      block_size: 16
      max_model_len: 8000
  device_mapping: list(range(8,16))

# Reference model shares inference NPUs
reference:
  strategy_args:
    strategy_name: hf_infer
    strategy_config: ~
  device_mapping: list(range(8,16))
```

Complete multi-node RLVR config example (2 nodes × 8 NPUs):

```yaml
defaults:
  - ../config/deepspeed_zero@_here_
  - ../config/deepspeed_zero3@_here_
  - ../config/deepspeed_zero3_cpuoffload@_here_

hydra:
  run:
    dir: .
  output_subdir: null

exp_name: "qwen2.5-7B-rlvr-npu-multinode"
seed: 42
logging_dir: /data/logs
output_dir: /data/output

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
whiten_advantages: true

pretrain: /data/models/Qwen2.5-7B
reward_pretrain: /data/models/Qwen2.5-7B

actor_train:
  model_args:
    attn_implementation: fa2
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
    dataset_dir: /data/datasets
    messages: messages
    interleave_probs: "1.0"
  strategy_args:
    strategy_name: deepspeed_train
    strategy_config: ${deepspeed_zero3_cpuoffload}
  device_mapping: list(range(0,8))    # Node0 NPUs 0-7 for training
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
    strategy_name: vllm
    strategy_config:
      gpu_memory_utilization: 0.8
      block_size: 16
      max_model_len: 8000
  device_mapping: list(range(8,16))   # Node1 NPUs 0-7 for inference
  infer_batch_size: 1

reference:
  model_args:
    attn_implementation: fa2
    disable_gradient_checkpointing: true
    dtype: bf16
    model_type: ~
  data_args:
    template: qwen2_5
  strategy_args:
    strategy_name: hf_infer
    strategy_config: ~
  device_mapping: list(range(8,16))   # Share inference NPUs
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

### Resource Allocation Patterns

When using 2 nodes, there are two common allocation strategies:

**Pattern 1: Training on Node0, Inference on Node1 (recommended for 2-node setups)**

| Component | Location | NPUs | Count |
| --------- | -------- | ---- | ----- |
| actor_train | Node0 | 0-7 | 8 |
| actor_infer | Node1 | 0-7 | 8 |
| reference | Node1 | 0-7 (shared) | - |
| device_mapping train | `list(range(0,8))` | | |
| device_mapping infer | `list(range(8,16))` | | |

**Pattern 2: Split both training and inference across nodes**

| Component | Location | NPUs | Count |
| --------- | -------- | ---- | ----- |
| actor_train | Node0 + Node1 | 0-3 on each | 4+4=8 |
| actor_infer | Node0 + Node1 | 4-7 on each | 4+4=8 |
| device_mapping train | `list(range(0,4)) + list(range(8,12))` | | |
| device_mapping infer | `list(range(4,8)) + list(range(12,16))` | | |

Pattern 1 has lower cross-node HCCL communication overhead during inference. Pattern 2 balances the load more evenly. Choose based on your workload characteristics.

## Device Mapping Reference

Since NPU does not support colocated mode, you must allocate separate NPUs for training and inference. Here are common allocation patterns:

### 8-NPU Single Node

| Component | NPUs | Count |
| --------- | ---- | ----- |
| actor_train | 0-3 | 4 |
| actor_infer | 4-7 | 4 |
| reference | 4-7 (shared) | - |

### 16-NPU Single Node (A3)

| Component | NPUs | Count |
| --------- | ---- | ----- |
| actor_train | 0-7 | 8 |
| actor_infer | 8-15 | 8 |
| reference | 8-15 (shared) | - |

### 2×8-NPU Multi-Node

| Component | NPUs | Count |
| --------- | ---- | ----- |
| actor_train | Node0: 0-7 | 8 |
| actor_infer | Node1: 0-7 | 8 |
| reference | Node1: 0-7 (shared) | - |

## Troubleshooting

### First Inference Request Is Very Slow

The first inference request after model loading triggers operator compilation, which can take several minutes. This is a one-time cost. To mitigate:

1. Enable operator compilation cache (see `ACL_OP_COMPILER_CACHE_MODE` above).
2. Run a warmup request before starting the actual training loop.

### OOM on 7B Model with 4 NPUs

If you encounter OOM with a 7B model on 4 NPUs:

1. Switch to `deepspeed_zero3_cpuoffload` strategy.
2. Reduce `per_device_train_batch_size` to 1.
3. Increase `gradient_accumulation_steps` accordingly.
4. Reduce `max_model_len` in vLLM config (e.g., from 8192 to 4096).

### Multi-Node HCCL Communication Failure

See [HCCL Communication Timeout or Failure](ascend_npu_faq.md#hccl-communication-timeout-or-failure) in the FAQ.

## Disclaimer

The Ascend support provided in ROLL is intended as a reference example. For production use, please consult official channels.
