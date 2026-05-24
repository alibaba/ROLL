# 昇腾 NPU 端到端配置样例

最后更新：2026/04/27。

本文档提供在华为昇腾 NPU 上运行 ROLL 的端到端配置样例，涵盖环境准备、资源切分和启动命令，适用于单机和多机场景。

## 前置条件

运行本样例前，请确保：

1. 已拉取与硬件匹配的预构建昇腾镜像（参见 [Docker 使用指南](ascend_docker_usage.md)）。
2. 已在容器内验证环境（参见 [验证环境](ascend_docker_usage.md#verify-the-environment)）。
3. 已将模型权重下载到容器可访问的目录。

当前仓库在 `examples/ascend_examples` 中提供可直接运行的昇腾示例，包括 `qwen3_8b_rlvr_deepspeed.yaml`、`qwen3_4B_dpo_deepspeed.yaml`、`run_rlvr_pipeline.sh` 和 `run_dpo_pipeline.sh`。

## GPU 与 NPU 的关键差异

将 GPU 配置适配到 NPU 时，**必须**进行以下修改：

| 项目 | GPU | NPU |
| ---- | --- | --- |
| 训练后端 | Megatron 或 DeepSpeed | 仅 DeepSpeed（不支持 Megatron） |
| 设备放置 | 支持 Colocated 模式 | **不支持** Colocated 模式；训练和推理必须使用不同的 NPU 卡 |
| 注意力实现 | `flash_attn` 或 `fa2` | 通过 `transformers` 使用 `fa2`（不能使用 `flash_attn` 包） |
| 通信后端 | NCCL | HCCL |
| 设备可见性 | `CUDA_VISIBLE_DEVICES` | `ASCEND_RT_VISIBLE_DEVICES` |

## 样例 1：单机 Agentic 流水线（Qwen2.5-0.5B）

本样例在单个 8 卡 NPU 节点上使用 DeepSpeed ZeRO-3 运行 FrozenLake Agentic 流水线。

### 步骤 1：启动容器

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

### 步骤 2：设置环境变量

```bash
# HCCL 通信
export HCCL_CONNECT_TIMEOUT=3600
export HCCL_DETERMINISTIC=false
export HCCL_OP_EXPANSION_MODE="AIV"

# NPU 显存
export NPU_MEMORY_FRACTION=0.96
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export MULTI_STREAM_MEMORY_REUSE=1
export TASK_QUEUE_ENABLE=2
export COMBINED_ENABLE=1

# CPU 调度
export CPU_AFFINITY_CONF=2
export OMP_NUM_THREADS=1

# vLLM-Ascend 推理
export VLLM_USE_V1=1
export VLLM_ASCEND_ENABLE_FLASHCOMM=1
export VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE=1
export VLLM_ASCEND_ENABLE_PREFETCH_MLP=1

# 算子编译缓存
export ACL_OP_COMPILER_CACHE_MODE=enable
export ACL_OP_COMPILER_CACHE_DIR=/tmp/npu_cache
export ASCEND_MAX_OP_CACHE_SIZE=5000

# 日志（生产环境）
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASDOPS_LOG_LEVEL=ERROR
export ATB_LOG_LEVEL=ERROR
```

### 步骤 3：创建 NPU 配置文件

创建 YAML 配置文件（如 `agentic_frozen_lake_npu.yaml`），以下为 NPU 专用配置。与 GPU 配置的关键差异以 `# NPU` 注释标记：

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
    attn_implementation: fa2          # NPU: 通过 transformers 使用 fa2，不能使用 flash_attn
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
    strategy_name: deepspeed_train    # NPU: 必须使用 DeepSpeed，不能用 megatron_train
    strategy_config: ${deepspeed_zero3}
  device_mapping: list(range(0,4))    # NPU: 训练使用 NPU 0-3
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
  device_mapping: list(range(4,8))    # NPU: 推理使用 NPU 4-7（与训练分离）
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
  device_mapping: list(range(4,8))    # NPU: 与 actor_infer 共享推理卡
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

### 步骤 4：启动训练

```bash
cd /workspace/ROLL
export PYTHONPATH="/workspace/ROLL:$PYTHONPATH"

python examples/start_agentic_pipeline.py \
    --config_path <config_dir> \
    --config_name agentic_frozen_lake_npu
```

## 样例 2：单机 RLVR 流水线（Qwen3-8B）

本样例使用仓库中的 `examples/ascend_examples/qwen3_8b_rlvr_deepspeed.yaml` 配置在昇腾 NPU 上运行 RLVR 流水线。

### 关键配置

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
    attn_implementation: fa2          # NPU: 通过 transformers 使用 fa2，不能使用 flash_attn
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
    strategy_name: deepspeed_train    # NPU: 必须使用 DeepSpeed
    strategy_config: ${deepspeed_zero3}
  device_mapping: list(range(0,8))    # NPU: 训练使用 NPU 0-7
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
  device_mapping: list(range(8,12))   # NPU: 推理使用 NPU 8-11
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
  device_mapping: list(range(12,16))  # NPU: Reference 使用 NPU 12-15
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

### 启动

```bash
cd /workspace/ROLL
export PYTHONPATH="/workspace/ROLL:$PYTHONPATH"

python examples/start_rlvr_pipeline.py \
    --config_path ascend_examples \
    --config_name qwen3_8b_rlvr_deepspeed
```

## 样例 3：多机分布式训练

本样例展示如何在多个昇腾 NPU 节点上运行 ROLL。ROLL 支持两种多机启动方式：

- **方式 A（推荐）：** 通过环境变量自动启动——在每个节点上设置 `RANK`、`WORLD_SIZE`、`MASTER_ADDR`、`MASTER_PORT`，ROLL 自动启动和管理 Ray 集群。
- **方式 B：** 手动 Ray 集群——在运行 ROLL 之前在每个节点上手动启动 Ray。

### 架构概览

```
┌──────────────────────────────────────────────────────┐
│  主节点 (RANK=0)                                      │
│  ┌────────────────────────────────────────────────┐   │
│  │ Docker 容器 (--net=host)                       │   │
│  │  ├─ Ray Head (端口 6379)                       │   │
│  │  ├─ Ray Dashboard (端口 8265)                  │   │
│  │  └─ 训练驱动 (python start_xxx.py)             │   │
│  └────────────────────────────────────────────────┘   │
└──────────────────────┬───────────────────────────────┘
                       │ HCCL (tcp)
         ┌─────────────┼─────────────┐
         ▼                           ▼
┌─────────────────────┐    ┌─────────────────────┐
│ 工作节点 1           │    │ 工作节点 2           │
│ (RANK=1)            │    │ (RANK=2)            │
│ ┌─────────────────┐ │    │ ┌─────────────────┐ │
│ │ Docker 容器      │ │    │ │ Docker 容器      │ │
│ │ Ray Worker      │ │    │ │ Ray Worker      │ │
│ │ ray start       │ │    │ │ ray start       │ │
│ │ --address=...   │ │    │ │ --address=...   │ │
│ └─────────────────┘ │    │ └─────────────────┘ │
└─────────────────────┘    └─────────────────────┘
```

### 多机前置条件

- 所有节点必须在同一二层网络内。
- 主节点的 `MASTER_PORT`（默认 6379）和 `DASHBOARD_PORT`（默认 8265）必须能被所有工作节点访问（关闭防火墙或开放这些端口）。
- 所有节点需要挂载共享存储（NFS 等）到相同路径，用于模型权重、数据和断点。
- 所有节点必须使用相同的 Docker 镜像和 CANN 版本。

### 网络接口识别

启动前，确认每个节点上 HCCL 使用的网卡名称：

```bash
# 列出可用网卡
ip addr

# 或用 NPU 工具查看 HCCL 接口
for i in {0..7}; do hccn_tool -i $i -ip -g; done

# NPU 设备 IP 通常在高速互联网络上（如 192.168.x.x）。
# 使用对应的以太网接口名称（如 enp194s0f0, eth0）作为 HCCL_SOCKET_IFNAME。
```

### 步骤 1：在所有节点启动容器

在**每个**节点上，使用 `--net=host` 启动 Docker 容器并挂载共享存储：

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

> **重要：** `-v /shared/storage:/data` 将共享存储挂载到容器内，用于模型权重、训练数据和断点。此目录必须在所有节点上以相同路径访问。可使用 NFS、HDFS 或其他共享文件系统方案。

### 步骤 2：验证 NPU 网络连通性

在**每个**节点上，验证 NPU 设备间通信：

```bash
# 检查链路状态（全部应显示 "up"）
for i in {0..7}; do hccn_tool -i $i -link -g; done

# 检查 TLS 一致性（所有卡应显示相同的 switch 值）
for i in {0..7}; do hccn_tool -i $i -tls -g; done | grep switch

# 若 TLS 不一致，在所有节点的所有卡上统一关闭：
for i in {0..7}; do hccn_tool -i $i -tls -s enable 0; done

# 查看 NPU 设备 IP
for i in {0..7}; do hccn_tool -i $i -ip -g; done

# 测试跨节点连通性（在节点 B 上执行，替换为节点 A 的 device IP）
hccn_tool -i 0 -ping -g address <节点A的device_ip>
```

### 步骤 3：设置环境变量

在**每个**节点上，设置所有环境变量。将 `<NODE_IP>`、`<HEAD_IP>` 和 `<网卡名称>` 替换为实际值：

```bash
# === Ray 集群变量（多机） ===
export RANK=<主节点为0，工作节点为1_2_3>
export WORLD_SIZE=2                  # 集群总节点数
export MASTER_ADDR=<HEAD_IP>         # 主节点 IP 地址
export MASTER_PORT=6379              # Ray 通信端口
export DASHBOARD_PORT=8265           # Ray Dashboard 端口

# === HCCL 多机通信 ===
export HCCL_CONNECT_TIMEOUT=3600
export HCCL_EXEC_TIMEOUT=3600
export HCCL_DETERMINISTIC=false
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_IF_IP=<NODE_IP>          # 当前节点 IP
export HCCL_SOCKET_IFNAME=<网卡名称>  # 例如 enp194s0f0
export HCCL_IF_BASE_PORT=23456

# === NPU 显存 ===
export NPU_MEMORY_FRACTION=0.96
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export MULTI_STREAM_MEMORY_REUSE=1
export TASK_QUEUE_ENABLE=2
export COMBINED_ENABLE=1

# === CPU 调度 ===
export CPU_AFFINITY_CONF=2
export OMP_NUM_THREADS=1

# === vLLM-Ascend 推理 ===
export VLLM_USE_V1=1
export VLLM_ASCEND_ENABLE_FLASHCOMM=1
export VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE=1
export VLLM_ASCEND_ENABLE_PREFETCH_MLP=1

# === 算子编译缓存 ===
export ACL_OP_COMPILER_CACHE_MODE=enable
export ACL_OP_COMPILER_CACHE_DIR=/tmp/npu_cache
export ASCEND_MAX_OP_CACHE_SIZE=5000

# === 日志（生产环境） ===
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASDOPS_LOG_LEVEL=ERROR
export ATB_LOG_LEVEL=ERROR
```

### 步骤 4：启动（方式 A — 自动启动，推荐）

在**所有**节点上同时运行 ROLL 流水线。ROLL 根据 `RANK` 自动启动或加入 Ray 集群：

运行下面的命令前，请先将本节中的多机配置保存为 `<config_dir>/rlvr_npu_multinode.yaml`。

**主节点**（RANK=0）：

```bash
cd /workspace/ROLL
export PYTHONPATH="/workspace/ROLL:$PYTHONPATH"

# 运行训练脚本 — ROLL 自动启动 Ray Head 并等待所有工作节点加入
python examples/start_rlvr_pipeline.py \
    --config_path <config_dir> \
    --config_name rlvr_npu_multinode
```

**工作节点**（RANK=1,2,3...）：

```bash
cd /workspace/ROLL
export PYTHONPATH="/workspace/ROLL:$PYTHONPATH"

# 运行相同脚本 — ROLL 自动加入 Ray 集群，然后 sys.exit(0)
python examples/start_rlvr_pipeline.py \
    --config_path <config_dir> \
    --config_name rlvr_npu_multinode
```

Ray 集群建立后，工作节点会自动退出。主节点继续执行训练流水线。你应能看到类似以下日志：

```
Starting ray cluster: ray start --head --port=6379 ...
1 nodes have joined so far, waiting for 1.
Current ray cluster resources: {'NPU': 16, 'CPU': ...}
```

### 步骤 4（备选）：启动（方式 B — 手动 Ray 集群）

如果你希望手动管理 Ray 集群：

在**主**节点上：

```bash
ray start --head --port=6379 --dashboard-port=8265
```

在所有**工作**节点上（将 `<HEAD_IP>` 替换为主节点 IP）：

```bash
ray start --address=<HEAD_IP>:6379
```

验证集群状态：

```bash
ray status
```

应能看到所有节点的 NPU 资源。然后在**主节点**上启动流水线：

```bash
cd /workspace/ROLL
export PYTHONPATH="/workspace/ROLL:$PYTHONPATH"

python examples/start_rlvr_pipeline.py \
    --config_path <config_dir> \
    --config_name rlvr_npu_multinode
```

### 步骤 5：监控集群

在任意节点上均可监控 Ray 集群：

```bash
# 查看集群状态
ray status

# 在浏览器中打开 Ray Dashboard
# http://<HEAD_IP>:8265
```

### 多机配置

多机配置中，调整 `device_mapping` 以覆盖跨节点的 NPU。例如 2 节点 × 8 卡：

```yaml
num_gpus_per_node: 8

# 训练在节点0的 NPU 0-7
actor_train:
  strategy_args:
    strategy_name: deepspeed_train
    strategy_config: ${deepspeed_zero3_cpuoffload}
  device_mapping: list(range(0,8))

# 推理在节点1的 NPU 0-7
actor_infer:
  strategy_args:
    strategy_name: vllm
    strategy_config:
      gpu_memory_utilization: 0.8
      block_size: 16
      max_model_len: 8000
  device_mapping: list(range(8,16))

# Reference 模型共享推理卡
reference:
  strategy_args:
    strategy_name: hf_infer
    strategy_config: ~
  device_mapping: list(range(8,16))
```

完整的多机 RLVR 配置示例（2 节点 × 8 卡）：

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
  device_mapping: list(range(0,8))    # 节点0 NPU 0-7 用于训练
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
  device_mapping: list(range(8,16))   # 节点1 NPU 0-7 用于推理
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
  device_mapping: list(range(8,16))   # 共享推理卡
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

### 资源分配策略

使用 2 个节点时，有两种常见的分配策略：

**策略 1：训练在节点0，推理在节点1（推荐用于 2 节点场景）**

| 组件 | 位置 | NPU | 数量 |
| ---- | ---- | --- | ---- |
| actor_train | 节点0 | 0-7 | 8 |
| actor_infer | 节点1 | 0-7 | 8 |
| reference | 节点1 | 0-7（共享） | - |
| device_mapping train | `list(range(0,8))` | | |
| device_mapping infer | `list(range(8,16))` | | |

**策略 2：训练和推理均跨节点分布**

| 组件 | 位置 | NPU | 数量 |
| ---- | ---- | --- | ---- |
| actor_train | 节点0 + 节点1 | 每节点 0-3 | 4+4=8 |
| actor_infer | 节点0 + 节点1 | 每节点 4-7 | 4+4=8 |
| device_mapping train | `list(range(0,4)) + list(range(8,12))` | | |
| device_mapping infer | `list(range(4,8)) + list(range(12,16))` | | |

策略 1 推理时跨节点 HCCL 通信开销更低。策略 2 负载更均衡。请根据实际工作负载特点选择。

## 设备映射参考

由于 NPU 不支持 colocated 模式，必须为训练和推理分配不同的 NPU 卡。以下是常见的分配方案：

### 8 卡单机

| 组件 | NPU 卡号 | 数量 |
| ---- | -------- | ---- |
| actor_train | 0-3 | 4 |
| actor_infer | 4-7 | 4 |
| reference | 4-7（共享） | - |

### 16 卡单机（A3）

| 组件 | NPU 卡号 | 数量 |
| ---- | -------- | ---- |
| actor_train | 0-7 | 8 |
| actor_infer | 8-15 | 8 |
| reference | 8-15（共享） | - |

### 2×8 卡多机

| 组件 | NPU 卡号 | 数量 |
| ---- | -------- | ---- |
| actor_train | 节点0: 0-7 | 8 |
| actor_infer | 节点1: 0-7 | 8 |
| reference | 节点1: 0-7（共享） | - |

## 常见问题

### 首次推理请求极慢

模型加载后的首次推理请求会触发算子编译，可能需要数分钟。这是一次性开销。缓解方法：

1. 启用算子编译缓存（参见上方 `ACL_OP_COMPILER_CACHE_MODE`）。
2. 在正式训练循环前发送一次预热请求。

### 7B 模型在 4 卡上 OOM

如果在 4 张 NPU 上运行 7B 模型遇到 OOM：

1. 切换到 `deepspeed_zero3_cpuoffload` 策略。
2. 将 `per_device_train_batch_size` 减小到 1。
3. 相应增大 `gradient_accumulation_steps`。
4. 减小 vLLM 配置中的 `max_model_len`（如从 8192 减到 4096）。

### 多机 HCCL 通信失败

参见 FAQ 中的 [HCCL 通信超时或失败](ascend_npu_faq.md#hccl-通信超时或失败)。

## 声明

ROLL 中提供的 Ascend 支持代码皆为参考样例，生产环境使用请通过官方正式途径沟通。
