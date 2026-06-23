# 在昇腾 NPU 上运行 RLVR 流水线

最后更新：2026/04/28。

本文档提供在华为昇腾 NPU 上运行 RLVR（Reinforcement Learning with Verifiable Rewards）流水线的端到端指南，涵盖环境准备、数据准备、模型下载、配置编写、训练启动、监控与评估，以及从 checkpoint 恢复训练。

## 工作流概览

从零开始在 NPU 上运行 RLVR 任务包含以下步骤：

```
1. 环境准备 → 2. 数据准备 → 3. 模型准备 → 4. 编写配置 → 5. 启动训练 → 6. 监控与评估 → 7. 从 Checkpoint 恢复
```

## 步骤 1：环境准备

### 1.1 硬件与驱动前置条件

请确保硬件和宿主机驱动已经准备就绪：

| 项目 | 要求 |
| ---- | ---- |
| 硬件 | Atlas 900 A2 PODc（Ascend 910B1）或 Atlas 900 A3 PODc（Ascend 910_9391） |
| 宿主机 OS | Ubuntu 22.04 |
| CANN | 9.0.0 |
| Ascend NPU 驱动 | 已在宿主机安装（`npu-smi info` 能看到设备） |
| Docker | >= 20.10 |

### 1.2 获取 Docker 镜像

请使用与硬件匹配的预构建昇腾镜像。官方 ROLL NPU 镜像标签可在 https://quay.io/repository/ascend/roll?tab=tags 查看。容器启动细节参见 [Ascend NPU Docker 使用指南](ascend_docker_usage.md)。

```bash
# A2 硬件
docker pull quay.io/ascend/roll:main-a2
docker tag quay.io/ascend/roll:main-a2 roll:ascend-a2

# A3 硬件
docker pull quay.io/ascend/roll:main-a3
docker tag quay.io/ascend/roll:main-a3 roll:ascend-a3
```

当前仓库提供 `docker/Dockerfile.A2` 和 `docker/Dockerfile.A3`，用于构建自定义镜像。如果你维护自定义镜像，请确保依赖版本与预构建镜像保持一致。

### 1.3 启动容器

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

> **注意：** `-v /path/to/models:/data/models` 和 `-v /path/to/data:/data` 分别挂载模型权重目录和训练数据目录。请根据实际环境调整路径。

### 1.4 验证环境

进入容器后，执行：

```bash
# 验证 NPU 可见性
npu-smi info

# 验证 CANN 环境是否已加载
env | grep -E "ASCEND|LD_LIBRARY_PATH|PATH"

# 验证 Python 包
python -c "import torch; import torch_npu; print(torch.npu.is_available())"
python -c "import vllm; print(f'vllm: {vllm.__version__}')"
python -c "import vllm_ascend; print(f'vllm_ascend available')"
```

如果以上验证均通过，说明环境已经准备就绪。环境变量的详细说明参见 [NPU 环境配置指南](ascend_npu_env_config.md)。

## 步骤 2：数据准备

RLVR 流水线使用 JSONL 格式的数据文件。不同奖励领域需要不同的数据字段。

### 2.1 数据格式

#### 通用字段（所有领域都必需）

| 字段 | 类型 | 是否必需 | 说明 |
| ---- | ---- | -------- | ---- |
| `id` | string/int | 是 | 数据样本的唯一标识 |
| `messages` 或 `prompt` | string | 是 | 输入 prompt；`messages` 是消息列表的 JSON 字符串 |
| `tag` | string | 是 | 奖励领域标签，用于决定使用哪个 Reward Worker |

#### 领域专属字段

| 领域 | tag 值 | 必需字段 | 说明 |
| ---- | ------ | -------- | ---- |
| 数学规则 | `math_rule` | `ground_truth` | 正确答案 |
| 代码沙箱 | `code_sandbox`（如 `KodCode`） | `test_cases`, `case_type` | 测试用例和类型（如 `pytest`） |
| LLM Judge | `llm_judge`（如 `RLVR`） | `ground_truth` | 参考答案或参考回复 |
| IFEval | `ifeval` | 无额外字段 | 基于规则的指令遵循评估 |
| CrossThinkQA | `crossthinkqa` | `ground_truth` | 跨学科推理答案 |

#### 数据样例

**数学领域（math_rule）：**

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

**代码领域（code_sandbox）：**

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

### 2.2 数据放置

将数据文件放到容器内的某个目录中（如 `/data/`），并在 `actor_train.data_args` 中指定路径：

```yaml
actor_train:
  data_args:
    file_name:
      - data/math_deepmath_deal.jsonl
      - data/code_KodCode_data.jsonl
    dataset_dir: data
```

### 2.3 验证数据

验证数据用于训练过程中的周期性评估。请在 `validation` 配置中指定：

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

验证数据中的 `tag` 字段应与训练数据中的 tag 保持一致，这样才能按领域报告准确率。

## 步骤 3：模型准备

### 3.1 下载模型权重

RLVR 流水线需要以下模型：

| 模型 | 配置键 | 说明 |
| ---- | ------ | ---- |
| Actor / Reference 模型 | `pretrain` | 用于训练和推理的策略模型 |
| Reward 模型 | `reward_pretrain` | Reward Worker 中使用的模型（如数学规则奖励中的答案抽取） |

以 Qwen2.5-7B 为例：

```bash
# 使用 ModelScope 下载（推荐中国大陆用户使用）
pip install modelscope
modelscope download --model Qwen/Qwen2.5-7B --local_dir /data/models/Qwen2.5-7B

# 或使用 HuggingFace 下载
huggingface-cli download Qwen/Qwen2.5-7B --local-dir /data/models/Qwen2.5-7B
```

### 3.2 在配置中指定模型路径

```yaml
pretrain: Qwen/Qwen2.5-7B           # 从 ModelScope/HuggingFace 自动下载
# 或使用本地路径
# pretrain: /data/models/Qwen2.5-7B

reward_pretrain: Qwen/Qwen2.5-7B
```

> **提示：** 如果容器内网络访问受限，请提前在宿主机下载模型，通过 `-v` 挂载到容器中，并在配置中使用本地路径。

## 步骤 4：编写 NPU 配置

### 与 GPU 的关键差异

将 GPU RLVR 配置适配到 NPU 时，**必须**进行以下修改：

| 项目 | GPU | NPU |
| ---- | --- | --- |
| 训练后端 | Megatron 或 FSDP2 | 仅 FSDP2（NPU 不支持 Megatron） |
| 推理后端 | vLLM | vLLM-Ascend |
| Reference 模型策略 | `megatron_infer` | `fsdp2_infer` |
| 注意力实现 | `flash_attn` 或 `fa2` | 通过 `transformers` 使用 `fa2`（不能使用 `flash_attn` 包） |
| 通信后端 | NCCL | HCCL |
| 设备可见性 | `CUDA_VISIBLE_DEVICES` | `ASCEND_RT_VISIBLE_DEVICES` |
| 分片配置 | FSDP2 或 Megatron 优化器分片 | FSDP2，7B+ 模型推荐 `offload_policy: true` |

### 完整 NPU 配置样例

下面是一个完整的 NPU 适配配置（改编自 `examples/ascend_examples/qwen3_30b_rlvr_fsdp2.yaml`），关键差异使用 `# NPU` 注释标记：

```yaml
hydra:
  run:
    dir: .
  output_subdir: null

exp_name: "qwen3-30BA3B-rlvr-npu"
seed: 42
logging_dir: ./output/logs
output_dir: ./output
system_envs:
  USE_MODELSCOPE: '1'
  HCCL_NPU_SOCKET_PORT_RANGE: auto       # NPU：允许同卡多进程 HCCL 自动分配 device 侧端口
  VLLM_ASCEND_ENABLE_NZ: '0'             # NPU：RL 权重刷新场景需禁用 FRACTAL_NZ

checkpoint_config:
  type: file_system
  output_dir: ./output/models/${exp_name}

track_with: tensorboard
tracker_kwargs:
  log_dir: ./output/tensorboard/rlvr_npu
rpc_timeout: 72000

num_gpus_per_node: 16

max_steps: 500
save_steps: 100
logging_steps: 1
eval_steps: 10
resume_from_checkpoint: false

rollout_batch_size: 32
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

pretrain: Qwen/Qwen3-30B-A3B
reward_pretrain: Qwen/Qwen3-30B-A3B

actor_train:
  model_args:
    disable_gradient_checkpointing: false
    dtype: bf16
    model_type: ~
  training_args:
    learning_rate: 1.0e-6
    weight_decay: 0
    per_device_train_batch_size: 2
    gradient_accumulation_steps: 8
    warmup_steps: 20
    num_train_epochs: 50
  data_args:
    template: qwen2_5
    file_name:
      - data/math_deepmath_deal.jsonl
    domain_interleave_probs:
      math_rule: 1
    dataset_dir: data
    messages: messages
    interleave_probs: "1.0"
    preprocessing_num_workers: 16
  strategy_args:
    strategy_name: fsdp2_train      # NPU：必须使用 FSDP2，不能用 megatron_train
    strategy_config:
      fsdp_size: 16                 # NPU：FSDP2 分片大小
      param_dtype: bf16
      reduce_dtype: bf16
      offload_policy: true          # NPU：大模型启用 CPU offloading
      apply_expert_patch: true      # NPU：MoE 模型必须启用
      apply_tiled_mlp: true         # NPU：TiledMLP 降低显存
      tiled_num_shards: 8
      reshard_after_forward: true
      wrap_policy:                  # NPU：MoE 专用 wrap policy
        wrap_embeddings: true
        wrap_lm_output: true
        moe_experts:
          - Qwen3MoeMLP
        transformer_layer_cls_to_wrap:
          - Qwen3MoeAttention
          - Qwen3MoeSparseMoeBlock
  use_remove_padding: true
  device_mapping: list(range(0,16))     # NPU：训练使用 NPU 0-15
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
    template: qwen2_5
  strategy_args:
    strategy_name: vllm                 # NPU：使用 vLLM-Ascend 推理
    strategy_config:
      gpu_memory_utilization: 0.8
      block_size: 16
      max_model_len: 6144
      tensor_parallel_size: 2
      enforce_eager: true
      load_format: dummy
  device_mapping: list(range(0,16))     # NPU：推理与训练共享 NPU
  infer_batch_size: 1

reference:
  model_args:
    disable_gradient_checkpointing: true
    dtype: bf16
    model_type: ~
  data_args:
    template: qwen2_5
  strategy_args:
    strategy_name: fsdp2_infer          # NPU：使用 fsdp2_infer，不能用 megatron_infer
    strategy_config:
      fsdp_size: 16
      param_dtype: bf16
      reduce_dtype: bf16
      apply_tiled_mlp: true
      tiled_num_shards: 8
      reshard_after_forward: true
      offload_policy: true
  device_mapping: list(range(0,16))     # NPU：Reference 与训练共享 NPU
  infer_batch_size: 2

rewards:
  math_rule:
    worker_cls: roll.pipeline.rlvr.rewards.math_rule_reward_worker.MathRuleRewardWorker
    model_args:
      model_name_or_path: ${reward_pretrain}
    data_args:
      template: qwen2_5
    tag_included: [deepmath_103k, aime]
    world_size: 8
    infer_batch_size: 1
```

### 关键配置变更说明

#### 1. 训练策略：使用 FSDP2 替代 Megatron

```yaml
# GPU（原始配置）
actor_train:
  strategy_args:
    strategy_name: megatron_train
    strategy_config:
      tensor_model_parallel_size: 1
      pipeline_model_parallel_size: 1

# NPU（适配后）
actor_train:
  strategy_args:
    strategy_name: fsdp2_train
    strategy_config:
      fsdp_size: 4
      param_dtype: bf16
      reduce_dtype: bf16
      reshard_after_forward: true
      offload_policy: true
```

在 4 张 NPU 上运行 7B 模型时，设置 `offload_policy: true` 可以启用 CPU offloading 避免 OOM。对于更小的模型（如 0.5B），`offload_policy: false` 可能已经足够。

#### 2. Reference 模型：使用 fsdp2_infer 替代 megatron_infer

```yaml
# GPU
reference:
  strategy_args:
    strategy_name: megatron_infer

# NPU
reference:
  strategy_args:
    strategy_name: fsdp2_infer
    strategy_config:
      fsdp_size: 4
      param_dtype: bf16
      reduce_dtype: bf16
      reshard_after_forward: true
      offload_policy: true
```

#### 3. 注意力实现

通过 `transformers` 库使用 `fa2`，不要使用 `flash_attn` 包：

```yaml
actor_train:
  model_args:
    attn_implementation: fa2    # 不能使用 flash_attn
```

#### 4. 系统环境变量

ROLL 会为 worker 注入设备可见性和 Ray 运行时变量，但生产运行时仍建议显式设置 HCCL、显存、vLLM-Ascend、缓存和日志相关变量。推荐的单机和多机环境变量设置参见 [NPU 环境配置指南](ascend_npu_env_config.md)。

## 步骤 5：启动训练

### 单机

运行仓库中提供的昇腾 RLVR 示例：

```bash
cd /workspace/ROLL
export PYTHONPATH="/workspace/ROLL:$PYTHONPATH"

python examples/start_rlvr_pipeline.py \
    --config_path ascend_examples \
    --config_name qwen3_30b_rlvr_fsdp2
```

如果你将上面的自定义配置保存为 `<config_dir>/rlvr_npu.yaml`，则使用 `--config_path <config_dir> --config_name rlvr_npu`。

### 多机

对于跨多个昇腾 NPU 节点的多机训练，ROLL 通过环境变量提供自动 Ray 集群管理。

#### 设置

启动前，请在**每个**节点上设置以下环境变量。请将占位符替换为实际值：

**Head 节点（RANK=0）：**

```bash
# Ray 集群
export RANK=0
export WORLD_SIZE=2
export MASTER_ADDR=10.0.0.1            # Head 节点 IP
export MASTER_PORT=6379
export DASHBOARD_PORT=8265

# HCCL 多机通信
export HCCL_CONNECT_TIMEOUT=3600
export HCCL_EXEC_TIMEOUT=3600
export HCCL_DETERMINISTIC=false
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_NPU_SOCKET_PORT_RANGE="auto"
export HCCL_IF_IP=10.0.0.1             # 当前节点 IP
export HCCL_SOCKET_IFNAME="enp194s0f0" # HCCL 网络接口
export HCCL_IF_BASE_PORT=23456

# vLLM-Ascend RL 场景
export VLLM_ASCEND_ENABLE_NZ=0

# NPU 显存、CPU、vLLM、缓存、日志等（同单机设置）
# 完整列表参见 NPU 环境配置指南
```

**Worker 节点（RANK=1）：**

```bash
# Ray 集群
export RANK=1
export WORLD_SIZE=2
export MASTER_ADDR=10.0.0.1            # Head 节点 IP（同上）
export MASTER_PORT=6379
export DASHBOARD_PORT=8265

# HCCL 多机通信
export HCCL_CONNECT_TIMEOUT=3600
export HCCL_EXEC_TIMEOUT=3600
export HCCL_DETERMINISTIC=false
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_NPU_SOCKET_PORT_RANGE="auto"
export HCCL_IF_IP=10.0.0.2             # 当前节点 IP
export HCCL_SOCKET_IFNAME="enp194s0f0"
export HCCL_IF_BASE_PORT=23456

# vLLM-Ascend RL 场景
export VLLM_ASCEND_ENABLE_NZ=0

# NPU 显存、CPU、vLLM、缓存、日志等（同单机设置）
```

#### 启动

在所有节点运行**相同**命令。ROLL 会读取 `RANK`，决定当前进程以 head 还是 worker 方式启动。

运行这些命令前，请先将多机配置保存为 `<config_dir>/rlvr_npu_multinode.yaml`。

**在 Head 节点上：**

```bash
cd /workspace/ROLL
export PYTHONPATH="/workspace/ROLL:$PYTHONPATH"

python examples/start_rlvr_pipeline.py \
    --config_path <config_dir> \
    --config_name rlvr_npu_multinode
```

**在每个 Worker 节点上：**

```bash
cd /workspace/ROLL
export PYTHONPATH="/workspace/ROLL:$PYTHONPATH"

python examples/start_rlvr_pipeline.py \
    --config_path <config_dir> \
    --config_name rlvr_npu_multinode
```

Worker 节点会输出已加入集群的日志，然后退出（`sys.exit(0)`）。对应的 Ray 进程会保持存活，用于服务训练任务。Head 节点会继续执行完整训练流水线。

:::tip
也可以在运行流水线前手动预启动 Ray（head 节点执行 `ray start --head`，worker 节点执行 `ray start --address=...`）。ROLL 会检测已有集群并跳过自动启动。
:::

#### 验证集群

在 Head 节点上检查所有节点是否已经加入：

```bash
ray status
```

输出中应能看到来自所有节点的 NPU 资源。例如，2 节点 × 8 NPU：

```
Resources
---------------------------------------------------------------
Total: 128.0 CPU, 16.0 NPU, ...
```

#### 多机配置

对于多机配置，需要调整 `device_mapping` 以覆盖跨节点的 NPU。例如，2 节点 × 8 NPU：

```yaml
num_gpus_per_node: 8

# 训练在 Node0 的 NPU 0-7 上执行
actor_train:
  strategy_args:
    strategy_name: fsdp2_train
    strategy_config:
      fsdp_size: 8
      param_dtype: bf16
      reduce_dtype: bf16
      reshard_after_forward: true
      offload_policy: true
  device_mapping: list(range(0,8))

# 推理在 Node1 的 NPU 0-7 上执行
actor_infer:
  strategy_args:
    strategy_name: vllm
    strategy_config:
      gpu_memory_utilization: 0.8
      max_model_len: 8000
  device_mapping: list(range(8,16))

# Reference 模型共享推理 NPU
reference:
  strategy_args:
    strategy_name: fsdp2_infer
    strategy_config:
      fsdp_size: 8
      param_dtype: bf16
      reduce_dtype: bf16
      reshard_after_forward: true
      offload_policy: true
  device_mapping: list(range(8,16))
```

包含数据准备和 Reward Worker 的完整多机配置样例参见 [NPU 端到端配置样例](ascend_npu_examples.md#样例-3多机分布式训练)。

#### 多机重要注意事项

- **需要共享存储：** 模型权重、训练数据和 checkpoint 必须能在所有节点以相同路径访问。请将 NFS 或其他共享文件系统挂载到每个容器中。
- **网络要求：** 所有节点必须位于同一二层网络中。所有 worker 节点都必须能访问 head 节点的 6379 端口。
- **HCCL 网络接口：** 所有节点上的 `HCCL_SOCKET_IFNAME` 必须一致，并且应对应高速互联网络（如 RoCE）。可使用 `ip addr` 或 `hccn_tool` 识别正确网卡。

## 步骤 6：监控与评估

### 6.1 训练监控

ROLL 内置支持 TensorBoard。请在配置中指定日志目录：

```yaml
track_with: tensorboard
tracker_kwargs:
  log_dir: ./output/tensorboard/rlvr_npu
```

启动 TensorBoard：

```bash
tensorboard --logdir ./output/tensorboard/rlvr_npu --port 6006
```

建议重点监控以下指标：

| 指标 | 说明 |
| ---- | ---- |
| `time/step_total` | 每步总耗时 |
| `time/step_generate` | 推理生成耗时 |
| `time/step_train` | 训练更新耗时 |
| `train/loss` | 训练损失 |
| `train/lr` | 当前学习率 |
| `reward/mean` | 平均奖励 |
| `response_length/mean` | 平均生成长度 |

### 6.2 验证评估

流水线会按 `eval_steps` 间隔自动运行验证评估。验证结果包括：

| 指标 | 说明 |
| ---- | ---- |
| `val_correct/all/mean` | 所有验证样本的准确率 |
| `val_correct/<tag>/mean` | 每个 tag 分组的准确率（如 `val_correct/math_rule/mean`） |

验证准确率是衡量 RLVR 训练效果的核心指标。随着训练推进，该指标通常应逐步提升。

### 6.3 生成样例

训练过程中，每隔 `logging_steps` 步会将生成样例打印到日志中，便于直观评估模型输出质量。

## 步骤 7：从 Checkpoint 恢复

### 7.1 Checkpoint 保存

流水线会按照 `save_steps` 间隔，自动将 checkpoint 保存到 `checkpoint_config.output_dir`：

```yaml
checkpoint_config:
  type: file_system
  output_dir: /data/models/${exp_name}

save_steps: 100
```

### 7.2 从 Checkpoint 恢复

将 `resume_from_checkpoint` 设置为 checkpoint 路径即可恢复训练：

```yaml
resume_from_checkpoint: ./output/models/qwen3-30BA3B-rlvr-npu/checkpoint-100
```

或者在启动命令中覆盖该参数：

```bash
python examples/start_rlvr_pipeline.py \
    --config_path <config_dir> \
    --config_name rlvr_npu \
    resume_from_checkpoint=./output/models/qwen3-30BA3B-rlvr-npu/checkpoint-100
```

## 设备映射参考

以下是 RLVR 常见的 NPU 资源分配模式。可以根据工作负载和硬件情况选择共卡模式（训练和推理共享 NPU）或分离模式：

### 8 卡单机（7B 模型）

| 组件 | NPU | 数量 | 说明 |
| ---- | --- | ---- | ---- |
| actor_train | 0-3 | 4 | FSDP2 + CPU offloading |
| actor_infer | 4-7 | 4 | vLLM-Ascend |
| reference | 4-7（共享） | - | fsdp2_infer，与 actor_infer 共享 |
| reward workers | CPU | - | 数学规则和代码沙箱运行在 CPU 上 |

### 16 卡单机（A3，7B 模型）

| 组件 | NPU | 数量 | 说明 |
| ---- | --- | ---- | ---- |
| actor_train | 0-7 | 8 | FSDP2 |
| actor_infer | 8-15 | 8 | vLLM-Ascend |
| reference | 8-15（共享） | - | fsdp2_infer，与 actor_infer 共享 |
| reward workers | CPU | - | 数学规则和代码沙箱运行在 CPU 上 |

### 2×8 卡多机（7B 模型）

| 组件 | NPU | 数量 | 说明 |
| ---- | --- | ---- | ---- |
| actor_train | Node0: 0-7 | 8 | FSDP2 + CPU offloading |
| actor_infer | Node1: 0-7 | 8 | vLLM-Ascend |
| reference | Node1: 0-7（共享） | - | fsdp2_infer，与 actor_infer 共享 |
| reward workers | CPU | - | 数学规则和代码沙箱运行在 CPU 上 |

## NPU 上支持的 Reward Worker

NPU 上支持以下 RLVR Reward Worker：

| Reward Worker | 类 | NPU 兼容性 | 说明 |
| ------------- | --- | ---------- | ---- |
| Math Rule Reward | `MathRuleRewardWorker` | ✅ 支持 | 基于规则评估，运行在 CPU 上 |
| Code Sandbox Reward | `CodeSandboxRewardWorker` | ✅ 支持 | 在沙箱中执行代码，运行在 CPU 上 |
| LLM Judge Reward | `LLMJudgeRewardWorker` | ✅ 支持 | 需要额外 NPU 用于 judge 模型推理 |
| IFEval Rule Reward | `GeneralRuleRewardWorker` | ✅ 支持 | 基于规则评估，运行在 CPU 上 |
| CrossThinkQA Reward | `CrossThinkQARuleRewardWorker` | ✅ 支持 | 基于规则评估，运行在 CPU 上 |

:::caution
使用 `LLMJudgeRewardWorker` 时，judge 模型需要独立的 NPU 设备进行推理。请确保在 `device_mapping` 中为 judge 模型分配独立 NPU，不要与 `actor_infer` 或 `actor_train` 共享。
:::

## GPU 到 NPU 配置迁移 Checklist

将已有 GPU RLVR 配置迁移到 NPU 时，可使用以下 checklist：

- [ ] 将 `actor_train.strategy_args.strategy_name` 从 `megatron_train` 改为 `fsdp2_train`
- [ ] 将 `actor_train.strategy_args.strategy_config` 改为 FSDP2 配置（7B+ 模型使用 `offload_policy: true`）
- [ ] 将 `reference.strategy_args.strategy_name` 从 `megatron_infer` 改为 `fsdp2_infer`
- [ ] 将 `reference.strategy_args.strategy_config` 设置为与 `actor_train` 一致的 FSDP2 配置
- [ ] 在 `actor_train.model_args` 和 `reference.model_args` 中添加 `attn_implementation: fa2`
- [ ] 移除所有 `flash_attn` 引用
- [ ] 移除所有 Megatron 专属配置（如 `tensor_model_parallel_size`、`pipeline_model_parallel_size`）
- [ ] 如果使用 `llm_judge` reward worker，确认它有独立的 NPU 分配

## 常见问题

### 首次推理请求极慢

模型加载后的首次推理请求会触发算子编译，可能需要几分钟。这是一次性开销。可通过以下方式缓解：

1. 启用算子编译缓存（参见 [NPU 环境配置指南](ascend_npu_env_config.md) 中的 `ACL_OP_COMPILER_CACHE_MODE`）。
2. 在正式训练循环开始前运行一次 warmup 请求。

### 7B 模型在 4 张 NPU 上 OOM

如果在 4 张 NPU 上运行 7B 模型时遇到 OOM：

1. 切换到 `fsdp2_train` 策略并设置 `offload_policy: true`。
2. 将 `per_device_train_batch_size` 降到 1。
3. 相应增大 `gradient_accumulation_steps`。
4. 降低 vLLM 配置中的 `max_model_len`（如从 8192 降到 4096）。

### HCCL 通信超时

参见 FAQ 中的 [HCCL 通信超时或失败](ascend_npu_faq.md#hccl-通信超时或失败)。

### vLLM-Ascend 导入错误

请确认 CANN 环境已经正确 source：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
```

### triton 冲突

在 NPU 上，`triton` 包会与 `triton-ascend` 冲突。可通过以下方式修复：

```bash
pip uninstall -y triton triton-ascend
pip install triton-ascend==3.2.1 --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi
```

更多排障建议参见 [Ascend NPU FAQ](ascend_npu_faq.md)。

## 声明

ROLL 中提供的 Ascend 支持代码皆为参考样例，生产环境使用请通过官方正式途径沟通。
