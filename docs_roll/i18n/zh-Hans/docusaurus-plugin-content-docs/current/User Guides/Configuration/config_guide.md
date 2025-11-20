# ROLL 配置指南

ROLL 框架通过 YAML 配置文件来定义实验参数。本文档将详细介绍配置项的含义和使用方法。

## Pipeline 配置

```yaml
exp_name: "agentic_pipeline"
seed: 42
rpc_timeout: 3600
logging_dir: ./output/logs
output_dir: ./output
render_save_dir: /data/oss_bucket_0/yali/output/render
system_envs:
  USE_MODELSCOPE: '1'

track_with: tensorboard
tracker_kwargs:
  log_dir: /data/oss_bucket_0/yali/llm/tensorboard/roll_exp/agentic_sokoban

num_gpus_per_node: 8

max_steps: 1024
save_steps: 10000
logging_steps: 1
eval_steps: 10
resume_from_checkpoint: false

rollout_batch_size: 64
prompt_length: 2048
response_length: 4096
num_return_sequences_in_group: 8
```

### 基本信息与通用配置
- `exp_name`: 当前实验的名称，用于标识和组织输出文件、日志等。默认是从 Python 文件名派生。
- `seed`: 用于初始化随机数生成器。设置固定的种子可以确保实验的可复现性。
- `rpc_timeout`: 远程过程调用（RPC）的超时时长，单位为秒。用于 Ray Actor 之间通信。如果一个调用在此时间内没有响应，则会抛出超时错误。
- `output_dir`: 模型预测结果和检查点（checkpoints）的输出目录。
- `logging_dir`: 存储日志文件的目录。
- `track_with`: 用于跟踪实验进度的工具类型。可选 wandb (Weights & Biases), tensorboard (TensorBoard), swanlab 或 stdout (标准输出)。
- `tracker_kwargs`: 传递给所选跟踪器类的额外关键字参数（字典）。例如，WandB 的 API 密钥、项目名称等。

### 训练/评估流程配置
- `max_steps`: 训练的最大步数。如果大于 0，则设置流水线执行的总步数。训练将在达到此步数时停止。
- `save_steps`: 保存模型检查点的频率。每隔 X 个更新步数保存一次模型检查点。
- `logging_steps`: 记录训练指标的频率。每隔 X 个更新步数记录一次训练信息（例如损失、指标等）。
- `eval_steps`: 评估频率。每隔 X 个更新步数执行一次评估。
- `resume_from_checkpoint`: 是否从检查点恢复训练。如果设置为 True，则从 output_dir 中最近的检查点恢复。
- `checkpoint_config`: 检查点相关的配置信息，这个字段会被写入 worker_config。

### 批处理大小与序列长度配置
- `rollout_batch_size`: 在每次推理批次中，要进行 Rollout 的样本数量。
- `val_batch_size`: 在每次验证批次中，要进行 Rollout 的样本数量。
- `prompt_length`: 提示（输入）的最大长度（以 token 为单位）。如果实际提示更短，会填充到此长度；如果更长，可能会被截断。
- `response_length`: LLM 生成的响应（输出）的最大长度（以 token 为单位）。如果 LLM 生成的响应更短，会填充；如果更长，会被截断。
- `sequence_length`: 要填充的最大序列长度（以 token 为单位）。这通常指 LLM 模型的总上下文窗口大小，包括提示和生成的响应。

### 分布式训练配置
- `local_rank`: 分布式训练中的本地排名（在当前节点内的排名）。对于分布式训练，通常由系统自动设置；如果不是分布式训练，则设置为 -1。
- `num_nodes`: 可用于分布式训练的节点（物理服务器）数量。如果设置为 1，则表示在单个节点上进行分布式训练。
- `num_gpus_per_node`: 指定每个节点上可用的 GPU 数量。当节点数量大于 1 时，此参数应请求整个节点上的 GPU 总数。确保在多节点设置中 GPU 资源分配与请求一致。

### 调度与请求管理
- `generate_opt_level`: 控制 LLM 生成（推理）的优化级别。设置为 0 时，使用基础批次生成接口；设置为 1 时，使用调度器处理请求。
- `is_num_return_sequences_expand`: 是否在提示（prompts）中复制 num_return_sequences 次。如果为 True，LLM 会为每个输入提示生成多个独立的响应，而不是只生成一个。
- `max_running_requests`: 在 LLM 推理服务器上可以同时处理的最大请求数量。这限制了并行推理的并发度。
- `is_use_additional_prompts`: 是否使用除常规批次大小之外的额外提示进行处理。
- `max_additional_running_prompts`: 在 batch_size 之外，可以额外运行的提示数量。这可能用于处理一些特殊或低优先级的请求，而不会阻塞主批次。

### RLVR Pipeline 常用配置

- `num_return_sequences_in_group`: 对于每个提示，LLM 要生成序列的数量。请注意，它的值会按比例扩大实际的训练全局批次样本量。换句话说，实际的训练全局批次大小等于 `num_return_sequences_in_group` * `rollout_batch_size`。

#### PPO算法核心参数
- `ppo_epochs`: 每个样本批次（即收集到一批经验数据后）进行优化的迭代次数。在一个 PPO 训练循环中，Agent 首先收集数据，然后利用这些数据进行多次梯度更新。
- `max_grad_norm`: 梯度裁剪的最大范数。用于防止梯度爆炸。
- `l2`: L2 正则化系数，用于惩罚大的权重值以防止过拟合。
- `lambd`: 广义优势估计（GAE, Generalized Advantage Estimation）中的 lambda 参数。控制偏差-方差权衡。值接近 1 减少方差，值接近 0 减少偏差。
- `gamma`: 强化学习中的折扣因子。用于折算未来奖励的重要性。值越接近 1，模型越重视长期奖励。
- `kl_penalty`: KL 散度惩罚的计算方式。KL 散度用于衡量新旧策略之间的差异，防止策略更新过大。
    - 'kl': model_logp - ref_logp (新策略 log 概率减去参考策略 log 概率)。
    - 'abs': abs(kl) (KL 散度的绝对值)。
    - 'mse': mse(kl) (KL 散度的均方误差)。
    - 'full': 分布中所有 token 的实际 KL 散度。
- `init_kl_coef`: 初始的 KL 惩罚系数（用于自适应和线性控制）。这个系数乘以 KL 散度项，作为损失的一部分。
- `kl_horizon`: 自适应 KL 控制的周期。

#### PPO奖励/优势处理
- `use_reward_scaling`: 是否对奖励进行缩放。
- `add_len_reward`: 是否添加基于序列长度的奖励。
- `reward_clip`: 对奖励进行裁剪的值，防止极端奖励影响训练。
- `difficulty_loss_weight`: 是否使用难度损失权重。
- `length_loss_weight`: 是否使用长度损失权重。
- `use_reward_norm`: 是否使用奖励归一化。仅当 use_reward_scaling 为 True 时适用。
- `whiten_rewards`: 在计算优势值之前，是否对奖励进行白化处理（使其均值为 0，方差为 1）。
- `whiten_advantages`: 是否对优势值进行白化处理。有助于稳定训练。
- `advantage_clip`: 优势值裁剪的范围。
- `adv_estimator`: 优势值的估计方法。
    - 'gae': 广义优势估计（GAE）。
    - 'reinforce': REINFORCE 算法中的优势估计。
    - 'grpo': Grouped Relative Policy Optimization 中的优势估计。
- `norm_mean_type`: 奖励归一化的均值计算方式。
    - 'batch': 批次内的所有奖励的均值。
    - 'group': 提示组内部的均值。
    - 'running': 使用动态更新的统计量进行均值计算。
    - None: 归一化的时候不减去均值。
- `norm_std_type`: 奖励归一化的标准差计算方式。
    - 'batch': 批次内的所有奖励的标准差。
    - 'group': 提示组内部的标准差。
    - 'running': 使用动态更新的统计量进行标准差计算。
    - None: 归一化的时候不除以标准差。

#### PPO 损失函数组件
- `add_token_level_kl`: 是否添加 token 级别的 KL 散度惩罚。
- `critic_warmup`: Critic 模型在正式训练开始前的预训练步数。
- `use_kl_loss`: 是否使用 KL 散度损失。
- `kl_loss_coef`: KL 散度损失项的系数。
- `entropy_loss_coef`: 熵损失项的系数。增加熵可以鼓励策略探索。
- `sft_loss_coef`: SFT (Supervised Fine-tuning) 损失的系数，用于正样本（例如，如果有监督微调的数据）。
- `use_topr_loss`: 是否使用 TPRO (Trigonometric Policy Regularization with Offset) 损失。
- `rl_loss_coef`: 强化学习损失项的系数。
- `dual_clip_loss`: 是否使用双裁剪损失。PPO 损失函数的一种变体。
- `loss_agg_mode`: 损失聚合的方式。
    - 'token-mean': Token 级别的均值。
    - 'seq-mean-token-sum': 序列级别的均值，token 级别的求和。
    - 'seq-mean-token-mean': 序列级别和 token 级别的均值。
    - 'seq-mean-token-sum-norm': 序列级别的均值，token 级别的归一化求和。

### Agentic Pipeline 配置

#### 奖励归一化配置
- `grouping`: 定义奖励归一化时的分组方式，可选 'state'、'batch'和'inductive'。
- `method`: 定义具体的归一化方法，可选 'asym_clip'、'identity'和'mean_std'。

#### 环境管理器配置
- env_groups: 训练期间环境组的数量。每个环境组可能并行运行。
- group_size: 在同一个组内，环境配置和环境种子（prompt）被确保是相同的。这对于控制实验变量和确保可复现性很重要。
- tags: 环境的标签列表，用于标识和选择要使用的环境类型（例如 "SimpleSokoban"）。
- n_groups: 如果未设置，所有环境名称将平均分配到组中。在同一个组中，环境配置和环境种子（prompt）在每次生成中都是相同的。
- max_traj_per_env: 每个环境可以 Rollout 的最大轨迹数量。-1 表示没有限制。
- format_penalty: 当 LLM 生成的响应不符合预期格式时所施加的惩罚值。这是一个负值，会降低不合格响应的奖励。
- worker_cls: 环境管理器将使用的具体工作器类的路径。这个类实现了环境交互的逻辑。

有关 RLVR/Agentic Pipeline配置和Reward设置的更多详细信息，还可以参阅 [RLVR Pipeline Start](../../User%20Guides/Pipeline/rlvr_pipeline_start.md) 和 [Agentic Pipeline Start](../../User%20Guides/Pipeline/agentic_pipeline_start.md)

## Worker配置

### ActorTrain/ActorInfer/Critic/Reference

```yaml
actor_train:
  model_args:
    dtype: bf16
    disable_gradient_checkpointing: False
    ...
  training_args:
    learning_rate: 1.0e-6
    weight_decay: 0
    per_device_train_batch_size: 1
    gradient_accumulation_steps: 32
    warmup_steps: 20
    ...
  data_args:
    template: native
    file_name: xxx/train.json
    prompt: instruction
  strategy_args:
    strategy_name: megatron_train  # 训练策略：deepspeed_train 或 megatron_train
    strategy_config:
      tensor_model_parallel_size: 1
      pipeline_model_parallel_size: 1
      expert_model_parallel_size: 1
  infer_batch_size: 4
  device_mapping: list(range(0,16))  # 使用的设备 ID 列表
actor_infer:
  model_args:
    ...
  generating_args:
    max_new_tokens: ${response_length}
    temperature: 0.99
    ...
  strategy_args:
    strategy_name: vllm  # 推理策略：vllm, sglang 或 hf_infer
    strategy_config:
      gpu_memory_utilization: 0.6
      block_size: 16
      max_model_len: 8000
  num_gpus_per_worker: 1  # 每个 worker 分配的 GPU 数量
  device_mapping: list(range(0,16))
reference:
  model_args:
    ...
  strategy_args:
    strategy_name: megatron_infer
    strategy_config:
      tensor_model_parallel_size: 1
  device_mapping: list(range(0,16))
```

- `world_size`: 参与这个特定角色集群的总数量。例如，如果有多个 actor_train 实例，这就是它们的总数。
- `device_mapping`: 当 worker 进行训练时要使用的设备 ID 列表。 配置所有使用 GPU 的 worker，包括`actor_train`、`actor_infer`、`critic` 和 `reference`。例如 list(range(0,16)) 表示使用 ID 为 0 到 15 的 GPU。
- `num_gpus_per_worker`: 分配给每个 worker 的 GPU 数量。 仅适用于`actor_infer`。如果一个 actor_infer 需要多个 GPU（例如用于模型并行），则设置此参数。
- `model_update_frequency`: 模型更新的频率。例如，每多少步或每多少个事件触发一次模型更新。
- `infer_batch_size`: 用于推理或计算 logprobs 时的批次大小。 这通常是单个推理请求的内部批次大小。

### 模型参数 (**model_args**)

- `model_args.dtype`: 设置模型的数据类型，可以是 fp32 (单精度浮点数), bf16 (BFloat16), 或 fp16 (半精度浮点数)。如果不设置，则使用配置的 torch_dtype。选择合适的数据类型可以平衡计算速度、内存消耗和精度。
- `model_args.disable_gradient_checkpointing`: 禁用梯度检查点。仅当 `actor_train` 的 `strategy_name` 为 `deepspeed_train` 时适用。梯度检查点是一种内存优化技术，通过在反向传播时重新计算部分激活值来减少显存占用。禁用它会增加内存消耗但可能略微加速计算。

### 数据参数 (**data_args**)

如何配置`actor_train`下的 data_args：

- `data_args.template`: 用于在训练和推理期间构建提示的聊天模板。设置为`native`时，将使用分词器（tokenizer）的默认聊天模板`tokenizer.apply_chat_template`来构建提示。
- `data_args.file_name`: 训练数据的文件路径。支持的格式包括 JSON、JSONL 和 CSV。
- `data_args.prompt`: 在数据文件中用作提示的列名。
- `data_args.messages`: 在数据文件中用作消息的列名（与 prompt 冲突，两者只能选其一）。

### 生成参数 (**generating_args**)

如何配置`actor_infer`下的 generating_args：

- `generating_args.max_new_tokens`: 生成文本的最大长度（以 token 为单位）。这限制了 LLM 每次推理调用可以输出多少新内容。
- `generating_args.temperature`: 用于采样的温度值。较高的温度会使生成结果更随机、更有创造性；较低的温度会使结果更确定、更保守。

### 策略参数 (**strategy_args**)

- `strategy_args.strategy_name`: 训练/推理策略的名称。
    - 用于训练的策略有：`deepspeed_train`（使用 DeepSpeed）或 `megatron_train`（使用 Megatron-LM）。
    - 用于推理的策略有：`vllm`、`sglang` 或 `hf_infer`（使用 Hugging Face 的默认推理）。
- `strategy_args.strategy_config`: 训练/推理策略的详细配置，它将作为参数传递给 `strategy_name`对应的构造函数。例如，`strategy_config.tensor_model_parallel_size` 用于 `megatron_train` 策略，`strategy_config.gpu_memory_utilization` 用于 `vllm` 策略。

以下列出了常用的策略配置：

#### Megatron 策略配置

- `tensor_model_parallel_size`: 张量模型并行度。将模型的层内（例如矩阵乘法）的计算和内存分割到多个 GPU 上。
- `pipeline_model_parallel_size`: 流水线模型并行度。将模型的不同层或层组分配到不同的 GPU 上，形成一个流水线，以并行处理不同批次的数据。
- `expert_model_parallel_size`: : 专家模型并行度。在 Mixture-of-Experts (MoE) 模型中，将不同的专家（expert）分配到不同的 GPU 上。
- `context_parallel_size`: 上下文并行度。 一种用于处理超长序列的并行策略，将序列分割后并行处理。
- `virtual_pipeline_model_parallel_size`: 流水线中的虚拟流水线数量。用于改善流水线并行的效率和负载均衡。
- `sequence_parallel`: 启用序列并行优化。针对 Transformer 模型中的序列处理进行优化，减少通信开销。
- `recompute_granularity`: 激活值重计算粒度。用于内存优化，在反向传播时重新计算激活值以节省显存。
    - full: 整个 Transformer 层都会被重新计算。
    - selective: 仅重新计算 Transformer 层中的核心注意力部分。
- `moe_layer_recompute`: 内存优化，对 MoE 层进行检查点以节省激活内存。
- `moe_token_dispatcher_type`: 使用的 token 调度器类型，选项有 'allgather' 和 'alltoall'。
- `moe_grouped_gemm`: 为 MoE 专家启用分组 GEMM (通用矩阵乘法)。
- `moe_shared_expert_overlap`: 启用共享专家计算与调度器通信之间的重叠。
- `overlap_grad_reduce`: 如果为 true，在分布式优化器中，将梯度 All-reduce 过程与反向传播计算重叠。

#### VLLM 策略配置

- `gpu_memory_utilization`: 用于模型执行器的 GPU 内存占比。 例如 0.6 表示使用 60% 的 GPU 内存。
- `block_size`: token 块大小，用于连续的 token 块。影响 VLLM 内部的内存管理效率。
- `max_model_len`: 模型上下文长度。如果未指定，将从模型配置中自动推导。
- `load_format`: 加载模型权重的格式。由于模型会在开始时进行"更新"，此值可以设置为 dummy。

#### SGLang 策略配置

- `mem_fraction_static`: 用于模型权重和 KV 缓存等静态内存的 GPU 内存占比。 如果 KV 缓存构建失败，请增加此值；如果 CUDA 内存不足，请减小此值。
- `load_format`: 加载模型权重的格式。（同 VLLM，可设为 dummy）

#### DeepSpeed 策略配置

在`./examples/config/` 中有 DeepSpeed 配置文件，可以在默认列表中重写以进行策略配置。
例如，要使用 deepspeed_zero2 策略，请将以下内容添加到您的配置中：

```yaml
defaults:
  - ../config/envs@_here_
  - ../config/deepspeed_zero@_here_
  - ../config/deepspeed_zero2@_here_   # 引入 deepspeed_zero2 策略配置
  - ../config/deepspeed_zero3@_here_
  - ../config/deepspeed_zero3_cpuoffload@_here_
actor_train:
  strategy_args:
    strategy_name: deepspeed_train
    strategy_config: ${deepspeed_zero2}
```

### 训练参数 (**training_args**)

用于配置训练参数，例如`learning_rate`(学习率)、`weight_decay`(权重衰减)、`warmup_steps`(预热步数) 等。

- `training_args.per_device_train_batch_size`: 在每个设备上进行训练时使用的批次大小。
- `training_args.gradient_accumulation_steps`: 梯度累积的步数。

在 DeepSpeed 训练中，全局训练批次大小是`per_device_train_batch_size` \* `gradient_accumulation_steps` \* world_size (即`actor_train`/`critic`的`device_mapping`长度)。

在 Megatron 训练中，全局训练批次大小是`per_device_train_batch_size` \* `gradient_accumulation_steps` \* world_size / `tensor_model_parallel_size` / `pipeline_model_parallel_size` / `context_parallel_size` (不需要除以`expert_model_parallel_size`)。

如果你想在每次 Rollout 中执行一次优化步骤，则应设置`gradient_accumulation_steps`为 `rollout_batch_size` \* `num_return_sequences_in_group` \* `tensor_model_parallel_size` \* `pipeline_model_parallel_size` \* `context_parallel_size`/ `per_device_train_batch_size` / world_size.