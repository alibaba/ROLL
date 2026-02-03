# ROLL Dynamic Batching

ROLL 框架支持对 Rollout Batch 做 **Dynamic Batching** 功能，尽量减少无效 token 计算，使得计算效率更高，本文档详细介绍如何使用这一功能。

## 术语列表

- attention_mask: rollout batch中的数据，其中 `1` 表示实际需要被计算的token，`0` 表示 pad_token；
- micro_batch (mbs): 模型前向处理时的微批次；
- num_micro_batches: 每个mini-batch中micro_batch数量；
- micro_batch_size: 每个微批次中序列数量； 
- micro_batch_seqlen: 每个微批次中序列长度；
- dp_size, dp_rank, shard: 数据并行时的并行数量，以及在并行组中的编号，每个数据并行组中的训练数据；
- vpp: Virtual Pipeline Model Parallel，Megatron-LM框架中支持的一种高效流水线并行技术；

## 简介

在RL训练场景中，每次rollout出来的数据具有十分显著的长尾效应，即序列长度不一致，尤其在Agentic Pipeline中，由于训练数据是多轮和Env相互产生的，导致这种长尾现象更为显著。

在训练时，通常会将一个rollout batch中的所有样本按照一个`max_len` pad到最长，这些pad_token也会参与计算，造成计算资源浪费；

为了解决这一问题，提高计算效率，Dynamic Batching技术核心思路是：
- 对整个rollout batch中的样本在DP Rank维度上按照token数进行划分，使得计算资源尽量均衡；
- 改变样本中序列的顺序，使得临近的样本，长度尽量接近，能够去掉尽量多的pad token；

## 示例
下面通过一个例子，简要说明 ROLL 中 Dynamic Batching 流程

假设 `dp_size=2`, `num_seqs=8`,  `max_tokens_microbatch=10`, `sequence_length_round=2`

原始输入 `attention_mask` 如下
```bash
attention_mask:
[1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
[1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
[1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
```
其对应的 `seq_lens` 如下:

```bash
seq_lens:
[7, 6, 8, 5, 1, 3, 8, 6]
```

可见序列之间的实际 token 数量是不均衡的，会浪费大量 GPU 时间在处理 `pad_token` 上

为了计算效率，ROLL Dynamic Batching 基于下面的步骤来消除 `micro_batch` 中的 pad_token，从而达到资源利用的最大化。

1. shard表示每个`dp_rank`中的训练数据，默认按照顺序切分，在Dynamic Batching中会基于序列实际长度排序并切分shard，使得 `dp_rank` 之间的tokens数均匀

```bash
# seq_lens 排序后:
[1, 3, 5, 6, 6, 7, 8, 8]
# 切分成dp_size个shard
shard0:
  [1, 5, 6, 8]
shard1:
  [3, 6, 7, 8]
```

2. 对于每个shard划分 `micro_batch`；

划分时需要考虑如下两个参数：
- max_tokens_per_microbatch: 每个micro_batch中最大token数量，`micro_batch_size * micro_batch_seqlen` 不能超过这个值，如果超过需要再生成一个新的 `micro_batch`；
- sequence_length_round: `micro_batch_seqlen` 需要能够被这个值整除；假设micro_batch中的序列长度为 `[200, 240]`，`sequence_length_round=64`，则这个micro_batch需要pad成`[256, 256]`；

Dynamic Batching的划分shard流程就是找到小于max_tokens_per_microbatch的micro_batch中tokens数量最大的划分。且保证每个micro_batch的序列长度需要根据实际长度pad到 `sequence_length_round` 的倍数；

具体如下所示：

```bash
shard0:
  mbs0: # padding长度6 
    [1, 0, 0, 0, 0, 0 
     1, 1, 1, 1, 1, 0]
  mbs1: # padding长度8
    [1, 1, 1, 1, 1, 1, 0, 0]
  mbs2: # padding长度8
    [1, 1, 1, 1, 1, 1, 1, 1]
shard1:
  mbs0: # padding长度6
    [1, 1, 1, 0, 0, 0
     1, 1, 1, 1, 1, 1]
  mbs1: # padding长度8
    [1, 1, 1, 1, 1, 1, 1, 0]
  mbs2: # padding长度8
    [1, 1, 1, 1, 1, 1, 1, 1]
```
在这个随机mask矩阵中，原来token总数为 `attention_mask.size(0) * attention_mask.size(1) = 80`，经过 Dynamic Batching 之后的 token 数量为：56，remove掉了 `30%` 的 pad_token

3. 支持Virtual Pipelie Model Parallel，优先拆分tokens数量多且micro_batch_size > 1的micro_batch，使得micro_batch数量为pp_size整除倍(支持megatron)

原来的这个例子中 `num_micro_batches` 不能够被 `pp_size` 整除，因此选择 `mbs0`，将其拆分成两个 mbs，如下所示：

```bash
shard0:
  mbs0: # padding长度6 
    [1, 0, 0, 0, 0, 0]
  mbs1: # padding长度6 
    [1, 1, 1, 1, 1, 0]
  mbs2: # padding长度8
    [1, 1, 1, 1, 1, 1, 0, 0]
  mbs3: # padding长度8
    [1, 1, 1, 1, 1, 1, 1, 1]
shard1:
  mbs0: # padding长度6
    [1, 1, 1, 0, 0, 0]
  mbs1: # padding长度6
    [1, 1, 1, 1, 1, 1]
  mbs2: # padding长度8
    [1, 1, 1, 1, 1, 1, 1, 0]
  mbs3: # padding长度8
    [1, 1, 1, 1, 1, 1, 1, 1]

```



## 参数配置

与 Dynamic Batching 相关的参数如下，分为 train 和 infer 两个部分
- Train
  - use_dynamic_batching_in_train: 是否在 `train_step` 时开启；
  - max_tokens_per_microbatch_in_train: 训练时每个 micro_batch 最大 token 数量；
  - sequence_length_round_in_train: 训练时每个 micro_batch 的序列长度需要能被这个参数整除，需要能够被 `tensor_model_parallel_size * context_parallel_size` 整除，一般取 128,64 即可；
- Infer
  - use_dynamic_batching_in_infer: 是否在 `compute_log_probs` 等不需要梯度更新的环节开启；
  - max_tokens_per_microbatch_in_infer: 与train中含义相同，根据显存消耗情况可以大一些；
  - sequence_length_round_in_infer: 与train中含义相同；



## 完整配置

```yaml
actor_train:
  # 同时开启 Dynamic Batching 和 Context Parallel 时推荐使用 flash_attn
  system_envs:
    NVTE_FLASH_ATTN: '1'
    NVTE_FUSED_ATTN: '0'
    NVTE_UNFUSED_ATTN: '0'
  model_args:
    attn_implementation: fa2
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
    strategy_name: megatron_train
    strategy_config:
      tensor_model_parallel_size: 1
      pipeline_model_parallel_size: 1
      expert_model_parallel_size: 1
      use_distributed_optimizer: true
  device_mapping: list(range(0,8))
  infer_batch_size: 2
  use_dynamic_batching_in_train: true
  max_tokens_per_microbatch_in_train: 8192
  sequence_length_round_in_train: 128
  use_dynamic_batching_in_infer: true
  max_tokens_per_microbatch_in_infer: 16384
  sequence_length_round_in_infer: 128

actor_infer:
  model_args:
    disable_gradient_checkpointing: true
    dtype: bf16
  generating_args:
    max_new_tokens: 128 # single-turn response length
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
  device_mapping: list(range(0,8))

reference:
  model_args:
    attn_implementation: fa2
    disable_gradient_checkpointing: true
    dtype: bf16
    model_type: ~
  data_args:
    template: qwen2_5
  strategy_args:
    strategy_name: megatron_infer
    strategy_config: ~
  device_mapping: list(range(0,8))
  infer_batch_size: 2
  use_dynamic_batching_in_infer: true
  max_tokens_per_microbatch_in_infer: 16384
  sequence_length_round_in_infer: 128

```
