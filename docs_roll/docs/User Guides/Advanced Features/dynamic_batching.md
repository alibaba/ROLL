# ROLL Dynamic Batching

The ROLL framework supports **Dynamic Batching** for rollout batches. This feature minimizes invalid token computation and improves overall computational efficiency. This document provides a detailed guide on how to use this feature.

## Glossary

- attention_mask: data in the rollout batch ,where `1` represents a real token and `0` represents a `pad_token`
- micro_batch (mbs): The micro-batch during the model forward pass.
- num_micro_batches: The number of micro_batch in one mini-batch.
- micro_batch_size: The number of sequences in the micro_batch.
- micro_batch_seqlen: The sequence length in the micro_batch.
- dp_size, dp_rank, shard: The size of data parallelism, the specific rank within the data parallel group and the training data in the data parallel group.
- vpp: Virtual Pipeline Model Parallelism; an efficient pipeline parallel technique supported by the Megatron-LM framework.

## Introduction

In Reinforcement Learning (RL) training, the data generated during rollout phase has a **long-tail** effect, that the sequence lengths vary significantly. This phenomenon is even more pronounced in **Agentic Pipelines**, where training data is generated through multi-turn interactions with an environment.

In the train step of RL, all samples in a rollout batch are typically padded to a fixed `max_len`. Consequently, these pad tokens are included in the calculation, leading to a waste of computational resources.

To address this and improve efficiency, the core idea of Dynamic Batching is:
- Partition the rollout batch across DP (Data Parallel) Ranks according to actual tokens and ensure a balanced workload.
- The sequence of samples is rearranged so that samples with similar lengths are grouped together, to remove as many pad tokens as possible.

## Example
The following example briefly illustrates the process of Dynamic Batching in ROLL.

**Assumptions:** `dp_size=2`, `num_seqs=8`, `max_tokens_microbatch=10`, `sequence_length_round=2`

Original input `attention_mask`
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
The corresponding `seq_lens` are:
```bash
seq_lens:
[7, 6, 8, 5, 1, 3, 8, 6]
```

As shown, the number of actual tokens varies significantly between sequences, causing the waste of GPU resources for processing `pad_tokens`.

To optimize efficiency, ROLL Dynamic Batching follows these steps to eliminate pad tokens within a `micro_batch`:

**1. Sort and Shard:**  A shard represents the training data within each dp_rank. By default, the data is sharded in order. In Dynamic Batching, sequences are first sorted by their actual length and then sharded to ensure that the number of tokens is balanced across dp_ranks.
```bash
# seq_lens after sorting:
[1, 3, 5, 6, 6, 7, 8, 8]

# Partition into dp_size shards:
shard0:
  [1, 5, 6, 8]
shard1:
  [3, 6, 7, 8]
```

**2. Micro-batch Partition:** 

The partition process consider the following two parameters:

- `max_tokens_per_microbatch`: The maximum number of tokens allowed in one micro_batch. `micro_batch_size * micro_batch_seqlen` cannot exceed this value. If it is exceeded, a new micro_batch must be created.
- `sequence_length_round`: The `micro_batch_seqlen` must be a multiple of this value. For example, the sequence lengths in a micro_batch is [200, 240] and `sequence_length_round` is 64, the sequences in this micro-batch must be padded to a length of 256.

The shard partition process for Dynamic Batching aims to find the split that maximizes the number of tokens in a micro-batch, while ensuring the numer of tokens in mirco_batch cannot exceed `max_tokens_per_microbatch`. It also ensures that the sequence length for each micro-batch is padded up to a multiple of `sequence_length_round`.

The process is detailed as follows:



```bash
shard0:
  mbs0: # Padding length 6 
    [1, 0, 0, 0, 0, 0 
     1, 1, 1, 1, 1, 0]
  mbs1: # Padding length 8
    [1, 1, 1, 1, 1, 1, 0, 0]
  mbs2: # Padding length 8
    [1, 1, 1, 1, 1, 1, 1, 1]

shard1:
  mbs0: # Padding length 6
    [1, 1, 1, 0, 0, 0
     1, 1, 1, 1, 1, 1]
  mbs1: # Padding length 8
    [1, 1, 1, 1, 1, 1, 1, 0]
  mbs2: # Padding length 8
    [1, 1, 1, 1, 1, 1, 1, 1]
```
In this example, the original total token count was `80` (`8 * 10`). After Dynamic Batching, the total token count is reduced to 56, removing 30% of the `pad_tokens`.

**3. Support Virtual Pipeline Model Parallel :** Split micro-batches with more tokens and `micro_batch_size > 1`. This ensures the number of micro-batches is an integer multiple of `pp_size` (compatible with Megatron).

Since the `num_microbatches` in the original example is not divisible by pp_size, mbs0 is selected and split into two mbs, as follows:

```bash
shard0:
  mbs0: # padding length 6 
    [1, 0, 0, 0, 0, 0]
  mbs1: # padding length 6 
    [1, 1, 1, 1, 1, 0]
  mbs2: # padding length 8
    [1, 1, 1, 1, 1, 1, 0, 0]
  mbs3: # padding length 8
    [1, 1, 1, 1, 1, 1, 1, 1]
shard1:
  mbs0: # padding length 6
    [1, 1, 1, 0, 0, 0]
  mbs1: # padding length 6
    [1, 1, 1, 1, 1, 1]
  mbs2: # padding length 8
    [1, 1, 1, 1, 1, 1, 1, 0]
  mbs3: # padding length 8
    [1, 1, 1, 1, 1, 1, 1, 1]

```

## Configuration Parameters

The Dynamic Batching parameters are divided into `train` and `infer`:

### Train
- `use_dynamic_batching_in_train`: Whether to enable this feature during the `train_step`.
- `max_tokens_per_microbatch_in_train`: The maximum number of tokens allowed per micro-batch during training.
- `sequence_length_round_in_train`: The sequence length of each micro-batch must be divisible by this value. It should also be divisible by `tensor_model_parallel_size * context_parallel_size`. Common values are 128 or 64.

### Infer
- `use_dynamic_batching_in_infer`: Whether to enable this during phases that do not require gradient update (e.g., `compute_log_probs`).
- `max_tokens_per_microbatch_in_infer`: Same as the train, usually be higher depending on gpu memory.
- `sequence_length_round_in_infer`: Same as train.

## Full Configuration

```yaml
actor_train:
  # Flash Attention is recommended when using both Dynamic Batching and Context Parallel
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