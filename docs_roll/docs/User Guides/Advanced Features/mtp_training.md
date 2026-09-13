# MTP (Multi-Token Prediction) Training Guide

## Overview

MTP (Multi-Token Prediction) is a technique that accelerates inference by predicting multiple future tokens in parallel. The ROLL framework supports training MTP models for both SFT (Supervised Fine-Tuning) and RL (Reinforcement Learning) scenarios.

## Speculative Decoding Principles

### The Bottleneck of Autoregressive Generation

Large language model text generation is an autoregressive process: each token generation requires a complete forward pass. For long-sequence generation (such as mathematical reasoning), this becomes a major performance bottleneck.

```
Traditional autoregressive generation:
  Step 1: Forward pass → Token 1
  Step 2: Forward pass → Token 2
  Step 3: Forward pass → Token 3
  ...
  Each token requires a complete forward pass
```

### The Idea Behind Speculative Decoding

Speculative Decoding breaks this bottleneck through a "predict-verify" approach:

1. **Draft Phase**: Use a small model to quickly generate K candidate tokens
2. **Verify Phase**: The main model verifies all K tokens in one forward pass
3. **Accept/Reject**: Accept tokens that match the main model's probability distribution, reject those that don't

```
Speculative Decoding:
  Draft: Small model quickly generates [Token 1, Token 2, Token 3, Token 4]
  Verify: Main model verifies all candidates in one forward pass
  Result: Accept first 3, reject the 4th

  Equivalent to: Generating 3 tokens with 2 forward passes (1 draft + 1 verify)
```

### Why Does This Speed Things Up?

Key insight: **The main model's forward pass can compute logits for multiple positions in parallel**.

In traditional generation, when generating a token, only the last position's logits are used, while computations for other positions are wasted. Speculative decoding leverages this by verifying multiple candidate tokens in a single main model forward pass, improving computational efficiency.

### What Determines the Speedup?

- **Acceptance Rate**: The closer the draft model's output distribution is to the main model, the higher the acceptance rate
- **Speculative Steps**: The number of candidate tokens generated per speculation
- **Draft Model Efficiency**: The inference speed of the draft model

An ideal draft model should:
1. Have an output distribution close to the main model (high acceptance rate)
2. Be fast at inference (low draft overhead)
3. Have small parameter count (low memory overhead)

## What is MTP?

MTP (Multi-Token Prediction) is an efficient draft model implementation. Unlike using an independent small model, MTP shares weights with the main model and has the following advantages:

### Difference from Regular LM

- **Regular LM**: Uses hidden state at position t to predict token at position t+1
- **MTP**: Uses hidden state at position t + token embedding at position t+1 to predict token at position t+2

```
Regular LM:    H(t) → predict(t+1)
MTP:           H(t) + E(t+1) → predict(t+2)
                ↑         ↑
          hidden state  embedding
```

### Advantages of MTP

1. **Weight Sharing**: MTP shares the main model's embedding and output layer, with minimal parameter increase (~5-10%)
2. **High Acceptance Rate**: MTP directly utilizes the main model's hidden states, naturally producing outputs close to the main model's distribution
3. **Simple Training**: Can be jointly trained with the main model without needing to train a separate draft model

### Use Cases for MTP

1. **Inference Acceleration**: As a draft model for speculative decoding to accelerate text generation
2. **RL Training Acceleration**: Accelerate rollout generation in scenarios like RLVR, improving training throughput

### Why Does RL Training Need MTP?

In RL training (such as RLVR), rollout generation is the main bottleneck:

1. **Large Generation Demand**: Each training round requires generating many samples
2. **Long-Sequence Generation**: Tasks like mathematical reasoning require long responses
3. **High Inference Engine Load**: The actor_infer worker is often the training bottleneck

Using MTP speculative decoding can significantly accelerate the rollout process and improve training throughput.

## Training Modes

ROLL supports three MTP training modes, configured via the `mtp_training_mode` parameter:

### 1. disabled (Default)

MTP weights are loaded but do not participate in training.

```yaml
actor_train:
  mtp_training_mode: disabled  # or omit the config
```

**Use Cases**:
- Only want to use pre-trained MTP for inference acceleration
- No need to update MTP weights

### 2. standalone (Recommended for RL)

MTP is trained independently with truncated gradients, not affecting the main model.

```yaml
actor_train:
  mtp_training_mode: standalone
```

**Characteristics**:
- MTP's gradient flow is truncated via `detach()`
- Main model gradients are not affected by MTP training
- Main model and MTP use different learning signals

**Use Cases**:
- **RL Training**: The main model needs to optimize based on rewards, while MTP needs to learn the main model's generation distribution
- Avoid RL instability affecting MTP

**How It Works**:

In standalone mode, the gradient flow between MTP and the main model is completely isolated:
- The main model optimizes based on RL rewards with normal gradient backpropagation
- MTP optimizes based on cross-entropy loss, but gradients do not backpropagate to the main model
- This allows MTP to stably learn the main model's generation distribution without being affected by RL training fluctuations

### 3. joint (Recommended for SFT)

MTP is trained jointly with the main model with complete gradient flow.

```yaml
actor_train:
  mtp_training_mode: joint
```

**Characteristics**:
- Main model and MTP share gradient flow
- MTP's loss affects main model parameters
- Main model and MTP optimize together

**Use Cases**:
- **SFT Training**: Want both main model and MTP to learn the target task simultaneously
- MTP serves as an auxiliary training objective

## Configuration Parameters

### Training Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mtp_training_mode` | `str` | `disabled` | MTP training mode: `disabled`, `standalone`, `joint` |
| `mtp_loss_scaling_factor` | `float` | See below | MTP loss scaling factor |

**mtp_loss_scaling_factor**:
- Default value is typically `0.3` (referencing DeepSeek-V3)
- In `standalone` mode, MTP loss is directly multiplied by this factor
- In `joint` mode, MTP loss participates in main model gradient updates

### Inference Engine Configuration (Speculative Decoding)

Currently, only vLLM in ROLL supports MTP speculative decoding. Configure it in `actor_infer`'s `strategy_config`:

```yaml
actor_infer:
  strategy_args:
    strategy_name: vllm
    strategy_config:
      tensor_parallel_size: 4
      # MTP speculative decoding config
      speculative_config:
        method: mtp
        num_speculative_tokens: 4
```

Note: Regardless of the training mode, when using MTP, you must configure `mtp_num_layers` (the corresponding value from the model's `config.json`) in `actor_train`'s `strategy_config`.

## Training Examples

### RLVR Pipeline with MTP

To enable MTP in RLVR training, configure `mtp_training_mode: standalone` in `actor_train` and `speculative_config` in `actor_infer`:

```yaml
actor_train:
  strategy_args:
    strategy_name: megatron_train
    strategy_config:
      tensor_model_parallel_size: 4
      pipeline_model_parallel_size: 2
      # ... other configs
  # MTP training config (uncomment to enable)
  #mtp_training_mode: standalone

actor_infer:
  strategy_args:
    strategy_name: vllm
    strategy_config:
      tensor_parallel_size: 4
      # ... other configs
      # Speculative decoding config (uncomment to enable)
      #speculative_config:
        # method: mtp
        # num_speculative_tokens: 3
```

For complete configuration examples, refer to:
- `examples/qwen3.5-27B-rlvr_megatron/rlvr_megatron_80GB.yaml` - Qwen3.5-27B Dense model
- `examples/qwen3.5-35BA3-rlvr_megatron/rlvr_megatron_80GB.yaml` - Qwen3.5-35B-A3B MoE model

### SFT Pipeline with MTP

SFT training uses `joint` mode for collaborative learning between main model and MTP:

```yaml
actor_train:
  model_args:
    model_name_or_path: Qwen/Qwen3.5-7B
    flash_attn: sdpa
    dtype: bf16
  training_args:
    learning_rate: 2.0e-5
    per_device_train_batch_size: 2
    gradient_accumulation_steps: 8
  data_args:
    file_name:
      - data/sft_data.jsonl
  strategy_args:
    strategy_name: megatron_train
    strategy_config:
      tensor_model_parallel_size: 2
      pipeline_model_parallel_size: 1
  # MTP joint training
  mtp_training_mode: joint
  mtp_loss_scaling_factor: 0.3
```

## Supported Models

ROLL currently supports MTP training for Qwen3.5 series models:

| Model | MTP Layers | Notes |
|-------|------------|-------|
| Qwen3.5-7B | 1 | Dense model |
| Qwen3.5-27B | 1 | Dense model |
| Qwen3.5-35B-A3B | 1 | MoE model |

MTP-related configuration is in the model checkpoint:

```json
// config.json
{
    "mtp_num_hidden_layers": 1,
    "mtp_use_dedicated_embeddings": false
}
```

## Notes

### 1. Mode Selection

| Scenario | Recommended Mode | Reason |
|----------|------------------|--------|
| RL Training | `standalone` | Isolate RL gradients, MTP learns main model distribution |
| SFT Training | `joint` | Joint optimization, MTP as auxiliary objective |
| Inference-only acceleration | `disabled` | Use pre-trained MTP, no training needed |

### 2. Performance Monitoring

Monitor the following metrics to evaluate MTP effectiveness:
- **Acceptance Rate**: The proportion of accepted tokens in speculative decoding
- **Average Acceptance Length**: The average number of tokens accepted per speculation
- **Throughput Improvement**: Speedup compared to non-speculative decoding

## Related Documentation

- [vLLM Configuration Guide](../Configuration/vllm.md) - vLLM inference engine detailed configuration
- [RLVR Pipeline](../Pipeline/rlvr_pipeline_start.md) - RLVR training pipeline
- [SFT Pipeline](../Pipeline/sft_pipeline_start.md) - SFT training pipeline
