# Off-Policy Algorithms Configuration Guide

The ROLL framework supports multiple Off-Policy algorithm variants for reinforcement learning training. This document provides detailed configuration methods and usage examples for various algorithms.

## Supported Algorithm Variants

The ROLL framework currently supports the following Off-Policy algorithms:

1. **vanilla** - Basic Policy Gradient algorithm
2. **ppo** - Proximal Policy Optimization
3. **tis** - Truncated Importance Sampling
4. **topr** - Tapered off-policy REINFORCE
5. **cispo** - Clipped Importance Sampling Policy Optimization
6. **kimi15** - Kimi15 algorithm

## Basic Configuration

### Core Parameters

Set the basic parameters for Off-Policy algorithms in the configuration file:

```yaml
# Select algorithm variant
pg_variant: topr  # Options: vanilla, tis, topr, cispo, kimi15, ppo

# Training configuration
max_steps: 500
save_steps: 100
logging_steps: 1
eval_steps: 10

# Data configuration
rollout_batch_size: 128
prompt_length: 2048
response_length: 8192
num_return_sequences_in_group: 8

# Common training parameters
ppo_epochs: 1
adv_estimator: "reinforce"
whiten_advantages: true
```

### Worker Configuration

Use the specialized ActorPGWorker to handle Off-Policy algorithms:

```yaml
actor_train:
  worker_cls: roll.pipeline.rlvr.actor_pg_worker.ActorPGWorker
  pg_variant: topr  # Keep consistent with global configuration
  model_args:
    flash_attn: fa2
    disable_gradient_checkpointing: false
    dtype: bf16
  training_args:
    learning_rate: 1.0e-6
    weight_decay: 0
    per_device_train_batch_size: 1
    gradient_accumulation_steps: 64
    warmup_steps: 20
    num_train_epochs: 50
  strategy_args:
    strategy_name: megatron_train
    strategy_config:
      tensor_model_parallel_size: 1
      pipeline_model_parallel_size: 1
      use_distributed_optimizer: true
      recompute_granularity: full
  device_mapping: list(range(0,16))
```

## Detailed Algorithm Configuration

### 1. Vanilla Policy Gradient

The most basic policy gradient algorithm, directly using the product of log probability and advantage function as loss.

**Configuration Features:**
- No additional parameters required
- High computational efficiency
- Suitable for simple reinforcement learning tasks

```yaml
pg_variant: vanilla

# No additional configuration parameters needed
```

### 2. PPO (Proximal Policy Optimization)

Proximal Policy Optimization algorithm that stabilizes training by clipping importance sampling ratios.

**Key Parameters:**
```yaml
pg_variant: ppo

# PPO specific parameters
pg_clip: 0.2                    # Clipping range
pg_clip_low: 0.2               # Lower bound clipping (optional)
pg_clip_high: 0.2              # Upper bound clipping (optional)
use_pg_clip_range: false       # Whether to use asymmetric clipping
dual_clip_loss: true           # Whether to enable dual clipping
```

**Configuration Example:**
```yaml
pg_variant: ppo
pg_clip: 0.2
dual_clip_loss: true
```

### 3. TIS (Truncated Importance Sampling)

Truncated Importance Sampling algorithm that limits importance sampling ratios to the range [0, 1].

**Key Parameters:**
```yaml
pg_variant: tis

# TIS specific parameters
tis_lower_bound: 0.0           # Lower bound
tis_upper_bound: 1.0           # Upper bound
```

**Configuration Example:**
```yaml
pg_variant: tis
tis_lower_bound: 0.0
tis_upper_bound: 1.0
```

### 4. TOPR (Tapered off-policy REINFORCE)

Tapered off-policy reinforcement learning algorithm that adopts different update strategies based on positive and negative rewards.

**Algorithm Features:**
- Positive samples: Direct SFT update without importance sampling
- Negative samples: TIS update with importance sampling ratio limited to [0, 1]

**Key Parameters:**
```yaml
pg_variant: topr

# TOPR specific parameters
topr_positive_weight: 1.0      # Positive sample weight
topr_negative_weight: 1.0      # Negative sample weight
```

**Configuration Example:**
```yaml
pg_variant: topr
topr_positive_weight: 1.0
topr_negative_weight: 1.0
```

### 5. CISPO (Clipped Importance Sampling Policy Optimization)

Clipped Importance Sampling Policy Optimization algorithm that uses clipped importance sampling weights and stop-gradient operations.

**Key Parameters:**
```yaml
pg_variant: cispo

# CISPO specific parameters
cispo_epsilon_low: 0.1         # Lower bound clipping parameter
cispo_epsilon_high: 0.1        # Upper bound clipping parameter
cispo_use_unified_mask: false  # Whether to use unified mask
```

**Configuration Example:**
```yaml
pg_variant: cispo
cispo_epsilon_low: 0.1
cispo_epsilon_high: 0.1
cispo_use_unified_mask: false
```

### 6. Kimi15

Policy gradient algorithm based on KL regularization, adding KL divergence regularization term to the policy gradient term.

**Key Parameters:**
```yaml
pg_variant: kimi15

# Kimi15 specific parameters
kimi15_tau: 0.1               # Regularization parameter
```

**Configuration Example:**
```yaml
pg_variant: kimi15
kimi15_tau: 0.1
```

## Complete Configuration Example

For a complete RLVR Off-Policy configuration example, please refer to:

**Configuration File**: `examples/qwen2.5-7B-rlvr-offpolicy/rlvr_config.yaml`

This configuration file contains all necessary parameter settings and supports switching between different algorithm variants by modifying the `pg_variant` parameter:

```yaml
pg_variant: topr  # Options: topr, vanilla, tis, cispo, kimi15, ppo
```

### Key Configuration Points

1. **Worker Configuration**: Use `ActorPGWorker` class to handle Off-Policy algorithms
2. **Algorithm Selection**: Specify algorithm variant through `pg_variant` parameter
3. **Model Configuration**: Support Megatron training and SGLang inference
4. **Reward Configuration**: Include mathematical rule reward model configuration

### Usage

1. Copy the configuration file to your working directory
2. Modify `pg_variant` and other parameters as needed
3. Run the training script:

```bash
python examples/start_rlvr_pipeline.py --config-path your_config.yaml
```

## Algorithm Selection Recommendations

### Selection Based on Task Characteristics

1. **Simple Tasks**: Use `vanilla` or `ppo`
   - Low computational overhead
   - Fast convergence

2. **Complex Reasoning Tasks**: Use `topr` or `cispo`
   - Better stability
   - Suitable for long sequence generation

3. **Tasks Requiring Exploration**: Use `tis` or `kimi15`
   - Better exploration capability
   - Suitable for sparse reward environments

### Selection Based on Data Distribution

1. **Balanced Positive/Negative Samples**: Use `ppo` or `vanilla`
2. **More Negative Samples**: Use `topr`, can adjust negative sample weights
3. **Need Regularization**: Use `kimi15`, control regularization intensity through tau parameter

## Monitoring and Debugging

### Key Metrics

Different algorithms output different monitoring metrics:

- **Common Metrics**: `pg_loss`, `kl_loss`, `entropy_loss`
- **PPO Specific**: `ppo_ratio_clipfrac`, `ppo_ratio_low_clipfrac`, `ppo_ratio_high_clipfrac`
- **TIS Specific**: `tis_lower_clipfrac`, `tis_upper_clipfrac`, `tis_total_clipfrac`
- **TOPR Specific**: `topr_positive_samples`, `topr_negative_samples`, `topr_negative_total_clipfrac`
- **CISPO Specific**: `cispo_total_clipfrac`, `cispo_clipped_ratio`
- **Kimi15 Specific**: `kimi15_policy_grad_magnitude`, `kimi15_kl_reg_magnitude`

### Debugging Recommendations

1. **Monitor Clipping Ratios**: High clipping ratios may indicate learning rate is too large
2. **Observe Sample Distribution**: TOPR algorithm focuses on positive/negative sample ratios
3. **Adjust Hyperparameters**: Tune algorithm-specific parameters based on task characteristics
4. **Use TensorBoard**: Visualize metric changes during training

## Frequently Asked Questions

### Q: How to choose the appropriate pg_variant?
A: It's recommended to start with `topr`, as it performs well on most tasks. Then adjust based on specific task characteristics.

### Q: What is the computational complexity of each algorithm?
A: `vanilla` < `ppo` < `tis` < `kimi15` < `cispo` < `topr`

### Q: Can I switch algorithms during training?
A: It's not recommended to switch algorithms during training, as this can cause training instability.

### Q: How to adjust algorithm-specific parameters?
A: Refer to the configuration examples for each algorithm and tune based on validation set performance. It's recommended to start with small adjustments.