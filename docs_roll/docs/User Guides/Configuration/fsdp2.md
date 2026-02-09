# FSDP2 Training and Inference Backend Configuration Guide

[FSDP2 (Fully Sharded Data Parallel 2](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html) is PyTorch's latest distributed training framework that provides efficient parameter sharding with [DTensor](https://docs.pytorch.org/docs/stable/distributed.tensor.html). This document will provide detailed instructions on how to configure and use the FSDP2 backend in the ROLL framework.

## FSDP2 with ROLL

ROLL support the following FSDP2 features:
1. **FSDP2 Sharding**: Shards model parameters, gradients, and optimizer with FSDP2 [fully_shard](https://docs.pytorch.org/docs/main/distributed.fsdp.fully_shard.html). Also support checkpoint management with [DCP](https://docs.pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html).
2. **Context Parallelism**: Supports integration with Context Parallel (Ulysses)
3. **Model Support**: Supports text models, Vision-Language (VL) models, and MoE (Mixture of Experts) models.

## Configuring FSDP2 Strategy

In the ROLL framework, FSDP2 training and inference strategies can be configured by setting `strategy_args` in the YAML configuration file.

### Training Configuration Example

The following is a typical FSDP2 training configuration example (from `examples_lixing/qwen3-8B-rlvr_fsdp2/rlvr_config.yaml`):

```yaml
actor_train:
  model_args:
    disable_gradient_checkpointing: false
    dtype: bf16
    model_type: ~
  training_args:
    learning_rate: 1.0e-6
    weight_decay: 0
    per_device_train_batch_size: 1
    gradient_accumulation_steps: 32
    warmup_steps: 20
    num_train_epochs: 50
  strategy_args:
    strategy_name: fsdp2_train
    strategy_config:
      fsdp_size: 16
      param_dtype: bf16
      reduce_dtype: float32
      reshard_after_forward: true
      offload_policy: false
  device_mapping: list(range(0,16))
  infer_batch_size: 4
```

### Inference Configuration Example

The following is a typical FSDP2 inference configuration example:

```yaml
reference:
  model_args:
    disable_gradient_checkpointing: true
    dtype: bf16
    model_type: ~
  strategy_args:
    strategy_name: fsdp2_infer
    strategy_config:
      fsdp_size: 4
      param_dtype: bf16
      reduce_dtype: float32
      reshard_after_forward: true
      offload_policy: false
  device_mapping: list(range(0,8))
  infer_batch_size: 1
```

### FSDP2 + Context Parallel Configuration Example

The following is a configuration example combining FSDP2 with Context Parallel (Ulysses) (from `examples_lixing/qwen3-4b-vl_fsdp2_lct/vl_fsdp2_lct_cp2.yaml`):

```yaml
actor_train:
  model_args:
    disable_gradient_checkpointing: false
    dtype: bf16
    model_type: ~
    ulysses_size: 2  # Context parallel size
  training_args:
    learning_rate: 1.0e-6
    weight_decay: 1.0e-2
    per_device_train_batch_size: 1
    gradient_accumulation_steps: 256
    warmup_steps: 0
    num_train_epochs: 50
  strategy_args:
    strategy_name: fsdp2_train
    strategy_config:
      fsdp_size: 4  # FSDP sharding size
      param_dtype: bf16
      reduce_dtype: float32
      reshard_after_forward: true
      offload_policy: false
  device_mapping: list(range(0,8))
  infer_batch_size: 1
```

In this example:
- Total GPUs: 8
- Context Parallel (Ulysses) size: 2
- FSDP size: 4
- Device mesh shape: (2, 4) [ddp, fsdp]
- 2 replicas, each with 4-way parameter sharding

### Configuration Parameter Details

1. **strategy_name**:
   - `fsdp2_train` for training
   - `fsdp2_infer` for inference

2. **strategy_config**: FSDP2-specific configuration parameters
   - `fsdp_size`: Number of FSDP shards
     - If `fsdp_size >= world_size` or `fsdp_size <= 1`: pure FSDP2 mode
     - If `fsdp_size < world_size`: HSDP mode with DDP replicas
   - `param_dtype`: Parameter data type (e.g., `bf16`, `fp16`, `float32`)
   - `reduce_dtype`: Data type for gradient reduction (e.g., `float32`)
   - `reshard_after_forward`: Whether to reshard parameters after forward pass
     - `true`: Reshard after forward
     - `false`: Keep parameters gathered
   - `offload_policy`: Whether to enable CPU offloading
     - `true`: Offload parameters to CPU when not in use (saves GPU memory)
     - `false`: Keep all parameters on GPU (faster but uses more memory)
   - `wrap_policy`: Module wrapping policy
     - `transformer_layer_cls_to_wrap`: List of transformer layer class names to wrap (e.g., `["Qwen3DecoderLayer"]`)
     - `wrap_embeddings`: Whether to wrap embedding layers (default: `false`)
     - `wrap_lm_output`: Whether to wrap LM head (default: `false`)
     - `moe_experts`: List of MoE expert block class names to wrap (for MoE models, we may want to wrap each experts seperately to avoid OOM during param. gather, but need dummy expert forward to avoid hang, see [example](../../../../roll/third_party/fsdp2/qwen3_moe_patch.py))
  
      if not sef the `wrap_policy`, by default will use the _no_splite_modules for transofmers models.
   - `apply_expert_patch`: Whether to apply MoE expert patch (for MoE models)
     - `true`: Apply patch to prevent deadlocks when different ranks activate different experts
     - `false`: Don't apply patch (may cause deadlocks in MoE models)
   - `apply_tiled_mlp`: Whether to apply TiledMLP optimization
     - `true`: Use tiled MLP computation to reduce memory usage
     - `false`: Use standard MLP computation
   - `tiled_num_shards`: Number of shards for TiledMLP (default: 4)
   - `async_save_ckpt`: Whether to save checkpoints asynchronously (default: `true`)

3. **ulysses_size**: Context parallel size (set in `model_args`)
   - Splits sequence dimension across multiple GPUs
   - Compatible with FSDP2 for hybrid parallelism
   - Useful for long-context training

4. **device_mapping**: Specify the list of GPU device IDs to use

5. **infer_batch_size**: Batch size during inference

## Device Mesh Configuration

FSDP2 supports different device mesh configurations based on `fsdp_size` and `ulysses_size`:

### Pure FSDP2 Mode

When `fsdp_size >= world_size` or `fsdp_size <= 1`:

```yaml
# Example: 16 GPUs, fsdp_size=16
strategy_config:
  fsdp_size: 16
# Device mesh: (16,) [fsdp]
# All 16 GPUs shard parameters
```

### HSDP Mode

When `fsdp_size < world_size`:

```yaml
# Example: 16 GPUs, fsdp_size=8
strategy_config:
  fsdp_size: 8
# ddp_size = 16 // 8 = 2
# Device mesh: (2, 8) [ddp, fsdp]
# 2 replicas, each with 8-way parameter sharding
```

### FSDP2 + Context Parallel (Ulysses)

When both `ulysses_size` and `fsdp_size` are configured:

```yaml
# Example: 8 GPUs, ulysses_size=2, fsdp_size=4
model_args:
  ulysses_size: 2
strategy_config:
  fsdp_size: 4
# ddp_size = 8 // 4 = 2
# Device mesh: (2, 4) [ddp, fsdp]
# 2 replicas, each with 4-way parameter sharding
# Ulysses: 2-way context parallel (sequence dimension split)
```

## Model-Specific Configurations

### Text Models (Qwen2.5, Qwen3, LLaMA)

```yaml
strategy_config:
  fsdp_size: 16
  param_dtype: bf16
  reduce_dtype: float32
  wrap_policy:
    transformer_layer_cls_to_wrap: ["Qwen3DecoderLayer"]
```

### Vision-Language Models (Qwen2.5-VL, Qwen3-VL)

VL models require special handling for the vision tower:

```yaml
actor_train:
  model_args:
    freeze_module_prefix: vision_model  # Freeze vision tower
    ulysses_size: 2  # Optional: context parallel
  strategy_args:
    strategy_name: fsdp2_train
    strategy_config:
      fsdp_size: 4
      param_dtype: bf16
      reduce_dtype: float32
      # Vision tower blocks automatically have cast_forward_inputs disabled
```

### MoE Models (Qwen3-MoE)

MoE models require the expert patch to prevent deadlocks:

```yaml
strategy_config:
  fsdp_size: 16
  param_dtype: bf16
  reduce_dtype: float32
  apply_expert_patch: true  # Critical for MoE models if wrap each expert separately
  wrap_policy:
    moe_experts: ["Qwen3MoeMLP"]
```


## Notes

1. **PyTorch Version**: FSDP2 requires PyTorch >= 2.4
2. **MoE Models**: Always enable `apply_expert_patch: true` for MoE models to prevent deadlocks if wrap experts seperately
3. **VL Models**: Vision tower blocks automatically handle precision issues
4. **Memory vs Performance**:
   - `offload_policy: true` saves memory but is slower
   - `reshard_after_forward: true` saves memory but may be slower
   - Balance based on your hardware and requirements