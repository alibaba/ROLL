# LoRA Fine-tuning Configuration Guide

LoRA (Low-Rank Adaptation) is an efficient parameter-efficient fine-tuning method that achieves parameter-efficient fine-tuning by adding low-rank matrices to pre-trained models. This document will provide detailed instructions on how to configure and use LoRA fine-tuning in the ROLL framework.

## LoRA Introduction

LoRA achieves parameter-efficient fine-tuning through the following approaches:
1. **Low-Rank Matrix Decomposition**: Decompose weight update matrices into the product of two low-rank matrices
2. **Parameter Efficiency**: Train only a small number of additional parameters instead of all model parameters
3. **Easy Deployment**: Fine-tuned models can be easily merged into the original model

## Configuring LoRA Fine-tuning

In the ROLL framework, LoRA fine-tuning can be configured by setting relevant parameters in the YAML configuration file.

### Configuration Example

The following is a typical LoRA configuration example (from `examples/qwen2.5-7B-rlvr_megatron/rlvl_lora_zero3.yaml`):

```yaml
# LoRA global configuration
lora_target: o_proj,q_proj,k_proj,v_proj
lora_rank: 32
lora_alpha: 32

actor_train:
  model_args:
    attn_implementation: fa2
    disable_gradient_checkpointing: true
    dtype: bf16
    lora_target: ${lora_target}
    lora_rank: ${lora_rank}
    lora_alpha: ${lora_alpha}
    model_type: ~
  training_args:
    learning_rate: 1.0e-5
    weight_decay: 0
    per_device_train_batch_size: 1
    gradient_accumulation_steps: 32
    warmup_steps: 20
    num_train_epochs: 50
  strategy_args:
    strategy_name: deepspeed_train
    strategy_config: ${deepspeed_zero3}
  device_mapping: list(range(0,16))
  infer_batch_size: 4

actor_infer:
  model_args:
    attn_implementation: fa2
    disable_gradient_checkpointing: true
    dtype: bf16
    lora_target: ${lora_target}
    lora_rank: ${lora_rank}
    lora_alpha: ${lora_alpha}
  generating_args:
    max_new_tokens: ${response_length}
    top_p: 0.99
    top_k: 100
    num_beams: 1
    temperature: 0.99
    num_return_sequences: ${num_return_sequences_in_group}
  strategy_args:
    strategy_name: vllm
    strategy_config:
      gpu_memory_utilization: 0.6
      enforce_eager: false
      block_size: 16
      max_model_len: 8000
  device_mapping: list(range(0,12))
  infer_batch_size: 1
```

### Configuration Parameter Details

1. **lora_target**: Specify the model layers to apply LoRA
   - For example: `o_proj,q_proj,k_proj,v_proj` means applying LoRA to the output projection and query, key, value projection layers in the attention mechanism
   - Can be adjusted according to the specific model structure

2. **lora_rank**: Rank of the LoRA matrix
   - Controls the size of the LoRA matrix
   - Smaller ranks can reduce the number of parameters but may affect performance
   - Usually set to 8, 16, 32, 64, etc.

3. **lora_alpha**: LoRA scaling factor
   - Controls the magnitude of LoRA updates
   - Usually set to the same as `lora_rank` or its multiple

4. **LoRA Parameters in model_args**:
   - `lora_target`: Specify the layers to apply LoRA
   - `lora_rank`: Rank of the LoRA matrix
   - `lora_alpha`: LoRA scaling factor

## LoRA Compatibility with Training Backends

Currently, LoRA fine-tuning only supports the DeepSpeed training backend:

```yaml
actor_train:
  strategy_args:
    strategy_name: deepspeed_train  # LoRA only supports deepspeed_train
```

This is because DeepSpeed provides optimization features that integrate well with LoRA.

## Performance Optimization Recommendations

1. **Selecting Appropriate LoRA Layers**:
   - Applying LoRA to attention mechanism-related layers usually works well
   - The best LoRA layer combination can be determined through experimentation

2. **Adjusting LoRA Parameters**:
   - `lora_rank`: Adjust according to model size and task complexity
   - `lora_alpha`: Usually set to `lora_rank` or its multiple

3. **Learning Rate Setting**:
   - LoRA fine-tuning usually requires a higher learning rate
   - Set to `1.0e-5` in the example

## Notes

1. LoRA fine-tuning currently only supports the DeepSpeed training backend
2. Ensure the model supports LoRA fine-tuning
3. Pay attention to compatibility with LoRA when using gradient checkpointing
4. LoRA fine-tuning performance may differ from full parameter fine-tuning and needs to be evaluated according to specific tasks

By properly configuring LoRA fine-tuning, you can significantly reduce the number of training parameters and computational resource consumption while maintaining model performance.