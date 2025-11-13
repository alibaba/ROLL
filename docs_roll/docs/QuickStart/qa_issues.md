# Frequently Asked Questions (Q&A)

This document compiles common issues that may be encountered when using the ROLL framework and their solutions.

## Model Conversion Related

### How to convert Megatron models to HF format?

Use the following command for format conversion:

```bash
python mcore_adapter/tools/convert.py --checkpoint_path path_to_megatron_model --output_path path_to_output_hf_model
```

## Resource Configuration Related

### What is colocate mode?

In colocate mode, multiple roles (such as `actor_train`, `actor_infer`, `reference`) can reuse the same GPU devices in their `device_mapping`. For example:

```yaml
actor_train:
  device_mapping: list(range(0,8))
actor_infer:
  device_mapping: list(range(0,8))
reference:
  device_mapping: list(range(0,8))
```

The framework's underlying resource management mechanism ensures GPU reuse between multiple roles, improving resource utilization.

### What is separate mode?

In separate mode, there is no intersection between different roles' `device_mapping`, and each role holds a set of independent GPU device resources. For example:

```yaml
actor_train:
  device_mapping: list(range(0,8))
actor_infer:
  device_mapping: list(range(8,16))
reference:
  device_mapping: list(range(16,24))
```

This approach can avoid resource competition between roles and improve system stability.

## Training Parameters Related

### What do `rollout_batch_size` and `num_return_sequences_in_group` mean?

- `rollout_batch_size`: The number of prompts in a batch
- `num_return_sequences_in_group`: The sampling count for each prompt, i.e., the n parameter in vLLM/SGLang inference

Actual number of samples in a batch = `rollout_batch_size` * `num_return_sequences_in_group`

For Megatron Backend, note:
```
rollout_batch_size * num_return_sequences_in_group must be an integer multiple of:
gradient_accumulation_steps * per_device_train_batch_size * (world_size/tensor_model_parallel_size/pipeline_model_parallel_size/context_parallel_size)
```

### How to set `gradient_accumulation_steps` and `per_device_train_batch_size`?

#### For DeepSpeed Backend:
```
global_batch_size = per_device_train_batch_size * gradient_accumulation_steps * world_size
```
Where `world_size` is the length of `device_mapping` for `actor_train`/`critic`

#### For Megatron Backend:
```
global_batch_size = per_device_train_batch_size * gradient_accumulation_steps * world_size / 
                    tensor_model_parallel_size / pipeline_model_parallel_size / context_parallel_size
```
Where `world_size` is the length of `device_mapping` for `actor_train`/`critic`

Note: No need to divide by `expert_model_parallel_size`

## Debugging and Performance Analysis Related

### How to get the training timeline?

You can try enabling profiling in YAML:

```yaml
system_envs:
  RAY_PROFILING: "1"
profiler_output_dir: /data/oss_bucket_0/yali/llm/profile/${exp_name}
```

Then use the [Perfetto UI](https://ui.perfetto.dev/) tool for analysis.

### How to debug code?

Set `"RAY_DEBUG": "legacy"` in Platform' env, and then you can use pdb for step-by-step debugging.

## Common Errors and Solutions

### Error: `self.node2pg[node_rank] KeyError: 1`

Check the total number of GPUs requested and the `device_mapping` configuration. This error generally occurs because `max(device_mapping)` is less than or greater than `total_gpu_nums`.

### Error: `assert self.lr_decay_steps > 0`

When ROLL distributes data, it will distribute `rollout_batch_size` samples to each `actor_train` worker according to DP size, and then calculate the samples for each gradient update according to `gradient_accumulation_steps`. The configuration results in 0 when divided.

For detailed configuration logic, refer to the manual: [Training Arguments](https://alibaba.github.io/ROLL/docs/QuickStart/config_guide#training-arguments-training_args)

### Error: `AssertionError: batch_size 32 < chunks 64`

`batch_size` is less than the DP size of `reference`/`actor_train`, causing insufficient data for splitting during dispatch. This can be resolved by adjusting `rollout_batch_size`.

### Error: `TypeError: BackendCompilerFailed.__init__() missing 1 required positional argument`

You can try adding a configuration item in YAML to resolve this:

```yaml
system_envs:
  NVTE_TORCH_COMPILE: '0'
```