# Configuration Guide

## Pipeline Config

Refer to [RLVR Pipeline Start](https://alibaba.github.io/ROLL/docs/StepByStep/rlvr_pipeline_start) and [Agentic Pipeline Start](https://alibaba.github.io/ROLL/docs/StepByStep/agent_pipeline_start) for more details about RLVR/Agentic pipeline configurations and reward settings.

```yaml
rollout_batch_size: 64
prompt_length: 2048
response_length: 4096
num_return_sequences_in_group: 8
```

- `rollout_batch_size`: The number of prompt samples to process in each inference batch.
- `num_return_sequences_in_group`: The number of sequences to generate for each prompt. Notice that its value proportionally scales the actual training samples. In other words, the actual training global batch size is equivalent to `num_return_sequences_in_group` * `rollout_batch_size`.

## Worker Config

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
    strategy_name: megatron_train  # deepspeed_train/megatron_train for training
    strategy_config:
      tensor_model_parallel_size: 1
      pipeline_model_parallel_size: 1
      expert_model_parallel_size: 1
  infer_batch_size: 4
  device_mapping: list(range(0,16))
actor_infer:
  model_args:
    ...
  generating_args:
    max_new_tokens: ${response_length}
    temperature: 0.99
    ...
  strategy_args:
    strategy_name: vllm  # vllm/sglang/hf_infer for inference
    strategy_config:
      gpu_memory_utilization: 0.6
      block_size: 16
      max_model_len: 8000
  num_gpus_per_worker: 1
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

- `device_mapping`: The list of device ids to use when training for the worker. Configure for any worker used gpu devices, including `actor_train`, `actor_infer`, `critic` and `reference`.
- `num_gpus_per_worker`: The number of GPUs assigned per worker. Applicable to `actor_infer` only.
- `infer_batch_size`: The batch size to used for inference or computing logprobs. 

### Model Arguments (**model_args**)

- `model_args.dtype`: Set model dtype as fp32, bf16, or fp16, otherwise use config's torch_dtype
- `model_args.disable_gradient_checkpointing`: Disable gradient checkpointing. Applicable only to `actor_train` when `strategy_name` is `deepspeed_train`

### Data Arguments (**data_args**)

Configure data_args under `actor_train`.

- `data_args.template`: The chat template used for constructing prompts during training and inference. Setting to `native` utilizes the default chat template `tokenizer.apply_chat_template` for prompt construction.
- `data_args.file_name`: The file path for training data. Supported formats include JSON, JSONL, and CSV.
- `data_args.prompt`: Which column in the file to use as prompt.
- `data_args.messages`: Which column in the file to use as messages. (conflicts with prompt)

### Generating Arguments (**generating_args**)

Configure generating_args under `actor_infer`.

- `generating_args.max_new_tokens`: The maximum length of the generated text.
- `generating_args.temperature`: The temperature to use for sampling.

### Strategy Arguments (**strategy_args**)

- `strategy_args.strategy_name`: The name of training/inference strategy. `deepspeed_train`/`megatron_train` for training, `vllm`/`sglang`/`hf_infer` for inference.
- `strategy_args.strategy_config`: The config of training/inference strategy. Will be passed to `strategy_name`'s constructor. E.g. `strategy_config.tensor_model_parallel_size` for `megatron_train` strategy and `strategy_config.gpu_memory_utilization` for `vllm` strategy.

Commonly used strategy configs are listed below:

#### Megatron Strategy Config

- `tensor_model_parallel_size`: Degree of tensor model parallelism.
- `pipeline_model_parallel_size`: Degree of pipeline model parallelism.
- `expert_model_parallel_size`: Degree of expert model parallelism.
- `context_parallel_size`: Degree of context parallelism.
- `virtual_pipeline_model_parallel_size`: Num of virtual pipeline in a pipeline.
- `sequence_parallel`: Enable sequence parallel optimization.
- `recompute_granularity`: Checkpoint activations to allow for training with larger models, sequences, and batch sizes. It is supported at two granularities 1) full: whole transformer layer is recomputed, 2) selective: core attention part of the transformer layer is recomputed.
- `moe_layer_recompute`: Memory optimization: checkpointing moe_layer to save activation memory.
- `moe_token_dispatcher_type`: The type of token dispatcher to use. Options are 'allgather' and 'alltoall'.
- `moe_grouped_gemm`: Enable grouped gemm for moe experts.
- `moe_shared_expert_overlap`: Enable overlapping between shared expert computations and dispatcher communications.
- `overlap_grad_reduce`: If true, overlap grad reduce-scatter with backward compute in distributed optimizer.


### VLLM Strategy Config

- `gpu_memory_utilization`: The fraction of GPU memory to be used for the model executor.
- `block_size`: Token block size for contiguous chunks of tokens.
- `max_model_len`: Model context length. If unspecified, will be automatically derived from the model config.
- `load_format`: The format of the model weights to load. Since there will be a `model update` in the beginning, this value should can be set to `dummy`.

#### SGLang Strategy Config

- `mem_fraction_static`: Fraction of the free GPU memory used for static memory like model weights and KV cache. Increase it if KV cache building fails. Decrease it if CUDA runs out of memory.
- `load_format`: The format of the model weights to load. Since there will be a `model update` in the beginning, this value should can be set to `dummy`.


#### DeepSpeed Strategy Config

There are DeepSpeed configurations in `./examples/config/` that can be overridden in the default list for strategy configuration.

For example, to use the deepspeed_zero2 strategy, add the following to your config:

```yaml
defaults:
  - ../config/envs@_here_
  - ../config/deepspeed_zero@_here_
  - ../config/deepspeed_zero2@_here_
  - ../config/deepspeed_zero3@_here_
  - ../config/deepspeed_zero3_cpuoffload@_here_
actor_train:
  strategy_args:
    strategy_name: deepspeed_train
    strategy_config: ${deepspeed_zero2}
```

### Training Arguments (**training_args**)

Used for configuring training parameters such as `learning_rate`, `weight_decay`, `warmup_steps`, etc.

- `training_args.per_device_train_batch_size`: The batch size to use when training.
- `training_args.gradient_accumulation_steps`: The number of gradient accumulation steps.

In deepspeed training the global train batch size is `per_device_train_batch_size` \* `gradient_accumulation_steps` \* world_size (a.k.a length of `device_mapping` for `actor_train`/`critic`).

In megatron training the global train batch size is `per_device_train_batch_size` \* `gradient_accumulation_steps` \* world_size / `tensor_model_parallel_size` / `pipeline_model_parallel_size` / `context_parallel_size` (don't need to divide `expert_model_parallel_size`).

If you want to perform one optimization step in each rollout, set `gradient_accumulation_steps` to `rollout_batch_size` \* `num_return_sequences_in_group` \* `tensor_model_parallel_size` \* `pipeline_model_parallel_size` \* `context_parallel_size`/ `per_device_train_batch_size` / world_size.
