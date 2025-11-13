# ROLL Resource Configuration

In the ROLL framework, resource settings are specified through the `device_mapping` parameter in YAML configuration files to determine which GPU devices each worker uses. This document will provide detailed instructions on how to configure resources, including colocated and disaggregated modes, multi-role resource configuration, and how worker counts are calculated.

## GPU Resource Configuration

In ROLL, GPU resource settings are configured by specifying the `device_mapping` parameter for each worker in the YAML configuration file. This parameter is a string that can be parsed by Python's `eval()` function into a list, where the values in the list represent global logical GPU RANKs.

For example:
```yaml
actor_train:
  device_mapping: list(range(0,16))
actor_infer:
  device_mapping: list(range(16,24))
```

In this example, the system requires a total of 24 GPUs, where `actor_train` is deployed on GPUs [0,16) and `actor_infer` is deployed on GPUs [16,24).

## CPU Resource Configuration

For workers that only use CPU resources, simply configure the `world_size` parameter, and the system will automatically deploy the corresponding number of workers (ray.Actor) on CPU resources.

For example:
```yaml
code_sandbox:
  world_size: 8
```

## Colocated and Disaggregated Modes

### Colocated Mode
In colocated mode, multiple roles share the same GPU resources. This approach can improve resource utilization and reduce resource waste.

For example, in `examples/docs_examples/example_grpo.yaml`:
```yaml
actor_infer:
  device_mapping: list(range(0,8)) # Shares GPUs [0,8) with actor_train, GPU time-division multiplexing
# ...
actor_train:
  device_mapping: list(range(0,8))
```

### Disaggregated Mode
In disaggregated mode, different roles use different GPU resources. This independent deployment approach is key to implementing asynchronous training.
ROLL directly implements disaggregated deployment by setting different `device_mapping` for different workers.

For example, in `examples/qwen2.5-7B-agentic_megatron/agentic_val_webshop_async.yaml`:
```yaml
# actor train uses GPUs [0, 1, 2, 3], actor_infer uses GPUs [4, 5, 6, 7]
# 
actor_train:
  device_mapping: list(range(0,4))
actor_infer:
  device_mapping: list(range(4,8))
```

## Flexibility in Multi-Role Resource Configuration

The ROLL framework supports configuring different resource strategies for different roles to meet the needs of various application scenarios:

1. Different roles can use different numbers of GPUs
2. Different roles can use different inference engines (such as vLLM, SGLang, etc.)
3. Different roles can set different `num_gpus_per_worker` parameters
   1. The num_gpus_per_worker for training roles is always 1
   2. The num_gpus_per_worker for inference roles is >=1 based on required resources, with vLLM automatically calculated based on parallel settings and SGLang directly specified

For example, when using vLLM as the inference engine, `num_gpus_per_worker` is automatically calculated based on tensor parallel (tensor_parallel_size) and pipeline parallel (pipeline_parallel_size) settings:
```yaml
actor_infer:
  strategy_args:
    strategy_name: vllm
    strategy_config:
      tensor_parallel_size: 2
      pipeline_parallel_size: 1
  num_gpus_per_worker: 2  # Automatically calculated as tensor_parallel_size * pipeline_parallel_size
```

## Worker Count Calculation

The number of workers (`world_size`) is automatically calculated based on the `device_mapping` and `num_gpus_per_worker` parameters:

```python
world_size = len(device_mapping) // num_gpus_per_worker
```

In the `WorkerConfig.__post_init__()` method, if `device_mapping` is not None, the following logic is executed:
1. Parse the string into a list through `eval(device_mapping)`
2. Verify that `len(device_mapping)` is divisible by `num_gpus_per_worker`
3. Calculate `world_size = len(device_mapping) // num_gpus_per_worker`

For workers that only use CPU, the worker count is directly specified through the `world_size` parameter, and `num_gpus_per_worker` is set to 0.

## Configuration Example

The following is a complete resource configuration example:

```yaml
num_gpus_per_node: 8

actor_train:
  device_mapping: list(range(0,16))  # Uses 16 GPUs
  # world_size automatically calculated as 16 // 1 = 16

actor_infer:
  num_gpus_per_worker: 2  # Each worker uses 2 GPUs
  device_mapping: list(range(0,12))  # Uses 12 GPUs
  # world_size automatically calculated as 12 // 2 = 6

rewards:
  code_sandbox:
    world_size: 8  # CPU-only, deploys 8 workers
```

By properly configuring these parameters, resources can be flexibly allocated to different roles to meet various training and inference requirements.