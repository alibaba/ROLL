# ROLL 资源配置

在 ROLL 框架中，资源设置是通过 YAML 配置文件中的 `device_mapping` 参数来指定每个 worker 使用哪些 GPU 设备。本文档将详细介绍如何配置资源，包括共置和分离模式、多角色资源配置以及 worker 数量的计算方式。

## GPU 资源配置

在 ROLL 中，GPU 资源的设置通过在 YAML 配置文件中为每个 worker 指定 `device_mapping` 参数来完成。该参数是一个能被 Python `eval()` 函数解析为列表的字符串，列表中的值表示全局逻辑 GPU RANK。

例如：
```yaml
actor_train:
  device_mapping: list(range(0,16))
actor_infer:
  device_mapping: list(range(16,24))
```

在这个例子中，系统总共需要 24 块 GPU，其中 `actor_train` 部署在 GPU [0,16) 上，`actor_infer` 部署在 GPU [16,24) 上。

## CPU 资源配置

对于仅使用 CPU 资源的 worker，只需配置 `world_size` 参数，系统会自动在 CPU 资源上部署相应数量的 worker (ray.Actor)。

例如：
```yaml
code_sandbox:
  world_size: 8
```

## 共置 (Colocated) 与分离 (Disaggregated) 模式

### 共置模式
在共置模式下，多个角色共享相同的 GPU 资源。这种方式可以提高资源利用率，减少资源浪费。

例如，在 `examples/docs_examples/example_grpo.yaml` 中：
```yaml
actor_infer:
  device_mapping: list(range(0,8)) # 与actor_train共享[0,8) GPU, GPU时分复用
# ...
actor_train:
  device_mapping: list(range(0,8))
```

### 分离模式
在分离模式下，不同角色使用不同的 GPU 资源。这种独立部署的方式是实现异步训练的关键。
ROLL里面通过为不同的worker设置不同的`device_mapping`直接实现分离部署。

例如，在 `examples/qwen2.5-7B-agentic_megatron/agentic_val_webshop_async.yaml` 中：
```yaml
# actor train 使用GPU[0, 1, 2, 3], actor_infer 使用GPU[4, 5, 6, 7]
# 
actor_train:
  device_mapping: list(range(0,4))
actor_infer:
  device_mapping: list(range(4,8))
```

## 多角色资源配置的灵活性

ROLL 框架支持为不同角色配置不同的资源策略，以满足各种应用场景的需求：

1. 不同角色可以使用不同数量的 GPU
2. 不同角色可以使用不同的推理引擎（如 vLLM、SGLang 等）
3. 不同角色可以设置不同的 `num_gpus_per_worker` 参数
   1. 训练角色的num_gpus_per_worker始终为1
   2. 推理角色的num_gpus_per_worker根据所需资源会>=1，vllm根据并行设置自动计算，sglang直接指定

例如，在使用 vLLM 作为推理引擎时，会根据张量并行 (tensor_parallel_size) 和流水线并行 (pipeline_parallel_size) 的设置自动计算 `num_gpus_per_worker`：
```yaml
actor_infer:
  strategy_args:
    strategy_name: vllm
    strategy_config:
      tensor_parallel_size: 2
      pipeline_parallel_size: 1
  num_gpus_per_worker: 2  # 自动计算为 tensor_parallel_size * pipeline_parallel_size
```

## Worker 数量计算方式

Worker 的数量 (`world_size`) 是根据 `device_mapping` 和 `num_gpus_per_worker` 参数自动计算的：

```python
world_size = len(device_mapping) // num_gpus_per_worker
```

在 `WorkerConfig.__post_init__()` 方法中，如果 `device_mapping` 不为 None，则会执行以下逻辑：
1. 通过 `eval(device_mapping)` 将字符串解析为列表
2. 验证 `len(device_mapping)` 能被 `num_gpus_per_worker` 整除
3. 计算 `world_size = len(device_mapping) // num_gpus_per_worker`

对于仅使用 CPU 的 worker，直接通过 `world_size` 参数指定 worker 数量，此时 `num_gpus_per_worker` 被设置为 0。

## 配置示例

以下是一个完整的资源配置示例：

```yaml
num_gpus_per_node: 8

actor_train:
  device_mapping: list(range(0,16))  # 使用 16 块 GPU
  # world_size 自动计算为 16 // 1 = 16

actor_infer:
  num_gpus_per_worker: 2  # 每个 worker 使用 2 块 GPU
  device_mapping: list(range(0,12))  # 使用 12 块 GPU
  # world_size 自动计算为 12 // 2 = 6

rewards:
  code_sandbox:
    world_size: 8  # 仅使用 CPU，部署 8 个 worker
```

通过合理配置这些参数，可以灵活地为不同角色分配资源，以满足各种训练和推理需求。