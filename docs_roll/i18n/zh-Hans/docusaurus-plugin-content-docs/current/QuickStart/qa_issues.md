# 常见问题解答 (Q&A)

本文档整理了使用 ROLL 框架时可能遇到的常见问题及其解决方案。

## 模型转换相关

### Megatron 模型如何转成 HF 格式？

使用如下命令进行格式转换：

```bash
python mcore_adapter/tools/convert.py --checkpoint_path path_to_megatron_model --output_path path_to_output_hf_model
```

## 资源配置相关

### 什么是 colocate（共置）模式？

在共置模式下，多个角色（如 `actor_train`、`actor_infer`、`reference`）的 `device_mapping` 可以复用相同的 GPU 设备。例如：

```yaml
actor_train:
  device_mapping: list(range(0,8))
actor_infer:
  device_mapping: list(range(0,8))
reference:
  device_mapping: list(range(0,8))
```

框架底层通过资源管理机制保证了多个角色间 GPU 的复用，提高资源利用率。

### 什么是分离模式？

在分离模式下，不同角色的 `device_mapping` 之间没有交集，每个角色持有一组独立的 GPU 设备资源。例如：

```yaml
actor_train:
  device_mapping: list(range(0,8))
actor_infer:
  device_mapping: list(range(8,16))
reference:
  device_mapping: list(range(16,24))
```

这种方式可以避免角色间的资源竞争，提高系统稳定性。

## 训练参数相关

### `rollout_batch_size` 和 `num_return_sequences_in_group` 是什么意思？

- `rollout_batch_size`：一个 batch 中的 prompt 数量
- `num_return_sequences_in_group`：针对每条 prompt 的采样数，即 vLLM/SGLang 推理中通常意义上的 n 参数

实际一个 batch 内的样本数 = `rollout_batch_size` * `num_return_sequences_in_group`

对于 Megatron Backend，需要注意：
```
rollout_batch_size * num_return_sequences_in_group 必须是以下值的整数倍：
gradient_accumulation_steps * per_device_train_batch_size * (world_size/tensor_model_parallel_size/pipeline_model_parallel_size/context_parallel_size)
```

### 如何设置 `gradient_accumulation_steps` 和 `per_device_train_batch_size`？

#### 对于 DeepSpeed Backend：
```
global_batch_size = per_device_train_batch_size * gradient_accumulation_steps * world_size
```
其中 `world_size` 即 `actor_train`/`critic` 的 `device_mapping` 长度

#### 对于 Megatron Backend：
```
global_batch_size = per_device_train_batch_size * gradient_accumulation_steps * world_size / 
                    tensor_model_parallel_size / pipeline_model_parallel_size / context_parallel_size
```
其中 `world_size` 即 `actor_train`/`critic` 的 `device_mapping` 长度

注意：不需要除以 `expert_model_parallel_size`

## 调试与性能分析相关

### 如何获取训练的 timeline？

可以尝试在 YAML 中开启 profile：

```yaml
system_envs:
  RAY_PROFILING: "1"
profiler_output_dir: /data/oss_bucket_0/yali/llm/profile/${exp_name}
```

然后利用 [Perfetto UI](https://ui.perfetto.dev/) 工具进行分析。

### 如何 debug 代码？

在 Platform 的 env 中设置 `"RAY_DEBUG": "legacy"`，就可以采用 pdb 进行单步调试。

## 常见错误及解决方案

### 错误：`self.node2pg[node_rank] KeyError: 1`

检查申请的 GPU 总数和 `device_mapping` 的配置。出现该错误一般是因为 `max(device_mapping)` 小于或大于 `total_gpu_nums`。

### 错误：`assert self.lr_decay_steps > 0`

ROLL 数据分配时，会将 `rollout_batch_size` 的样本按 DP size 分发到每个 `actor_train` worker 上，然后再按 `gradient_accumulation_steps` 计算每次梯度更新的样本。配置一除就是 0。

详细配置逻辑可以参考手册：[Training Arguments](https://alibaba.github.io/ROLL/docs/QuickStart/config_guide#training-arguments-training_args)

### 错误：`AssertionError: batch_size 32 < chunks 64`

`batch_size` 小于 `reference`/`actor_train` 的 DP size，导致 dispatch 时数据不够切分，可以调整 `rollout_batch_size` 解决。

### 错误：`TypeError: BackendCompilerFailed.__init__() missing 1 required positional argument`

可以尝试在 YAML 中增加配置项解决：

```yaml
system_envs:
  NVTE_TORCH_COMPILE: '0'
```