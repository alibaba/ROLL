# 常见 Q&A

0. **Megatron 模型如何转成 HF**

使用如下命令进行格式转换

```bash
python mcore_adapter/tools/convert.py --checkpoint_path path_to_megatron_model --output_path path_to_output_hf_model
```

0. **什么是colocate模式**

actor_train、actor_infer、reference多个角色之间的device_mapping可以复用，比如actor_train配置device_mapping: list(range(0,8)), actor_infer配置device_mapping: list(range(0,8)), reference配置device_mapping: list(range(0,8)) , 框架底层通过对保证了多个角色间GPU的复用


0. **什么是分离模式**

actor_train、actor_infer、reference多个角色之间的device_mapping 之间没有交集，每个角色持有一组独立的GPU device资源，比如actor_train配置device_mapping: list(range(0,8)), actor_infer配置device_mapping: list(range(8,16)), reference配置device_mapping: list(range(16,24)) 


0. **rollout_batch_size/num_return_sequences_in_group是什么意思**

rollout_batch_size: 一个batch中的prompt数量

num_return_sequences_in_group: 针对每条prompt采样数，也就是vllm/sglang推理中通常意义上的n参数

也就是实际一个batch内样本数 = rollout_batch_size * num_return_sequences_in_group

对于Megatron Backend, 需要注意: 
 
rollout_batch_size * num_return_sequences_in_group 整数倍于 
gradient_accumulation_steps * per_device_train_batch_size * (world_size/tensor_model_parallel_size/pipeline_model_parallel_size/context_parallel_size)


0. **如何设置gradient_accumulation_steps/per_device_train_batch_size**

***对于DeepSpeed Backend***

global_batch_size = per_device_train_batch_size * gradient_accumulation_steps * world_size 

world_size 即actor_train/critic的device_mapping长度

***对于Megatron Backend***

global_batch_size = per_device_train_batch_size * gradient_accumulation_steps * world_size / tensor_model_parallel_size / pipeline_model_parallel_size / context_parallel_size 

world_size 即actor_train/critic的device_mapping长度

注意: 不需要除以expert_model_parallel_size


0. **如何获取训练的timeline**

可以尝试在yaml中开启profile

```yaml
system_envs:
  RAY_PROFILING: "1"
profiler_output_dir: /data/oss_bucket_0/yali/llm/profile/${exp_name}
```

然后利用https://ui.perfetto.dev/ 工具进行分析

0. **如何debug代码**

在对应后端的Platform的env中设置 "RAY_DEBUG": "legacy" ， 就可以采用pdb进行单步调试


0. **如果出现这种错误: self.node2pg[node_rank] KeyError: 1**

检查申请的GPU总数和device_mapping的配置，出现该错误一般是max(device_mapping) < 或者 > total_gpu_nums

0. **如果出现这种错误：assert self.lr_decay_steps > 0**

roll数据分配的时候，会将rollout_batch_size的样本，按dp size 分发到每个actor_train worker上，然后再按gradient_accumulation_steps计算每次梯度更新的样本。配置一除就是0; 

详细配置逻辑可以参考手册：https://alibaba.github.io/ROLL/docs/User%20Guides/Configuration/config_guide#training-arguments-training_args


1. **如果出现这种错误：AssertionError: batch_size 32 < chunks 64**

batch_size 小于reference/actor_train 的DP size，导致dispatch时数据不够切分，可以调整rollout_batch_size解决


0. **如果出现这种错误：TypeError: BackendCompilerFailed.__init__() missing 1 required positional argument**

可以尝试在yaml增加配置项解决:

```yaml
system_envs:
  NVTE_TORCH_COMPILE: '0'
```

