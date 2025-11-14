# 快速上手：单机版部署指南

## 准备环境
1. 购买配备GPU的机器，并同步安装GPU驱动
2. 远程连接GPU实例，进入机器终端
3. 运行以下命令安装 Docker环境 和 NVIDIA容器工具包
```shell
curl -fsSL https://github.com/alibaba/ROLL/blob/main/scripts/install_docker_nvidia_container_toolkit.sh | sudo bash
```

## 环境配置
从[镜像地址](https://alibaba.github.io/ROLL/docs/QuickStart/image_address)中选择你需要的Docker镜像，下文均以 *torch2.6.0 + vLLM0.8.4* 为例
```shell
# 1. 启动一个docker容器，指定GPU支持，暴露容器端口，并始终保持容器运行
sudo docker run -dit \
  --gpus all \
  -p 9001:22 \
  --ipc=host \
  --shm-size=10gb \
  roll-registry.cn-hangzhou.cr.aliyuncs.com/roll/pytorch:nvcr-24.05-py3-torch260-vllm084 \
  /bin/bash

# 2. 进入docker容器
#    您可以使用 `sudo docker ps` 命令查找运行中的容器ID或名称。
sudo docker exec -it <container_id> /bin/bash

# 3. 验证GPU是否可见
nvidia-smi

# 4. 克隆项目代码
git clone https://github.com/alibaba/ROLL.git

# 5. 安装项目依赖（选择对应镜像的requirements文件）
cd ROLL
pip install -r requirements_torch260_vllm.txt -i https://mirrors.aliyun.com/pypi/simple/
```

## pipeline运行
```shell
bash examples/agentic_demo/run_agentic_pipeline_frozen_lake_single_node_demo.sh
```

pipeline运行中的log截图示例：
![log_pipeline_start](https://img.alicdn.com/imgextra/i2/O1CN015wRcUb1EkfZQEjpEI_!!6000000000390-2-tps-2868-650.png)

![log_pipeline_in_training](https://img.alicdn.com/imgextra/i2/O1CN01Iaiv7H1HYi4hu8Fat_!!6000000000770-2-tps-2876-904.png)

![log_pipeline_complete](https://img.alicdn.com/imgextra/i1/O1CN016YSeUs1F5jmtyFA5V_!!6000000000436-2-tps-1904-206.png)


## 参考：单卡V100显存 config修改要点
```yaml
# 将系统预期的GPU数量从8块减少到你实际拥有的1块V100
num_gpus_per_node: 1 
# 训练进程现在只映射到 GPU 0
actor_train.device_mapping: list(range(0,1))
# 推理进程现在只映射到 GPU 0
actor_infer.device_mapping: list(range(0,1))
# 参考模型进程现在只映射到 GPU 0
reference.device_mapping: list(range(0,1))

# 大幅减小Rollout阶段/Validation阶段的批量大小，防止单GPU处理大批次时显存不足
rollout_batch_size: 16
val_batch_size: 16

# V100 对 FP16 有较好的原生支持，而对 BF16 的支持不如 A100/H100，切换到 FP16 可以提高兼容性和稳定性，同时节省显存。
actor_train.model_args.dtype: fp16
actor_infer.model_args.dtype: fp16
reference.model_args.dtype: fp16

# 大模型训练框架从 DeepSpeed 切换到 Megatron-LM，参数可以批量发送，运行速度更快
strategy_name: megatron_train
strategy_config:
  tensor_model_parallel_size: 1
  pipeline_model_parallel_size: 1
  expert_model_parallel_size: 1
  use_distributed_optimizer: true
  recompute_granularity: full

# 在 Megatron 训练中，全局训练批次大小是 per_device_train_batch_size * gradient_accumulation_steps * world_size
actor_train.training_args.per_device_train_batch_size: 1
actor_train.training_args.gradient_accumulation_steps: 16  

# 减少每条轨迹的最大动作数，使得每个Rollout轨迹更短，减少了LLM生成内容的长度
max_actions_per_traj: 10    

# 减少并行运行的训练环境组和验证环境组，以适配单GPU资源
train_env_manager.env_groups: 1
train_env_manager.n_groups: 1
val_env_manager.env_groups: 2
val_env_manager.n_groups: [1, 1]
val_env_manager.tags: [SimpleSokoban, FrozenLake]

# 减少总的训练步骤，以便更快运行一个完整的训练流程，用于快速调试
max_steps: 100
```
