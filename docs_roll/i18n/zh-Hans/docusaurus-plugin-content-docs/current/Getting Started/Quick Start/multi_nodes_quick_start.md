# 快速上手：多节点部署指南

## 准备环境
1. 购买多台配备GPU的机器，并同步安装GPU驱动，其中一台作为主节点，其它作为工作节点（下文示例为2台机器各2个GPU）
2. 远程连接GPU实例，进入机器终端
3. 分别运行以下命令安装 Docker环境 和 NVIDIA容器工具包
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
1. 在主节点配置环境变量:
```shell
export MASTER_ADDR="ip of master node"
export MASTER_PORT="port of master node"  # Default: 6379
export WORLD_SIZE=2
export RANK=0
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
```
注意事项：
- `MASTER_ADDR` 和 `MASTER_PORT` 定义了分布式集群的通信端点。
- `WORLD_SIZE` 指定了集群中的节点总数（例如 2 个节点）。
- `RANK` 用于标识节点的角色（0 表示主节点，1、2、3 等表示工作节点）。
- `NCCL_SOCKET_IFNAME` 和 `GLOO_SOCKET_IFNAME` 指定了 GPU/集群通信使用的网络接口（通常为 eth0）。

2. 在主节点上运行pipeline
```shell
bash examples/agentic_demo/run_agentic_pipeline_frozen_lake_multi_nodes_demo.sh
```
ray集群启动后，会看到下面的log示例：
![log_ray_multi_nodes](https://img.alicdn.com/imgextra/i3/O1CN01sq2KKz1vBQZViSB0E_!!6000000006134-2-tps-2758-204.png)

1. 在工作节点配置环境变量
```shell
export MASTER_ADDR="ip of master node"
export MASTER_PORT="port of master node" # Default: 6379
export WORLD_SIZE=2
export RANK=1
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
```

1. 在工作节点上连接主节点启动的ray集群：
```shell
ray start --address='ip of master node:port of master node' --num-gpus=2
```

## 参考：多卡V100显存 config修改要点
```yaml
# 将系统预期的GPU数量从8块减少到你实际拥有的2块V100
num_gpus_per_node: 2
# 训练进程现在映射到 GPU 0-3
actor_train.device_mapping: list(range(0,4))
# 推理进程现在映射到 GPU 0-3
actor_infer.device_mapping: list(range(0,4))
# 参考模型进程现在映射到 GPU 0-3
reference.device_mapping: list(range(0,4))

# 大幅减小Rollout阶段/Validation阶段的批量大小，防止单GPU处理大批次时显存不足
rollout_batch_size: 64
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

# 在 Megatron 训练中，全局训练批次大小是 per_device_train_batch_size * gradient_accumulation_steps * world_size，这里的world_size = 4
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
