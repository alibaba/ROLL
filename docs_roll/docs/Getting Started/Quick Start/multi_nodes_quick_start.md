# Quick Start: Multi-Node Deployment Guide

## Environment Preparation
1. Purchase multiple machines equipped with GPUs and install GPU drivers synchronously, with one machine as the master node and others as worker nodes (the example below uses 2 machines with 2 GPUs each)
2. Connect to the GPU instances remotely and enter the machine terminal
3. Run the following command on each machine to install the Docker environment and NVIDIA container toolkit
```shell
curl -fsSL https://github.com/alibaba/ROLL/blob/main/scripts/install_docker_nvidia_container_toolkit.sh | sudo bash
```

## Environment Configuration
Choose your desired Docker image from the [image addresses](https://alibaba.github.io/ROLL/docs/QuickStart/image_address). The following example uses *torch2.6.0 + vLLM0.8.4*
```shell
# 1. Start a Docker container with GPU support, expose container ports, and keep the container running
sudo docker run -dit \
  --gpus all \
  -p 9001:22 \
  --ipc=host \
  --shm-size=10gb \
  roll-registry.cn-hangzhou.cr.aliyuncs.com/roll/pytorch:nvcr-24.05-py3-torch260-vllm084 \
  /bin/bash

# 2. Enter the Docker container
#    You can use the `sudo docker ps` command to find the running container ID or name.
sudo docker exec -it <container_id> /bin/bash

# 3. Verify that GPUs are visible
nvidia-smi

# 4. Clone the project code
git clone https://github.com/alibaba/ROLL.git

# 5. Install project dependencies (choose the requirements file corresponding to your image)
cd ROLL
pip install -r requirements_torch260_vllm.txt -i https://mirrors.aliyun.com/pypi/simple/
```

## Pipeline Execution
1. Configure environment variables on the master node:
```shell
export MASTER_ADDR="ip of master node"
export MASTER_PORT="port of master node"  # Default: 6379
export WORLD_SIZE=2
export RANK=0
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
```
Notes:
- `MASTER_ADDR` and `MASTER_PORT` define the communication endpoint of the distributed cluster.
- `WORLD_SIZE` specifies the total number of nodes in the cluster (e.g., 2 nodes).
- `RANK` identifies the role of the node (0 represents the master node, 1, 2, 3, etc. represent worker nodes).
- `NCCL_SOCKET_IFNAME` and `GLOO_SOCKET_IFNAME` specify the network interface used for GPU/cluster communication (usually eth0).

2. Run the pipeline on the master node
```shell
bash examples/agentic_demo/run_agentic_pipeline_frozen_lake_multi_nodes_demo.sh
```
After the Ray cluster starts, you will see log examples like the following:
![log_ray_multi_nodes](https://img.alicdn.com/imgextra/i3/O1CN01sq2KKz1vBQZViSB0E_!!6000000006134-2-tps-2758-204.png)

1. Configure environment variables on the worker node
```shell
export MASTER_ADDR="ip of master node"
export MASTER_PORT="port of master node" # Default: 6379
export WORLD_SIZE=2
export RANK=1
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
```

1. Connect to the Ray cluster started by the master node on the worker node:
```shell
ray start --address='ip of master node:port of master node' --num-gpus=2
```

## Reference: Multi-GPU V100 Memory Configuration Key Points
```yaml
# Reduce the expected number of GPUs from 8 to the 2 V100s you actually have
num_gpus_per_node: 2
# Training processes now map to GPUs 0-3
actor_train.device_mapping: list(range(0,4))
# Inference processes now map to GPUs 0-3
actor_infer.device_mapping: list(range(0,4))
# Reference model processes now map to GPUs 0-3
reference.device_mapping: list(range(0,4))

# Significantly reduce the batch size during Rollout/Validation stages to prevent out-of-memory errors when a single GPU processes large batches
rollout_batch_size: 64
val_batch_size: 16

# V100 has better native support for FP16 than BF16 (unlike A100/H100). Switching to FP16 can improve compatibility and stability while saving GPU memory.
actor_train.model_args.dtype: fp16
actor_infer.model_args.dtype: fp16
reference.model_args.dtype: fp16

# Switch the large model training framework from DeepSpeed to Megatron-LM, where parameters can be sent in batches for faster execution
strategy_name: megatron_train
strategy_config:
  tensor_model_parallel_size: 1
  pipeline_model_parallel_size: 1
  expert_model_parallel_size: 1
  use_distributed_optimizer: true
  recompute_granularity: full

# In Megatron training, the global training batch size is per_device_train_batch_size * gradient_accumulation_steps * world_size, where world_size = 4
actor_train.training_args.per_device_train_batch_size: 1
actor_train.training_args.gradient_accumulation_steps: 16  

# Reduce the maximum number of actions per trajectory to make each Rollout trajectory shorter, reducing the length of LLM-generated content
max_actions_per_traj: 10    

# Reduce the number of parallel training environment groups and validation environment groups to accommodate single GPU resources
train_env_manager.env_groups: 1
train_env_manager.n_groups: 1
val_env_manager.env_groups: 2
val_env_manager.n_groups: [1, 1]
val_env_manager.tags: [SimpleSokoban, FrozenLake]

# Reduce the total number of training steps to run a complete training process faster for quick debugging
max_steps: 100
```