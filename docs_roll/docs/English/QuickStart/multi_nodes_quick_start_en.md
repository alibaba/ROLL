# Quickstart: Multi Nodes Deployment Guide

## Environment Preparation
1. Purchase multiple machines equipped with GPU and install GPU drivers simultaneously. One machine serves as the master node, and the others serve as worker nodes. (e.g., 2 machines with 2 GPUs each).
2. Connect remotely to the GPU instance and access the machine terminal
3. Install Docker environment and NVIDIA Container Toolkit on each machine
```shell
curl -fsSL https://github.com/alibaba/ROLL/blob/main/scripts/install_docker_nvidia_container_toolkit.sh  | sudo bash   
```

## Environment Configuration
Choose your desired image from the [image addresses](https://alibaba.github.io/ROLL/docs/English/QuickStart/image_address). The following example will use *torch2.6.0 + vLLM0.8.4*.
```shell
# 1. Start a Docker container with GPU support, expose the port, and keep the container running.
sudo docker run -dit \
  --gpus all \
  -p 9001:22 \
  --ipc=host \
  --shm-size=10gb \
  roll-registry.cn-hangzhou.cr.aliyuncs.com/roll/pytorch:nvcr-24.05-py3-torch260-vllm084 \
  /bin/bash

# 2. Enter the Docker container
#    You can find your running container's ID or name using `sudo docker ps`.
sudo docker exec -it <container_id> /bin/bash

# 3. Verify GPU visibility
nvidia-smi

# 4. Clone the project repo
git clone https://github.com/alibaba/ROLL.git

# 5. Install dependencies (select the requirements file corresponding to your chosen image)
cd ROLL
pip install -r requirements_torch260_vllm.txt -i https://mirrors.aliyun.com/pypi/simple/
```

## pipeline运行
1. Set environment variables on the master node:
```shell
export MASTER_ADDR="ip of master node"
export MASTER_PORT="port of master node"  # Default: 6379
export WORLD_SIZE=2
export RANK=0
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
```

Notes:
- `MASTER_ADDR` and `MASTER_PORT` define the communication endpoint for the distributed cluster.
- `WORLD_SIZE` specifies the total number of nodes in the cluster (e.g., 2 nodes).
- `RANK` identifies the node's role (0 for master, 1、2、3 etc. for worker nodes).
- `NCCL_SOCKET_IFNAME` and `GLOO_SOCKET_IFNAME` specify the network interface for GPU/cluster communication (typically eth0).

2. Run the pipeline on the master node:
```shell
bash examples/agentic_demo/run_agentic_pipeline_frozen_lake_multi_nodes_demo.sh
```
After the Ray cluster starts, you will see log examples like:
![log_ray_multi_nodes](../../../static/img/log_ray_multi_nodes.png)

3. Set environment variables on the worker node:
```shell
export MASTER_ADDR="ip of master node"
export MASTER_PORT="port of master node" # Default: 6379
export WORLD_SIZE=2
export RANK=1
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
```

4. Connect the worker node to the Ray cluster on the master node:
```shell
ray start --address='ip of master node:port of master node' --num-gpus=2
```

## Reference: V100 Multi-GPU Memory Configuration Optimization
```yaml
# Reduce the system's expected number of GPUs from 8 to your actual 2 V100
num_gpus_per_node: 2
# Training processes are now mapped to GPU 0-3
actor_train.device_mapping: list(range(0,4))
# Inference processes are now mapped only to GPU 0-3
actor_infer.device_mapping: list(range(0,4))
# Reference model processes are now mapped to GPU 0-3
reference.device_mapping: list(range(0,4))

# Significantly reduce the batch sizes for Rollout and Validation stages to prevent out-of-memory errors on a single GPU
rollout_batch_size: 64
val_batch_size: 16

# V100 has better native support for FP16 than BF16 (unlike A100/H100). Switching to FP16 improves compatibility and stability, while also saving GPU memory.
actor_train.model_args.dtype: fp16
actor_infer.model_args.dtype: fp16
reference.model_args.dtype: fp16

# Switch the large model training framework from DeepSpeed to Megatron-LM. Parameters can be sent in batches, resulting in faster execution.
strategy_name: megatron_train
strategy_config:
  tensor_model_parallel_size: 1
  pipeline_model_parallel_size: 1
  expert_model_parallel_size: 1
  use_distributed_optimizer: true
  recompute_granularity: full

# In megatron training the global train batch size is equivalent to per_device_train_batch_size * gradient_accumulation_steps * world_size, and here world_size is 4
actor_train.training_args.per_device_train_batch_size: 1
actor_train.training_args.gradient_accumulation_steps: 16  

# Reduce the maximum number of actions per trajectory, making each Rollout trajectory shorter that reduces the length of LLM-generated content.
max_actions_per_traj: 10    

# Reduce the number of parallel training and validation environment groups to accommodate single-GPU resources.
train_env_manager.env_groups: 1
train_env_manager.n_groups: 1
val_env_manager.env_groups: 2
val_env_manager.n_groups: [1, 1]
val_env_manager.tags: [SimpleSokoban, FrozenLake]

# Reduce the total number of training steps for quicker full pipeline runs, useful for rapid debugging.
max_steps: 100
```
