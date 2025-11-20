# Quick Start: Single-Node Deployment Guide

## Environment Preparation
1. Purchase a machine equipped with GPU and install GPU drivers synchronously
2. Connect to the GPU instance remotely and enter the machine terminal
3. Run the following command to install the Docker environment and NVIDIA container toolkit
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
```shell
bash examples/agentic_demo/run_agentic_pipeline_frozen_lake_single_node_demo.sh
```

Example log screenshots during pipeline execution:
![log_pipeline_start](https://img.alicdn.com/imgextra/i2/O1CN015wRcUb1EkfZQEjpEI_!!6000000000390-2-tps-2868-650.png)

![log_pipeline_in_training](https://img.alicdn.com/imgextra/i2/O1CN01Iaiv7H1HYi4hu8Fat_!!6000000000770-2-tps-2876-904.png)

![log_pipeline_complete](https://img.alicdn.com/imgextra/i1/O1CN016YSeUs1F5jmtyFA5V_!!6000000000436-2-tps-1904-206.png)

## Reference: Single V100 GPU Memory Configuration Key Points
```yaml
# Reduce the expected number of GPUs from 8 to the 1 V100 you actually have
num_gpus_per_node: 1 
# Training processes now only map to GPU 0
actor_train.device_mapping: list(range(0,1))
# Inference processes now only map to GPU 0
actor_infer.device_mapping: list(range(0,1))
# Reference model processes now only map to GPU 0
reference.device_mapping: list(range(0,1))

# Significantly reduce the batch size during Rollout/Validation stages to prevent out-of-memory errors when a single GPU processes large batches
rollout_batch_size: 16
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

# In Megatron training, the global training batch size is per_device_train_batch_size * gradient_accumulation_steps * world_size
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