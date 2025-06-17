# Quickstart: Singel Node based on Alibaba Cloud

## Environment Preparation
1. Purchase an Alibaba Cloud Server
- For a single-machine setup, consider a GPU instance with **NVIDIA V100**.
- **Recommendation:** When purchasing a GPU instance via the ECS console, it's advised to select the option to automatically install GPU drivers.
2. Remote Connect to the GPU Instance and access the machine terminal
3. Install NVIDIA Container Toolkit
```shell
curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | \
  sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo
  
sudo yum install -y nvidia-container-toolkit
```
4. Install Docker Environment：refer to https://developer.aliyun.com/mirror/docker-ce/
```shell
# step 1: install necessary system tools
sudo yum install -y yum-utils

# Step 2: add software repository information
sudo yum-config-manager --add-repo https://mirrors.aliyun.com/docker-ce/linux/centos/docker-ce.repo

# Step 3: install Docker
sudo yum install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Step 4: start Docker service
sudo service docker start

# Verify that GPUs are visible
docker version
```


## Environment Configuration
```shell
# 1. Pull Docker image
sudo docker pull <image_address>

# Image Addresses (choose based on your needs)
# torch2.6.0 + SGlang0.4.6: roll-registry.cn-hangzhou.cr.aliyuncs.com/roll/pytorch:nvcr-24.05-py3-torch260-sglang046
# torch2.6.0 + vLLM0.8.4: roll-registry.cn-hangzhou.cr.aliyuncs.com/roll/pytorch:nvcr-24.05-py3-torch260-vllm084
# torch2.5.1 + SGlang0.4.3: roll-registry.cn-hangzhou.cr.aliyuncs.com/roll/pytorch:nvcr-24.05-py3-torch251-sglang043
# torch2.5.1 + vLLM0.7.3: roll-registry.cn-hangzhou.cr.aliyuncs.com/roll/pytorch:nvcr-24.05-py3-torch251-vllm073

# 2. Start a Docker container with GPU support and keep the container running
sudo docker images
sudo docker run -dit \
  --gpus all \
  --network=host \
  --ipc=host \
  --shm-size=2gb \
  <image_id> \
  /bin/bash

# 3. Enter the Docker container (execute this command every time you reconnect)
sudo docker ps
sudo docker exec -it <container_id> /bin/bash

# 4. Verify GPU visibility
nvidia-smi

# 5. Download the code

# Install git（this image is ubuntu-based）and clone the repo
apt update && apt install git -y
git clone https://github.com/alibaba/ROLL.git

# If Github is not accessible, download the zip file and unzip it
wget https://github.com/alibaba/ROLL/archive/refs/heads/main.zip
unzip main.zip

# 5. Install dependencies (select the requirements file corresponding to your chosen image)
cd ROLL-main
pip install -r requirements_torch260_sglang.txt -i https://mirrors.aliyun.com/pypi/simple/
```

## Pipeline Execution
```shell
# If you encounter "ModuleNotFoundError: No module named 'roll'", you need to add environment variables
export PYTHONPATH="/workspace/ROLL-main:$PYTHONPATH"

# Method 1: Specify the YAML file path, with the script directory (examples) as the root
python examples/start_agentic_pipeline.py --config_path qwen2.5-0.5B-agentic_ds  --config_name agent_val_frozen_lake

# Method 2: Execute the shell script directly
bash examples/qwen2.5-0.5B-agentic_ds/run_agentic_pipeline_frozen_lake.sh

# Modify the configuration as needed
vim examples/qwen2.5-0.5B-agentic_ds/agent_val_frozen_lake.yaml
```

Key Configuration Modifications for Single V100 GPU Memory:
```yaml
# Reduce the system's expected number of GPUs from 8 to your actual 1 V100
num_gpus_per_node: 1 
# Training processes are now mapped only to GPU 0
actor_train.device_mapping: list(range(0,1))
# Inference processes are now mapped only to GPU 0
actor_infer.device_mapping: list(range(0,1))
# Reference model processes are now mapped only to GPU 0
reference.device_mapping: list(range(0,1))

# Significantly reduce the batch sizes for Rollout and Validation stages to prevent out-of-memory errors on a single GPU
rollout_batch_size: 16
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

# In megatron training the global train batch size is equivalent to per_device_train_batch_size * gradient_accumulation_steps * world_size
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

Example Log Screenshots during Pipeline Execution:
<img src="../../../../static/img/log_1.png" width="100%" alt="log1" />

<img src="../../../../static/img/log_2.png" width="100%" alt="log2" />

<img src="../../../../static/img/log_3.png" width="100%" alt="log3" />

