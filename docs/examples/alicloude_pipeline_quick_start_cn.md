# 阿里云ROLL实践手册

## 准备环境
1. 购买阿里云服务器
- 单机版本 选择 GPU：NVIDIA V100
- 建议您通过ECS控制台购买GPU实例时，同步选中安装GPU驱动
2. 选中分配公网IPv4地址，带宽计费方式选择按使用流量，建议带宽峰值选择100 Mbps，以加快模型下载速度。
3. 远程连接GPU实例，进入机器终端
4. 安装 Docker 环境：参考 https://developer.aliyun.com/mirror/docker-ce/?spm=a2c6h.25603864.0.0.59637e39XhdIj0
```shell
# step 1: 安装必要的一些系统工具
sudo yum install -y yum-utils

# Step 2: 添加软件源信息
yum-config-manager --add-repo https://mirrors.aliyun.com/docker-ce/linux/centos/docker-ce.repo

# Step 3: 安装Docker
sudo yum install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Step 4: 开启Docker服务
sudo service docker start

# 安装校验
docker version
```
5. 安装 NVIDIA容器工具包
```shell
curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | \
  sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo
  
# 安装 NVIDIA Container Toolkit 软件包
sudo yum install -y nvidia-container-toolkit

# 重启docker
sudo systemctl restart docker
```

## 环境配置
参考 https://code.alibaba-inc.com/openlm/ScaleAligner/blob/open/roll/README.md
```shell
# 1. 拉取docker镜像
sudo docker pull roll-registry.cn-hangzhou.cr.aliyuncs.com/roll/pytorch:nvcr-24.05-py3-torch260-sglang046
# 或者
sudo docker pull roll-registry.cn-hangzhou.cr.aliyuncs.com/roll/pytorch:nvcr-24.05-py3-torch260-vllm084

# 2. 启动一个docker容器，指定GPU支持，并始终保持容器运行
sudo docker images
sudo docker exec -it <image_id> /bin/bash
sudo docker run -d \
  --gpus all \
  --network=host \
  --ipc=host \
  --shm-size=2gb \
  <image_id> \
  /bin/bash -c 'tail -f /dev/null'

# 3. 进入docker容器（下次重新连接也要执行这条命令）
sudo docker ps
sudo docker exec -it <container_id> /bin/bash

# 4. 验证GPU是否可见
nvidia-smi

# 5. 下载代码

# 安装git（该镜像是ubuntu系统），并clone代码
apt update && apt install git -y
git clone https://github.com/alibaba/ROLL.git

# 若无法访问github，直接下载zip文件并解压
wget https://github.com/alibaba/ROLL/archive/refs/heads/main.zip
unzip main.zip

# 5. 安装依赖（选择对应镜像的requirements文件）
cd ROLL-main
pip install -r requirements_torch260_sglang.txt -i https://mirrors.aliyun.com/pypi/simple/
```

## pipeline运行
```shell
# 若执行报错 ModuleNotFoundError: No module named 'roll'，需要添加环境变量
export PYTHONPATH="/workspace/ROLL-main:$PYTHONPATH"

# 方法一：指定yaml文件路径，需要以脚本目录即examples为根目录
python examples/start_rlvr_pipeline.py --config_path qwen2.5-7B-rlvr_megatron  --config_name rlvr_config

# 方法二：直接执行sh脚本
bash examples/qwen2.5-7B-rlvr_megatron/run_rlvr_pipeline.sh
bash examples/qwen2.5-0.5B-agentic_ds/run_agentic_pipeline_frozen_lake.sh

# 根据需要修改config
vim examples/qwen2.5-0.5B-agentic_ds/agent_val_frozen_lake.yaml
```

单卡V100显存config修改要点：
```yaml
# 将系统预期的 GPU 数量从 8 块减少到你实际拥有的 1 块 V100
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

# 大模型训练框架从 DeepSpeed 切换到 Megatron-LM，参数可以批量发送
strategy_name: megatron_train
strategy_config:
  tensor_model_parallel_size: 1
  pipeline_model_parallel_size: 1
  expert_model_parallel_size: 1
  use_distributed_optimizer: true
  recompute_granularity: full

# 减少并行运行的训练环境组和验证环境组，以适配单 GPU 资源
train_env_manager.env_groups: 1
train_env_manager.n_groups: 1
val_env_manager.env_groups: 2
val_env_manager.n_groups: [1, 1]
val_env_manager.tags: [SimpleSokoban, FrozenLake]
```


