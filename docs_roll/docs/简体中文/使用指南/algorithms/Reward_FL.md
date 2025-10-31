# Reward Feedback Learning (Reward FL)

## 简介

奖励反馈学习（Reward Feedback Learning, Reward FL） 是一种强化学习算法，用于针对特定评分器对扩散模型进行优化。Reward FL 的工作流程如下：

1. **采样**: 对于给定的提示词（prompt）和首帧隐变量（latent），模型生成对应的视频。
2. **奖励计算**: 根据生成视频中的人脸信息，对其进行评估并赋予相应的奖励值。
3. **模型更新**: 模型根据生成视频所获得的奖励信号更新其参数，强化那些能够获得更高奖励的生成策略。


## Reward FL 配置参数

在 ROLL 中，使用Reward FL算法特有的配置参数如下： (`roll.pipeline.diffusion.reward_fl.reward_fl_config.RewardFLConfig`):

```yaml
# reward fl
learning_rate: 2e-6
lr_scheduler_type: constant
per_device_train_batch_size: 1
gradient_accumulation_steps: 1
warmup_steps: 10
num_train_epochs: 1

model_name: "wan2_2"

# wan2_2 related
model_paths: ./examples/wan2.2-14B-reward_fl_ds/wan22_paths.json
reward_model_path: /data/models/antelopev2/
tokenizer_path: /data/models/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/
model_id_with_origin_paths: null
trainable_models: dit2
use_gradient_checkpointing_offload: true
extra_inputs: input_image
max_timestep_boundary: 1.0
min_timestep_boundary: 0.9
num_inference_steps: 8
```

### 核心参数描述

- `learning_rate`: 学习率
- `gradient_accumulation_steps`: 梯度累积步数。
- `weight_decay`: 权重衰减大小。
- `warmup_steps`: lr 预热步数
- `lr_scheduler_type`: lr scheduler 类型

### Wan2_2 相关参数

Wan2_2 相关参数如下：
- `model_paths`: 模型权重路径，例如 `wan22_paths.json`，包括 high_noise_model、low_noise_model、text_encoder、vae。
- `tokenizer_path`: Tokenizer 路径，留空将会自动下载。
- `reward_model_path`: 奖励模型路径，例如人脸模型。
- `max_timestep_boundary`: Timestep 区间最大值，范围为 0～1，默认为 1，仅在多 DiT 的混合模型训练中需要手动设置，例如 [Wan-AI/Wan2.2-I2V-A14B](https://modelscope.cn/models/Wan-AI/Wan2.2-I2V-A14B)。[Wan-AI/Wan2.2-I2V-A14B](https://modelscope.cn/models/Wan-AI/Wan2.2-I2V-A14B).
- `min_timestep_boundary`: Timestep 区间最小值，范围为 0～1，默认为 1，仅在多 DiT 的混合模型训练中需要手动设置，例如 [Wan-AI/Wan2.2-I2V-A14B](https://modelscope.cn/models/Wan-AI/Wan2.2-I2V-A14B)。
- `model_id_with_origin_paths`: 带原始路径的模型 ID，例如 Qwen/Qwen-Image:transformer/diffusion_pytorch_model*.safetensors。用逗号分隔。
- `trainable_models`: 可训练的模型，例如 dit、vae、text_encoder。
- `extra_inputs`: 额外的模型输入，以逗号分隔。
- `use_gradient_checkpointing_offload`: 是否将 gradient checkpointing 卸载到内存中
- `num_inference_steps`: 推理步数，默认值为 8 (蒸馏 wan2_2 模型)


## 注意事项
- 奖励模型分数是基于人脸信息，因此请确保视频的第一帧包含人脸。
- 将人脸模型相关 onnx 文件下载到 `reward_model_path` 目录.
- 下载官方 Wan2.2 pipeline 和 蒸馏 Wan2.2 safetensors, 并放在 `model_paths` 目录，例如 `wan22_paths.json` 文件。
- 根据 data/example_video_dataset/metadata.csv 文件，将你的视频数据集适配到对应的格式

## 模型引用
- `官方 Wan2.2 pipeline`: [Wan-AI/Wan2.2-I2V-A14B](https://modelscope.cn/models/Wan-AI/Wan2.2-I2V-A14B)
- `蒸馏 Wan2.2 模型参数`: [lightx2v/Wan2.2-Lightning](https://huggingface.co/lightx2v/Wan2.2-Lightning/tree/main)
- `奖励模型`: [deepinsight/insightface](https://github.com/deepinsight/insightface/tree/master/model_zoo) 

## 权重预处理
- 运行`merge_model.py`来分别合并`Official Wan2.2 pipeline` 高噪模型和低噪模型的多个文件为一个
- 运行`merge_lora.py`来合并`Distilled Wan2.2 DiT safetensors`蒸馏加速lora分别到`Official Wan2.2 pipeline`高噪模型和低噪模型

## 环境配置
```
pip install -r requirements_torch260_diffsynth.txt
```

## 参考示例

可以参考以下配置文件来设置 Reward FL 训练：

- `./examples/docs_examples/example_reward_fl.yaml`

运行`run_reward_fl_ds_pipeline.sh`快速开始

## 参考文献
[1]: Identity-Preserving Image-to-Video Generation via Reward-Guided Optimization. https://arxiv.org/abs/2510.14255