# Reward Feedback Learning (Reward FL)

## Introduction

Reward Feedback Learning (Reward FL) is a reinforcement learning algorithm that optimize diffusion models against a scorer. Reward Fl works as follows:

1. **Sampling**: For a given prompt and first frame latent, the model generates a corresponding video.
2. **Reward Assignment**: Each video is evaluated and assigned a reward based on its face informations.
3. **Model Update**: The model updates its parameters based on reward signals from the generated videos, reinforcing strategies that obtain higher rewards.

## Reward FL Configuration Parameters

In ROLL, the Reward FL algorithm-specific configuration parameters are as follows (`roll.pipeline.diffusion.reward_fl.reward_fl_config.RewardFLConfig`):

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

### Core Parameter Descriptions

- `num_train_epochs`: Number of optimization rounds per batch of samples
- `train_batch_size`: Batch size for one train step.In deepspeed training the global train batch size is `per_device_train_batch_size` \* `gradient_accumulation_steps` \* world_size
- `learning_rate`: Learning rate
- `per_device_train_batch_size`: Training batch size per device
- `gradient_accumulation_steps`: Gradient accumulation steps
- `weight_decay`: Weight decay coefficient
- `warmup_steps`: Learning rate warmup steps
- `lr_scheduler_type`: Learning rate scheduler type

### Wan2_2 Related Parameters

The following parameters related to Wan2_2 are as follows:
- `model_paths`: Model path of json file, e.g., `wan22_paths.json`, including high_noise_model, low_noise_model, text_encoder, vae.
- `tokenizer_path`: Tokenizer path. Leave empty to auto-download.
- `reward_model_path`: Reward model path, e.g., face model.
- `max_timestep_boundary`: Maximum value of the timestep interval, ranging from 0 to 1. Default is 1. This needs to be manually set only when training mixed models with multiple DiTs, for example, [Wan-AI/Wan2.2-I2V-A14B](https://modelscope.cn/models/Wan-AI/Wan2.2-I2V-A14B).
- `min_timestep_boundary`: Minimum value of the timestep interval, ranging from 0 to 1. Default is 1. This needs to be manually set only when training mixed models with multiple DiTs, for example, [Wan-AI/Wan2.2-I2V-A14B](https://modelscope.cn/models/Wan-AI/Wan2.2-I2V-A14B).
- `model_id_with_origin_paths`: Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.
- `trainable_models`: Models to train, e.g., dit, vae, text_encoder.
- `extra_inputs`: Additional model inputs, comma-separated.
- `use_gradient_checkpointing_offload`: Whether to offload gradient checkpointing to CPU memory.
- `num_inference_steps`: Number of inference steps, default is 8 for the distilled wan2_2 model.


## Note
- The reward model is constructed based on facial information, Please ensure that the first frame of the video contains a human face.
- Download the reward model(antelopev2.zip) and unzip the onnx files to `reward_model_path` directory.
- Download the official Wan2.2 pipeline and Distilled Wan2.2 DiT safetensors. Put them in the `model_paths` directory, e.g., `wan22_paths.json` file.
- According to the data/example_video_dataset/metadata.csv file, adapt your video dataset to the corresponding format

## Refernece Model
- `Official Wan2.2 pipeline`: [Wan-AI/Wan2.2-I2V-A14B](https://modelscope.cn/models/Wan-AI/Wan2.2-I2V-A14B)
- `Distilled Wan2.2 DiT safetensors`: [lightx2v/Wan2.2-Lightning](https://huggingface.co/lightx2v/Wan2.2-Lightning/tree/main)
- `Reward Model`: [deepinsight/insightface](https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip) 

## Preprocess checkpoints
- Run `merge_model.py` to merge multiple files of `Official Wan2.2 pipeline` high noise model and low noise model into one file, respectively.
- Run `merge_lora.py` to merge `Distilled Wan2.2 DiT safetensors` lora to the base model of `Official Wan2.2 pipeline` high noise model and low noise model, respectively.

## Setup environments
```
pip install -r requirements_torch260_diffsynth.txt
```

## Reference Example

You can refer to the following configuration file to set up Reward FL training:

- `./examples/docs_examples/example_reward_fl.yaml`

Run `run_reward_fl_ds_pipeline.sh` to get start.

## Reference
[1]: Identity-Preserving Image-to-Video Generation via Reward-Guided Optimization. https://arxiv.org/abs/2510.14255
