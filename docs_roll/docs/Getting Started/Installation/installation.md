# Installation

## 🐳 Install from Docker

We provide pre-built Docker images both on CUDA and ROCm for a quick start. Choose your desired image from the [Image Addresses](https://alibaba.github.io/ROLL/docs/QuickStart/image_address).

## 🛠️ Install from Custom Environment

If our pre-built Docker images are not compatible with your environment, you can install ROLL and its dependencies in your Python environment. Please ensure you meet the following prerequisites:

```bash
# Prerequisites
CUDA Version >= 12.4
cuDNN Version >= 9.1.0
PyTorch >= 2.5.1
SGlang >= 0.4.3
vLLM >= 0.7.3

# Clone the repository and install
git clone https://github.com/alibaba/ROLL.git
cd ROLL
pip install -r requirements.txt # Or follow your specific installation steps
# For development, consider: pip install -e .
```

For AMD users, please ensure you meet the following prerequisites:

```bash
# Prerequisites
ROCm Version >= 6.3.4
PyTorch >= 2.6.0
vLLM >= 0.8.4
# Clone the repository and install
git clone https://github.com/alibaba/ROLL.git
cd ROLL
pip install -r requirements.txt # Or follow your specific installation steps
```
We highly suggest to use pre-built Docker images from [Image Addresses](https://alibaba.github.io/ROLL/docs/QuickStart/image_address) instead of installation from Custom Environment for ROCm users.

## 🔄 About Model Checkpoint Format

For `MegatronStrategy`, model checkpoints are saved in Megatron format by default. To convert them back to HuggingFace format, please use the following command:

```bash
python mcore_adapter/tools/convert.py --checkpoint_path path_to_megatron_model --output_path path_to_output_hf_model
```
