# Installation

## 🐳 Install from Docker

We provide pre-built Docker images for a quick start (Links will be updated):

* `torch2.6.0 + SGlang0.4.6`: roll-registry.cn-hangzhou.cr.aliyuncs.com/roll/pytorch:nvcr-24.05-py3-torch260-sglang046
* `torch2.6.0 + vLLM0.8.4`: roll-registry.cn-hangzhou.cr.aliyuncs.com/roll/pytorch:nvcr-24.05-py3-torch260-vllm084
* `torch2.5.1 + SGlang0.4.3`: roll-registry.cn-hangzhou.cr.aliyuncs.com/roll/pytorch:nvcr-24.05-py3-torch251-sglang043
* `torch2.5.1 + vLLM0.7.3`: roll-registry.cn-hangzhou.cr.aliyuncs.com/roll/pytorch:nvcr-24.05-py3-torch251-vllm073

You can also find Dockerfiles under the `docker/` directory to build your own images.

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
## 🔄 About Model Checkpoint Format

For `MegatronStrategy`, model checkpoints are saved in Megatron format by default. To convert them back to HuggingFace format, please use the following command:

```bash
python mcore_adapter/tools/convert.py --checkpoint_path path_to_megatron_model --output_path path_to_output_hf_model
```
