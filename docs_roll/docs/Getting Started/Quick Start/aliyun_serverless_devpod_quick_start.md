# Quick Start: Alibaba Cloud Function Compute DevPod for Rapid Development

> DevPod is a rapid development environment provided by Alibaba Cloud Function Compute that can help you quickly deploy and run the ROLL project.

## Prerequisites

Before you begin, ensure you have completed the following preparations:

- You have an Alibaba Cloud account.
- You are logged into the [Function Compute FunModel Console](https://fcnext.console.aliyun.com/fun-model).
- Complete the required RAM role authorization and configuration as guided by the console.

## Create a Training DevPod Environment

1.  Click **Create Model - Custom Development**.
2.  Select **Custom Environment** and configure as follows:
    *   **Container Image**: Choose a non-public custom image - Container Image Address - `roll-registry.cn-hangzhou.cr.aliyuncs.com/roll/pytorch:nvcr-24.05-py3-torch260-vllm084`
    *   **Model Name**: Enter a name, e.g., `roll-dev`
    *   **Model Source**: Select `No Model`
    *   **Startup Command**: Keep the default, no modification needed
    *   **Instance Specification**: Select `GPU Performance Type`
    *   Click the **DevPod Development & Debugging** button (**Note**: Do *not* click "Create Model Service")
3.  Wait for deployment to complete (typically 1–2 minutes).

## Configure and Test

### Download the ROLL Project and Install Dependencies

```bash
# /mnt/ModelName is the default NAS mount point; replace 'roll-dev' with your actual model name
cd /mnt/roll-dev

# 1. Clone the project code
git clone https://github.com/alibaba/ROLL.git  

# 2. Install project dependencies
cd ROLL
pip install -r requirements_torch260_vllm.txt -i https://mirrors.aliyun.com/pypi/simple/  
```

### Run a Pipeline Example

```bash
bash examples/agentic_demo/run_agentic_pipeline_frozen_lake_single_node_demo.sh
```

For more DevPod usage guidelines, see [https://fun-model-docs.devsapp.net/user-guide/devpod/](https://fun-model-docs.devsapp.net/user-guide/devpod/)
