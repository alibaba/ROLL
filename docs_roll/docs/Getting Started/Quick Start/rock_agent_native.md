# ROCK Agent Native Quick Start Guide

This guide will walk you through running a Reinforcement Learning example based on iflow-cli (Agent) using ROLL (Training Framework) and ROCK (Environment Management).

## Prerequisites

- ROCK Service: Ensure you have an available ROCK service. For local server setup, refer to [ROCK Installation Guide](https://alibaba.github.io/ROCK/docs/Getting%20Started/installation)

- For instructions on starting ROCK and ROLL on a single machine, refer to[ROCK & ROLL Quick Start Guide](https://alibaba.github.io/ROLL/docs/Getting%20Started/rockroll)


## Usage Examples

ROLL provides configuration examples based on iflow-cli, located in the *examples/agentic_demo* directory of the ROLL repository:

```
examples/agentic_demo
├── agent_rollout_rock_swe.yaml    # Rollout only (Inference/Sampling) 
└── agent_val_rock_swe.yaml        # Full pipeline (Train & Val)
```

To run an example:
```bash
bash examples/agentic_demo/run_agentic_rollout_pipeline_rock_swe.sh

bash examples/agentic_demo/run_agentic_pipeline_rock_swe.sh
```

## Data Preparation

This example uses the SWE-bench Verified evaluation set, converted into the Terminal-bench format.

- [Git Repo](https://github.com/laude-institute/terminal-bench-datasets/tree/main/datasets/swebench-verified)
- [Data Description](https://www.tbench.ai/registry/swebench-verified/head/sympy__sympy-18199)
- [Image Registry](https://hub.docker.com/r/slimshetty/swebench-verified/tags)

The full evaluation set must be downloaded locally beforehand:
```bash
cd / && git clone https://github.com/laude-institute/terminal-bench-datasets.git
```

The repository provides 10 task samples: *data/swe_bench_verified_example.jsonl*

Example configuration (modify as needed):
```yaml
custom_envs:
  swebench_native_verified:
    env_config:
      dataset_name: /ROLL/data/swe_bench_verified_example.jsonl
      test_files: ["/terminal-bench-datasets/datasets/swebench-verified"]
```

## ROCK Service Setup

1. Install ROCK SDK
```bash
pip install rl-rock -i https://mirrors.aliyun.com/pypi/simple/
```

2. Configure ROCK Service Address
```yaml
env_config:
    # Replace with your actual ROCK service address
    # e.g., 'http://192.168.1.10:8000'
    sandbox_base_url: 'http://<ip_address>:<port>'
```

## Agent Configuration

This example uses iflow-cli as the execution Agent:

```yaml
agent_config_common:
  agent_type: "default"
  
  # Startup command; placeholders (e.g., <<PROMPT>>) are parsed in the code
  run_cmd: 'iflow -p <<PROMPT>> --yolo'
  
  # Dependency pre-installation; modify based on your sandbox image
  pre_init_cmds:
    - command: "apt-get update"
      timeout_seconds: 600
    - command: "apt-get install -y curl git wget xz-utils"
      timeout_seconds: 600
    - command: "apt-get install -y build-essential libc6-dev patch procps"
      timeout_seconds: 600
    # Install helper tools like 'uv'
    - command: "wget -q https://xrl-sandbox-bucket.oss-cn-hangzhou.aliyuncs.com/uv-files/uv-x86_64-unknown-linux-gnu.tar.gz && tar -xzf uv-x86_64-unknown-linux-gnu.tar.gz --strip-components=1 -C /usr/local/bin && uv --version"
      timeout_seconds: 600 

  model_service_config: 
    type: "local"
    enabled: True
  
  # 运行时环境  
  runtime_env_config:
    type: node
    npm_registry: "https://registry.npmmirror.com"
    # Install specific iflow versions as needed
    custom_install_cmd: "wget --retry-connrefused --tries=10 --waitretry=2 -O ~/iflow-cli.tgz 'http://cloud.iflow.cn/iflow-cli/iflow-ai-iflow-cli-for-roll-0-4-4-v5.tgz' && npm i -g ~/iflow-cli.tgz"
  
  env:
    # Configure iflow parameters as needed
    IFLOW_apiKey: "test"
    IFLOW_baseUrl: "http://localhost:8080/v1"
    IFLOW_modelName: "ROME"
    IFLOW_searchApiKey: "88888888"
    IFLOW_selectedAuthType: "openai-compatible"
    IFLOW_disableAutoUpdate: "true"
    IFLOW_tokensLimit: "128000"
    IFLOW_shellTimeout: "360000"
    IFLOW_coreTools: "Edit,exit_plan_mode,glob,list_directory,multi_edit,plan,read plan,read_file,read_many_files,save_memory,Search,Shell,task,web_fetch,web_search,write_file,xml_escape"
```

ROCK also supports other Agents. For more details, refer to the[ROCK Agent](https://alibaba.github.io/ROCK/docs/References/Python%20SDK%20References/rock-agent)


## Key Module Index
- Environment Implementation: *roll/pipeline/agentic/env/terminal_env/rock_tb_native_env.py* 
  - Responsible for RL flow control, reward calculation, and task distribution.
- Sandbox Management: *roll/pipeline/agentic/env/rock/sandbox_manager_v2.py* 
  - Responsible for communication with ROCK services, file uploads, and session management.
- Agent Management: *roll/pipeline/agentic/env/rock/agent_manager.py* 
  - Responsible for configuring the environment and binaries required by the Agent upon sandbox startup.

For more information on the principles of the [Model Service](https://alibaba.github.io/ROCK/docs/References/Python%20SDK%20References/model-service)