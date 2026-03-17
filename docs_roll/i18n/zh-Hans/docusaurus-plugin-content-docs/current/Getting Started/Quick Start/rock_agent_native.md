# ROCK Agent Native 快速开始指南

本指南将引导您使用 ROLL (训练框架) 和 ROCK (环境管理) 来运行一个基于 iflow-cli（Agent）的强化学习示例。

## 前置条件

- 确保有可用的ROCK服务, 如果需要本地拉起服务端, 参考[ROCK快速启动](https://alibaba.github.io/ROCK/zh-Hans/docs/Getting%20Started/installation)

- 如果需要单机启动ROCK服务并运行ROLL，参考[ROCK & ROLL 快速开始指南](https://alibaba.github.io/ROLL/zh-Hans/docs/Getting%20Started/rockroll)


## 使用示例

ROLL提供了基于iflow-cli（Agent）的配置示例，位于ROLL仓库的*examples/agentic_demo*目录下:

```
examples/agentic_demo
├── agent_rollout_rock_swe.yaml    # 仅运行 Rollout（推理/采样）
└── agent_val_rock_swe.yaml        # 包含训练（Train）和验证（Val）全流程
```

可以选择一个示例运行：
```bash
bash examples/agentic_demo/run_agentic_rollout_pipeline_rock_swe.sh

bash examples/agentic_demo/run_agentic_pipeline_rock_swe.sh
```

## 数据准备

本示例使用的是 SWE-bench Verified 评测集，转成 Terminal-bench 格式
- [git地址](https://github.com/laude-institute/terminal-bench-datasets/tree/main/datasets/swebench-verified)
- [数据介绍](https://www.tbench.ai/registry/swebench-verified/head/sympy__sympy-18199)
- [镜像仓库](https://hub.docker.com/r/slimshetty/swebench-verified/tags)

最终的评测集需要提前下载到本地
```bash
cd / && git clone https://github.com/laude-institute/terminal-bench-datasets.git
```

仓库中提供了10条样例数据：*data/swe_bench_verified_example.jsonl*

示例配置如下，你可以按照自己的需要进行修改
```yaml
custom_envs:
  swebench_native_verified:
    env_config:
      dataset_name: /ROLL/data/swe_bench_verified_example.jsonl
      test_files: ["/terminal-bench-datasets/datasets/swebench-verified"]
```

## ROCK服务相关

1. 安装ROCK SDK
```bash
pip install rl-rock -i https://mirrors.aliyun.com/pypi/simple/
```

2. 配置ROCK服务地址
```yaml
env_config:
    # 将这里的地址修改为您的 ROCK 服务地址
    # 例如: sandbox_base_url: 'http://192.168.1.10:8000'
    sandbox_base_url: 'http://<ip_address>:<port>'
```

## Agent配置

本示例以iflow-cli作为执行Agent:

```yaml
agent_config_common:
  agent_type: "default"
  # 启动命令，特殊符号会在代码中解析
  run_cmd: 'iflow -p <<PROMPT>> --yolo'
  # 依赖预装，请根据你的镜像进行修改
  pre_init_cmds:
    - command: "apt-get update"
      timeout_seconds: 600
    - command: "apt-get install -y curl git wget xz-utils"
      timeout_seconds: 600
    - command: "apt-get install -y build-essential libc6-dev patch procps"
      timeout_seconds: 600
    # 安装 uv 等辅助工具
    - command: "wget -q https://xrl-sandbox-bucket.oss-cn-hangzhou.aliyuncs.com/uv-files/uv-x86_64-unknown-linux-gnu.tar.gz && tar -xzf uv-x86_64-unknown-linux-gnu.tar.gz --strip-components=1 -C /usr/local/bin && uv --version"
      timeout_seconds: 600 
  model_service_config: 
    type: "local"
    enabled: True
  # 运行时环境  
  runtime_env_config:
    type: node
    npm_registry: "https://registry.npmmirror.com"
    # 根据需要安装自己所需iflow版本
    custom_install_cmd: "wget --retry-connrefused --tries=10 --waitretry=2 -O ~/iflow-cli.tgz 'http://cloud.iflow.cn/iflow-cli/iflow-ai-iflow-cli-for-roll-0-4-4-v5.tgz' && npm i -g ~/iflow-cli.tgz"
  env:
    # 根据需要设置iflow参数
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

ROCK服务也支持其他Agent，配置可参考[ROCK Agent](https://alibaba.github.io/ROCK/zh-Hans/docs/References/Python%20SDK%20References/rock-agent)


## 重要模块索引
- 环境实现：roll/pipeline/agentic/env/terminal_env/rock_tb_native_env.py 
  - 负责 RL 流程控制、奖励计算和任务分发
- 沙盒管理：roll/pipeline/agentic/env/rock/sandbox_manager_v2.py 
  - 负责与 ROCK 服务通信、文件上传、Session 管理。
- Agent 管理：roll/pipeline/agentic/env/rock/agent_manager.py 
  - 负责在沙盒启动瞬间配置 Agent 所需的环境和二进制文件

Model Service的原理可参考[文档](https://alibaba.github.io/ROCK/zh-Hans/docs/References/Python%20SDK%20References/model-service)
