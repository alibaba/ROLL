# 公开数据的本地 Tinker runtime

本 package 将 Tinker 采样与训练接入 ROLL。runtime type 为 `ROLL`，runtime 配置段为 `tinker_runtime`，嵌入 engine 配置段为 `roll`。GPU runtime 从本仓库启动，使用既有训练与推理 strategy。

已安装的 `rl-rock` SDK/backend 使用独立控制面环境。原 Tinker cookbook
通过 ROCK ModelService 编排任务环境，ROLL runtime 领取模型采样与训练
action。控制面与 GPU runtime 应使用各自的解释器。

## 安装 GPU runtime

使用 quick-start 指定的 Python 3.12 / PyTorch 2.11 / vLLM 0.23 / CUDA 13
基础镜像，并准备 `uv`。在 ROLL 根目录执行：

```bash
bash examples/tinker_backend_runtime/setup_public_runtime.sh \
  --roll-root "$PWD" \
  --venv "$PWD/.venv-tinker-runtime" \
  --python /usr/bin/python3
```

脚本继承基础镜像的 CUDA 包，按
[requirements_tinker.txt](../../requirements_tinker.txt) 安装补充依赖，
并以 `--no-deps` 安装 SGLang 工具解析器。它校验受保护 GPU 包没有被替换，
然后执行 CPU 导入检查。venv 内的 `setup-public-runtime.json` 记录版本与
启动 runtime 所需的 CUDA 库环境。这些检查不运行模型。

克隆两个仓库、安装 SDK/backend、下载公开资产、生成本地路径配置，以及
通过原 cookbook 运行 rollout 和 PPO/PPO+KL/GRPO 训练，统一参见
[Tinker quick-start](../../../ROCK/docs/tinker-quick-start.md)。相对链接要求
`ROCK` 与 `ROLL` 为同级目录。运行命令、验收边界与结果均维护在该文档。

## Runtime 接口

[runtime_pipeline.py](../../roll/pipeline/tinker_backend_runtime/runtime_pipeline.py)
通过 `python -m roll.pipeline.tinker_backend_runtime.runtime_pipeline`
启动，避免相邻 `types.py` 遮蔽 Python 标准库。

结构化 chat prompt 使用模型 tokenizer 的 chat template，并传入 tools。
采样保留 prompt token IDs、生成 token IDs、logprobs、原始解码文本与
解析后的 tool calls。只有经末尾 token 校验的传输 EOS 会从响应正文中去除，
底层采样证据保持原样。模型路径应为已有本地目录或公开模型来源，
内部 `OPENLM_HUB` 下载会被拒绝。

GPU 采样成功不代表 SWE-bench 修复通过或获得训练奖励。结果由 cookbook
的任务环境与 verifier 判定；完整 ModelService 流程及清理检查参见 quick-start。
