# 快速上手：使用阿里云函数计算 DevPod 进行快速开发

> DevPod 是阿里云函数计算提供的一种快速开发环境，可以帮助您快速部署和运行 ROLL 项目。

## 先决条件

在开始之前，请确保您已完成以下准备工作：

- 拥有一个阿里云账号
- 登录 [函数计算 FunModel 控制台](https://fcnext.console.aliyun.com/fun-model)
- 根据控制台指引，完成 RAM 相关的角色授权等配置

## 创建训练 DevPod 环境

1.  点击 **创建模型 - 自定义开发**。
2.  选择 **自定义环境**，并按如下配置：
    *   **容器镜像**：选择不可公开访问的自定义镜像 - 容器镜像地址 - `roll-registry.cn-hangzhou.cr.aliyuncs.com/roll/pytorch:nvcr-24.05-py3-torch260-vllm084`
    *   **模型名称**：输入一个名称，例如 `roll-dev`
    *   **模型来源**：选中 `无模型`
    *   **启动命令**：保持默认，无需修改
    *   **实例规格**：选择 `GPU性能型`
    *   点击 **DevPod开发调试** 按钮（**注意**：不要点击"创建模型服务"）
3.  等待部署成功（通常 1–2 分钟）。

## 配置和测试

### 下载 ROLL 项目并安装依赖

```bash
# /mnt/模型名称 为默认的 NAS 挂载点，请将 roll-dev 修改为实际的模型名称
cd /mnt/roll-dev

# 1. 克隆项目代码
git clone https://github.com/alibaba/ROLL.git

# 2. 安装项目依赖
cd ROLL
pip install -r requirements_torch260_vllm.txt -i https://mirrors.aliyun.com/pypi/simple/
```

### 运行 pipeline 示例

```bash
bash examples/agentic_demo/run_agentic_pipeline_frozen_lake_single_node_demo.sh
```

更多 DevPod 使用指南，见 [https://fun-model-docs.devsapp.net/user-guide/devpod/](https://fun-model-docs.devsapp.net/user-guide/devpod/)
