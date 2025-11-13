# ROLL 调试指南

在开发和使用 ROLL 框架时，调试是必不可少的环节。本文档将介绍几种有效的调试方法，帮助您快速定位和解决问题。

## 1. 使用 Ray Debugger

ROLL 基于 Ray 构建，因此可以使用 Ray 提供的调试工具。Ray Debugger 是一个强大的工具，可以帮助您调试分布式应用。

### 启用 Ray Debugger

在 `roll/utils/ray_utils.py` 文件中，可以通过设置环境变量来启用 Ray Debugger：

```python
# 用于调试
env_vars["RAY_DEBUG"] = "legacy"
```

您可以在启动训练脚本前设置此环境变量：

```bash
export RAY_DEBUG=legacy
```

### 使用 Ray Debugger

启用 Ray Debugger 后，您可以使用标准的 Python 调试器（如 pdb）进行单步调试。当程序运行到断点时，调试器会暂停执行，允许您检查变量、调用栈等信息。

有关 Ray Debugger 的详细使用方法，请参考官方文档：
[Ray Debugging Documentation](https://docs.ray.io/en/latest/ray-observability/user-guides/debug-apps/ray-debugging.html)

## 2. pipeline里的代码debug
由于Pipeline直接运行在ray的driver中，因此可以直接使用pdb进行调试，使用方法是是需要加断点的地方使用：
```python
import pdb; pdb.set_trace()
```
不要在pipeline中使用breakpoint()，在ray debug模式中，driver中的breakpoint()不会进入pdb。


## 3. 本地调试 Agentic 多轮交互过程

ROLL 框架的一个特色是支持 Agentic 多轮交互过程的调试。这对于开发和优化 Agentic 应用非常有帮助。

### 使用测试脚本

在 `tests/agentic/env_manager/test_traj_env_manager.py` 文件中，提供了本地调试 Agentic 多轮交互过程的测试demo脚本，可以自行扩展新的测试。

该脚本包含以下功能：
1. `test_debug_traj_env_manager()` - 调试轨迹环境管理器
2. `test_debug_vl_traj_env_manager()` - 调试视觉-语言轨迹环境管理器
3. `test_debug_step_env_manager()` - 调试步骤环境管理器

### 运行调试脚本

要运行调试脚本，请按照以下步骤操作：

1. 创建并激活 Python 环境：
```bash
conda create -n python310_torch260_em python=3.10
conda activate python310_torch260_em
```

2. 安装依赖：
```bash
pip3 install torch torchvision torchaudio py-cpuinfo
pip install -r requirements_em_local_debug.txt
```

3. 运行测试脚本：
```bash
python tests/agentic/env_manager/test_traj_env_manager.py
```

通过这种方式，您可以本地调试 Agentic 多轮交互过程，大幅提升 Agentic 应用的实际开发效率。

## 3. 其他调试技巧

### 日志调试[TODO]

ROLL 框架内置了详细的日志系统。您可以通过调整日志级别来获取更多调试信息：

```yaml
# 在配置文件中设置日志级别
system_envs:
  ROLL_LOG_LEVEL: "DEBUG"
```

### 性能分析

要获取训练的 timeline，可以在 YAML 配置中开启 profile：

```yaml
system_envs:
  RAY_PROFILING: "1"
profiler_output_dir: /data/oss_bucket_0/llm/profile/${exp_name}
```

然后使用 [Perfetto UI](https://ui.perfetto.dev/) 工具进行分析。

通过合理使用这些调试方法，您可以更高效地开发和优化基于 ROLL 框架的应用。