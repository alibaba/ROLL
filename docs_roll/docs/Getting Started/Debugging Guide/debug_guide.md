# ROLL Debugging Guide

When developing and using the ROLL framework, debugging is an essential step. This document will introduce several effective debugging methods to help you quickly locate and resolve issues.

## 1. Using Ray Debugger

ROLL is built on Ray, so you can use the debugging tools provided by Ray. Ray Debugger is a powerful tool that can help you debug distributed applications.

### Enabling Ray Debugger

In the `roll/utils/ray_utils.py` file, you can enable Ray Debugger by setting environment variables:

```python
# For debugging
env_vars["RAY_DEBUG"] = "legacy"
```

You can set this environment variable before starting the training script:

```bash
export RAY_DEBUG=legacy
```

### Using Ray Debugger

After enabling Ray Debugger, you can use standard Python debuggers (such as pdb) for step-by-step debugging. When the program reaches a breakpoint, the debugger will pause execution, allowing you to inspect variables, call stacks, and other information.

For detailed usage of Ray Debugger, please refer to the official documentation:
[Ray Debugging Documentation](https://docs.ray.io/en/latest/ray-observability/user-guides/debug-apps/ray-debugging.html)

## 2. Debugging Code in Pipeline

Since the Pipeline runs directly in Ray's driver, you can directly use pdb for debugging. To add breakpoints, use:

```python
import pdb; pdb.set_trace()
```

Do not use breakpoint() in the pipeline. In Ray debug mode, breakpoint() in the driver will not enter pdb.

## 3. Local Debugging of Agentic Multi-Round Interaction Process

One feature of the ROLL framework is support for debugging the Agentic multi-round interaction process. This is very helpful for developing and optimizing Agentic applications.

### Using Test Scripts

In the `tests/agentic/env_manager/test_traj_env_manager.py` file, test demo scripts for locally debugging the Agentic multi-round interaction process are provided, which you can extend with new tests.

The script includes the following functions:
1. `test_debug_traj_env_manager()` - Debug trajectory environment manager
2. `test_debug_vl_traj_env_manager()` - Debug vision-language trajectory environment manager
3. `test_debug_step_env_manager()` - Debug step environment manager

### Running Debug Scripts

To run the debug script, follow these steps:

1. Create and activate a Python environment:
```bash
conda create -n python310_torch260_em python=3.10
conda activate python310_torch260_em
```

2. Install dependencies:
```bash
pip3 install torch torchvision torchaudio py-cpuinfo
pip install -r requirements_em_local_debug.txt
```

3. Run the test script:
```bash
python tests/agentic/env_manager/test_traj_env_manager.py
```

Through this approach, you can locally debug the Agentic multi-round interaction process, significantly improving the actual development efficiency of Agentic applications.

## 4. Other Debugging Techniques

### Log Debugging [TODO]

The ROLL framework has a built-in detailed logging system. You can obtain more debugging information by adjusting the log level:

```yaml
# Set log level in configuration file
system_envs:
  ROLL_LOG_LEVEL: "DEBUG"
```

### Performance Analysis

To get the training timeline, you can enable profiling in the YAML configuration:

```yaml
system_envs:
  RAY_PROFILING: "1"
profiler_output_dir: /data/oss_bucket_0/llm/profile/${exp_name}
```

Then use the [Perfetto UI](https://ui.perfetto.dev/) tool for analysis.

By properly using these debugging methods, you can more efficiently develop and optimize applications based on the ROLL framework.