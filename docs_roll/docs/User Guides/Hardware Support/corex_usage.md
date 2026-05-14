# ROLL x CoreX

Last updated: 03/20/2026.

This document records the CoreX-specific adaptations currently integrated in the local ROLL workspace for Iluvatar devices such as `Iluvatar BI-V150`.

## Current Scope

The current adaptation targets a CUDA-like CoreX software stack where:

- `torch.cuda` is available
- Ray exposes the accelerator as `GPU`
- the NVML-compatible monitoring interface is provided by `libixml.so`
- vendor-patched `torch`, `megatron-core`, and `vllm` may diverge from upstream behavior

This is a practical compatibility layer for running ROLL on the current machine. It is not yet a full official upstream hardware support package.

## What Was Adapted

### 1. Platform Detection

ROLL previously treated `Iluvatar BI-V150` as an unknown CUDA device. The platform initialization logic now detects CoreX-style device names and creates a dedicated `CorexPlatform` instead of falling back to `UnknownPlatform`.

Current detection keywords include:

- `ILUVATAR`
- `COREX`
- `BI-V`

Implementation:

- `roll/platforms/corex.py`
- `roll/platforms/__init__.py`

### 2. Safe CUDA Platform Initialization

On this vendor stack, subprocesses can hit a state where:

- `torch.cuda.is_available()` is effectively usable
- but `device_count() == 0` in the current visibility scope

Directly calling `torch.cuda.get_device_name()` in that state can raise `AssertionError: Invalid device id`.

The platform bootstrap now checks `device_count()` first and only queries the CUDA device name when at least one visible device exists.

Implementation:

- `roll/platforms/__init__.py`

### 3. Ray GPU Resource Registration

Ray did not automatically register CoreX GPUs as `GPU` resources on this machine, even though `torch` could see the devices. That caused the scheduler to believe the cluster had zero usable GPU nodes.

ROLL now starts Ray with explicit accelerator resources so the cluster exposes the expected `GPU` count.

Implementation:

- `roll/distributed/scheduler/initialize.py`
- `roll/distributed/scheduler/resource_manager.py`

### 4. NVML-Compatible Memory Monitoring Through `libixml.so`

The vendor stack does not provide `libnvidia-ml.so.1`, so upstream `torch.cuda.device_memory_used()` fails when it tries to initialize NVML. However, CoreX exposes an NVML-compatible API through `libixml.so`.

ROLL now:

1. Tries the upstream `torch.cuda.device_memory_used()`
2. If that fails, tries to load an NVML-compatible library
3. Falls back in this order:
   - standard NVML if present
   - `libixml.so`
4. Calls:
   - `nvmlInit_v2` / `nvmlInit`
   - `nvmlDeviceGetHandleByIndex_v2` / `nvmlDeviceGetHandleByIndex`
   - `nvmlDeviceGetMemoryInfo`
5. Maps logical device index through `CUDA_VISIBLE_DEVICES` before querying the physical handle

An override is also supported:

```bash
export ROLL_NVML_COMPAT_LIB=/path/to/libixml.so
```

Implementation:

- `roll/platforms/platform.py`

### 5. Megatron Optimizer Compatibility

The installed vendor-patched `megatron-core` on this machine uses a newer `_get_param_groups_and_buffers()` signature than the previous ROLL compatibility layer expected.

ROLL now inspects the runtime function signature and passes the new required arguments when needed, while staying compatible with older variants.

Implementation:

- `roll/third_party/megatron/optimizer.py`

### 6. vLLM Sleep/Offload Compatibility

The current CoreX stack does not expose the allocator backend required by vLLM sleep mode (`cumem`). When ROLL forced sleep mode on this machine, vLLM crashed during initialization.

ROLL now:

- disables vLLM sleep/offload automatically when the allocator backend is unavailable
- warns explicitly when this means `actor_infer` will stay resident on GPU
- warns again if the user is still using a colocated train/infer layout

Implementation:

- `roll/third_party/vllm/__init__.py`
- `roll/distributed/strategy/vllm_strategy.py`

### 7. Single-Node Demo Layout Adjustment

On this CoreX stack, colocating `actor_train` and `actor_infer` on the same GPU was not stable for the default frozen-lake agentic demo after vLLM sleep mode became unavailable. The main failure mode was OOM during the Megatron optimizer step.

The single-node frozen-lake demo was adjusted to a 2-GPU disaggregated layout:

- `actor_train` on GPU 0
- `actor_infer` on GPU 1
- lower vLLM `gpu_memory_utilization`

Implementation:

- `examples/agentic_demo/agent_val_frozen_lake_single_node_demo.yaml`

## Validation Performed

The following checks were performed on the current machine:

- `torch.cuda.get_device_name(0)` returned `Iluvatar BI-V150`
- `ldconfig -p` exposed `libixml.so`
- direct `ctypes` calls to `libixml.so` succeeded for:
  - `nvmlInit_v2`
  - `nvmlDeviceGetHandleByIndex_v2`
  - `nvmlDeviceGetMemoryInfo`
- `current_platform.device_memory_used()` successfully reported memory through the NVML-compatible path
- the frozen-lake single-node pipeline ran past step 0 and step 1 after the disaggregated layout change

## Tests Added or Updated

- `tests/platforms/test_platform_init.py`
- `tests/platforms/test_platform_memory.py`
- `tests/distributed/scheduler/test_initialize.py`
- `tests/distributed/scheduler/test_resource_manager.py`
- `tests/third_party/megatron/test_optimizer_compat.py`

## Known Limitations

- CoreX is currently integrated as a CUDA-like platform, not as a fully separate backend with vendor-specific kernels or scheduling behavior.
- vLLM sleep mode is still disabled on the current stack because the required allocator backend is unavailable.
- The current adaptation favors reliable execution over preserving the original single-GPU colocated demo topology.
- Existing long-running processes must be restarted to pick up the latest platform and monitoring changes.

## Recommended Run Command

```bash
conda activate ROLL
ray stop
bash examples/agentic_demo/run_agentic_pipeline_frozen_lake_single_node_demo.sh
```
