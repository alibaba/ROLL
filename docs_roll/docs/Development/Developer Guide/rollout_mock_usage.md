---
sidebar_position: 4
---

# Rollout Dump Mock Usage Guide

## Overview

Rollout Dump Mock is a powerful debugging tool in the ROLL framework designed to **eliminate randomness in the rollout phase of RL training**, enabling numerical precision alignment verification. By saving and replaying rollout data, it helps developers quickly validate the correctness of computational optimizations.

### Core Value

- **Eliminate Randomness**: Enable numerical precision alignment verification
- **Fast Iteration**: Mock mode skips expensive environment rollout
- **Reproducible Debugging**: Capture problematic rollout data for repeated debugging
- **Transparent Architecture**: Implemented at the Scheduler layer, completely transparent to the Pipeline

### Use Cases

| Scenario | Description |
|----------|-------------|
| **Computation Optimization Verification** | Verify numerical consistency of optimizations like dynamic_batching, sequence_packing |
| **Model Parallelism Verification** | Verify precision alignment of TP, PP, EP and other parallel strategies |
| **Regression Testing** | Automated precision testing in CI/CD pipelines |

---

## Quick Start

### Typical Workflow

```
[1. Dump Mode] → [2. Modify Code] → [3. Mock Mode] → [4. Precision Verification]
    ↓                ↓                   ↓                   ↓
 Capture baseline  Optimize compute   Deterministic      Numerical
     data             logic             replay           comparison
```

### Step 1: Dump Mode - Capture Baseline Data

Before modifying code, capture correct rollout data as a baseline.

**Configuration File** (`agentic_sokoban_rollout_mock_dump.yaml`):
```yaml
exp_name: "sokoban_precision_test_dump"
max_steps: 50

# Rollout Mock Configuration - DUMP MODE
rollout_mock:
  enable: true
  mode: dump
  dump_dir: ./output/rollout_dumps/baseline_v1

# Environment variables for deterministic execution
system_envs:
  NCCL_ALGO: Ring
  NVTE_ALLOW_NONDETERMINISTIC_ALGO: '0'
  CUBLAS_WORKSPACE_CONFIG: ':4096:8'
  DETERMINISTIC_MODE: '1'

# ... other configurations ...
```

**Command**:
```bash
python examples/start_agentic_pipeline.py \
  --config_name agentic_sokoban_rollout_mock_dump \
  --config_path examples/qwen2.5-0.5B-agentic
```

**Output**:
```
./output/rollout_dumps/baseline_v1/
  └── train/
      ├── step_000000.pkl  (~5MB)
      ├── step_000001.pkl
      ├── step_000002.pkl
      ├── ...
      └── step_000049.pkl
```

**Log Example**:
```
[Rollout Mock] Rollout Mock enabled: mode=dump, dir=./output/rollout_dumps/baseline_v1
[Rollout Mock] Dumped step 0: ./output/rollout_dumps/baseline_v1/train/step_000000.pkl (samples=128, size=4.82MB)
[Rollout Mock] Dumped step 1: ./output/rollout_dumps/baseline_v1/train/step_000001.pkl (samples=128, size=4.85MB)
```

### Step 2: Modify Code

Implement your computational optimizations, such as:
- Adding dynamic_batching
- Implementing sequence_packing
- Migrating to new parallel strategies

### Step 3: Mock Mode - Deterministic Replay

Use pre-recorded rollout data to verify that modified code maintains numerical consistency.

**Configuration File** (`agentic_sokoban_rollout_mock_mock.yaml`):
```yaml
exp_name: "sokoban_precision_test_mock"
max_steps: 50

# Rollout Mock Configuration - MOCK MODE
rollout_mock:
  enable: true
  mode: mock
  dump_dir: ./output/rollout_dumps/baseline_v1  # Same path as dump mode

# Environment variables for deterministic execution (keep consistent with dump mode)
system_envs:
  NCCL_ALGO: Ring
  NVTE_ALLOW_NONDETERMINISTIC_ALGO: '0'
  CUBLAS_WORKSPACE_CONFIG: ':4096:8'
  DETERMINISTIC_MODE: '1'

# ... other configurations (keep consistent with dump mode) ...
```

**Command**:
```bash
python examples/start_agentic_pipeline.py \
  --config_name agentic_sokoban_rollout_mock_mock \
  --config_path examples/qwen2.5-0.5B-agentic
```

**Behavior**:
- ✅ Directly loads DataProto from disk for each step
- ✅ All subsequent computations (advantages, losses, gradients) are fully deterministic

**Log Example**:
```
[Rollout Mock] Rollout Mock enabled: mode=mock, dir=./output/rollout_dumps/baseline_v1
[Rollout Mock] Loaded step 0: ./output/rollout_dumps/baseline_v1/train/step_000000.pkl (samples=128)
[Rollout Mock] Loaded step 1: ./output/rollout_dumps/baseline_v1/train/step_000001.pkl (samples=128)
```


### Step 4: Numerical Precision Verification

Compare training metrics between baseline and optimized versions to ensure complete numerical consistency. You can verify that both runs produce identical results by examining key metrics (such as pg_loss, total_loss, value_loss, approx_kl, grad_norm, etc.) in the logs.
---

## Configuration Parameters

### Configuration Schema

Add the `rollout_mock` section to your YAML configuration file:

```yaml
rollout_mock:
  enable: bool              # Enable rollout dump/mock mechanism
  mode: "dump" | "mock"     # dump: save data, mock: load data
  dump_dir: str             # Data storage directory
```

### Configuration Examples

**Dump Mode Configuration**:
```yaml
rollout_mock:
  enable: true
  mode: dump
  dump_dir: ./rollout_dumps/precision_test_v1
```

**Mock Mode Configuration**:
```yaml
rollout_mock:
  enable: true
  mode: mock
  dump_dir: ./rollout_dumps/precision_test_v1  # Same path as dump mode
```

### Environment Variables for Deterministic Execution

To ensure complete numerical reproducibility, the following environment variables should be configured:

```yaml
system_envs:
  NCCL_ALGO: Ring                           # Use Ring algorithm for NCCL
  NVTE_ALLOW_NONDETERMINISTIC_ALGO: '0'     # Disable non-deterministic algorithms in Transformer Engine
  CUBLAS_WORKSPACE_CONFIG: ':4096:8'        # Enable deterministic CUDA operations
  DETERMINISTIC_MODE: '1'                   # Enable PyTorch deterministic mode
```

**DETERMINISTIC_MODE Effects**:
- Sets `torch.backends.cudnn.deterministic = True` for reproducible cuDNN operations
- Sets `torch.backends.cudnn.benchmark = False` to disable auto-tuning that causes non-determinism
- Calls `torch.use_deterministic_algorithms(True)` to enforce deterministic PyTorch algorithms

**Important**: These environment variables must be kept consistent between dump and mock modes to ensure numerical precision alignment.

### Key Considerations

1. **dump_dir must match**: Dump and Mock modes must use the same `dump_dir` path
2. **mode must match**: Scheduler mode (train/val) must match the dump mode
3. **max_steps cannot exceed**: Mock mode `max_steps` cannot exceed the value used in Dump mode
4. **system_envs must be consistent**: Environment variables for deterministic execution should be identical between dump and mock modes

---

## Common Issues and Troubleshooting

### Issue 1: Mock File Not Found

**Error Message**:
```
FileNotFoundError: [Rollout Mock] Mock file not found: ./dumps/baseline/train/step_000005.pkl
Possible reasons:
  1. Step 5 was not run in dump mode
  2. dump_dir configuration is incorrect: ./dumps/baseline
  3. mode mismatch (current: train)
Please run in dump mode first to ensure all step data is generated.
```

**Troubleshooting Steps**:

1. Check if enough steps were run in dump mode:
   ```bash
   ls -lh ./output/rollout_dumps/baseline_v1/train/
   # Should see step_000000.pkl ~ step_000049.pkl
   ```

2. Confirm `max_steps` consistency:
   ```bash
   # Dump: max_steps=50
   # Mock: max_steps=50 (must match or be smaller)
   ```

3. Verify `dump_dir` path is correct:
   ```yaml
   # Dump mode
   dump_dir: ./output/rollout_dumps/baseline_v1

   # Mock mode (must be same)
   dump_dir: ./output/rollout_dumps/baseline_v1
   ```

### Issue 2: Mode Mismatch

**Problem**: Used train mode during dump, but accidentally used val mode during mock.

**File Structure**:
```
dumps/baseline/
  ├── train/       # Generated during dump
  │   └── step_*.pkl
  └── val/         # Empty directory
      └── (no files)
```

**Solution**: Ensure dump and mock use the same scheduler mode (train/val).

### Issue 3: Insufficient Disk Space

**Symptom**: Error during dump:
```
OSError: [Errno 28] No space left on device
```

**Disk Usage Estimation**:
```
Single step file size ≈ batch_size × seq_len × data type size
                      ≈ 128 × 512 × 4 bytes (float32)
                      ≈ 256KB ~ 10MB (depending on sequence length and metadata)

Total disk usage ≈ single step size × max_steps
                ≈ 5MB × 100 steps = 500MB
```

**Solutions**:
- Increase disk space
- Reduce `max_steps`
- Use network storage (OSS, etc.)

### Issue 4: Pickle Version Incompatibility

**Symptom**: Error when loading across different Python versions:
```
pickle.UnpicklingError: invalid load key, '\x00'
```

**Cause**: Pickle compatibility issues between different Python versions.

**Solutions**:
- Ensure dump and mock use the same Python version
- Or use a lower protocol version during dump (requires source code modification)
