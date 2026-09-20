# Qwen3.8-Flash-Next validation dependencies

The Flash-Next implementation is under validation. Reproducing its dependency
source does not establish numerical parity, full SFT/RL acceptance, or production
support. The initial training scope is LoRA and full text-backbone training with
the N-gram tables frozen. Trainable N-gram tables remain a separate capability.

## Observed validation environment

The September 17, 2026 validation uses eight H800 GPUs, approximately 2 TiB of
host RAM, CPU FP32 Adam, and the following installed distributions:

| Distribution | Version |
| --- | --- |
| torch | `2.13.0+cu130` |
| megatron-core | `0.16.0rc0` plus the patches below |
| transformer-engine | `2.18.0` |
| transformers | `5.15.1` |
| ray | `2.48.0` |
| vllm | `0.1.dev20073+g8e685d198` |
| flash-linear-attention | `0.5.2` |
| flashinfer-python | `0.6.17` |
| peft | `0.12.0` plus the adapter-key patch below |

This is an observed environment, not a standalone package lock. In particular,
the vLLM version string alone does not identify all wheel/build inputs. Replacing
it with the latest PyPI release has not been validated. The repository's
`Dockerfile.torch2100` uses different PyTorch and floating dependency versions;
building it unchanged does not reproduce this environment.

## Reproduce the Megatron source

Use a fresh dependency directory in an existing compatible CUDA/PyTorch
environment. Run these commands from the ROLL repository root:

```bash
ROLL_QWEN38_ROOT="$PWD"
ROLL_QWEN38_MEGATRON="$PWD/../Megatron-LM-qwen38"
git clone --branch core_dev_r0.16.0 --single-branch \
  https://github.com/NVIDIA/Megatron-LM.git "$ROLL_QWEN38_MEGATRON"
git -C "$ROLL_QWEN38_MEGATRON" checkout --detach \
  bfa1d3163804eb8ea65b77d1c0e807a3fcb959e9

for patch in \
  block_factory \
  gdn_decay \
  gdn_lora_checkpoint \
  gdn_gate_precision \
  gdn_convolution \
  gradient_clipping \
  moe_cache \
  ddp_memory \
  cpu_grad_staging \
  hybrid_checkpoint \
  checkpoint_padding \
  checkpoint_initialization
do
  python "$ROLL_QWEN38_ROOT/scripts/qwen38/patch_megatron_${patch}.py" \
    "$ROLL_QWEN38_MEGATRON" || exit 1
done
```

Order matters: GDN checkpoint support builds on the decay patch, followed by
the gate-precision and convolution hooks. Optimizer checkpoint padding builds
on CPU gradient staging and hybrid checkpoint support. Start from the
pinned clean source for each reproduction. Replaying the entire sequence on a
fully patched tree is not supported: an early patch can reject the final hash
produced by a later patch to the same file. An unknown-source error requires
inspecting the dependency, not bypassing its guard.

Verify the resulting files before using them:

```bash
python - "$ROLL_QWEN38_ROOT" "$ROLL_QWEN38_MEGATRON" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root, dependency = map(Path, sys.argv[1:])
manifest = json.loads(
    (root / "scripts/qwen38/megatron_016_patch_manifest.json").read_text()
)
for relative, checksums in manifest["files"].items():
    actual = hashlib.sha256((dependency / relative).read_bytes()).hexdigest()
    if actual != checksums["after_sha256"]:
        raise SystemExit(f"Patched source mismatch: {relative}")
print(f"Verified {len(manifest['files'])} patched Megatron files")
PY
```

On September 19, all 12 patches were replayed from the pinned upstream source
on both the local machine and the validation host. The seven modified files
matched this manifest, and all 393 `megatron/core/**/*.py` files in the rebuilt
dependency matched the frozen dependency used by the active RL validation.
This verifies the current source recipe, including the later GDN
gate/convolution and deterministic checkpoint-padding fixes. It does not verify
compiled extensions or a fresh environment installation.

To install in an environment where the required dependencies are already
present, preserve the patched source and avoid resolving newer dependencies:

```bash
python -m pip install --no-build-isolation --no-deps -e "$ROLL_QWEN38_MEGATRON"
python -m pip install --no-deps -e "$ROLL_QWEN38_ROOT/mcore_adapter"
export PYTHONPATH="$ROLL_QWEN38_ROOT/mcore_adapter/src:$ROLL_QWEN38_ROOT:$ROLL_QWEN38_MEGATRON${PYTHONPATH:+:$PYTHONPATH}"
python - <<'PY'
import importlib.metadata
import megatron.core
print(importlib.metadata.version("megatron-core"))
print(megatron.core.__file__)
PY
```

These installation commands have not been exercised as a fresh container build.
The editable/source installation form follows the [Megatron installation
documentation](https://github.com/NVIDIA/Megatron-LM/blob/main/docs/get-started/install.md).
Do not select `mcore_adapter[dev]` for this reproduction: that extra requests
`megatron-core==0.18.0.dev0`. The adapter's base dependency range is
`>=0.15.0,<0.19.0`; that broad package range is not a Flash-Next support matrix.

## PEFT adapter keys

The validation dependency also uses
`patch_peft_adapter_state_keys.py PATH_TO_PEFT/peft/utils/save_and_load.py`.
This patch inserts the adapter name only before the final tensor suffix,
preserving GR module names such as `input_mix_weight_down`. It accepts the known
source hash (or its own already-patched form) and rejects unknown source.
Apply it to an isolated dependency copy before starting training processes.

## Validation resource policy

The current RL capacity probe uses TP4/EP8/ETP1, CPU Adam, frozen N-gram tables,
and sequential role loading/offloading. Low instantaneous GPU utilization can
occur during CPU work and weight transfers; follow phase and optimizer-update
receipts to distinguish progress from a stalled worker.

On this approximately 2 TiB host, Ray 2.48's default 95% used-memory threshold
killed a rollout worker with approximately 99 GiB still available. The next
probe sets `RAY_memory_usage_threshold=0.97` **before starting its Ray head** and
retains an independent supervisor that stops only that task below 64 GiB
`MemAvailable`. This is a host-specific capacity experiment, not a portable
default. A shared GPU lock serializes validation jobs, and cleanup selects only
processes carrying the run's exact output-directory identity.

Synchronous distributed model and optimizer saves use MCA's streaming DCP
writer. Megatron's default synchronous entry point preloads the whole
checkpoint into host memory; the installed PyTorch serial writer also retains
written tensors. The streaming path writes and releases each tensor in turn,
with staging bounded by the largest individual write item. It preserves native
DCP metadata, serialization, and collective failure handling. CPU and bounded
CUDA round trips have passed on the validation environment. A real full-backbone
RL checkpoint containing model and Adam state has also been saved and all
285,140 storage items passed type, dtype, shape and finite-value checks. Its
cold restore reproduced the saved actor/reference sentinel outputs and resumed
updates. The complete continuation run and final checkpoint remain under
validation; the sentinel check is not an all-state tensor comparison.

A full text-backbone checkpoint with FP32 Adam and master parameters occupies
approximately 1.79 TB in this environment. Use fast local storage for the first
save and cold restore, and keep its staging and final checkpoint directories on
the same filesystem so the verified hard-link upload path can avoid a second
copy. Budget the measured allocated checkpoint size plus at least 128 GiB of
free space on each destination filesystem. A single mechanical disk sustaining
about 100 MiB/s needs several hours for this checkpoint; GPU utilization can
remain zero while its files grow. Check disk throughput, file growth, and host
memory before diagnosing that phase as a training hang.

Completion still requires the real-model update, save/restore, natural
cross-framework parity, export, and performance acceptance results. CPU tests,
dependency hashes, or a successful partial optimizer step do not satisfy those
gates.
