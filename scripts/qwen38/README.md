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

### Isolated FLA backward regression

The September 22 H800 / Triton 3.7.1 diagnostic reproduced incorrect local
value gradients at 64-token chunk boundaries with a 32-wide value head. The
bounded candidate uses a minimum value tile of 64 on Hopper, only in
`chunk_bwd_dv_local`; the model's 128-wide value heads are unaffected.
`patch_fla_hopper_dv.py` accepts only the captured source's before/after hashes
and rejects other revisions. Do not apply it to running validation snapshots.

The regression includes an independent FP64 recurrence, nonzero initial-state
gradients, packed sequences, both state layouts and the model's 1:3 grouped
heads. To verify an isolated source copy:

```bash
python scripts/qwen38/patch_fla_hopper_dv.py /path/to/isolated-fla
PYTHONPATH=/path/to/isolated-fla \
QWEN38_FLA_SOURCE=/path/to/isolated-fla/fla/ops/common/chunk_o.py \
RUN_QWEN38_FLA_TESTS=1 \
python -m pytest -q mcore_adapter/tests/test_qwen4_exp_fla_backward.py
```

CPU-only checks do not exercise the seven CUDA cases. This workaround remains
separate from full-model inference/training probability parity and is not
automatically enabled by the dependency recipe below.

On September 23, the isolated H800 regression reproduced the unpatched failure
and passed all 10 tests after the patch, including 36 gradient comparisons.
The installed FLA dependency was unchanged, and no full model was loaded by
this regression.

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

### Export a native LoRA checkpoint

Use the checkpoint directory containing the named adapter directories, and a
fresh export destination. The public converter writes Qwen4 per-expert 2D
adapters with `roll_lora_layout=qwen4_exp_vllm_2d` for the native vLLM file loader.
Generic Transformers/PEFT reload of this layout remains unsupported; inference
also requires the original base model and its frozen N-gram assets.

```bash
export MODEL_PATH=/path/to/Qwen3.8-Flash-Next
export QWEN38_ADAPTER_CHECKPOINT=/path/to/checkpoint-19
export QWEN38_ADAPTER_EXPORT=/path/to/new-adapter-export
CUDA_VISIBLE_DEVICES= python - <<'PY'
import os
import torch
from mcore_adapter.models.converter.post_converter import LoRAHFConverter

LoRAHFConverter(
    hf_base_model_path=os.environ["MODEL_PATH"],
    adapter_name_or_path=os.environ["QWEN38_ADAPTER_CHECKPOINT"],
    save_directory=os.environ["QWEN38_ADAPTER_EXPORT"],
    torch_dtype=torch.bfloat16,
).convert()
PY
```

On September 25, the real checkpoint from two 8192-token SFT updates with the
current gated-normalization fix exported 148,808 finite BF16 tensors in about
29 seconds, using approximately 8 GiB peak host memory and no CUDA context.
The native CPU file loader and packing path then reproduced every selected
adapter value and its scaling exactly on all eight expert partitions, including
9,892 projections and 276 packed modules per partition. That CPU diagnostic
disabled pinned-memory allocation in its own process. Exported-adapter GPU
forward, natural probability parity, and complete training acceptance remain
separate checks.

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

Ray also writes its session logs and object-store spill files to its temporary
filesystem. Set `ROLL_RAY_TEMP_DIR` to a dedicated local filesystem before
launching a Qwen3.8 run to make the head and worker use the same location; ROLL
passes it to `ray start --temp-dir` and checks free space before starting the
cluster. Keep the configured directory path short enough for Ray's Unix socket
limit (for example `/tmp/roll-qwen38-ray`); ROLL rejects an overlong path before
starting a cluster. The default minimum is 128 GiB and can be changed with
`ROLL_RAY_TEMP_MIN_FREE_BYTES`. An unset variable preserves Ray's normal
`/tmp/ray` behavior. Keep this directory separate from checkpoint output and
remove only completed run sessions after the run has exited.

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

The deterministic optimizer-padding patch also has real checkpoint evidence:
a September 25 CPU check of the saved full-backbone OPD continuation found all
864 known 52-element alignment gaps exactly zero. It read approximately 1.5 MB
of serialized padding, with no CUDA context. The earlier SFT continuation's
original comparison still records 864 padding differences; its failed raw-state
comparison has not been rewritten as a pass. Zero padding in a later save does
not establish equality of model or Adam state between training runs.

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

## GDN convolution arithmetic

The native BF16 convolution cache path rounds each input/weight product to
BF16, accumulates those products in FP32, applies SiLU in FP32, and casts the
result to BF16. Rounding the completed convolution before SiLU agrees only at
the first causal token. The model-specific differentiable convolution preserves
the product casts and uses `torch.compile` with `emulate_precision_casts=True`
so kernel fusion does not remove them. Parameter shapes and checkpoint keys
are unchanged. This path handles independent padded batches, without packing
or recurrent state.

Run the CUDA regression in the pinned dependency environment:

```bash
RUN_QWEN4_GDN_DECAY_TESTS=1 python -m pytest -q \
  mcore_adapter/tests/test_qwen4_exp_gdn_convolution.py \
  mcore_adapter/tests/test_qwen4_exp_causal_convolution.py
```

On September 24, all five tests passed on the H800/PyTorch 2.13 validation
environment, including gradients, strided batches, bias and an actual
8192-token convolution. Two captured native input prefixes also matched
exactly after the fix. This does not establish whole-model numerical parity.

For a `[1, 8192, 1280]` BF16 convolution in a fresh process, measured median
forward / forward-plus-backward latency was 0.112 / 0.419 ms versus
0.150 / 0.452 ms for the previous FLA path, excluding compilation. The compiled
training path uses more temporary memory. Unusually many stride/gradient
variants can exhaust the compiler cache; normal eager fallback preserves
correctness but costs more time and memory. Full-model throughput and memory
acceptance remain pending.

## GDN projection precision and distributed gradients

The training implementation computes native `qkvz` and `ba` projections
separately while retaining the fused checkpoint parameter. It preserves TP
input-gradient reduction, sequence-parallel gathering and LoRA dtype promotion.
LoRA's column projection already reduces its input gradient; its surrounding
sequence gather therefore scatters that gradient without reducing it again.

The GDN output projection keeps BF16 operands for the Tensor Core GEMM, returns
FP32 partial results, performs the TP reduction in FP32, and then casts once to
BF16. This avoids amplifying rank-local output rounding in the later PLE gate.
It retains parameter objects, adapter wrappers and checkpoint names. PyTorch
2.13 supports `mm(out_dtype=torch.float32)` on CUDA but lacks its autograd
implementation; the scoped helper supplies a first-order backward. It requires
the immediate cast after the TP sum and is not a general FP32-output linear.

Run each distributed test file in a separate process group:

```bash
NVIDIA_TF32_OVERRIDE=0 RUN_QWEN4_NATIVE_PROJECTION_TESTS=1 \
torchrun --nproc_per_node=2 -m pytest -q \
  mcore_adapter/tests/test_qwen4_exp_native_projection.py
NVIDIA_TF32_OVERRIDE=0 RUN_QWEN4_OUTPUT_PROJECTION_TESTS=1 \
torchrun --nproc_per_node=2 -m pytest -q \
  mcore_adapter/tests/test_qwen4_exp_output_projection.py
NVIDIA_TF32_OVERRIDE=0 RUN_QWEN4_DISTRIBUTED_TESTS=1 \
torchrun --nproc_per_node=2 -m pytest -q \
  mcore_adapter/tests/test_qwen4_exp_distributed.py
```

On September 25, actual TE/LoRA/TP/SP/DDP projection tests passed over two
updates. The complete small-model TP1 versus TP2/SP gradient comparison improved
from relative L2 `0.0216556` to `0.00124173`, passing the unchanged `0.02` bound.
These tests do not establish real-model inference/training probability parity.

Qwen3.8's sigmoid-gated GDN RMSNorm uses the native `rsqrt` operation in
its fused forward. FLA's `1 / sqrt` can round differently at a BF16 midpoint:
a real layer-0 replay isolated one changed normalized element, which produced
74 changed elements after the output projection. The scoped forward saves its
reciprocal RMS for FLA's existing fused backward; other FLA models and the
non-sigmoid gate path are unaffected. No checkpoint parameter names change.

Run the forward and gradient regression with the pinned CUDA/vLLM dependencies:

```bash
python -m pytest -q mcore_adapter/tests/test_qwen4_exp_gated_norm.py
```

On September 25, the old forward failed two strict native-rounding cases. The
scoped implementation passed all 12 tests, including native forward equality
and independent FP64 checks of input, gate and norm-weight gradients. The
complete small-model TP1 versus TP2/SP test also passed with aggregate gradient
relative L2 `0.00124173`. Real-model probability parity remains unverified;
matching the first GDN output does not establish agreement in subsequent MoE
layers or after training.

ROLL's vLLM factory defaults the GDN prefill backend to `triton` when the model
directory or HF repository name begins with `Qwen3.8-Flash-Next` (ignoring
punctuation). An explicit `additional_config.gdn_prefill_backend` is preserved.
For generic names such as `checkpoint-99`, set that option explicitly; parent
experiment directories are not used to identify the model.

The SFT validation runner requires at least one complete heldout batch before
initializing workers: `DP * gradient_accumulation_steps * infer_batch_size`
records. With the supplied TP1/EP8 LoRA configuration, use at least eight
heldout records. A second check rejects an empty loader after preprocessing.

## Native DCP payload validation

After a training process exits, validate every declared model and optimizer
payload with bounded tensor memory:

```bash
python scripts/qwen38/validate_backbone_payloads.py \
  /path/to/checkpoint-19 \
  --output /path/to/payload-validation.json
```

The validator checks logical chunk coverage, actual tensor type, dtype, shape,
finite values, Adam step values, and native common state. It groups items by
storage file and reads increasing offsets through one file handle, so a large
checkpoint scan does not reopen the same rank file for every tensor. This is a
payload gate; architecture inventory, state equality, resumed updates, export,
parity, and performance remain separate checks.

## RLVR and OPD configuration candidates

The `configs` directory includes `rlvr_lora.yaml`, `rlvr_backbone.yaml`,
`opd_lora.yaml`, and `opd_backbone.yaml`. They use the real validation role
topologies: TP1/EP8 for LoRA training, TP4/EP8 for backbone training and the
separate teacher, and TP8 native vLLM rollout. Both training scopes use CPU Adam
and frozen N-gram tables. OPD explicitly names all three model paths so that the
configured teacher cannot be replaced by the student's base checkpoint.

These configurations use ordinary ROLL workers and have no dependency on the
private validation observers under `output/`. Each runs 20 steps and saves a
final checkpoint. Reserve the checkpoint capacity described above before
launching a backbone configuration. The independent memory supervisor described
above is part of the validation environment, not installed by these YAML files.

After setting up the pinned dependencies, run from the repository root:

```bash
export MODEL_PATH=/path/to/Qwen3.8-Flash-Next
python scripts/qwen38/prepare_rl_validation_data.py \
  --output-dir "$PWD/output/qwen38-rl-data"
export ROLL_RL_DATA="$PWD/output/qwen38-rl-data/train.jsonl"
export ROLL_RL_OUTPUT_DIR=/path/to/fresh-rl-output
NVIDIA_TF32_OVERRIDE=0 CUDA_DEVICE_MAX_CONNECTIONS=1 \
python examples/start_rlvr_pipeline.py \
  --config_path ../scripts/qwen38/configs --config_name rlvr_lora
```

Use `rlvr_backbone` to train the text backbone. For pure OPD, set
`ROLL_TEACHER_PATH` to a compatible trained Flash-Next model directory, choose a
new output directory, and select `opd_lora` or `opd_backbone` with the same
entry point. A native model-only teacher view must retain the matching HF
configuration, tokenizer and external N-gram assets. An optimizer checkpoint
directory alone is not a portable teacher model directory.

The example data uses the existing `messages`, `ground_truth`, and
`gpqa_diamond_boxed` tag format and the strict boxed-choice reward. Prompt and
response budgets are 256 and 128 tokens; these are lifecycle examples, not a
GPQA accuracy evaluation or an 8K RL configuration. The September 25 validation
used a fixed 80-question answer-only derivative of that dataset.
The data preparation command verifies the bundled source and recorded split
hashes, reproduces those 80 training records byte for byte, and writes 16
disjoint heldout records and a manifest into a fresh directory. It changes only
the instruction to request a boxed answer without an explanation; questions,
choices, and ground-truth answers remain unchanged.

On the observed validation installation, Hydra composition and the actual
`RLVRConfig` parser accepted all four configurations with CUDA uninitialized;
the training topology matched the executed validation configurations.
LoRA RL validation of the gated-RMSNorm fix (`9064b72`) subsequently completed
20 updates on every rank and transferred versions 0 through 20. A fresh process
restored checkpoint 19, completed updates 20 and 21 on all eight ranks, and
transferred versions 20 through 22. Both runs exited successfully. The final
checkpoints each passed an actual payload scan of 297,936 optimizer storage
items and eight adapter files. Reference and frozen-backbone sentinels remained
unchanged, and the restored sentinels matched the baseline's final values.

This verifies the instrumented LoRA RL lifecycle and positive-count cold
continuation. It does not establish complete state equality, whole-model
probability parity, performance, or full acceptance of the public examples.
The corresponding 20-step OPD validation and cold continuation remain pending.
