# ROLL Qwen3.8-Flash-Next Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a capability-gated Qwen3.8-Flash-Next training path to ROLL with exact GR/PLE/QSA semantics, bounded N-gram memory handling, and reproducible SFT/RL validation on `.181`.

**Architecture:** Extend the vendored mcore adapter with a dedicated `qwen4_exp` model, transformer-native GR/PLE modules, and loss-aware QSA indexer. Keep N-gram storage outside optimizer state by default and expose explicit trainable-table mode only after sparse-row semantics are implemented. Add ROLL lifecycle hooks for capability reports, CPU optimizer forwarding, and versioned weight updates; use vLLM's model mapper exactly once.

**Tech Stack:** Python 3.10+, PyTorch, Megatron-Core 0.16.x, Transformer Engine, flash-linear-attention/FlashQLA, ROLL Ray pipelines, vLLM qwen3_8_flash_next adapter.

**Spec:** `docs/superpowers/specs/2026-09-12-roll-qwen38-flash-next-design.md`

## Global Constraints

- Preserve unrelated dirty files in the parent checkout and keep PR #497 separate.
- Use `/data_hdd/Qwen3.8-Flash-Next` on `.181` as the authoritative BF16 source.
- Default first delivery is architecture-complete LoRA and main-text full-parameter training with the 51B N-gram table frozen; label trainable scope explicitly.
- Reject unsupported long-context sparse training, CP, MTP/vision training, and FP8 rollout until their stated checks pass.
- Production code requires a failing test before implementation; every `.181` claim must include a command and artifact.

---

### Task 1: Establish adapter package and failing contract tests

**Files:**
- Create: `mcore_adapter/src/mcore_adapter/models/qwen4_exp/{__init__.py,config_qwen4_exp.py,modeling_qwen4_exp.py,template_qwen4_exp.py}`
- Create: `mcore_adapter/tests/models/test_qwen4_exp_contract.py`
- Modify: adapter model/config registries and package exports.

**Interfaces:**
- `Qwen4ExpConfig` exposes `hc_count`, `hc_lowrank`, `layer_types`, QSA and PLE metadata without silently dropping unknown checkpoint fields.
- `Qwen4ExpModel` builds a 48-layer hybrid spec and rejects a missing or mismatched pattern.
- Template conversion reports every checkpoint key as trainable, frozen-external, preserved-auxiliary, or derived.

- [ ] Write tests for config registration, 36/12 layer pattern, PP-local layer numbering, and full 1,658-key coverage.
- [ ] Run the focused tests and confirm they fail because registration/module paths are absent.
- [ ] Implement the minimum config, model, template, and registry wiring using the existing qwen3_next/qwen3_5_moe patterns.
- [ ] Run focused tests and inspect the coverage report for zero unknown keys.
- [ ] Commit the isolated adapter change.

### Task 2: Implement exact GR residual path

**Files:**
- Create: `mcore_adapter/src/mcore_adapter/models/qwen4_exp/hyperconnection.py`
- Create: `mcore_adapter/src/mcore_adapter/models/qwen4_exp/hyperconnection_layer.py`
- Create: `mcore_adapter/tests/models/test_qwen4_exp_hyperconnection.py`

**Interfaces:**
- `HyperConnection.mix(stream) -> (stream, block_input, injection_logits)`.
- `HyperConnection.combine(stream, block_output, injection_logits) -> stream`.
- `HyperConnectionTransformerLayer` consumes and returns `[sequence,batch,4*hidden]` and never delegates ordinary residual BDA.

- [ ] Write a zero-block identity test and an FP32 reference test for read/write gates and gradients.
- [ ] Run tests and observe the old implementation's duplicate residual failure reproduced by the fixture.
- [ ] Implement grouped Gemma RMSNorm, SiLU(down / hc_count), sigmoid read mean, and `2*sigmoid(inject/4)` write.
- [ ] Run focused tests, including attention/MLP/hyperconnection input gradient checks.
- [ ] Commit the GR implementation.

### Task 3: Implement exact PLE and frozen N-gram backend

**Files:**
- Create: `mcore_adapter/src/mcore_adapter/models/qwen4_exp/ngram_embedding.py`
- Create: `mcore_adapter/src/mcore_adapter/models/qwen4_exp/ple_layer.py`
- Create: `mcore_adapter/tests/models/test_qwen4_exp_ple.py`

**Interfaces:**
- `FrozenNGramEmbedding.forward(input_ids) -> [batch,seq,ple_embed_dim]`, with checkpoint-provided hash constants and EOS segmentation.
- `PLELayer.forward(stream, input_ids) -> stream_delta` with key width `4*hidden`, value width `hidden`, and dilation `3`.

- [ ] Write tests for hash IDs, EOS isolation, checkpoint-shaped parameters, and residual gradient flow.
- [ ] Run them against the old adapter and confirm shape/semantic failures.
- [ ] Implement a non-persistent frozen table with bounded CPU/pinned staging hooks; add a separate explicit row-sharded trainable interface without enabling it by default.
- [ ] Run focused tests and verify table is absent from named parameters, state dict, and optimizer groups.
- [ ] Commit the PLE/N-gram implementation.

### Task 4: Implement QSA indexer loss and sparse training guard

**Files:**
- Create: `mcore_adapter/src/mcore_adapter/models/qwen4_exp/qsa.py`
- Create: `mcore_adapter/tests/models/test_qwen4_exp_qsa.py`
- Modify: model forward/loss interface and ROLL training config validation.

**Interfaces:**
- `QSAIndexer.forward(hidden, position, visible_mask) -> scores, selected_indices`.
- `qsa_indexer_kl_loss(scores, teacher_attention, complete_blocks, selected_blocks) -> scalar`.
- `Qwen4ExpModel` rejects sequences >2048 when sparse training kernel is unavailable.

- [ ] Write boundary tests for lengths 2047/2048/2049/2052/4096, padding, EOS, tie-breaking, and KL stop-gradient.
- [ ] Run tests and confirm no dense fallback is accepted above the configured budget.
- [ ] Implement FP32 reference selection and an autograd-safe path; keep the full `[B,S,S]` mask out of the training implementation.
- [ ] Run focused tests and compare selected sets and gradients with the official Transformers reference.
- [ ] Commit QSA support.

### Task 5: Fix conversion, optimizer, and ROLL/vLLM lifecycle

**Files:**
- Modify: mcore adapter converter and `roll/third_party/megatron/model_update.py`.
- Modify: `roll/distributed/strategy/megatron_strategy.py` and offload patch.
- Modify: `roll/third_party/vllm/worker.py` and compatibility layer.
- Create: focused conversion/lifecycle regression tests.

**Interfaces:**
- Conversion uses stacked GDN/SwiGLU components, TP=2/EP=8 constraints, cloned expert slices, and complete coverage reports.
- `OptimizerConfig` receives CPU-offload fields and distinguishes true CPU optimizer from phase state offload.
- Weight updates carry version/layout metadata, apply model-owned mapper once, invalidate QSA/GDN/PLE caches, and acknowledge all ranks.

- [ ] Write failing tests for optimizer field forwarding, duplicate mapper application, unknown keys, and duplicate residual lifecycle.
- [ ] Implement minimal fixes and run focused tests.
- [ ] Run TP1/TP2 conversion and roundtrip checks on small fixtures; verify source tensor slices bytewise.
- [ ] Commit lifecycle fixes.

### Task 6: Build `.181` validation runner and execute full acceptance matrix

**Files:**
- Create: `scripts/roll_qwen38_flash_next_validate.sh` and Python probes under `output/roll-qwen38-flash-next-research/validation/`.
- Create: `output/roll-qwen38-flash-next-research/validation-report.md`.
- Create: `.181` configs for LoRA SFT/RL and frozen-table text training.

- [ ] Run V0-V3 component/distributed checks on idle `.181` GPUs.
- [ ] Run V4 48-layer real-weight 2K/8K forward/logprob parity and memory sampling.
- [ ] Run V5 at least 100 optimizer steps for SFT in both declared modes, with save/resume.
- [ ] Run V6 at least 20 RL rollout/update/reload cycles and verify versions, advantages, and reference stability.
- [ ] Run V8 20-step throughput repeats for baseline and optimized kernels.
- [ ] Record all failures and unsupported combinations; do not label incomplete modes as supported.
- [ ] Commit validation runner and report.

### Task 7: Review, upstream PR, and final handoff

- [ ] Run full relevant local tests and verification-before-completion checks.
- [ ] Review code diff and generated coverage/memory reports.
- [ ] Split generic ROLL fixes and Qwen3.8 model support into reviewable commits/PRs.
- [ ] Push branch and create official ROLL PR only after `.181` evidence meets the acceptance matrix.
