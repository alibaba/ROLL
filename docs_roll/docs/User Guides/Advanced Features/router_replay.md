# ROUTER REPLAY IN ROLL

The ROLL framework supports **Router Replay**, a feature that addresses the training-inference mismatch caused by inconsistent expert routing in MoE (Mixture-of-Experts) RL training. By forcing the training-side MoE Router to use a pre-recorded set of routing decisions, Router Replay eliminates routing-level discrepancies at their source and substantially stabilizes training.

> **Note**: ROLL supports both the **R2** mode (Vanilla Routing Replay: Megatron old-policy forward records, Megatron training replays) and the **R3** mode (Rollout Routing Replay: SGLang / vLLM inference + Megatron training). Both modes work with TP / PP / VPP / CP parallelism, dynamic batching, and `sequence_packing`.

## 1. Background

### 1.1 Routing Inconsistency in MoE RL

In each MoE layer the Router selects top-k experts per token. In RL training, the same set of weights is used by three different roles:

- **Rollout policy**: the policy used by the inference engine (e.g., SGLang) for sampling.
- **Old policy**: the training-side model state right before this batch's gradient updates.
- **Training policy**: the training-side model that is actively being updated.

Ideally, all three should produce identical routing, but in practice:

- **Training vs. inference**: the inference and training engines differ in kernel implementations, numerical precision, and parallelism layouts. Even with identical weights, the two sides may select different top-k experts for the same input.
- **Across gradient steps**: as mini-batch updates proceed, routing decisions also drift along with the weights.

### 1.2 Why It Matters

Routing is a discrete choice that gets amplified by the downstream expert outputs. When the rollout-side and training-side selected experts disagree, per-token output probabilities diverge significantly, which leads to:

- Inflated importance sampling ratios — many samples in PPO/GRPO become heavily clipped or contribute high-variance updates;
- Training collapse in highly off-policy regimes;
- IS correction or TIS-style loss-side compensation alone is often insufficient to recover stability.

The idea behind Router Replay is simple: **rather than trying to fix the discrepancy at the loss layer, fix the routing mask at the architecture layer so that the training side directly reuses a "reference" routing**, removing this source of mismatch entirely.

## 2. Design

### 2.1 The Replay Formula

Both R2 and R3 share the same mechanism: during the training forward, replace the top-k mask normally produced by router logits with an externally provided mask $I_{\text{ref}}$, and renormalize using the training-side logits $s_{\text{train}}$:

$$
g_i = \frac{I_{\text{ref}, i} \cdot \exp(s_{\text{train}, i})}{\sum_j I_{\text{ref}, j} \cdot \exp(s_{\text{train}, j})}
$$

Key properties:

- The **selection** of experts is dictated by $I_{\text{ref}}$, not by training-side argmax.
- The **softmax** is still computed over training-side logits, so router weights still receive gradients normally.

R2 and R3 differ only in where $I_{\text{ref}}$ comes from.

### 2.2 R2 — Vanilla Routing Replay

R2's recorded routing comes from the training engine itself. At the start of each batch, Megatron first runs an old-policy forward to recompute the old log probs; R2 records the top-k experts chosen by every MoE layer during this pass and replays them verbatim in the batch's subsequent gradient updates. This way, no matter how many update rounds the same batch goes through, the routing stays identical, removing the noise caused by routing drift as the weights change. For the first mini-batch the model has not been updated yet, so the replayed forward matches the original one exactly — effectively on-policy.

To use it, simply set `router_replay.mode: R2` on `actor_train` and enable `enable_old_logprobs_recompute: true` at the pipeline level (without the latter, nothing is recorded and the training step fails for lack of routing data); the inference side needs no configuration at all. Note that R2 only guarantees routing consistency within the training side — it does not address the routing gap between the inference and training engines. If you need alignment with the rollout routing, use R3.

### 2.3 R3 — Rollout Routing Replay (Recommended)

- $I_{\text{ref}}$ comes **directly from the routing recorded by the inference engine during rollout** (SGLang ≥ 0.5.6.post3, or ROLL's patched vLLM integration).
- The training side uses an expert selection that is exactly aligned with the sampled trajectory, so the inference-vs-training routing gap is eliminated entirely.
- This also constrains routing drift across gradient steps (same benefit as R2).

R3 mitigates both the training-inference discrepancy and policy staleness simultaneously, and is the recommended path in ROLL.

### 2.4 End-to-End R3 Flow in ROLL

```
┌─────────────────────────────┐         ┌──────────────────────────────┐
│   Rollout (SGLang / vLLM)   │         │      Megatron Training       │
│                             │         │                              │
│  generate(...)              │         │  forward()                   │
│   └─ MoE Router top-k       │         │   └─ MoE RouterReplay        │
│        └─ export indices    │ ──────► │        └─ replay indices     │
│           [seq, layers, k]  │ batch   │           in forward         │
│                             │  data   │                              │
│  return routed_experts      │         │  forward → backward          │
└─────────────────────────────┘         └──────────────────────────────┘
```

1. **Sampling**: while generating tokens, the inference engine additionally records the top-k experts per MoE layer and returns them with the response as a `routed_experts` tensor of shape `[seq_len, num_layers, top_k]`.
2. **Data movement**: ROLL attaches `routed_experts` to each sample in the batch, which then flows through the standard data path (DP / mini-batch / micro-batch) into the training workers — no user action required.
3. **Training**: before each forward, Megatron loads the recorded indices into every MoE layer; the Router skips its own top-k computation and replays them. The same routing is reused throughout the backward pass and activation recomputation.

### 2.5 Router Replay + Sequence Packing

**R2/R3 can be enabled together with `sequence_packing`.** The recorded routing is laid out per sample, while sequence packing concatenates samples into packed sequences; ROLL automatically converts between the two layouts and repacks `routed_experts` together with `input_ids`, so every token keeps its recorded experts under any TP / CP configuration.

No extra configuration is needed: keep `use_sequence_packing` as you would without Router Replay.

### 2.6 Compatibility Matrix

| Feature                            | R2                             | R3                                   |
|------------------------------------|--------------------------------|--------------------------------------|
| Megatron `megatron_train`          | Required                       | Required                             |
| Rollout engine                     | Not involved (keep `disable`)  | `sglang` ≥ 0.5.6.post3 or patched vLLM |
| Tensor Parallelism (TP)            | Supported                      | Supported                            |
| Pipeline Parallelism (PP)          | Supported                      | Supported                            |
| Virtual Pipeline Parallelism (VPP) | Supported                      | Supported                            |
| Context Parallelism (CP)           | Supported                      | Supported                            |
| Dynamic Batching                   | Supported                      | Supported                            |
| **Sequence Packing**               | **Supported**                  | **Supported**                        |
| GSPO                               | Orthogonal, can be combined    | Orthogonal, can be combined          |
| TIS / IS correction                | Coexists; gains are workload-dependent | Coexists; gains are workload-dependent |
| FSDP / DeepSpeed training          | Not supported                  | Not supported                        |

## 3. Implementation

### 3.1 Rollout Side (SGLang / vLLM)

When `router_replay.mode: R3` is set on `actor_infer`, ROLL automatically:

- launches the SGLang server with routed-experts export enabled and assembles per-sample `routed_experts` records from every response (requires SGLang `>= 0.5.6.post3`);
- or, with ROLL's patched vLLM integration, converts the `routed_experts` carried by each completion into the same per-sample tensor format.

### 3.2 Training Side (Megatron)

When `router_replay.mode` is not `disable`, the Megatron training side automatically:

- enables routing replay on every MoE layer at initialization;
- **R3**: any forward whose batch carries `routed_experts` replays it; batches without it (e.g., the reference model) run with the normal router. The training step requires `routed_experts` and fails fast if it is missing;
- **R2**: the old-policy forward records the routing of every MoE layer, and the records from all parallel ranks are merged into a per-sample `routed_experts` tensor attached to the batch, which the training step then replays exactly like R3;
- reuses the same recorded indices for activation recomputation during backward, keeping forward and backward consistent.

To contain the extra memory and transfer cost of `routed_experts` (tens of MB per batch on large MoE models), ROLL keeps it aside during intermediate stages (reorder, grouping, advantage computation) and re-attaches it only for actor training, and transfers it in chunks rather than as one large object.

### 3.3 Core Utilities

The implementation lives in `roll/third_party/megatron/router_replay_utils.py`, which takes care of the details for you: distributing recorded indices across sequence-parallel ranks, collecting and merging R2 recordings across pipeline stages, and picking the smallest integer dtype (`uint8` / `uint16` / `uint32`) based on the model's expert count to reduce memory and transfer cost.

## 4. Configuration

### 4.1 How to Enable

Set `router_replay.mode` on every worker that participates:

- **R3** must be enabled symmetrically on both the rollout (`actor_infer`) and the training side (`actor_train`). Enabling only the inference side has no effect; enabling only the training side makes training fail, because no routing data ever arrives.
- **R2** is enabled on `actor_train` only; the rollout side stays `disable`.

### 4.2 Parameters

#### `router_replay.mode`

- **`disable`** (default): Router Replay is off.
- **`R2`**: Vanilla Routing Replay — record routing during the Megatron old-policy forward and replay it in the gradient updates of the same batch. Requires `enable_old_logprobs_recompute: true`.
- **`R3`**: Rollout Routing Replay — replay routing recorded by the SGLang / vLLM rollout engine.

### 4.3 Configuration Examples

R3 with SGLang rollout (works with or without `sequence_packing`):

```yaml
actor_train:
  router_replay:
    mode: R3
  strategy_args:
    strategy_name: megatron_train
    strategy_config:
      moe_enable_routing_replay: true  # build every MoE layer with replay support

actor_infer:
  router_replay:
    mode: R3
  strategy_args:
    strategy_name: sglang  # requires sglang >= 0.5.6.post3; or use the patched vllm

reference:
  router_replay:
    mode: disable
  strategy_args:
    strategy_name: megatron_infer
```

R2 (record on the Megatron old-policy forward; no rollout involvement):

```yaml
# R2 records during the old-log-probs recompute
enable_old_logprobs_recompute: true

actor_train:
  router_replay:
    mode: R2
  strategy_args:
    strategy_name: megatron_train
    strategy_config:
      moe_enable_routing_replay: true

actor_infer:
  router_replay:
    mode: disable

reference:
  router_replay:
    mode: disable
```

### 4.4 Usage Recommendations

1. **Environment & strategies**: for R3, `actor_infer` must use `sglang` (≥ 0.5.6.post3) or the patched `vllm` integration; `actor_train` must use `megatron_train`. For R2, only `actor_train` is involved. In both modes, also set `moe_enable_routing_replay: true` in `actor_train`'s `strategy_config` so that every MoE layer is built with replay support.
2. **Symmetric enablement (R3)**: configure `mode: R3` on both rollout and training workers. Enabling only the inference side is a no-op; enabling only the training side makes training fail on missing routing data.
3. **Reference model**: keep `mode: disable`. When `routed_experts` is missing from the batch, ROLL automatically skips the replay logic.
4. **Sequence packing**: R2/R3 work with `use_sequence_packing` enabled or disabled — no special action required.
5. **Resource overhead**: the `routed_experts` tensor (`[seq_len, num_layers, top_k]`) introduces extra memory and inter-worker transfer cost; ROLL automatically uses the most compact integer dtype and minimizes how long the tensor travels with the batch, so the overhead stays modest.
6. **Relationship with IS / TIS**: Router Replay fixes routing at the architecture level, while IS / TIS correct probability divergence at the loss level. They are complementary and can be used together depending on the workload.
7. **Troubleshooting**: start with the startup logs — training workers print `Router Replay <mode> mode: REPLAY enabled`, and with R2 you should additionally see `RECORD enabled`. If training-inference mismatch persists, verify (a) the rollout responses actually carry `routed_experts`; (b) for R3, `mode: R3` is set on both `actor_train` and `actor_infer`, and `sglang >= 0.5.6.post3` (or the patched vLLM) is installed; (c) for R2, `enable_old_logprobs_recompute: true` is set — otherwise recording never happens and the training step fails.

With Router Replay enabled, ROLL guarantees strict alignment of MoE routing under TP / PP / VPP / CP parallelism — R3 aligns rollout and training, and R2 constrains routing drift across gradient steps — removing a class of mismatch that loss-level correction alone cannot fully address.
