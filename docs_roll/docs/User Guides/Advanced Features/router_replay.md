# ROUTER REPLAY IN ROLL

The ROLL framework supports **Router Replay**, a feature that addresses the training-inference mismatch caused by inconsistent expert routing in MoE (Mixture-of-Experts) RL training. By forcing the training-side MoE Router to use a pre-recorded set of routing decisions, Router Replay eliminates routing-level discrepancies at their source and substantially stabilizes training.

> **Note**: ROLL currently implements only the **R3** mode (Rollout Routing Replay: SGLang inference + Megatron training). R2 is not yet implemented; the combination of R3 with `sequence_packing` is also not yet supported. Both will be added in future releases — please do not enable R3 together with `sequence_packing` for now.

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

### 2.2 R2 — Vanilla Routing Replay (Not Yet Implemented)

- $I_{\text{ref}}$ comes from a **forward pass that the training engine itself runs with old policy weights**.
- For the first mini-batch, $\theta = \theta_{\text{old}}$, so the replayed forward matches the original forward (effectively on-policy).
- For subsequent off-policy mini-batches, fixing the routing constrains policy staleness.

R2 addresses "routing drift across gradient steps within the training engine" but **does not solve the discrepancy between inference and training engines**. R2 is currently unimplemented in ROLL; selecting `mode: R2` raises `NotImplementedError`.

### 2.3 R3 — Rollout Routing Replay (Supported in ROLL)

- $I_{\text{ref}}$ comes **directly from the routing recorded by the inference engine during rollout**.
- The training side uses an expert selection that is exactly aligned with the sampled trajectory, so the inference-vs-training routing gap is eliminated entirely.
- This also constrains routing drift across gradient steps (same benefit as R2).

R3 mitigates both the training-inference discrepancy and policy staleness simultaneously, and is the recommended path in ROLL.

### 2.4 End-to-End R3 Flow in ROLL

```
┌─────────────────────────────┐         ┌──────────────────────────────┐
│        SGLang Rollout       │         │      Megatron Training       │
│                             │         │                              │
│  generate(...)              │         │  forward()                   │
│   └─ MoE Router top-k       │         │   └─ MoE RouterReplay        │
│        └─ export indices    │ ──────► │        └─ replay indices     │
│           [seq, layers, k]  │ batch   │           in forward         │
│                             │  data   │                              │
│  return routed_experts      │         │  forward → backward          │
└─────────────────────────────┘         └──────────────────────────────┘
```

1. **Sampling**: while generating tokens, SGLang additionally records the top-k experts per MoE layer, producing a `routed_experts` tensor of shape `[seq_len, num_layers, top_k]` returned with the response.
2. **Data movement**: the rollout postprocessor attaches `routed_experts` to each sample in the batch; it then flows through ROLL's standard data path (DP / mini-batch / micro-batch) into the training worker.
3. **Training**: before each forward, Megatron sets per-layer RouterReplay indices from `routed_experts`; the MoE Router skips its top-k computation and replays the external indices; after forward, the action flips to "replay-on-backward" so activation recomputation uses the same routing.

### 2.5 R3 + Sequence Packing Is Not Yet Supported

`routed_experts` is laid out **per-sample on the original (unpacked) attention_mask**, while `sequence_packing` concatenates sequences, repads them to a multiple of `2 × CP_SIZE × TP_SIZE`, and re-chunks across CP ranks. The two layouts are not yet reconciled, so:

- ROLL currently disallows enabling R3 and `sequence_packing` simultaneously;
- When R3 is enabled, please keep `use_sequence_packing: False`;
- A future release will add the index-remapping logic required to make the two work together.

### 2.6 Compatibility Matrix

| Feature                            | R3 Compatibility               |
|------------------------------------|--------------------------------|
| Megatron `megatron_train`          | Supported                      |
| SGLang `sglang` rollout            | Supported (required)           |
| Tensor Parallelism (TP)            | Supported                      |
| Pipeline Parallelism (PP)          | Supported                      |
| Virtual Pipeline Parallelism (VPP) | Supported                      |
| Dynamic Batching                   | Supported                      |
| **Sequence Packing**               | **Not yet (planned)**          |
| GSPO                               | Orthogonal, can be combined    |
| TIS / IS correction                | Coexists; gains are workload-dependent |
| vLLM rollout                       | Not supported                  |
| FSDP / DeepSpeed training          | Not supported                  |

## 3. Implementation

### 3.1 Rollout Side (SGLang)

In `sglang_strategy.py`, when `router_replay.mode != "disable"`:

- The SGLang server is launched with `enable_return_routed_experts=True`;
- Each generate request sets `return_routed_experts=True`;
- The chunked outputs collect `routed_experts` from each chunk's `meta_info` and assemble per-sample records;
- SGLang `>= 0.5.6.post3` is required.

### 3.2 Training Side (Megatron)

In `megatron_strategy.py`:

- During init, `moe_enable_routing_replay` is force-set to `True` so each MoE layer carries a `RouterReplay` instance;
- For `compute_log_probs`-style forwards: replay is enabled only when the batch contains `routed_experts`; otherwise (e.g., reference model) the global action is cleared and the router runs as usual;
- For `train_step`: the strategy asserts that `routed_experts` is present, sets `REPLAY_FORWARD` globally, runs forward/backward, and finally clears global state;
- In `inner_forward_step`, `set_router_replay_data` scatters the recorded indices to the current SP rank; after the forward, the action flips to `REPLAY_BACKWARD` so activation recomputation stays consistent.

### 3.3 Core Utilities

`roll/third_party/megatron/router_replay_utils.py` provides:

- `set_router_replay_data` — scatter top-k indices to the local SP rank;
- `RouterReplayHelper` — query and toggle per-layer RouterReplay state across PP / VPP;
- `get_routed_experts_dtype` — pick `uint8` / `int16` automatically based on the number of experts to reduce memory and transfer cost.

## 4. Configuration

### 4.1 How to Enable

Set `router_replay.mode: R3` on every worker that participates in R3. R3 **must** be enabled symmetrically on both the rollout and the training side — enabling it on only one side has no effect.

### 4.2 Parameters

#### `router_replay.mode`

- **`disable`** (default): Router Replay is off.
- **`R2`**: Vanilla Routing Replay. **Not yet implemented** — configuring this mode raises `NotImplementedError`.
- **`R3`**: Rollout Routing Replay, the only supported mode in ROLL today.

### 4.3 Full Configuration Example

```yaml
actor_train:
  router_replay:
    mode: R3

  strategy_args:
    strategy_name: megatron_train

  # R3 + sequence_packing is not yet supported; keep this disabled
  use_sequence_packing: False

actor_infer:
  router_replay:
    mode: R3

  strategy_args:
    strategy_name: sglang  # requires sglang >= 0.5.6.post3

reference:
  router_replay:
    mode: disable
  strategy_args:
    strategy_name: megatron_infer
```

### 4.4 Usage Recommendations

1. **Environment & strategies**: `actor_infer` must use `sglang` (≥ 0.5.6.post3); `actor_train` must use `megatron_train`.
2. **Symmetric enablement**: configure `mode: R3` on both rollout and training workers — enabling only one side is a no-op.
3. **Reference model**: keep `mode: disable`. When `routed_experts` is missing from the batch, ROLL automatically skips the replay logic.
4. **Disable sequence packing for now**: until R3 + `sequence_packing` is supported, keep `use_sequence_packing: False` on every worker that uses R3.
5. **Resource overhead**: the `routed_experts` tensor (`[seq_len, num_layers, top_k]`) introduces extra memory and inter-worker transfer cost; ROLL automatically selects the smallest integer dtype to mitigate this.
6. **Relationship with IS / TIS**: Router Replay fixes routing at the architecture level, while IS / TIS correct probability divergence at the loss level. They are complementary and can be used together depending on the workload.
7. **Troubleshooting**: if training-inference mismatch persists, verify (a) the rollout response actually carries `routed_experts`; (b) `moe_enable_routing_replay` is `True` on the training side; (c) sequence packing is disabled on every relevant worker.

With Router Replay (R3) enabled, ROLL guarantees strict alignment of MoE routing between rollout and training under TP / PP / VPP parallelism, removing a class of mismatch that loss-level correction alone cannot fully address. Future releases will progressively add R2 and the R3 + sequence_packing combination.
