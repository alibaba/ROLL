# Guide to Implementing Custom `loss_func`

When implementing a custom `loss_func` in ROLL, the most critical aspects are **how the loss is aggregated** and **how `loss_scale` is handled**. Mishandling these two points can cause the final computed loss or gradients to **deviate from the result that would be obtained by performing a single forward pass over the entire global batch**, thereby introducing training bias—especially severe in complex training scenarios involving **data parallelism (DP) + gradient accumulation (GA) + sequence packing**.

---

## 1. Common Loss Aggregation Strategies

Consider a **global batch** containing $B$ sequences. Let the length of the $i$-th sequence be $T_i$, with a per-token mask $m_{i,t} \in \{0,1\}$ indicating whether position $t$ participates in loss computation. The number of valid tokens is:

$$
N_i = \sum_{t=1}^{T_i} m_{i,t}, \quad N_{\text{all}} = \sum_{i=1}^{B} N_i
$$

Let $\mathcal{L}_{i,t}$ denote the per-token loss at position $t$ of sequence $i$ (e.g., NLL, CE, KL divergence, policy loss, etc.).

### 1.1 Token-level Loss (`token-mean`)

Compute the average loss over **all valid tokens in the global batch**:

$$
\mathcal{L}_{\text{token}} = \frac{1}{N_{\text{all}}} \sum_{i=1}^{B} \sum_{t=1}^{T_i} m_{i,t} \mathcal{L}_{i,t}
$$

**Property**: Each token has equal weight; longer sequences contribute more due to having more valid tokens.

### 1.2 Sequence-level Loss (`seq-mean`)

First aggregate within each sequence, then average across sequences. ROLL commonly uses two variants:

**(a) `seq-mean-token-sum`**  
Sum losses over tokens within each sequence, then average across sequences:
$$
\mathcal{L}_{\text{seq-sum}} = \frac{1}{B} \sum_{i=1}^{B} \left( \sum_{t=1}^{T_i} m_{i,t} \mathcal{L}_{i,t} \right)
$$

**(b) `seq-mean-token-mean`**  
Average losses over tokens within each sequence, then average across sequences:
$$
\mathcal{L}_{\text{seq-mean}} = \frac{1}{B} \sum_{i=1}^{B} \left( \frac{1}{N_i} \sum_{t=1}^{T_i} m_{i,t} \mathcal{L}_{i,t} \right)
$$

**Property**: Each sequence has equal weight, avoiding bias due to sequence length differences.

---

## 2. Micro-batch Partitioning in Distributed Training

In practice, a single global training step typically involves:

- **Data Parallelism (DP)**: The global batch is split across multiple DP ranks;
- **Gradient Accumulation (GA)**: Each rank further splits its data into multiple micro-batches, processed sequentially;
- **Sequence Packing**: To reduce padding and improve GPU utilization, multiple samples are concatenated into fixed-length packed sequences.

Let:
- DP world size be $D$,
- Gradient accumulation steps be $A$,
- Then the total number of micro-batches per global step is $M = D \times A$.

Denote the set of samples in the $k$-th micro-batch as $\mathcal{S}_k$. Its number of valid tokens is:
$$
N_k = \sum_{(i,t) \in \mathcal{S}_k} m_{i,t}, \quad N_{\text{all}} = \sum_{k=1}^{M} N_k
$$
The number of sequences (samples) in this micro-batch is $B_k$, satisfying:
$$
B = \sum_{k=1}^{M} B_k
$$

### 2.1 Why Does Sequence Packing Cause $B_k$ to Vary?

With sequence packing enabled, frameworks typically construct micro-batches based on a **token budget** rather than a fixed number of samples:

- Short sequences can be densely packed → some micro-batches contain many samples ($B_k$ large);
- Long sequences consume more space → some micro-batches contain few samples ($B_k$ small).

Thus, under packing, the number of samples per micro-batch $B_k$ is typically **uneven and unpredictable**, posing challenges for correct sequence-level loss aggregation.

---

## 3. Core Issue: Why You Should Not Normalize Using Local Statistics Within Micro-batches

ROLL’s goal is: **regardless of training configuration (DP/GA/Packing), the final loss used for backpropagation must be mathematically equivalent to computing the loss over the entire global batch in one go** (as defined in Section 1).

If each micro-batch uses its own local statistics (e.g., $N_k$ or $B_k$) for normalization, and gradients are accumulated via the backend, the result is generally **not equivalent**.

### 3.1 Token-level: Incorrect Normalization Within Micro-batches

**Wrong approach** (normalize by micro-batch’s own token count):
$$
\ell_k^{\text{wrong}} = \frac{1}{N_k} \sum_{(i,t) \in \mathcal{S}_k} m_{i,t} \mathcal{L}_{i,t}
$$

If micro-batches are equally weighted during averaging (e.g., via gradient averaging), the total loss becomes:
$$
\frac{1}{M} \sum_{k=1}^{M} \ell_k^{\text{wrong}} = \frac{1}{M} \sum_{k=1}^{M} \left( \frac{1}{N_k} \sum_{(i,t) \in \mathcal{S}_k} m_{i,t} \mathcal{L}_{i,t} \right)
$$

But the correct global `token-mean` loss is:
$$
\mathcal{L}_{\text{token}} = \frac{1}{N_{\text{all}}} \sum_{k=1}^{M} \sum_{(i,t) \in \mathcal{S}_k} m_{i,t} \mathcal{L}_{i,t}
$$

These are only equal when all $N_k$ are identical. Under variable-length sequences or packing, $N_k$ varies significantly, causing bias.

### 3.2 Sequence-level: Micro-batch `seq-mean` Causes Sample Weight Imbalance

Take `seq-mean-token-mean` as an example:

**Wrong approach** (normalize by micro-batch’s sample count $B_k$):
$$
\ell_k^{\text{wrong}} = \frac{1}{B_k} \sum_{i \in \mathcal{S}_k} \bar{\mathcal{L}}_i, \quad \text{where } \bar{\mathcal{L}}_i = \frac{1}{N_i} \sum_t m_{i,t} \mathcal{L}_{i,t}
$$

After equal-weight averaging across micro-batches:
$$
\frac{1}{M} \sum_{k=1}^{M} \ell_k^{\text{wrong}} = \frac{1}{M} \sum_{k=1}^{M} \left( \frac{1}{B_k} \sum_{i \in \mathcal{S}_k} \bar{\mathcal{L}}_i \right)
$$

But the correct global `seq-mean` is:
$$
\mathcal{L}_{\text{seq-mean}} = \frac{1}{B} \sum_{i=1}^{B} \bar{\mathcal{L}}_i
$$

The former treats each micro-batch equally; the latter treats each sequence equally. When $B_k$ varies (common under packing), they are not equivalent.

---

## 4. Correct Approach: Use Global Denominator + Sum Across Micro-batches

ROLL follows these design principles:

1. **Within each micro-batch, use global statistics as the denominator**;
2. **Each micro-batch’s returned loss should represent a partial contribution to the global loss**;
3. **The sum of all micro-batch losses must exactly equal the global loss**;
4. **Use `loss_scale` to counteract the backend’s default normalization behavior** (see Section 5).

### 4.1 Correct Implementation for Token-level Loss

For the $k$-th micro-batch:
$$
\ell_k = \frac{1}{N_{\text{all}}} \sum_{(i,t) \in \mathcal{S}_k} m_{i,t} \mathcal{L}_{i,t}
$$

Then:
$$
\sum_{k=1}^{M} \ell_k = \frac{1}{N_{\text{all}}} \sum_{k=1}^{M} \sum_{(i,t) \in \mathcal{S}_k} m_{i,t} \mathcal{L}_{i,t} = \mathcal{L}_{\text{token}}
$$

✅ Mathematically exact.

### 4.2 Correct Implementation for Sequence-level Loss (e.g., `seq-mean-token-mean`)

For the $k$-th micro-batch:
$$
\ell_k = \frac{1}{B} \sum_{i \in \mathcal{S}_k} \bar{\mathcal{L}}_i
$$

Then:
$$
\sum_{k=1}^{M} \ell_k = \frac{1}{B} \sum_{i=1}^{B} \bar{\mathcal{L}}_i = \mathcal{L}_{\text{seq-mean}}
$$

✅ Holds exactly even when $B_k$ varies (common under packing).

---

## 5. `loss_scale`: Compensating for Backend Normalization

Most training frameworks (e.g., Megatron, FSDP) implicitly normalize gradients under DP + GA to stabilize scale:

- **GA dimension**: Average gradients over $A$ micro-steps (equivalent to `loss /= A`);
- **DP dimension**: Divide by $D$ after AllReduce (equivalent to averaging across ranks).

The combined effect is:
$$
g \propto \frac{1}{M} \sum_{k=1}^{M} \nabla \ell_k, \quad M = D \times A
$$

However, ROLL’s aggregation design requires **summation semantics** across micro-batches:
$$
\nabla \mathcal{L}_{\text{global}} = \sum_{k=1}^{M} \nabla \ell_k
$$

To cancel the backend’s $1/M$ normalization, multiply each micro-batch’s loss by:
$$
\text{loss\_scale} = M
$$

Thus:
$$
\frac{1}{M} \sum_{k=1}^{M} \nabla (M \cdot \ell_k) = \sum_{k=1}^{M} \nabla \ell_k
$$

✅ Recovers correct summation semantics.

---

## 6. ROLL Interface: Global Stat Injection and `loss_scale` Control

To enable **globally equivalent loss aggregation** at the micro-batch level, ROLL automatically injects global batch statistics (e.g., total valid tokens, total valid samples) into each training step. These statistics are **computed based entirely on user-specified `loss_mask_keys`**.

### 6.1 `loss_mask_keys`: Define Loss Participation Scope and Drive Global Stat Injection

`loss_mask_keys` is a list of strings declaring **which mask fields identify "valid tokens participating in loss computation."** This configuration not only guides how the loss function masks invalid positions but—more importantly—**determines how the strategy computes and injects global aggregation quantities**.

You must set this in your pipeline’s data preprocessing or worker initialization:
```python
data.meta_info['loss_mask_keys'] = ['response_mask', 'labels_mask']
```

For each key in `loss_mask_keys` (e.g., `'response_mask'`), ROLL’s strategy will:

1. **Extract the corresponding mask tensor** from `data.batch` (typically shape `[batch_size, seq_len]`);
2. **Gather this mask across all DP ranks and GA steps**;
3. **Compute two global statistics**:
   - **`batch_num_tokens[key]`**: Total sum of this mask over the entire global batch, i.e.,  
     $$
     N_{\text{all}}^{(\text{key})} = \sum_{\text{all samples}} \sum_{t} \text{mask}_{i,t}^{(\text{key})}
     $$
   - **`global_valid_samples[key]`**: Number of sequences with **at least one valid token**, i.e.,  
     $$
     B^{(\text{key})} = \sum_{i=1}^{B} \mathbb{I}\left( \sum_{t} \text{mask}_{i,t}^{(\text{key})} > 0 \right)
     $$

These statistics are injected into `data.meta_info` for use in `loss_func`.

> ⚠️ **Critical Consistency Requirement**: The mask you use in `loss_func` for loss computation, weighting, or aggregation **must have identical semantics to the mask specified in `loss_mask_keys`**.  
> For example, if `loss_mask_keys = ['response_mask']`, your loss must be computed **only** using `response_mask`. Using a different mask (e.g., `attention_mask`) will cause a mismatch between the numerator (loss computation) and denominator (global stats), breaking equivalence.

### 6.2 Using Injected Global Statistics in `loss_func`

In your custom `loss_func`, access global statistics as follows:

```python
# Assume 'response_mask' is in loss_mask_keys
mask_key = 'response_mask'

N_all = data.meta_info['batch_num_tokens'][mask_key]        # Global valid token count
B_all = data.meta_info['global_valid_samples'][mask_key]    # Global valid sample count
```

Then use these global values as denominators during aggregation (see Section 4) to ensure micro-batch computations exactly reconstruct the global loss.

### 6.3 `apply_loss_scale`: Controlling Gradient Scale Correction

Since training backends (e.g., Megatron/FSDP) typically apply implicit $1/(D \times A)$ normalization under DP + GA, while ROLL relies on **summation semantics**, compensation via `loss_scale = D \times A` is needed.

In `worker_config`, the parameter `apply_loss_scale` controls whether this scaling is applied automatically:

- **Default: `True`** (recommended to keep enabled)
- **Effect**: Framework automatically multiplies the loss returned by `loss_func` by `loss_scale`
- **When to disable**: Only if you manually implement the full global loss (including scale) inside `loss_func`—generally not advised.

---

## 7. Metrics Logging: Use `@sum` Semantics

For losses aggregated using global denominators, metrics should be **summed—not averaged—during multi-worker reduction**.

ROLL supports specifying reduction behavior via an `@operator` suffix in metric names:

```python
metrics = {
    "actor/kl_loss@sum": kl_loss.detach().item(),
}
reduce_metrics(metrics)
```

- `@sum`: Sum values across all workers during reduction;
- `@mean` (default): Average across workers;
- The logger automatically strips everything from `@` onward, so it displays as `actor/kl_loss`.

---

## 8. Code Example: Globally Equivalent KL Loss Implementation in Actor

### 8.1 Compute Per-Token KL

```python
kl_loss = compute_approx_kl(
    log_probs=log_probs,
    log_probs_base=ref_log_probs,
    action_mask=final_response_mask,
    kl_penalty="k3"
)
```

### 8.2 Aggregate Using Global Denominator

```python
kl_loss = agg_loss(
    loss_mat=kl_loss,
    loss_mask=final_response_mask,
    loss_agg_mode=self.pipeline_config.loss_agg_mode,
    batch_num_tokens=batch_num_tokens['final_response_mask'],
    global_valid_samples=global_valid_samples['final_response_mask'],
)
```

### 8.3 Key Implementation of `agg_loss`

```python
def agg_loss(loss_mat, loss_mask, loss_agg_mode, batch_num_tokens=None, global_valid_samples=None, weights=None):
    if batch_num_tokens is None:
        batch_num_tokens = loss_mask.sum()
    if global_valid_samples is None:
        global_valid_samples = loss_mat.size(0)

    if loss_agg_mode == "token-mean":
        loss = (loss_mat * loss_mask).sum() / batch_num_tokens
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = (loss_mat * loss_mask).sum(dim=-1)
        valid = (loss_mask.sum(dim=-1) > 0).float()
        loss = (seq_losses * valid).sum() / (global_valid_samples + 1e-8)
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_means = masked_mean(loss_mat, loss_mask, dim=-1)  # Custom function supporting mask
        valid = (loss_mask.sum(dim=-1) > 0).float()
        loss = (seq_means * valid).sum() / (global_valid_samples + 1e-8)
    else:
        raise ValueError(f"Unsupported loss_agg_mode: {loss_agg_mode}")
    
    return loss
```

### 8.4 Log Metrics

```python
pg_metrics = {"actor/kl_loss@sum": kl_loss.detach().item()}
```

---

## 9. Design Checklist: Custom Loss Implementation (⚠️ Summary of Critical Points)

To ensure mathematical equivalence and training stability under any configuration, strictly follow this checklist:

### ✅ **Loss Granularity and Aggregation Mode**
- Clearly decide whether your loss is **token-level** or **sequence-level**.
- Choose the correct `loss_agg_mode` (e.g., `"token-mean"`, `"seq-mean-token-mean"`).

### ✅ **Use Global Denominators (Critical!)**
- **Never** use local micro-batch statistics (e.g., `loss_mask.sum()` or `loss_mat.shape[0]`) as denominators.
- **Always** use global statistics from `data.meta_info['batch_num_tokens'][key]` and `data.meta_info['global_valid_samples'][key]`.

### ✅ **`loss_mask_keys` Configuration and Consistency (Common Pitfall!)**
- Explicitly set `data.meta_info['loss_mask_keys']` in your pipeline.
- **Ensure** the mask used in `loss_func` for computation/masking/weighting **exactly matches** the key(s) in `loss_mask_keys`.
- If using multiple masks (e.g., response + labels), include all in `loss_mask_keys` and handle them separately.

### ✅ **`apply_loss_scale` Setting**
- **Keep default `True`** unless you fully understand and manage scaling logic yourself.
- Disabling incorrectly causes gradients to be implicitly scaled down by $1/(D \times A)$, leading to divergence or extremely slow convergence.

### ✅ **Metrics Logging Convention**
- For losses using global denominators, **always** append `@sum` to metric names (e.g., `"loss@sum"`).
- Otherwise, reduction by averaging will log incorrect (underestimated by $M\times$) loss values.

### ✅ **Special Care Under Packing**
- Never assume fixed $B_k$ (sample count) or $N_k$ (token count) per micro-batch.
- All aggregation logic must **avoid dependence on micro-batch-local statistics** and rely solely on injected global values.