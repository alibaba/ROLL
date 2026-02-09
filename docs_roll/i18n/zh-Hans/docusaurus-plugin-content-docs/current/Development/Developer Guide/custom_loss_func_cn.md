# 自定义 `loss_func` 实现指南

在 ROLL 中实现自定义 `loss_func` 时，最关键的是 **loss 的聚合方式（aggregate）** 与 **`loss_scale` 的处理**。如果这两点处理不当，会导致最终计算出的 loss 或梯度 **不等价于对整个 global batch 一次性前向计算的结果**，从而引入训练偏差——这在 **数据并行（DP） + 梯度累积（Gradient Accumulation, GA） + 序列打包（Sequence Packing）** 的复杂训练场景下尤为严重。

---

## 1. 常用 Loss 聚合方式

设一个 **global batch** 包含 $B$ 个序列。第 $i$ 个序列长度为 $T_i$，其 token 级 mask 为 $m_{i,t} \in \{0,1\}$，表示该位置是否参与 loss 计算。有效 token 数为：

$$
N_i = \sum_{t=1}^{T_i} m_{i,t}, \quad N_{\text{all}} = \sum_{i=1}^{B} N_i
$$

令 $\mathcal{L}_{i,t}$ 表示第 $i$ 个序列第 $t$ 个位置的逐 token loss（如 NLL、CE、KL 散度、策略损失等）。

### 1.1 Token-level Loss（token-mean）

对 global batch 中 **所有有效 token 求平均**：

$$
\mathcal{L}_{\text{token}} = \frac{1}{N_{\text{all}}} \sum_{i=1}^{B} \sum_{t=1}^{T_i} m_{i,t} \mathcal{L}_{i,t}
$$

**特点**：每个 token 权重相同，长序列因包含更多有效 token 而贡献更大。

### 1.2 Sequence-level Loss（seq-mean）

先对每条序列内部做聚合，再对所有序列求平均。ROLL 中常用两种变体：

**(a) seq-mean-token-sum**  
序列内对 token 求和，再对序列求平均：
$$
\mathcal{L}_{\text{seq-sum}} = \frac{1}{B} \sum_{i=1}^{B} \left( \sum_{t=1}^{T_i} m_{i,t} \mathcal{L}_{i,t} \right)
$$

**(b) seq-mean-token-mean**  
序列内对 token 求平均，再对序列求平均：
$$
\mathcal{L}_{\text{seq-mean}} = \frac{1}{B} \sum_{i=1}^{B} \left( \frac{1}{N_i} \sum_{t=1}^{T_i} m_{i,t} \mathcal{L}_{i,t} \right)
$$

**特点**：每条序列权重相同，不会因长度不同而产生偏差。

---

## 2. 分布式训练中的 micro-batch 划分

实际训练中，一个 global step 通常同时涉及：

- **数据并行（DP）**：global batch 被划分到多个 DP rank 上；
- **梯度累积（GA）**：每个 rank 将其数据进一步划分为多个 micro-batch，逐次前向/反向；
- **序列打包（Sequence Packing）**：为减少 padding、提升 GPU 利用率，将多个样本拼接成固定长度的 packed 序列。

设：
- DP world size 为 $D$，
- Gradient accumulation steps 为 $A$，
- 则一个 global step 内共有 $M = D \times A$ 个 micro-batch。

第 $k$ 个 micro-batch 包含的样本集合记为 $\mathcal{S}_k$，其有效 token 数为：
$$
N_k = \sum_{(i,t) \in \mathcal{S}_k} m_{i,t}, \quad N_{\text{all}} = \sum_{k=1}^{M} N_k
$$
其包含的序列数量（即样本数）为 $B_k$，满足：
$$
B = \sum_{k=1}^{M} B_k
$$

### 2.1 为什么 sequence packing 会导致 $B_k$ 不固定？

开启 sequence packing 后，框架通常按 **token 预算**（而非固定样本数）来构建 micro-batch：

- 短序列可被密集打包 → 某些 micro-batch 包含较多样本（$B_k$ 较大）；
- 长序列占用更多空间 → 某些 micro-batch 只能容纳较少样本（$B_k$ 较小）。

因此，在 packing 场景下，各 micro-batch 的样本数 $B_k$ 通常是**不均衡且不可预测的**。这对 sequence-level loss 的正确聚合提出了挑战。

---

## 3. 核心问题：为何不能在 micro-batch 内使用局部统计量做归一化？

ROLL 的目标是：**无论训练配置如何（DP/GA/Packing），最终用于反向传播的 loss 必须严格等价于对整个 global batch 一次性计算的结果**（见第 1 节）。

若在每个 micro-batch 内使用其自身的统计量（如 $N_k$ 或 $B_k$）进行归一化，再依赖 backend 进行梯度累积，通常会导致**非等价结果**。

### 3.1 Token-level：错误的 micro 内归一化

**错误做法**（用 micro 自身 token 数归一化）：
$$
\ell_k^{\text{wrong}} = \frac{1}{N_k} \sum_{(i,t) \in \mathcal{S}_k} m_{i,t} \mathcal{L}_{i,t}
$$

若 micro-batch 之间被等权平均（如通过梯度平均实现），则总 loss 为：
$$
\frac{1}{M} \sum_{k=1}^{M} \ell_k^{\text{wrong}} = \frac{1}{M} \sum_{k=1}^{M} \left( \frac{1}{N_k} \sum_{(i,t) \in \mathcal{S}_k} m_{i,t} \mathcal{L}_{i,t} \right)
$$

而正确的 global token-mean 应为：
$$
\mathcal{L}_{\text{token}} = \frac{1}{N_{\text{all}}} \sum_{k=1}^{M} \sum_{(i,t) \in \mathcal{S}_k} m_{i,t} \mathcal{L}_{i,t}
$$

二者仅在所有 $N_k$ 相等时才一致。在变长序列或 packing 场景下，$N_k$ 差异显著，导致偏差。

### 3.2 Sequence-level：micro 内 seq-mean 导致样本权重失衡

以 `seq-mean-token-mean` 为例：

**错误做法**（用 micro 自身样本数 $B_k$ 归一化）：
$$
\ell_k^{\text{wrong}} = \frac{1}{B_k} \sum_{i \in \mathcal{S}_k} \bar{\mathcal{L}}_i, \quad \text{其中 } \bar{\mathcal{L}}_i = \frac{1}{N_i} \sum_t m_{i,t} \mathcal{L}_{i,t}
$$

micro 间等权平均后得到：
$$
\frac{1}{M} \sum_{k=1}^{M} \ell_k^{\text{wrong}} = \frac{1}{M} \sum_{k=1}^{M} \left( \frac{1}{B_k} \sum_{i \in \mathcal{S}_k} \bar{\mathcal{L}}_i \right)
$$

而正确的 global seq-mean 是：
$$
\mathcal{L}_{\text{seq-mean}} = \frac{1}{B} \sum_{i=1}^{B} \bar{\mathcal{L}}_i
$$

前者等价于“每个 micro-batch 等权”，后者是“每个序列等权”。当 $B_k$ 不固定时（packing 常见），两者不等价。

---

## 4. 正确做法：使用全局分母 + micro 间求和

ROLL 的设计原则是：

1. **在 micro-batch 内部聚合时，直接使用 global 统计量作为分母**；
2. **每个 micro-batch 返回的 loss 应设计为 global loss 的一部分**；
3. **所有 micro-batch 的 loss 相加后，应精确等于 global loss**；
4. **通过 `loss_scale` 抵消 backend 的默认归一化行为**（见第 5 节）。

### 4.1 Token-level 的正确实现

对第 $k$ 个 micro-batch：
$$
\ell_k = \frac{1}{N_{\text{all}}} \sum_{(i,t) \in \mathcal{S}_k} m_{i,t} \mathcal{L}_{i,t}
$$

则：
$$
\sum_{k=1}^{M} \ell_k = \frac{1}{N_{\text{all}}} \sum_{k=1}^{M} \sum_{(i,t) \in \mathcal{S}_k} m_{i,t} \mathcal{L}_{i,t} = \mathcal{L}_{\text{token}}
$$

✅ 严格等价。

### 4.2 Sequence-level 的正确实现（以 seq-mean-token-mean 为例）

对第 $k$ 个 micro-batch：
$$
\ell_k = \frac{1}{B} \sum_{i \in \mathcal{S}_k} \bar{\mathcal{L}}_i
$$

则：
$$
\sum_{k=1}^{M} \ell_k = \frac{1}{B} \sum_{i=1}^{B} \bar{\mathcal{L}}_i = \mathcal{L}_{\text{seq-mean}}
$$

✅ 即使 $B_k$ 不固定（packing 场景），仍严格成立。

---

## 5. `loss_scale`：抵消 backend 的默认归一化

大多数训练框架（如 Megatron、FSDP）为保证梯度尺度稳定，在 DP + GA 下会对梯度做隐式归一化：

- **GA 维度**：对 $A$ 次 micro-step 的梯度取平均（等效于 `loss /= A`）；
- **DP 维度**：AllReduce 后除以 $D$（等效于跨 rank 求平均）。

综合效果等价于：
$$
g \propto \frac{1}{M} \sum_{k=1}^{M} \nabla \ell_k, \quad M = D \times A
$$

但 ROLL 的 aggregate 设计要求 **micro 间是求和语义**：
$$
\nabla \mathcal{L}_{\text{global}} = \sum_{k=1}^{M} \nabla \ell_k
$$

为抵消 backend 的 $1/M$ 归一化，需在每个 micro-batch 的 loss 上乘以：
$$
\text{loss\_scale} = M
$$

这样：
$$
\frac{1}{M} \sum_{k=1}^{M} \nabla (M \cdot \ell_k) = \sum_{k=1}^{M} \nabla \ell_k
$$

✅ 恢复了正确的求和语义。

---

## 6. ROLL 接口：全局统计量注入机制与 `loss_scale` 控制

在 ROLL 中，为了支持在 micro-batch 级别实现**全局等价的 loss 聚合**，框架会自动为每个训练 step 注入当前 global batch 的全局统计信息（如总有效 token 数、总有效样本数）。这些信息的**计算方式完全由用户通过 `loss_mask_keys` 指定**。

### 6.1 `loss_mask_keys`：定义 loss 参与范围，并驱动全局统计注入

`loss_mask_keys` 是一个字符串列表，用于声明 **哪些 mask 字段应被用于识别“参与 loss 计算的有效 token”**。该配置不仅指导 loss 函数如何屏蔽无效位置，更重要的是——**它直接决定了 strategy 如何统计并注入全局聚合量**。

你需要在 pipeline 的数据预处理或 worker 初始化阶段设置：
```python
data.meta_info['loss_mask_keys'] = ['response_mask', 'labels_mask']
```

对于 `loss_mask_keys` 中的每一个 key（例如 `'response_mask'`），ROLL 的 strategy 会：

1. **从 `data.batch` 中提取对应的 mask 张量**（形状通常为 `[batch_size, seq_len]`）；
2. **跨所有 DP rank 和 GA steps 收集该 mask**；
3. **计算两个全局统计量**：
   - **`batch_num_tokens[key]`**：该 mask 在整个 global batch 中的 **总和**，即  
     $$
     N_{\text{all}}^{(\text{key})} = \sum_{\text{all samples}} \sum_{t} \text{mask}_{i,t}^{(\text{key})}
     $$
   - **`global_valid_samples[key]`**：该 mask **至少有一个有效 token 的序列数量**，即  
     $$
     B^{(\text{key})} = \sum_{i=1}^{B} \mathbb{I}\left( \sum_{t} \text{mask}_{i,t}^{(\text{key})} > 0 \right)
     $$

这些统计量会被注入到 `data.meta_info` 中，供 `loss_func` 使用。

> ⚠️ **关键一致性要求**：你在 `loss_func` 中用于计算 loss、加权或聚合的 mask，**必须与 `loss_mask_keys` 中指定的 key 对应的 mask 语义完全一致**。  
> 例如，若 `loss_mask_keys = ['response_mask']`，则你的 loss 必须且只能基于 `response_mask` 来屏蔽 token；若实际使用了其他 mask（如 `attention_mask`），会导致分子（loss 计算）与分母（全局统计）不匹配，破坏等价性。

### 6.2 在 `loss_func` 中使用注入的全局统计量

在自定义 `loss_func` 中，你可以通过以下方式获取对应 mask 的全局统计量：

```python
# 假设 loss_mask_keys 包含 'response_mask'
mask_key = 'response_mask'

N_all = data.meta_info['batch_num_tokens'][mask_key]        # 全局有效 token 数
B_all = data.meta_info['global_valid_samples'][mask_key]    # 全局有效样本数
```

然后在聚合时直接使用这些全局值作为分母（见第 4 节），确保 micro-batch 的局部计算能精确还原 global loss。

### 6.3 `apply_loss_scale`：控制是否应用梯度尺度校正

由于训练 backend（如 Megatron/FSDP）在 DP + GA 下通常会对梯度做 $1/(D \times A)$ 的隐式归一化，而 ROLL 的聚合设计依赖**求和语义**，因此需要通过 `loss_scale = D \times A` 进行补偿。

在 `worker_config` 中，参数 `apply_loss_scale` 控制是否自动应用此缩放：

- **默认值：`True`**（推荐保持开启）
- **作用**：框架会自动将 `loss_func` 返回的 loss 乘以 `loss_scale`
- **何时关闭**：仅当你在 `loss_func` 内部已手动完成完整 global loss（含 scale）时才设为 `False`，一般不建议。

---

## 7. Metrics 记录：使用 `@sum` 语义

对于通过全局分母聚合的 loss，其 metrics 在多 worker reduce 时**不应取平均**，而应**求和**。

ROLL 支持在 metric 名称后添加 `@操作符` 来指定 reduce 方式：

```python
metrics = {
    "actor/kl_loss@sum": kl_loss.detach().item(),
}
reduce_metrics(metrics)
```

- `@sum`：reduce 时对所有 worker 的值求和；
- `@mean`（默认）：求平均；
- 日志记录时会自动过滤 `@` 及之后的内容，最终显示为 `actor/kl_loss`。

---

## 8. 代码示例：Actor 中 KL Loss 的全局等价实现

### 8.1 计算逐 token KL

```python
kl_loss = compute_approx_kl(
    log_probs=log_probs,
    log_probs_base=ref_log_probs,
    action_mask=final_response_mask,
    kl_penalty="k3"
)
```

### 8.2 调用聚合函数（使用全局分母）

```python
kl_loss = agg_loss(
    loss_mat=kl_loss,
    loss_mask=final_response_mask,
    loss_agg_mode=self.pipeline_config.loss_agg_mode,
    batch_num_tokens=batch_num_tokens['final_response_mask'],
    global_valid_samples=global_valid_samples['final_response_mask'],
)
```

### 8.3 `agg_loss` 关键实现

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
        seq_means = masked_mean(loss_mat, loss_mask, dim=-1)  # 自定义函数，支持 mask
        valid = (loss_mask.sum(dim=-1) > 0).float()
        loss = (seq_means * valid).sum() / (global_valid_samples + 1e-8)
    else:
        raise ValueError(f"Unsupported loss_agg_mode: {loss_agg_mode}")
    
    return loss
```

### 8.4 记录指标

```python
pg_metrics = {"actor/kl_loss@sum": kl_loss.detach().item()}
```

---

## 9. 设计建议：自定义 loss 实现 Checklist（⚠️ 所有注意事项汇总）

为确保 loss 在任意训练配置下保持数学等价性和训练稳定性，请严格遵循以下 checklist：

### ✅ **Loss 粒度与聚合模式**
- 明确你的 loss 是 **token-level** 还是 **sequence-level**。
- 根据需求选择正确的 `loss_agg_mode`（如 `"token-mean"`、`"seq-mean-token-mean"`）。

### ✅ **全局分母使用（核心！）**
- **禁止**在 micro-batch 内使用局部统计量（如 `loss_mask.sum()` 或 `loss_mat.shape[0]`）作为分母。
- **必须**使用 `data.meta_info['batch_num_tokens'][key]` 和 `data.meta_info['global_valid_samples'][key]` 提供的**全局统计量**。

### ✅ **`loss_mask_keys` 配置与一致性（极易出错！）**
- 在 pipeline 中显式设置 `data.meta_info['loss_mask_keys']`。
- **确保** `loss_func` 中用于计算/屏蔽/加权的 mask **与 `loss_mask_keys` 中指定的 key 完全对应**。
- 若使用多个 mask（如 response + labels），需全部列入 `loss_mask_keys`，并分别处理。

### ✅ **`apply_loss_scale` 设置**
- **保持默认 `True`**，除非你完全理解并接管了 scale 逻辑。
- 错误关闭会导致梯度被 backend 隐式缩小 $1/(D \times A)$ 倍，训练发散或收敛极慢。

### ✅ **Metrics 记录方式**
- 对使用全局分母聚合的 loss，**必须**在 metric 名称后加 `@sum`（如 `"loss@sum"`）。
- 否则 reduce 时取平均会导致 logged loss 值错误（偏小 $M$ 倍）。

### ✅ **Packing 场景特别注意**
- 不要假设 micro-batch 的样本数 $B_k$ 或 token 数 $N_k$ 固定。
- 所有聚合逻辑必须**不依赖 micro 内部统计量**，只依赖全局注入值。

---