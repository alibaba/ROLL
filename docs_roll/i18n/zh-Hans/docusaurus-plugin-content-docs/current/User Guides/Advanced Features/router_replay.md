# ROLL ROUTER REPLAY

ROLL 框架支持 **Router Replay**（路由复放）功能，用于解决 MoE（混合专家）模型在 RL 训练中由专家路由不一致引起的训练-推理失配问题。该特性在训练阶段强制使用预先记录的路由决策，从源头消除 MoE Router 在不同执行环境下的偏差，使训练更加稳定。

> **注意**：当前 ROLL 同时支持 **R2** 模式（Vanilla Routing Replay：Megatron old policy forward 记录 + Megatron 训练复放）与 **R3** 模式（Rollout Routing Replay：SGLang / vLLM 推理 + Megatron 训练）。两种模式均兼容 TP / PP / VPP / CP 并行、Dynamic Batching 与 `sequence_packing`。

## 1. 背景

### 1.1 MoE RL 中的路由不一致

MoE 模型每一层 Router 会为每个 token 选择 top-k 个专家。在 RL 训练里，同一份模型权重会被三个角色使用：

- **Rollout policy**：推理引擎（如 SGLang）中负责采样的 policy。
- **Old policy**：训练引擎中、本批次更新前的模型状态。
- **Training policy**：训练引擎中、正在做梯度更新的模型。

理想情况下三者应当给出一致的路由结果，但实际上：

- **训练 vs 推理**：推理引擎与训练引擎在算子实现、精度、并行布局上存在差异，即使权重相同，对同一输入也可能选出不同的 top-k 专家。
- **多次梯度更新内部**：随着 mini-batch 不断更新，路由也会随权重漂移。

### 1.2 路由不一致带来的影响

路由的离散选择会被后续 expert 的输出放大。当推理与训练侧选中的专家不同，token 级别的输出概率会出现明显偏差，进而：

- 放大 importance sampling 比，使 PPO/GRPO 的有效样本被严重 clip 或方差激增；
- 在 off-policy 比例较大的场景下出现训练崩溃；
- 让 IS 修正、TIS 等 loss 层补偿手段难以单独解决问题。

Router Replay 的思路是：**与其在 loss 层做事后修正，不如在模型架构层固定路由 mask，让训练侧直接复用一份"参考路由"**，从源头去掉这一差异。

## 2. 实现原理

### 2.1 复放公式

无论是 R2 还是 R3，做法都可以抽象成：在训练 forward 中，把原本由 router logits 决定的 top-k mask 替换为外部传入的 mask $I_{\text{ref}}$，再用训练侧的 logits $s_{\text{train}}$ 重新归一化得到专家权重：

$$
g_i = \frac{I_{\text{ref}, i} \cdot \exp(s_{\text{train}, i})}{\sum_j I_{\text{ref}, j} \cdot \exp(s_{\text{train}, j})}
$$

要点：

- 选哪几个专家由 $I_{\text{ref}}$ 决定，不再由训练侧 argmax 决定；
- 但 softmax 仍作用在训练侧 logits 上，因此 router 权重的梯度可以正常回传。

R2 与 R3 的差别仅在于 $I_{\text{ref}}$ 的来源不同。

### 2.2 R2 — Vanilla Routing Replay

R2 的路由记录来自训练引擎自身。每个批次开始时，Megatron 会先用 old policy 重算一遍 old log probs，R2 正是在这次 forward 中把每个 MoE 层选出的 top-k 专家记录下来，并在该批次随后的多次梯度更新中原样复放。这样同一批数据无论经历多少轮更新，使用的都是同一份路由，消除了路由随权重漂移带来的噪声；其中第一个 mini-batch 的模型尚未更新，复放结果与原始 forward 完全一致，等价于 on-policy。

使用上只需在 `actor_train` 设置 `router_replay.mode: R2`，并在 pipeline 级开启 `enable_old_logprobs_recompute: true`（缺少该开关不会发生任何记录，训练步会因拿不到路由数据而失败），推理侧无需任何配置。需要注意的是，R2 只保证训练侧内部的路由一致性，并不解决推理引擎与训练引擎之间的路由差异；如需与 rollout 路由对齐，请使用 R3。

### 2.3 R3 — Rollout Routing Replay（推荐）

- $I_{\text{ref}}$ 直接来自 **推理引擎在 rollout 时记录的路由**（SGLang ≥ 0.5.6.post3，或 ROLL 打过 patch 的 vLLM 集成）。
- 训练侧拿到的是与采样轨迹严格对齐的专家选择，因此 training-inference 的路由差异被彻底消除。
- 同时也限制了多次更新中的路由漂移（与 R2 同方向受益）。

R3 是同时缓解 training-inference discrepancy 与 policy staleness 的更强方案，是 ROLL 当前的默认推荐路径。

### 2.4 R3 在 ROLL 中的端到端流程

```
┌─────────────────────────────┐         ┌──────────────────────────────┐
│  Rollout（SGLang / vLLM）   │         │      Megatron Training       │
│                             │         │                              │
│  generate(...)              │         │  forward()                   │
│   └─ MoE Router 计算 top-k   │         │   └─ MoE RouterReplay        │
│        └─ 同步导出索引       │ ──────► │        └─ 用导出的索引       │
│           [seq, layers, k]  │ batch   │           参与 forward       │
│                             │  data   │                              │
│  返回 routed_experts        │         │  forward → backward          │
└─────────────────────────────┘         └──────────────────────────────┘
```

1. **采样阶段**：推理引擎在生成 token 时，额外记录每一层 MoE 的 top-k 专家，形成形如 `[seq_len, num_layers, top_k]` 的 `routed_experts` 张量并随响应返回。
2. **数据搬运**：ROLL 自动把 `routed_experts` 挂到 batch 的每条样本上，沿标准数据通路（DP / mini-batch / micro-batch）流向训练 worker，无需用户干预。
3. **训练阶段**：Megatron 在 forward 前把记录的索引装载到每个 MoE 层；Router 跳过自己的 top-k 计算，直接复放这份索引，且在 backward 与激活重计算中也使用同一份路由。

### 2.5 Router Replay 与 Sequence Packing 的组合

**R2/R3 可以与 `sequence_packing` 同时启用。** 路由记录按样本组织，而 sequence packing 会把多条样本拼接成 packed 序列；ROLL 会自动完成两套布局的转换，把 `routed_experts` 随 `input_ids` 一起重新打包，保证在任意 TP / CP 配置下每个 token 都保留自己的路由记录。

无需任何额外配置：`use_sequence_packing` 与不开 Router Replay 时保持一致即可。

### 2.6 兼容性矩阵

| 特性                               | R2                          | R3                                    |
|------------------------------------|-----------------------------|---------------------------------------|
| Megatron `megatron_train`          | 必需                        | 必需                                  |
| Rollout 引擎                       | 不参与（保持 `disable`）    | `sglang` ≥ 0.5.6.post3 或 patched vLLM |
| 张量并行 TP                        | 支持                        | 支持                                  |
| 流水并行 PP                        | 支持                        | 支持                                  |
| Virtual Pipeline Parallelism (VPP) | 支持                        | 支持                                  |
| Context Parallelism (CP)           | 支持                        | 支持                                  |
| Dynamic Batching                   | 支持                        | 支持                                  |
| **Sequence Packing**               | **支持**                    | **支持**                              |
| GSPO                               | 正交，可叠加                | 正交，可叠加                          |
| TIS / IS 修正                      | 可共存，收益视场景而定      | 可共存，收益视场景而定                |
| FSDP / DeepSpeed 训练              | 不支持                      | 不支持                                |

## 3. 实现流程

### 3.1 Rollout 端（SGLang / vLLM）

在 `actor_infer` 上设置 `router_replay.mode: R3` 后，ROLL 会自动：

- 以导出路由记录的方式启动 SGLang 服务，并从每个响应中收集、按样本拼装 `routed_experts`（要求 SGLang `>= 0.5.6.post3`）；
- 或者，在使用 ROLL patched vLLM 集成时，把每个 completion 携带的 `routed_experts` 转换为相同的按样本张量格式。

### 3.2 训练端（Megatron）

当 `router_replay.mode` 不为 `disable` 时，Megatron 训练侧会自动：

- 初始化时在每个 MoE 层上启用路由复放；
- **R3**：forward 时 batch 中带有 `routed_experts` 就复放；未携带的 batch（如 reference 模型）仍按默认路由运行；训练步要求 `routed_experts` 必须存在，缺失时会直接报错；
- **R2**：old policy forward 记录每个 MoE 层的路由，各并行 rank 上的记录被合并为按样本组织的 `routed_experts` 张量挂回 batch，训练步随后按 R3 相同的方式复放；
- backward 的激活重计算复用同一份记录，保证前向与反向一致。

为控制 `routed_experts`（大型 MoE 模型下每批次可达几十 MB）带来的显存与传输开销，ROLL 会在 reorder、分组、advantage 计算等中间阶段将其暂存旁路，仅在 actor 训练时重新挂回，并按分块传输而非一次性发送整个对象。

### 3.3 核心工具

实现集中在 `roll/third_party/megatron/router_replay_utils.py`，内部细节均由其处理：把记录的索引按 sequence parallel rank 分发、跨 pipeline stage 收集合并 R2 记录，并按模型专家数自动选择最小的整数类型（`uint8` / `uint16` / `uint32`）以压缩显存与传输开销。

## 4. 配置参数

### 4.1 如何启用

在需要参与的 worker 上设置 `router_replay.mode`：

- **R3** 必须**同时**作用于 rollout（`actor_infer`）与训练端（`actor_train`）。只在推理侧开启没有任何效果；只在训练侧开启则会因为收不到路由数据而导致训练失败。
- **R2** 只在 `actor_train` 上开启；rollout 侧保持 `disable`。

### 4.2 参数说明

#### `router_replay.mode`

- **`disable`**（默认）：关闭路由复放。
- **`R2`**：Vanilla Routing Replay —— Megatron old policy forward 记录路由，并在同一批数据的梯度更新中复放。要求 `enable_old_logprobs_recompute: true`。
- **`R3`**：Rollout Routing Replay —— 复放 SGLang / vLLM rollout 引擎记录的路由。

### 4.3 配置示例

R3 + SGLang rollout（`sequence_packing` 开关与否均可）：

```yaml
actor_train:
  router_replay:
    mode: R3
  strategy_args:
    strategy_name: megatron_train
    strategy_config:
      moe_enable_routing_replay: true  # 让每个 MoE 层以支持复放的方式构建

actor_infer:
  router_replay:
    mode: R3
  strategy_args:
    strategy_name: sglang  # 需要 sglang >= 0.5.6.post3；也可使用 patched vllm

reference:
  router_replay:
    mode: disable
  strategy_args:
    strategy_name: megatron_infer
```

R2（在 Megatron old policy forward 上记录，rollout 不参与）：

```yaml
# R2 在 old log probs 重算阶段记录
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

### 4.4 使用建议

1. **环境与策略**：R3 下 `actor_infer` 必须使用 `sglang`（≥ 0.5.6.post3）或 patched `vllm` 集成；`actor_train` 必须使用 `megatron_train`，R2 只涉及 `actor_train`。两种模式下都需要在 `actor_train` 的 `strategy_config` 中设置 `moe_enable_routing_replay: true`，让每个 MoE 层以支持复放的方式构建。
2. **对称启用（R3）**：rollout 与训练两端需同时配置 `mode: R3`。只在推理侧开启等于没开；只在训练侧开启会因缺少路由数据而训练失败。
3. **Reference 模型**：保持 `mode: disable`。batch 中没有 `routed_experts` 时，ROLL 自动跳过复放逻辑。
4. **Sequence packing**：R2/R3 与 `use_sequence_packing` 开或关都兼容，无需额外操作。
5. **资源开销**：`routed_experts` 张量为 `[seq_len, num_layers, top_k]`，会带来额外显存与跨 worker 传输；ROLL 自动选用最紧凑的整数类型，并尽量减少它随 batch 流转的时间，实际开销较为有限。
6. **与 IS / TIS 的关系**：Router Replay 在架构层消除路由差异，IS / TIS 在 loss 层修正概率差异，二者并不冲突，可视场景共存使用。
7. **排查路径**：先看启动日志——训练 worker 会打印 `Router Replay <mode> mode: REPLAY enabled`，R2 下还应看到 `RECORD enabled` 字样。若仍出现训练-推理不一致，依次检查 (a) rollout 响应是否真的带回了 `routed_experts`；(b) R3 下 `actor_train` 与 `actor_infer` 是否都配置了 `mode: R3`，且 SGLang 版本 ≥ 0.5.6.post3（或使用 patched vLLM）；(c) R2 下是否设置了 `enable_old_logprobs_recompute: true`，否则不会发生任何记录，训练步会因缺少路由数据而失败。

启用 Router Replay 后，ROLL 在 TP / PP / VPP / CP 等并行配置下都能保证 MoE 路由严格对齐 —— R3 对齐 rollout 与训练，R2 约束多次梯度更新中的路由漂移 —— 从模型架构层面消除一类难以靠 loss 修正解决的不一致来源。
