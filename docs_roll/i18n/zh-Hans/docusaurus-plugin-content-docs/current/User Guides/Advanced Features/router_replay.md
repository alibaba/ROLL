# ROLL ROUTER REPLAY

ROLL 框架支持 **Router Replay**（路由复放）功能，用于解决 MoE（混合专家）模型在 RL 训练中由专家路由不一致引起的训练-推理失配问题。该特性在训练阶段强制使用预先记录的路由决策，从源头消除 MoE Router 在不同执行环境下的偏差，使训练更加稳定。

> **注意**：当前 ROLL 仅实现了 **R3** 模式（Rollout Routing Replay：SGLang 推理 + Megatron 训练）。R2 暂未实现；R3 与 `sequence_packing` 的组合也暂未支持，后续版本会逐步补齐，目前请勿同时启用。

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

### 2.2 R2 — Vanilla Routing Replay（暂未实现）

- $I_{\text{ref}}$ 来自 **训练引擎自己用 old policy 跑一次 forward** 记录下的路由。
- 第一个 mini-batch 时 $\theta = \theta_{\text{old}}$，复放结果与原始 forward 一致，等价于 on-policy。
- 后续 mini-batch 在 off-policy 场景下，复放固定了路由，从而约束 policy staleness。

R2 解决的是"训练侧自身在多次更新中路由漂移"的问题，但**无法解决推理引擎与训练引擎之间的差异**。在 ROLL 中 R2 暂未实现，配置 `mode: R2` 会抛出 `NotImplementedError`。

### 2.3 R3 — Rollout Routing Replay（ROLL 当前支持）

- $I_{\text{ref}}$ 直接来自 **推理引擎在 rollout 时记录的路由**。
- 训练侧拿到的是与采样轨迹严格对齐的专家选择，因此 training-inference 的路由差异被彻底消除。
- 同时也限制了多次更新中的路由漂移（与 R2 同方向受益）。

R3 是同时缓解 training-inference discrepancy 与 policy staleness 的更强方案，是 ROLL 当前的默认推荐路径。

### 2.4 R3 在 ROLL 中的端到端流程

```
┌─────────────────────────────┐         ┌──────────────────────────────┐
│        SGLang Rollout       │         │      Megatron Training       │
│                             │         │                              │
│  generate(...)              │         │  forward()                   │
│   └─ MoE Router 计算 top-k   │         │   └─ MoE RouterReplay        │
│        └─ 同步导出索引       │ ──────► │        └─ 用导出的索引       │
│           [seq, layers, k]  │ batch   │           参与 forward       │
│                             │  data   │                              │
│  返回 routed_experts        │         │  forward → backward          │
└─────────────────────────────┘         └──────────────────────────────┘
```

1. **采样阶段**：SGLang 在生成 token 时，额外记录每一层 MoE 的 top-k 专家，形成形如 `[seq_len, num_layers, top_k]` 的 `routed_experts` 张量并随响应返回。
2. **数据搬运**：rollout 后处理把 `routed_experts` 挂到 batch 的每条样本上，沿 ROLL 标准的数据通路（DP / mini-batch / micro-batch）流向训练 worker。
3. **训练阶段**：Megatron 侧在 forward 前根据 `routed_experts` 设置每层 RouterReplay 的索引；MoE Router 跳过 top-k 计算，直接复放外部索引；forward 完成后切换为"backward 复放"，确保激活重计算下也用同一份路由。

### 2.5 R3 与 Sequence Packing 暂未支持组合使用

`routed_experts` 是按"每条样本独立、原始 attention_mask 布局"组织的，而 `sequence_packing` 会把多个序列拼接、按 `2 × CP_SIZE × TP_SIZE` 重新对齐、并按 CP 交错切分。两套布局当前没有打通，因此：

- ROLL 暂不支持 R3 与 `sequence_packing` 同时启用；
- 启用 R3 时请保持 `use_sequence_packing: False`；
- 后续版本会补齐两者间的索引重映射，使其可以协同工作。

### 2.6 兼容性矩阵

| 特性                               | R3 兼容性                  |
|------------------------------------|----------------------------|
| Megatron `megatron_train`          | 支持                       |
| SGLang `sglang` rollout            | 支持（必需）               |
| 张量并行 TP                        | 支持                       |
| 流水并行 PP                        | 支持                       |
| Virtual Pipeline Parallelism (VPP) | 支持                       |
| Dynamic Batching                   | 支持                       |
| **Sequence Packing**               | **暂未支持，后续会支持**   |
| GSPO                               | 正交，可叠加               |
| TIS / IS 修正                      | 可共存，收益视场景而定     |
| vLLM rollout                       | 不支持                     |
| FSDP / DeepSpeed 训练              | 不支持                     |

## 3. 实现流程

### 3.1 Rollout 端（SGLang）

`sglang_strategy.py` 中，当 `router_replay.mode != "disable"` 时：

- SGLang 服务以 `enable_return_routed_experts=True` 启动；
- 每个 generate 请求带上 `return_routed_experts=True`；
- 多 chunk 输出从 `meta_info` 中收集 `routed_experts`，按样本拼装；
- 要求 SGLang `>= 0.5.6.post3`。

### 3.2 训练端（Megatron）

`megatron_strategy.py` 中：

- 初始化时强制 `moe_enable_routing_replay=True`，让每个 MoE 层带上 `RouterReplay` 实例；
- `compute_log_probs` 类 forward：仅当 batch 包含 `routed_experts` 时才进入复放，否则（如 reference model）走默认路由；
- `train_step`：断言 `routed_experts` 存在，全局设置 `REPLAY_FORWARD`，forward/backward 完成后清理全局状态；
- `inner_forward_step`：通过 `set_router_replay_data` 把记录的索引按 SP rank 分发；forward 后切换为 `REPLAY_BACKWARD`，保证激活重计算正确。

### 3.3 核心工具

`roll/third_party/megatron/router_replay_utils.py` 提供：

- `set_router_replay_data` — 将 top-k 索引按 SP rank 分发；
- `RouterReplayHelper` — 跨 PP / VPP 查询和切换每层 RouterReplay 状态；
- `get_routed_experts_dtype` — 按专家数量自动选 `uint8` / `int16`，压缩传输与显存。

## 4. 配置参数

### 4.1 如何启用

在需要参与 R3 的 worker 上设置 `router_replay.mode: R3`。R3 必须**同时**作用于 rollout 与训练两端，缺一不可。

### 4.2 参数说明

#### `router_replay.mode`

- **`disable`**（默认）：关闭路由复放。
- **`R2`**：Vanilla Routing Replay。**当前未实现**，配置该模式会抛出 `NotImplementedError`。
- **`R3`**：Rollout Routing Replay，ROLL 当前唯一可用的模式。

### 4.3 完整配置示例

```yaml
actor_train:
  router_replay:
    mode: R3

  strategy_args:
    strategy_name: megatron_train

  # R3 暂未支持与 sequence_packing 组合，请保持关闭
  use_sequence_packing: False

actor_infer:
  router_replay:
    mode: R3

  strategy_args:
    strategy_name: sglang  # 需要 sglang >= 0.5.6.post3

reference:
  router_replay:
    mode: disable
  strategy_args:
    strategy_name: megatron_infer
```

### 4.4 使用建议

1. **环境与策略**：`actor_infer` 必须使用 `sglang`（≥ 0.5.6.post3）；`actor_train` 必须使用 `megatron_train`。
2. **对称启用**：rollout 与训练两端同时配置 `mode: R3`；只在一端开启没有意义。
3. **Reference 模型**：保持 `mode: disable`。batch 中没有 `routed_experts` 时，ROLL 自动跳过复放逻辑。
4. **暂时关闭 sequence packing**：在 R3 + sequence_packing 支持上线前，所有相关 worker 都需保持 `use_sequence_packing: False`。
5. **资源开销**：`routed_experts` 张量为 `[seq_len, num_layers, top_k]`，会带来额外显存与跨 worker 传输；ROLL 自动选择最小整数 dtype 以缓解。
6. **与 IS / TIS 的关系**：Router Replay 在架构层消除路由差异，IS / TIS 在 loss 层修正概率差异，二者并不冲突，可视场景共存使用。
7. **排查路径**：若仍出现训练-推理不一致的现象，依次检查 (a) rollout 响应是否真的带回了 `routed_experts`；(b) 训练侧 `moe_enable_routing_replay` 是否为 `True`；(c) 是否所有相关 worker 都关闭了 sequence packing。

启用 Router Replay（R3）后，ROLL 在 TP / PP / VPP 等并行配置下都能保证 rollout 与训练的 MoE 路由严格对齐，从模型架构层面消除一类难以靠 loss 修正解决的不一致来源。后续版本会陆续补齐 R2 与 R3 + sequence packing 等组合能力。
