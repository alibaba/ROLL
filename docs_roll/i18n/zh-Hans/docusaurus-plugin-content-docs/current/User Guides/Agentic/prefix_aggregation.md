# 前缀聚合 (Prefix Aggregation)

## 概述

在 Agentic 训练场景中，一个 episode 通常包含多轮交互 (multi-turn)。默认的 step-wise 模式为每个 step 生成一个独立的训练样本，每个样本的 prompt 包含完整的历史前缀。当 episode 较长时（如 30-turn 的 SWE 任务），会导致 **样本爆炸** 和 **重复 tokenization** 问题。

前缀聚合方案通过 `MessageTracker` 组件，将多个 step 聚合为 **traj-wise** 的单条训练样本，同时支持增量 tokenization 和历史分叉检测。

### 核心能力

- **Traj-wise 聚合**：将整个 episode 的多轮交互聚合为一条训练样本，消除前缀重复
- **增量 Tokenization**：通过哈希链缓存已处理的消息，避免每一步对完整历史重复 tokenize
- **分叉检测**：自动检测 agent 框架修改历史消息的情况，支持多分支训练样本生成
- **在离线一致性**：assistant 的 token_ids 直接使用推理引擎输出，保证训练与生成一致

## 问题背景

### 问题 1: Step-wise 样本爆炸

默认模式下，`ProxyEnvManager.formulate_rollouts()` 为每个 step 生成一个独立的 DataProto 样本。每个样本的 prompt 包含从第一轮到当前轮的所有历史消息。30-turn episode 会产生 30 个样本，总 token 数约为 `O(n²)` 级别。

```
Step 1: [sys, user] → asst1                     # prompt: 2 msgs
Step 2: [sys, user, asst1, tool1] → asst2        # prompt: 4 msgs (重复 step1 的 2 msgs)
Step 3: [sys, user, asst1, tool1, asst2, tool2] → asst3  # prompt: 6 msgs (重复 step1-2 的 4 msgs)
...
Step N: [全部历史] → asstN                        # prompt: 2N msgs
```

Traj-wise 模式将上述 N 个样本聚合为一条：

```
[sys, user, asst1, tool1, asst2, tool2, ..., asstN]  # 单条样本，无重复
```

### 问题 2: 重复 Tokenization

每次 `process_request()` 调用时，需要对完整消息历史执行 `apply_chat_template()` 进行 tokenize。随着轮次增长，重复计算量线性增加。

`MessageTracker` 通过哈希链缓存每条消息的 token_ids，仅对新增消息执行 tokenize。

### 问题 3: 历史分叉 (Fork)

Agent 框架（如 claude-code）可能在 step 之间修改历史消息（截断、规整、总结等），导致消息历史发生分叉。

```
Step 1: [sys, user] → asst1
Step 2: [sys, user, asst1, tool1] → asst2              ← 正常延续
Step 3: [sys, user, asst1_modified, tool1'] → asst3     ← 分叉! asst1 被修改
Step 4: [sys, user, asst1_modified, tool1', asst3, tool2] → asst4
```

此时产生两个分支的训练样本：

- **分支 1** (dead branch): `sys → user → asst1 → tool1 → asst2`
- **分支 2** (final branch): `sys → user → asst1_modified → tool1' → asst3 → tool2 → asst4`

### 在离线一致性分析

前缀聚合中，assistant 的 token_ids 直接使用推理引擎缓存的输出（`response_ids`），而非重新 tokenize，保证 assistant 部分训练与生成完全一致。

Left context 使用逐条消息 dummy prefix + slice 方式独立 tokenize 后拼接。与推理时 sglang 对完整消息列表做 `apply_chat_template` 的结果可能存在 tokenizer 边界差异，但由于 chat template 结构化的特性，实际影响可控。

## 配置参数

前缀聚合通过 `env_config.config` 配置，相关参数如下：

```yaml
custom_envs:
  my_env:
    env_type: "my_env_type"
    max_steps: 30
    max_tokens_per_step: 4096
    env_manager_cls: ProxyEnvManager
    config:
      trajectory_mode: traj       # "step" (默认) | "traj"
      enable_thinking: false      # 是否启用 thinking 模式
      enable_fork: false          # 是否启用分叉多分支输出
```

### 参数说明

- `trajectory_mode`: 轨迹输出模式
  - `"step"` (默认): 每个 assistant response 产生一条训练样本，与原有逻辑兼容
  - `"traj"`: 将整个 episode 聚合为 traj-wise 样本，消除前缀重复
- `enable_thinking`: 是否在 `apply_chat_template` 中启用 thinking 模式，影响 tokenization
- `enable_fork`: 分叉处理策略
  - `false` (默认): 仅保留最长分支，丢弃 dead branches
  - `true`: 保留所有分支，每个分支输出一条训练样本

## 实现方案

### 核心组件: MessageTracker

`MessageTracker` 位于 `roll/pipeline/agentic/env_manager/message_tracker.py`，负责单个 episode 的消息追踪、增量 tokenization 和训练数据生成。

#### 数据结构

```python
@dataclass
class MessageItem:
    index: int                                # 消息在会话中的位置
    message: dict                             # 统一化后的消息 dict
    token_ids: Optional[List[int]] = None     # 该消息的 token IDs
    logprobs: Optional[List[float]] = None    # 仅模型生成的 assistant 消息有
```

`MessageTracker` 内部维护以下核心数据结构：

- `msg_item_dict: Dict[str, MessageItem]` — 哈希 → 消息项映射
- `prev_hash_dict: Dict[str, str]` — 哈希链，记录每条消息的前驱哈希
- `tool_calls_dict: Dict[str, dict]` — tool_call ID → 规范化 tool_call 对象
- `step_response_hashes: List[str]` — 每个 step 的 response 哈希，用于分叉检测

#### 消息统一化

所有消息经过 pydantic `ChatCompletionMessageParam` 校验后统一化：

1. `unify_content()` — content 归一化为 list-of-dicts 格式，合并连续 text 项
2. `unify_tool_calls()` — 用 `tool_calls_dict` 中的规范版本替换 tool_calls
3. `get_unified_message()` — 组合以上步骤

#### 哈希链去重

每条消息通过 SHA-256 链式哈希标识：`hash(index, message_fields, pre_hash)`。

当新请求到达时，从头遍历消息列表，逐条计算哈希并查找 `msg_item_dict`。匹配的消息直接复用缓存的 token_ids，仅对新增消息执行 tokenize。

#### 增量 Tokenization

单条消息 tokenize 使用 dummy prefix + slice 方法：

```python
# system 消息直接 tokenize
token_ids = tokenizer.apply_chat_template([system_msg], ...)

# 其他消息: dummy prefix + slice
full_ids = tokenizer.apply_chat_template(
    [*base_chat_history, msg],   # base_chat_history = [dummy_sys, dummy_user]
    ...
)
token_ids = full_ids[base_offset:]   # 去掉 dummy prefix 部分
```

`base_offset` 是 dummy prefix 的 token 长度，依赖于 tools 定义（因为 `apply_chat_template` 会将 tool 定义注入 system 部分）。当 tools 变化时自动重新计算。

#### 分叉检测

每次 `process_messages()` 调用时，将当前匹配的消息数与上一步记录的总消息数对比：

- 匹配数 ≥ 上一步消息数 → 正常延续
- 匹配数 < 上一步消息数 → 历史被修改，发生分叉

分支识别算法：按顺序检查 `step_response_hashes`，连续 step 属于同一分支**当且仅当**前一个 step 的 response_hash 是后一个 step 哈希链的祖先。

#### Response Mask 规则

训练样本中的 `response_mask` 决定哪些 token 参与 loss 计算：

- **模型生成的 assistant 消息** (通过 `record_response()` 记录，有 logprobs) → `response_mask = 1`
- **所有其他消息** (system, user, tool, 以及分叉后被 agent 框架修改的 assistant 消息) → `response_mask = 0`

这确保了分叉分支中，agent 框架修改的 "继承" 消息不会被错误地算入 loss。

### 与 ProxyEnvManager 集成

#### 生命周期

```
每个 episode 开始
  → 创建新的 MessageTracker 实例

每次 process_request() 调用
  → message_tracker.process_messages(messages, tools)     # 增量 tokenize
  → LLM 推理
  → message_tracker.record_response(resp_msg, response_ids, logprobs)  # 记录输出

episode 结束 (formulate_rollouts)
  → message_tracker.get_trajectory_data(mode=trajectory_mode)  # 提取训练数据
  → 构建 DataProto 样本
```

#### 奖励分配

- **Step 模式**: reward 仅分配给最后一步，前面的步骤 reward 为 0
- **Traj 模式**: 每个分支共享 `episode_score`，reward 放在样本最后一个 token 位置

#### Newline Token 处理

在 assistant 消息后面紧邻的消息前插入一个 `\n` token，保持与 chat template 的格式一致。

## 两种模式对比

| 特性         | Step 模式 (`"step"`)           | Traj 模式 (`"traj"`)                    |
| ------------ | -------------------------------- | ----------------------------------------- |
| 样本数量     | 每个 step 一条                   | 每个 episode 一条 (或分叉时多条)          |
| 前缀重复     | 有重复 (O(n²) tokens)           | 无重复 (O(n) tokens)                      |
| Tokenization | 每步全量 `apply_chat_template` | 增量 tokenize，缓存复用                   |
| 在离线一致性 | 完全一致                         | assistant 部分一致，left context 微小差异 |
| 分叉支持     | 不涉及                           | 支持 (`enable_fork` 控制)               |
| 适用场景     | 短交互、调试                     | 长交互 (SWE, 多轮 agent)                  |

## 使用示例

### 基本配置 (Traj 模式)

```yaml
custom_envs:
  swe_env:
    env_type: "swe_env"
    max_steps: 50
    max_tokens_per_step: 8192
    env_manager_cls: ProxyEnvManager
    config:
      trajectory_mode: traj
      enable_thinking: false
      enable_fork: false
      dataset_name: my_swe_dataset
```

### 启用分叉检测

```yaml
custom_envs:
  complex_agent_env:
    env_type: "complex_agent"
    max_steps: 30
    max_tokens_per_step: 4096
    env_manager_cls: ProxyEnvManager
    config:
      trajectory_mode: traj
      enable_fork: true
      dataset_name: my_agent_dataset
```

## 注意事项

1. **Tools 变化**: 如果 episode 过程中 tools 定义发生变化，`base_offset` 会自动重新计算。但频繁变化的 tools 可能影响缓存命中率
2. **序列长度**: traj 模式下整个 episode 的所有 token 聚合在一条样本中，需要确保 `sequence_length` 配置足够大
3. **Step 模式兼容**: 设置 `trajectory_mode: step` 时行为与原有逻辑完全一致，可用于回退和调试
4. **Assistant 消息 tokenize 警告**: 如果 `process_messages()` 中遇到未通过 `record_response()` 记录的 assistant 消息（如 agent 框架修改后的消息），会记录 warning 日志并使用 tokenizer 重新 tokenize
