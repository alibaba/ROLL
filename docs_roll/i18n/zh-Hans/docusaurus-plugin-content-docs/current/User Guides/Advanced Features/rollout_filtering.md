# Rollout 数据过滤与动态补采样

## 1. 概述

在强化学习训练中，并非所有 rollout 数据都适合直接进入训练。例如：

- 同一 prompt 的所有回答奖励完全相同，无法提供有效的组内相对优势；
- 轨迹因环境超时、执行失败或格式错误而不完整；
- 推理概率与训练概率偏差过大，部分 token 不适合参与梯度计算；
- 输入 prompt 超过模型允许的最大长度。

ROLL 为这些场景提供了多层过滤能力。需要特别区分以下三种机制：

| 过滤机制 | 适用 Pipeline | 过滤单位 | 是否丢弃 rollout | 是否动态补采样 |
| --- | --- | --- | --- | --- |
| RLVR 动态过滤 | RLVR | response 或同一 prompt 的 response group | 是 | 是 |
| Agentic 轨迹组过滤 | Agentic | 同一环境配置和 seed 的 trajectory group | 是 | 是 |
| Train-Infer Correction Filter | RLVR、Agentic | token、segment 或 sequence mask | 否 | 否 |

前两种机制发生在 rollout scheduler 中：不合格的数据不会提交给训练端，scheduler 会继续采样，直到获得足够的数据。第三种机制发生在训练计算阶段，只修改 loss mask，不会从 batch 中删除样本。

## 2. RLVR 动态过滤

### 2.1 数据层次

RLVR 通常对一个 prompt 生成多个 response：

```text
Prompt
├── Response 0 ── response reward
├── Response 1 ── response reward
├── ...
└── Response N ── response reward
       │
       ├── response_filter：检查单条 response
       └── query_filter：检查同一 prompt 的完整 response group
```

默认 rollout loop 位于：

```text
roll/distributed/scheduler/user_defined_rollout_loop.py
```

它依次完成以下工作：

1. 从数据集取得 prompt；
2. 生成一个或多个 response；
3. 计算 reward；
4. 执行 response 级过滤；
5. 执行 prompt/query 级过滤；
6. 将保留的数据提交到 `ReplayBuffer`。

任何阶段返回 `None`，都会终止当前 prompt 的事务，并让 scheduler 继续处理新的 prompt。

### 2.2 Query 级过滤

Query 级过滤检查同一 prompt 下所有 response 的 `response_level_rewards`。默认入口为：

```python
def query_filter(data_list: list[DataProto], config: RLVRConfig) -> bool:
    ...
```

其返回值语义为：

- `True`：保留这个 prompt group；
- `False`：过滤这个 prompt group。

过滤配置属于具体 reward domain：

```yaml
rewards:
  math:
    query_filter_config:
      type: mean_filter
      filter_args:
        threshold_down: 0.0
        threshold_up: 1.0
```

ROLL 根据样本 `non_tensor_batch["domain"]` 找到对应的 `rewards.<domain>` 配置，因此不同 domain 可以使用不同规则。

当前内置类型如下：

| `type` | 行为 |
| --- | --- |
| `no_filter` | 始终保留 |
| `mean_filter` | reward 均值小于等于下界或大于等于上界时过滤 |
| `std_filter` | reward 标准差小于等于阈值时过滤 |

`mean_filter` 常用于 GRPO/RLVR：过滤全部答错或全部答对的 group，避免训练缺少有效的组内相对优势信号。

`std_filter` 示例：

```yaml
rewards:
  math:
    query_filter_config:
      type: std_filter
      filter_args:
        std_threshold: 0.01
```

:::note
默认 `query_filter()` 在输入列表只有一个元素时直接保留。使用该接口时，应传入按 response 拆分后的 `list[DataProto]`，而不是把包含多条 response 的单个 `DataProto` 包在长度为 1 的列表中。
:::

### 2.3 Response 级过滤

Reward 计算完成后，默认 rollout loop 会逐条调用：

```python
response_filter(batch_item, pipeline_config)
```

如果某些 response 被过滤，ROLL 会重新发送原始请求，只补采被过滤的 response 数量。为避免异常过滤规则造成死循环，默认实现最多重试 5 轮。

当前版本中，默认函数仍为：

```python
def response_filter(data_item, config):
    return True
```

因此默认情况下不会过滤任何单条 response。虽然 `RewardConfig` 已定义 `response_filter_config`，但默认函数尚未读取该配置。仅在 YAML 中填写 `response_filter_config` 不会自动生效。

需要 response 级过滤时，应自定义 `UserDefinedRolloutLoop`，并在生成、reward 与返回结果之间实现对应规则。例如：

```python
from roll.distributed.scheduler.user_defined_rollout_loop import UserDefinedRolloutLoop


class MyRolloutLoop(UserDefinedRolloutLoop):
    async def _generate_and_reward_impl(self, context, req, domain):
        responses = await super()._generate_and_reward_impl(context, req, domain)
        if responses is None:
            return None

        return [
            item
            for item in responses
            if not item.meta_info.get("invalid_format", False)
        ]
```

自定义后通过顶层配置加载：

```yaml
user_defined_rollout_loop_cls: my_package.rollout_loop.MyRolloutLoop
```

实现自定义逻辑时，应确保返回数量满足 scheduler 对 `num_return_sequences` 的约束。若需要动态改变返回数量，还需要正确配置 additional prompts 机制。

### 2.4 Prompt 或数据集级过滤

输入数据也可以在生成前过滤。例如 RLVR VLM Pipeline 会检查 tokenized prompt 长度：

```python
request_data, domain = context.get_request_data(meta_info=context.meta_info)
if request_data.batch["input_ids"].shape[1] > context.prompt_length:
    return None
```

返回 `None` 后，当前 prompt 不会进入生成和训练流程，scheduler 会继续取下一条数据。

:::caution
`RLVRConfig.dataset_filter` 当前仅定义了配置结构，尚未接入默认 RLVR rollout 主调用链。通用数据过滤应在数据集构造阶段完成，或通过自定义 `UserDefinedRolloutLoop.process_new_prompt()` 实现。
:::

### 2.5 过滤后的补采样

RLVR 使用 `ReplayBuffer` 管理 prompt 的事务状态：

```text
process_new_prompt
        │
        ├── 返回 responses
        │       └── ReplayBuffer.commit(prompt_id, responses)
        │
        └── 返回 None
                └── ReplayBuffer.abort(prompt_id)
                            │
                            └── 释放并发配额，调度新 prompt
```

因此 RLVR filter 不是对已经构造好的训练 batch 做布尔索引。被过滤的 prompt 不会提交到 ReplayBuffer，scheduler 会持续读取并处理新 prompt，直到收集到要求的 batch size。

Validation 路径默认跳过 query 和 response filter，避免过滤改变评估数据分布。

## 3. Agentic 轨迹组过滤

### 3.1 Group 语义

Agentic Pipeline 的过滤单位是 trajectory group：

```text
(group_id, episode_id)
├── Trajectory 0
├── Trajectory 1
├── ...
└── Trajectory N
       │
       └── GroupFilter.filter(group_id, episode_id, group)
```

同一 `group_id` 中的环境实例具有相同的环境配置和 seed，也就是针对同一个任务产生多条完整交互轨迹。每条 trajectory 可以包含多轮模型响应、工具调用、环境状态和 reward。

### 3.2 配置过滤类

通过 `EnvManagerConfig.group_filter_cls` 配置过滤实现：

```yaml
train_env_manager:
  num_env_groups: 128
  group_size: 8
  group_filter_cls: my_package.filters.MyGroupFilter
```

过滤类接口如下：

```python
class MyGroupFilter:
    def __init__(self, config, env_manager_config, mode):
        self.mode = mode

    def filter(
        self,
        group_id: int,
        episode_id: int,
        group: list[DataProto],
    ) -> bool:
        # True：过滤整个 group
        # False：保留整个 group
        return False
```

默认类为：

```text
roll.pipeline.agentic.agentic_pipeline.GroupFilter
```

默认实现始终返回 `False`，即不过滤。

:::caution
应以实际调用代码为准：`filter()` 返回 `True` 表示过滤，返回 `False` 表示保留。当前 `group_filter_cls` 配置字段中的 help 文本与实际返回值语义不一致。
:::

### 3.3 过滤时机与动态补采样

EnvManager 完成 trajectory 后，将 `DataProto` 写入 `GroupQueueManager`。当一个 episode 收集到 `group_size` 条 trajectory 时，`GroupQueue` 才执行 group filter。

```text
EnvManager
    │ put(group_id, episode_id, rollout)
    ▼
GroupQueue
    │ 收集到 group_size 条 trajectory
    ▼
GroupFilter.filter(...)
    │
    ├── False：标记 group 完成，交给 get_batch()
    │
    └── True ：删除当前 episode
                创建 replacement episode
                环境继续 rollout
```

被过滤的 group 不会进入训练 batch。过滤次数通过以下 metric 上报：

```text
scheduler/group_filter_count
```

与 RLVR 不同，Agentic scheduler 在 validation 模式下仍然会调用所配置的 `GroupFilter`，并通过构造参数传入 `mode="val"`。如果只希望过滤训练数据，自定义实现必须主动判断：

```python
def filter(self, group_id, episode_id, group):
    if self.mode != "train":
        return False
    ...
```

### 3.4 自定义过滤示例

下面的实现过滤环境超时、失败或 reward 无组内差异的轨迹组：

```python
import torch


class MyGroupFilter:
    def __init__(self, config, env_manager_config, mode):
        self.mode = mode

    def filter(self, group_id, episode_id, group):
        if self.mode != "train":
            return False

        valid_items = [item for item in group if item is not None]
        if len(valid_items) != len(group):
            return True

        if any(
            item.meta_info.get("env_timeout", False)
            or item.meta_info.get("env_failed", False)
            for item in valid_items
        ):
            return True

        rewards = torch.cat([
            item.batch["response_level_rewards"].reshape(-1)
            for item in valid_items
        ])
        return rewards.numel() > 1 and rewards.std() <= 0.01
```

实际 EnvManager 输出的字段可能不同。自定义 filter 前，应检查目标 manager 的 `formulate_rollouts()`，确认 reward、停止原因和环境指标位于 `DataProto.batch`、`non_tensor_batch` 还是 `meta_info`。

### 3.5 `drop_flag` 与过滤比例保护

部分 Agentic EnvManager 提供了基于 `meta_info["drop_flag"]` 的特殊 `GroupFilter`。其基本行为是：

1. group 中任意 trajectory 带有 `drop_flag=True`，则将整组标记为待过滤；
2. 维护全局 `total` 和 `filtered` 统计；
3. 当过滤后可能使全局过滤比例超过 50% 时，保留该 group。

相关实现可参考：

```text
roll/pipeline/agentic/env_manager/agent_native_env_manager.py
roll/pipeline/agentic/env_manager/traj_env_manager_tb.py
```

这类比例保护可以避免 dirty data 过多时 scheduler 长时间无法凑齐 batch。需要注意，当前算法在第一个待过滤 group 到来时，预测过滤率为 `1 / 1`，因此会先保留该 group；它是在全局处理过程中限制累计过滤比例，并不是严格的滑动窗口比例控制。

### 3.6 `group_size_redundancy`

Agentic 支持为每个 group 启动额外 trajectory：

```yaml
train_env_manager:
  group_size: 8
  group_size_redundancy: 2
```

这表示同一 episode 最多可以启动 10 条 trajectory，但最先完成的 8 条到达后即可执行过滤：

- group 被接受时，最终只取 `group_size` 条；
- group 被过滤时，当前 episode 被删除并创建新 episode；
- 已删除 episode 后续到达的冗余 trajectory 会被忽略。

该配置主要用于降低慢环境和长尾 trajectory 带来的等待时间，不是“过滤一条 trajectory 后用冗余结果替换一条”的机制。Agentic 的通用过滤单位仍然是整个 group。

## 4. Train-Infer Correction Filter

训练模型与推理引擎可能因为精度、算子或执行后端不同，对同一 token 给出不同概率。ROLL 可以根据 `old_log_probs` 与 `infer_logprobs` 的偏差生成 filter mask。

配置示意：

```yaml
train_infer_correction:
  filters:
    - enabled: true
      agg_type: token
      ratio_enabled: true
      ratio_low: 0.8
      ratio_high: 1.2
      diff_enabled: false
```

支持的聚合粒度包括：

- `token`
- `segment`
- `geometric`
- `sequence`

框架计算 ratio 或 probability difference 是否落在配置区间内，并将结果乘到 `response_mask`、`final_response_mask` 等训练 mask。

```text
rollout batch
    │
    ├── old_log_probs
    ├── infer_logprobs
    └── response_mask
            │
            ▼
  compute_train_infer_correction
            │
            └── filter_mask
                    │
                    └── 更新训练 mask
```

这种 filter 不会：

- 删除 `DataProto` 中的样本；
- 调用 ReplayBuffer abort；
- 创建新的 Agentic episode；
- 触发额外 rollout。

它只决定哪些 token 参与 loss，因此不应与 rollout 动态过滤混用概念。

## 5. 过滤策略设计建议

### 5.1 明确过滤单位

选择过滤层次时，先确定数据是否仍然有训练价值：

- 单个 response 格式错误，但同组其他 response 有效：优先 response 级过滤与补采样；
- 整个 reward group 没有相对优势：过滤完整 prompt/group；
- 环境异常使整条 trajectory 不可信：在 Agentic group filter 中处理；
- 只有部分 token 存在 train-infer 偏差：使用 correction mask，不要丢弃整条轨迹。

### 5.2 避免无法凑齐 batch

过滤规则过严会显著增加 rollout 成本，甚至让 scheduler 长时间无法获得完整 batch。建议：

- 记录各过滤原因及比例；
- 对异常数据过滤与低信息量数据过滤分别计数；
- 为高过滤率设置保护上限或退化策略；
- 对 response 补采样设置最大重试次数；
- 在小规模数据上先评估保留率。

### 5.3 保持 group 的算法语义

GRPO、GiGPO 等算法依赖 group 内样本关系。不要在 scheduler filter 中返回不完整或语义混乱的 group。自定义 RLVR response filter 后，尤其需要确认补采后的 response 仍属于同一 prompt，并且最终数量与下游 advantage 计算一致。

### 5.4 正确处理 Validation

- RLVR 默认在 validation 中跳过 query/response filter；
- Agentic 自定义 `GroupFilter` 应显式检查 `mode`；
- 如果 validation 确实需要过滤，应同时上报原始数量、过滤数量和有效数量，否则评估指标可能出现选择偏差。

### 5.5 RemoteBatch 注意事项

如果 EnvManager 在写入 output queue 前手动调用 `DataProto.to_remote()`，当前 filter 路径不会自动对被过滤的远程数据调用 `drop()`，可能造成远程存储数据泄漏。

因此，在该清理逻辑完善前：

- 使用 Agentic group filter 时，不要在 EnvManager 中提前手动 `to_remote()`；
- 保持默认由 RolloutScheduler 在 batch 合并后执行远程传输。

## 6. 相关代码

| 功能 | 代码位置 |
| --- | --- |
| RLVR query/response filter | `roll/distributed/scheduler/user_defined_rollout_loop.py` |
| RLVR ReplayBuffer 与动态调度 | `roll/distributed/scheduler/generate_scheduler.py` |
| RLVR filter 配置 | `roll/pipeline/rlvr/rlvr_config.py` |
| Agentic GroupQueue 与补采样 | `roll/distributed/scheduler/rollout_scheduler.py` |
| Agentic 默认 GroupFilter | `roll/pipeline/agentic/agentic_pipeline.py` |
| Agentic EnvManager 配置 | `roll/pipeline/agentic/agentic_config.py` |
| Train-Infer Correction Filter | `roll/utils/train_infer_corrections.py` |

