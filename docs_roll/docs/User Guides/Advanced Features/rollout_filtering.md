# Rollout Filtering and Dynamic Resampling

## 1. Overview

Not every rollout is suitable for training. Common examples include:

- all responses to the same prompt receive the same reward and provide no useful within-group relative advantage;
- a trajectory is incomplete because of an environment timeout, execution failure, or format error;
- training and inference probabilities diverge so much that some tokens should not contribute to the gradient;
- an input prompt exceeds the maximum supported length.

ROLL provides filtering at several levels. It is important to distinguish the following three mechanisms:

| Mechanism | Pipeline | Filtering unit | Drops rollouts | Dynamically resamples |
| --- | --- | --- | --- | --- |
| RLVR dynamic filtering | RLVR | A response or all responses to one prompt | Yes | Yes |
| Agentic trajectory-group filtering | Agentic | Trajectories with the same environment configuration and seed | Yes | Yes |
| Train-Infer Correction Filter | RLVR and Agentic | Token, segment, or sequence mask | No | No |

The first two mechanisms run in the rollout scheduler. Rejected data is not submitted to training, and the scheduler continues sampling until it has enough accepted data. The third mechanism runs during training and only modifies loss masks; it does not remove samples from the batch.

## 2. RLVR Dynamic Filtering

### 2.1 Data hierarchy

RLVR usually generates multiple responses for one prompt:

```text
Prompt
├── Response 0 ── response reward
├── Response 1 ── response reward
├── ...
└── Response N ── response reward
       │
       ├── response_filter: inspect one response
       └── query_filter: inspect the complete response group for the prompt
```

The default rollout loop is implemented in:

```text
roll/distributed/scheduler/user_defined_rollout_loop.py
```

It performs the following operations:

1. Fetch a prompt from the dataset.
2. Generate one or more responses.
3. Compute rewards.
4. Apply response-level filtering.
5. Apply prompt/query-level filtering.
6. Commit accepted data to the `ReplayBuffer`.

Returning `None` at any stage terminates the transaction for the current prompt and lets the scheduler process another prompt.

### 2.2 Query-level filtering

Query-level filtering examines the `response_level_rewards` of all responses to one prompt. Its default entry point is:

```python
def query_filter(data_list: list[DataProto], config: RLVRConfig) -> bool:
    ...
```

The return value means:

- `True`: keep the prompt group;
- `False`: filter out the prompt group.

The filter configuration belongs to a particular reward domain:

```yaml
rewards:
  math:
    query_filter_config:
      type: mean_filter
      filter_args:
        threshold_down: 0.0
        threshold_up: 1.0
```

ROLL reads `non_tensor_batch["domain"]` from the sample and selects the corresponding `rewards.<domain>` configuration. Different domains can therefore use different filtering rules.

The built-in filter types are:

| `type` | Behavior |
| --- | --- |
| `no_filter` | Always keep the group |
| `mean_filter` | Filter when the mean reward is less than or equal to the lower bound, or greater than or equal to the upper bound |
| `std_filter` | Filter when the reward standard deviation is less than or equal to the threshold |

`mean_filter` is commonly used with GRPO/RLVR to remove all-wrong or all-correct groups, for which there is no useful within-group relative advantage signal.

Example using `std_filter`:

```yaml
rewards:
  math:
    query_filter_config:
      type: std_filter
      filter_args:
        std_threshold: 0.01
```

:::note
The default `query_filter()` immediately keeps an input list containing only one element. Pass a `list[DataProto]` whose elements have been split by response; do not wrap one multi-response `DataProto` in a list of length one.
:::

### 2.3 Response-level filtering

After reward computation, the default rollout loop invokes the following function for each response:

```python
response_filter(batch_item, pipeline_config)
```

If responses are rejected, ROLL resends the original request and generates only the number of replacement responses required. The default implementation limits this process to five retry rounds to avoid an infinite loop caused by an invalid filtering rule.

In the current version, the default function is still:

```python
def response_filter(data_item, config):
    return True
```

No individual response is therefore filtered by default. Although `RewardConfig` defines `response_filter_config`, the default function does not currently read that configuration. Setting only `response_filter_config` in YAML has no effect.

To implement response-level filtering, define a custom `UserDefinedRolloutLoop` and apply the rule between generation, reward computation, and result return. For example:

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

Load the custom class through the top-level configuration:

```yaml
user_defined_rollout_loop_cls: my_package.rollout_loop.MyRolloutLoop
```

The returned response count must satisfy the scheduler's `num_return_sequences` contract. If the custom loop changes the response count dynamically, configure the additional-prompts mechanism accordingly.

### 2.4 Prompt- or dataset-level filtering

Input data may also be filtered before generation. For example, the RLVR VLM Pipeline checks the tokenized prompt length:

```python
request_data, domain = context.get_request_data(meta_info=context.meta_info)
if request_data.batch["input_ids"].shape[1] > context.prompt_length:
    return None
```

After `None` is returned, the prompt does not enter generation or training, and the scheduler fetches another dataset item.

:::caution
`RLVRConfig.dataset_filter` currently defines only the configuration structure and is not wired into the default RLVR rollout path. Implement general dataset filtering while constructing the dataset, or override `UserDefinedRolloutLoop.process_new_prompt()`.
:::

### 2.5 Resampling after filtering

RLVR uses `ReplayBuffer` to manage the transaction state of each prompt:

```text
process_new_prompt
        │
        ├── returns responses
        │       └── ReplayBuffer.commit(prompt_id, responses)
        │
        └── returns None
                └── ReplayBuffer.abort(prompt_id)
                            │
                            └── release concurrency capacity and schedule a new prompt
```

RLVR filtering is therefore not a Boolean index applied to an already constructed training batch. A rejected prompt is never committed to the ReplayBuffer. The scheduler continues fetching and processing prompts until it collects the requested batch size.

The validation path skips query- and response-level filtering by default so filtering does not change the evaluation distribution.

## 3. Agentic Trajectory-Group Filtering

### 3.1 Group semantics

The filtering unit in the Agentic Pipeline is a trajectory group:

```text
(group_id, episode_id)
├── Trajectory 0
├── Trajectory 1
├── ...
└── Trajectory N
       │
       └── GroupFilter.filter(group_id, episode_id, group)
```

Environment instances in the same `group_id` use the same environment configuration and seed. They produce multiple complete interaction trajectories for the same task. Each trajectory may contain multiple model turns, tool calls, environment states, and rewards.

### 3.2 Configuring the filter class

Set the implementation through `EnvManagerConfig.group_filter_cls`:

```yaml
train_env_manager:
  num_env_groups: 128
  group_size: 8
  group_filter_cls: my_package.filters.MyGroupFilter
```

The class contract is:

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
        # True: filter out the entire group
        # False: keep the entire group
        return False
```

The default class is:

```text
roll.pipeline.agentic.agentic_pipeline.GroupFilter
```

Its default implementation always returns `False`, so no group is filtered.

:::caution
Follow the actual call-site semantics: returning `True` filters out the group, while returning `False` keeps it. The help text currently attached to the `group_filter_cls` configuration field states the opposite behavior.
:::

### 3.3 Filtering time and dynamic resampling

After an EnvManager completes a trajectory, it writes the `DataProto` to `GroupQueueManager`. `GroupQueue` invokes the group filter when an episode has collected `group_size` trajectories.

```text
EnvManager
    │ put(group_id, episode_id, rollout)
    ▼
GroupQueue
    │ collect group_size trajectories
    ▼
GroupFilter.filter(...)
    │
    ├── False: mark the group complete and expose it through get_batch()
    │
    └── True : delete the current episode
                create a replacement episode
                continue environment rollouts
```

A rejected group never enters the training batch. The scheduler reports the number of rejected groups through:

```text
scheduler/group_filter_count
```

Unlike RLVR, the Agentic scheduler still calls the configured `GroupFilter` in validation mode and passes `mode="val"` to its constructor. A filter intended only for training must check the mode explicitly:

```python
def filter(self, group_id, episode_id, group):
    if self.mode != "train":
        return False
    ...
```

### 3.4 Custom filtering example

The following implementation rejects groups containing environment timeouts, environment failures, or no within-group reward variation:

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

The exact fields emitted by EnvManagers vary. Before implementing a custom filter, inspect the selected manager's `formulate_rollouts()` method and determine whether rewards, stop reasons, and environment metrics are stored in `DataProto.batch`, `non_tensor_batch`, or `meta_info`.

### 3.5 `drop_flag` and filtering-ratio protection

Some Agentic EnvManagers provide specialized `GroupFilter` implementations based on `meta_info["drop_flag"]`. Their basic behavior is:

1. Mark the entire group as a rejection candidate when any trajectory has `drop_flag=True`.
2. Maintain global `total` and `filtered` counters.
3. Keep the group if rejecting it could make the global filtering ratio exceed 50%.

See the following implementations:

```text
roll/pipeline/agentic/env_manager/agent_native_env_manager.py
roll/pipeline/agentic/env_manager/traj_env_manager_tb.py
```

This ratio guard prevents excessive dirty data from making it difficult for the scheduler to fill a batch. With the current algorithm, the first rejection candidate is retained because its prospective ratio is `1 / 1`, which exceeds 50%. The implementation limits the accumulated global ratio; it is not a strict sliding-window limiter.

### 3.6 `group_size_redundancy`

Agentic can start extra trajectories for each group:

```yaml
train_env_manager:
  group_size: 8
  group_size_redundancy: 2
```

Up to ten trajectories may run for the same episode, but filtering can begin as soon as the first eight complete:

- if the group is accepted, only `group_size` trajectories are returned;
- if the group is rejected, the current episode is removed and a new episode is created;
- redundant trajectories that later arrive for a removed episode are ignored.

This setting primarily reduces waiting caused by slow environments and long-tail trajectories. It does not replace one rejected trajectory with one redundant trajectory; the general Agentic filtering unit remains the entire group.

## 4. Train-Infer Correction Filter

Different precision, operators, or execution backends can cause the training model and inference engine to assign different probabilities to the same token. ROLL can generate a filter mask from the divergence between `old_log_probs` and `infer_logprobs`.

Example configuration:

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

Supported aggregation levels include:

- `token`
- `segment`
- `geometric`
- `sequence`

ROLL checks whether the ratio or probability difference falls within the configured interval and multiplies the result into training masks such as `response_mask` and `final_response_mask`.

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
                    └── update training masks
```

This filter does not:

- remove samples from `DataProto`;
- abort a ReplayBuffer transaction;
- create a replacement Agentic episode;
- trigger additional rollouts.

It only determines which tokens contribute to the loss and should not be confused with dynamic rollout filtering.

## 5. Filter Design Guidelines

### 5.1 Choose the filtering unit deliberately

Determine whether any training value remains before selecting the filtering layer:

- If one response has an invalid format but the other responses remain useful, prefer response-level filtering and replacement sampling.
- If an entire reward group provides no relative advantage, filter the complete prompt/group.
- If an environment failure makes a trajectory unreliable, handle it in an Agentic group filter.
- If only some tokens have excessive train-infer divergence, apply a correction mask instead of dropping the complete trajectory.

### 5.2 Avoid making the batch impossible to fill

An overly strict filter increases rollout cost and can prevent the scheduler from obtaining a complete batch. Recommended practices include:

- record counts and rates for each rejection reason;
- track invalid-data filtering separately from low-information filtering;
- use a maximum ratio or fallback policy when rejection rates are high;
- bound response replacement retries;
- estimate the retention rate on a small dataset before a large run.

### 5.3 Preserve group semantics

Algorithms such as GRPO and GiGPO depend on relationships between samples in a group. Do not return incomplete or semantically inconsistent groups from a scheduler filter. After implementing a custom RLVR response filter, verify that replacement responses still belong to the same prompt and that the final count matches the downstream advantage computation contract.

### 5.4 Handle validation explicitly

- RLVR skips query/response filtering during validation by default.
- An Agentic custom `GroupFilter` should inspect `mode` explicitly.
- If validation filtering is required, report original, rejected, and accepted counts so the evaluation metrics do not hide selection bias.

### 5.5 RemoteBatch considerations

If an EnvManager manually invokes `DataProto.to_remote()` before writing to the output queue, the current filter path does not automatically call `drop()` for rejected remote data. This can leak data in remote storage.

Until that cleanup path is implemented:

- do not manually call `to_remote()` inside an EnvManager when using an Agentic group filter;
- keep the default behavior in which RolloutScheduler performs remote transfer after merging the batch.

## 6. Related Code

| Feature | Location |
| --- | --- |
| RLVR query/response filtering | `roll/distributed/scheduler/user_defined_rollout_loop.py` |
| RLVR ReplayBuffer and dynamic scheduling | `roll/distributed/scheduler/generate_scheduler.py` |
| RLVR filter configuration | `roll/pipeline/rlvr/rlvr_config.py` |
| Agentic GroupQueue and replacement sampling | `roll/distributed/scheduler/rollout_scheduler.py` |
| Default Agentic GroupFilter | `roll/pipeline/agentic/agentic_pipeline.py` |
| Agentic EnvManager configuration | `roll/pipeline/agentic/agentic_config.py` |
| Train-Infer Correction Filter | `roll/utils/train_infer_corrections.py` |

