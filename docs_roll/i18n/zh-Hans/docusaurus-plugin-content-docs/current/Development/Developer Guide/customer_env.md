# 自定义环境

ROLL 的 Agentic Pipeline 使用 [GEM](https://github.com/axon-rl/gem) 环境。一个环境是注册到某个 `env_type` 的 `gem.Env` 实现；环境管理器通过 `gem.make()` 创建实例，并不断把模型输出传给 `step()`。

本文描述 `roll/pipeline/agentic/env` 与 `roll/pipeline/agentic/env_manager` 当前采用的契约。旧版文档中的 `BaseDiscreteActionEnv` / `BaseLanguageBasedEnv` 分类已经不再使用。

## 运行流程

标准 `TrajEnvManager` 中的单个 Episode 流程如下：

```text
gem.make(env_type, **env_config)
  -> reset(seed)
  -> 根据 observation 和 info["env_instruction"] 构造提示词
  -> 模型生成动作字符串
  -> step(action_string)
  -> 重复，直到 terminated 或达到 max_steps
```

环境实例运行在 Environment Worker 中，并且可能并发执行。不要依赖可变的全局状态，并保证 `reset(seed)` 可复现。若环境持有外部资源，应实现可重复调用的 `close()`；具体是在每个 Episode 后还是退出时调用取决于所选环境管理器，因此有严格生命周期要求的环境还应在终止和异常路径中清理资源。

## 环境接口契约

继承 `gem.Env` 并实现以下方法：

```python
from typing import Any
from gem import Env


class MyEnv(Env):
    def __init__(self, max_steps: int = 10, **kwargs): ...

    def reset(self, seed: int | None = None) -> tuple[Any, dict[str, Any]]: ...

    def step(
        self, action: str
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]: ...

    def close(self) -> None: ...
```

### `__init__(**env_config)`

`custom_envs.<tag>.env_config` 下的配置会由下面的代码直接传给构造函数：

```python
gem.make(env_id=env_config["env_type"], **env_config["config"])
```

构造函数应只接收有意义的选项并尽早校验。包装第三方环境时，可以保留 `**kwargs` 以传递其参数。

### `reset(seed)`

适用时先调用 GEM 基类实现，重置 Episode 的所有状态，然后返回：

```python
(observation, info)
```

- `observation` 是当前要展示给 Agent 的状态；对 `TrajEnvManager` 而言通常是字符串。
- `info` 必须是字典。可将稳定的任务说明或动作格式要求放入 `info["env_instruction"]`，标准管理器会在第一轮将其加入提示词。
- 仅在没有可用 Episode 时返回 `(None, info)`；管理器会将其视为本次没有 Rollout。
- 所有影响任务生成的随机源都应使用传入的 seed。同一个 Rollout Group 内的环境会收到相同的 Episode seed。

### `step(action)`

标准管理器传入的是模型解码后的完整响应，而不是离散动作 ID。环境需自行解析并校验文本，然后返回 Gymnasium 风格的五元组：

```python
(observation, reward, terminated, truncated, info)
```

- `observation`：动作执行后的状态。
- `reward`：本轮的标量奖励；ROLL 会将各轮奖励相加得到 Episode score。
- `terminated`：任务因成功或失败等自然终止条件结束。
- `truncated`：因超时等外部限制结束。在当前标准 `TrajEnvManager` 中，只有 `truncated=True` 不会停止循环；环境自身发生超时时，应同时返回 `terminated=True, truncated=True`。达到 `custom_envs.<tag>.max_steps` 时，管理器也会强制结束并设置 truncation。
- `info`：本轮附加信息。

内置环境通常使用以下可选 `info` 字段：

```python
info = {
    "action_desc": "动作结果的可读描述",
    "metrics": {
        "action_is_valid": True,
        "action_is_effective": True,
        "success": False,
    },
    "metrics_agg_mode": {
        "action_is_valid": "mean",
        "action_is_effective": "mean",
        "success": "last",
    },
    # "suffix": "当 agent_template 包含 {suffix} 时渲染的额外状态",
}
```

`metrics` 的值应为数值或布尔值，`metrics_agg_mode` 指定 ROLL 如何在整条轨迹上聚合各项指标。遇到格式错误的动作时，通常保持 observation 不变、返回格式惩罚、设置 `action_is_valid=False`，并允许 Episode 继续。

### Observation 格式

`TrajEnvManager` 期望文本类 observation，并由管理器构造对话历史。`AgentNativeStepEnvManager` 用于由环境自行维护对话的场景，此类环境返回 OpenAI 风格的消息列表，例如 `[{'role': 'user', 'content': '...'}]`。环境和管理器必须成对选择，不要混用两种契约。使用 Tool Call 的环境还应配置对应的 `ToolCallRunner` 或 Native Runner。

## 多模态 Observation

环境需要返回图片或视频时，使用 `roll.pipeline.agentic.env_manager.vl_traj_env_manager.VLTrajEnvManager`。它沿用相同的 `reset()` 和五元组 `step()` 契约，但支持以下 observation 形式：

- `str`：纯文本的一轮。
- `numpy.ndarray`：单张 RGB 图片。数组必须能传给 `PIL.Image.fromarray(obs, mode="RGB")`，通常为 `H x W x 3` 的 `uint8` 数组。
- `dict`：多模态的一轮。`prompt` 保存文本或 Chat 风格内容，`image` 和/或 `video` 保存模型 Transformers processor 能接收的媒体对象。

典型的图片 observation 如下：

```python
from PIL import Image


class VisualQuestionEnv(Env):
    image_placeholder = "<image>"

    def reset(self, seed=None):
        super().reset(seed)
        image = Image.open(self.image_path).convert("RGB")
        observation = {
            "prompt": "<image>\nWhat object is highlighted?",
            "image": [image],
        }
        return observation, {
            "env_instruction": "Answer directly, or request a visual tool action."
        }
```

字典的 key 使用单数形式：`image` 和 `video`。值可以是单个对象或 list/tuple；建议始终使用列表，当 prompt 中包含多个媒体占位符时也必须使用列表。媒体值的顺序必须与 prompt 中占位符的顺序一致。

如果 prompt 使用显式占位符，应声明对应的类属性：

```python
class MyMultimodalEnv(Env):
    image_placeholder = "<image>"
    video_placeholder = "<video>"
```

`VLTrajEnvManager` 会将环境占位符替换成 `DataCollatorWithPaddingForMM` 要求的特殊 token。它按照轨迹顺序累积所有轮次的媒体，使用策略模型的 `ProcessorMixin` 构造模型输入，并将处理后的多模态字段保留在 Rollout 中。环境应该返回原始媒体和文本，不要自行返回预计算的 `pixel_values`、token ID 或设备上的 tensor。

需要结构化 Chat 内容时，`prompt` 也可以是 content item 列表：

```python
observation = {
    "prompt": [
        {"type": "text", "text": "Inspect this image:"},
        {"type": "image"},
        {"type": "text", "text": "What changed?"},
    ],
    "image": [image],
}
```

视觉语言环境需要配置 `VLTrajEnvManager` 及其两个轮次模板：

```yaml
custom_envs:
  VisualQuestion:
    env_type: visual_question
    env_manager_cls: roll.pipeline.agentic.env_manager.vl_traj_env_manager.VLTrajEnvManager
    max_steps: 4
    max_tokens_per_step: 256
    agent_system_template: "You are a visual reasoning agent."
    pre_step_template: "\nTurn {turn_idx}:\n"
    next_step_template: |
      You have {actions_left} actions left.
      Keep the response within {max_response_length} tokens.
    env_config:
      image_path: /path/to/image.png
```

Actor 必须是拥有兼容 tokenizer/processor 的视觉语言模型。`VLTrajEnvManager` 当前只为图片和视频提供了明确的 Collation 路径；音频或新的媒体类型需要扩展管理器和多模态 Collator。媒体加载可能失败时，应在 `reset()`/`step()` 中校验并返回受控的终止结果，避免 prompt 与错误媒体静默配对。

## 最小实现

```python
import random
import re
from typing import Any

from gem import Env


class GuessNumberEnv(Env):
    def __init__(self, low: int = 1, high: int = 10, format_penalty: float = -0.1, **kwargs):
        self.low = low
        self.high = high
        self.format_penalty = format_penalty
        self.target = None
        self.done = False

    def reset(self, seed: int | None = None) -> tuple[str, dict[str, Any]]:
        super().reset(seed)
        self.target = random.Random(seed).randint(self.low, self.high)
        self.done = False
        return (
            f"Guess an integer from {self.low} to {self.high}.",
            {"env_instruction": "Reply with <answer>number</answer>."},
        )

    def step(self, action: str) -> tuple[str, float, bool, bool, dict[str, Any]]:
        match = re.search(r"<answer>\s*(-?\d+)\s*</answer>", action)
        valid = match is not None
        guess = int(match.group(1)) if valid else None
        success = valid and guess == self.target
        self.done = success

        if not valid:
            observation, reward = "Invalid format; try again.", self.format_penalty
        elif guess < self.target:
            observation, reward = "Too small.", 0.0
        elif guess > self.target:
            observation, reward = "Too large.", 0.0
        else:
            observation, reward = "Correct.", 1.0

        info = {
            "metrics": {"action_is_valid": valid, "success": success},
            "metrics_agg_mode": {"action_is_valid": "mean", "success": "last"},
        }
        return observation, reward, self.done, False, info

    def close(self) -> None:
        pass
```

## 注册环境

在环境包中导出该类，并在 `roll/pipeline/agentic/env/__init__.py` 中添加 GEM 延迟注册：

```python
gem.register(
    "guess_number",
    entry_point="roll.pipeline.agentic.env.guess_number:GuessNumberEnv",
)
```

注册名必须与 `env_type` 完全一致。不要在包的 `__init__.py` 中导入重量级或可选依赖；只有 `gem.make()` 创建环境时才应加载 entry point。若环境依赖可选组件，可参考现有可选环境的受保护注册方式。

## 配置环境

在 `custom_envs` 下添加一个 tag（`examples/config` 中部分公共配置片段使用单数 `custom_env`，由上层配置合并到 `custom_envs`）：

```yaml
env_manager_cls: roll.pipeline.agentic.env_manager.traj_env_manager.TrajEnvManager

custom_envs:
  GuessNumber:
    env_type: guess_number
    env_manager_cls: ${env_manager_cls}
    agent_runner_cls: null
    max_steps: 8
    max_tokens_per_step: 32
    agent_system_template: "You are a careful game-playing agent."
    agent_template: |
      Turn {turn_idx}:
      Observation: {observation}
      You have {actions_left} actions left.
      Respond with one action only.
    env_config:
      low: 1
      high: 20
      format_penalty: -0.1

train_env_manager:
  num_env_groups: 32
  group_size: 4
  tags: [GuessNumber]
  num_groups_partition: [32]
```

配置分为两层：

- `max_steps`、`max_tokens_per_step`、模板、`env_manager_cls` 和 `agent_runner_cls` 等字段由 Rollout 框架使用。
- 只有 `env_config` 内的字段会传给环境构造函数。
- `tags` 按 `custom_envs` 的 key 选择配置，`env_type` 则选择已注册的 GEM 类。
- `num_groups_partition` 必须与 `tags` 一一对应，且总和等于 `num_env_groups`；同一 Group 的成员共享配置和 seed。

初始化或执行步骤不是线程安全时，设置 `use_thread_lock: true`。昂贵的共享后端可用 `max_env_step_concurrent` 限制并发调用数。

## 验证清单

在完整训练前先验证环境契约：

```python
import roll.pipeline.agentic.env  # 执行 ROLL 的 GEM 注册
import gem

env = gem.make(env_id="guess_number", low=1, high=3)
obs, info = env.reset(seed=42)
assert isinstance(info, dict)

obs, reward, terminated, truncated, info = env.step("<answer>2</answer>")
assert isinstance(reward, (int, float))
assert isinstance(terminated, bool) and isinstance(truncated, bool)
assert isinstance(info, dict)
env.close()
```

还需检查：

- 相同 seed 是否生成相同初始任务；
- 合法动作、非法动作、自然终止、超时和最大步数路径；
- 每次 `step()` 是否严格返回五个值；
- 各步的 metric key 和聚合模式是否保持一致；
- 异常和 `close()` 后是否正确释放资源；
- 使用所选 Environment Manager 与 Runner 的小规模 Rollout 是否能完整运行。
