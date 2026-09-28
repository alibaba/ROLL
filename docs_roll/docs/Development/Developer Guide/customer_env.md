# Custom Environment

ROLL's Agentic Pipeline uses [GEM](https://github.com/axon-rl/gem) environments. An environment is a `gem.Env` implementation registered under an `env_type`; the environment manager creates it with `gem.make()` and repeatedly sends model output to `step()`.

This document describes the contract used by the current code in `roll/pipeline/agentic/env` and `roll/pipeline/agentic/env_manager`. The old `BaseDiscreteActionEnv` / `BaseLanguageBasedEnv` split is no longer used.

## Runtime flow

For the standard `TrajEnvManager`, one episode is:

```text
gem.make(env_type, **env_config)
  -> reset(seed)
  -> build prompt from observation and info["env_instruction"]
  -> model generates an action string
  -> step(action_string)
  -> repeat until terminated or max_steps
```

Environment instances run inside environment workers, potentially concurrently. Avoid global mutable state and make `reset(seed)` reproducible. Implement idempotent cleanup in `close()` when the environment owns external resources; whether it is called per episode or at shutdown depends on the selected environment manager, so an environment with strict lifecycle requirements should also clean up on terminal/error paths.

## Environment contract

Inherit from `gem.Env` and implement these methods:

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

The keys under `custom_envs.<tag>.env_config` are passed directly to the constructor by:

```python
gem.make(env_id=env_config["env_type"], **env_config["config"])
```

Accept only meaningful options, validate them early, and consider accepting `**kwargs` when wrapping a third-party environment.

### `reset(seed)`

Call the GEM base implementation when appropriate, reset all episode state, and return:

```python
(observation, info)
```

- `observation` is the current state shown to the agent, normally a string for `TrajEnvManager`.
- `info` is a dictionary. Put a stable task description or action-format instruction in `info["env_instruction"]`; the standard manager prepends it on the first turn.
- Return `(None, info)` only when no episode is available. The manager treats this as no rollout.
- Use the supplied seed for every random source that affects task generation. Environments in the same rollout group receive the same episode seed.

### `step(action)`

The standard manager passes the model's decoded response, not a discrete action ID. Parse and validate the text inside the environment, then return the Gymnasium-style five-tuple:

```python
(observation, reward, terminated, truncated, info)
```

- `observation`: state after the action.
- `reward`: scalar reward for this turn. ROLL sums turn rewards into the episode score.
- `terminated`: the task reached a natural terminal state, including success or failure.
- `truncated`: the environment ended for an external limit such as timeout. In the current standard `TrajEnvManager`, `truncated=True` alone does not stop the loop: return both `terminated=True` and `truncated=True` for an environment-owned timeout. The manager also forces termination and sets truncation when `custom_envs.<tag>.max_steps` is reached.
- `info`: per-turn metadata.

The built-in environments use the following optional `info` fields:

```python
info = {
    "action_desc": "Human-readable result of the action",
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
    # "suffix": "Extra state rendered by an agent_template containing {suffix}",
}
```

Keep `metrics` numeric or boolean. `metrics_agg_mode` tells ROLL how to aggregate each metric over the trajectory. If an action is malformed, normally return an unchanged observation, a format penalty, `action_is_valid=False`, and keep the episode running.

### Observation formats

`TrajEnvManager` expects text-like observations and constructs the chat history itself. `AgentNativeStepEnvManager` is for environments that own the conversation and return OpenAI-style message lists such as `[{'role': 'user', 'content': '...'}]`. Pair the environment and manager deliberately; do not mix the two contracts. Tool-call environments should use the matching `ToolCallRunner` or native runner configuration.

## Multimodal observations

Use `roll.pipeline.agentic.env_manager.vl_traj_env_manager.VLTrajEnvManager` when an environment returns images or videos. It preserves the same `reset()` and five-value `step()` contract, but accepts the following observation forms:

- `str`: a text-only turn.
- `numpy.ndarray`: one RGB image. It must be compatible with `PIL.Image.fromarray(obs, mode="RGB")`, normally an `H x W x 3` `uint8` array.
- `dict`: a multimodal turn. `prompt` contains text or chat-style content, while `image` and/or `video` contain the media objects accepted by the model's Transformers processor.

A typical image observation is:

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

The dictionary keys are singular: `image` and `video`. Their values can be a single item or a list/tuple; using a list is recommended and is required when the prompt contains multiple media placeholders. The order of media values must match placeholder order in the prompt.

If the prompt contains explicit placeholders, declare the matching class attributes:

```python
class MyMultimodalEnv(Env):
    image_placeholder = "<image>"
    video_placeholder = "<video>"
```

`VLTrajEnvManager` replaces these environment placeholders with the special tokens required by `DataCollatorWithPaddingForMM`. It accumulates media from all turns in trajectory order, uses the policy model's `ProcessorMixin` to build model inputs, and retains the processed multimodal fields in the rollout. The environment should return raw media and text, not precomputed `pixel_values`, token IDs, or device tensors.

For structured chat content, `prompt` may also be a list of content items:

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

Configure a vision-language environment with `VLTrajEnvManager` and its two turn templates:

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

The actor must be a vision-language model with a compatible tokenizer/processor. `VLTrajEnvManager` currently has explicit collation paths for images and videos; audio or a new media type requires extending the manager and multimodal collator. When media loading can fail, validate it in `reset()`/`step()` and return a controlled terminal result rather than silently pairing the prompt with the wrong media.

## Minimal implementation

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

## Register the environment

Export the class from its package and add a lazy GEM registration in `roll/pipeline/agentic/env/__init__.py`:

```python
gem.register(
    "guess_number",
    entry_point="roll.pipeline.agentic.env.guess_number:GuessNumberEnv",
)
```

The registration name must exactly match `env_type`. Keep heavyweight or optional imports out of the package `__init__.py`; the entry point is imported when `gem.make()` creates the environment. If an environment has optional dependencies, follow the guarded-registration pattern used by the existing optional environments.

## Configure the environment

Add a tag under `custom_envs` (some shared fragments in `examples/config` use the singular key `custom_env` and are merged into `custom_envs` by their parent configuration):

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

Configuration has two levels:

- Fields such as `max_steps`, `max_tokens_per_step`, templates, `env_manager_cls`, and `agent_runner_cls` are consumed by the rollout framework.
- Only fields nested under `env_config` are passed to the environment constructor.
- `tags` selects entries by their `custom_envs` key, while `env_type` selects the registered GEM class.
- `num_groups_partition` must align with `tags` and sum to `num_env_groups`; members of a group share configuration and seed.

Use `use_thread_lock: true` for initialization or steps that are not thread-safe. Use `max_env_step_concurrent` to bound concurrent calls for an expensive shared backend.

## Validation checklist

Test the environment contract before a full training run:

```python
import roll.pipeline.agentic.env  # performs ROLL's GEM registrations
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

Also verify:

- the same seed produces the same initial task;
- valid, invalid, terminal, timeout, and maximum-step paths;
- every `step()` returns exactly five values;
- metric keys and aggregation modes stay consistent across steps;
- resources are cleaned up after exceptions and `close()`;
- a small rollout with the selected environment manager and runner completes.
