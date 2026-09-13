# AgentRunner -- Agent Interaction Loop Abstraction

## Overview

ROLL's Agentic training framework supports various types of agent-environment interactions: from local gym-like game environments (Sokoban, FrozenLake) to SWE-agents running in remote sandboxes. In ROLL, the entire agent rollout process is fully user-customizable -- the framework does not impose a fixed interaction pattern, but instead provides flexible extension points. However, the framework should not simply leave users to figure out "how to write a rollout loop" on their own. Therefore, ROLL provides common rollout loop implementations for the most accessible multi-turn interaction scenarios (`TrajEnvManager` / `StepEnvManager`), allowing users to quickly understand the framework protocol based on these examples and then customize for their specific needs.

As scenarios have grown more diverse, the existing `EnvManager` system has exposed bottlenecks in responsibility boundaries and extensibility. The introduction of the `AgentRunner` abstraction decouples the "how agents interact with environments" logic from `EnvManager`, enabling different types of agents to share a unified interface while preserving `EnvManager`'s core capability in training sample construction.

The AgentRunner abstraction design is derived from practical experience across numerous real-world business scenarios. Thanks to the blog post [Getting Started with AgenticRL Using ROLL](https://zhuanlan.zhihu.com/p/2023433301961049206) for its discussion on Agentic training architecture, which provided valuable reference for this design.

## 1. Architecture Background: The EnvManager System

Before understanding AgentRunner, it's important to first understand the responsibilities and existing architecture of EnvManager in ROLL.

### 1.1 Core Responsibilities of EnvManager

EnvManager is the core component of Agentic training, responsible for three things:

1. **Interaction Loop**: Driving multi-turn agent-environment interactions (`reset` / `step` / `make_decision`)
2. **Message Formatting**: Constructing environment observations into LLM-understandable prompts (`format_messages`)
3. **Training Sample Construction**: Converting interaction trajectories into `DataProto` required for training (`formulate_rollouts`)

### 1.2 Existing Inheritance Hierarchy

```
BaseEnvManager
├── TrajEnvManager                     # Trajectory-level concatenation, suitable for gym-like environments
│   ├── StepEnvManager                 # Step-level decomposition, suitable for GiGPO algorithm
│   │   └── StepConcatEnvManager       # Step-level + observation concatenation variant
│   ├── AgentNativeStepEnvManager      # SWE/TerminalBench native environments
│   └── VLTrajEnvManager               # Vision-Language model variant
```

### 1.3 TrajEnvManager

`TrajEnvManager` is the most basic implementation. It completes all work in a compact loop:

```python
# TrajEnvManager core loop (simplified)
def run_rollout_loop(self, data):
    while self.running:
        rollout_cache = self.reset()              # env.reset(seed)
        while not done:
            lm_output = self.make_decision(cache)  # format_messages → LLM inference
            rollout_cache = self.step(lm_output)   # env.step(action)
        rollout = self.formulate_rollouts(cache)   # concatenate full trajectory → DataProto
        output_queue.put(rollout)
```

**Trajectory-level sample construction**: Concatenates all steps' `prompt_ids + response_ids` within an episode into a single training sequence, with the episode reward placed on the last token. The entire trajectory is one training sample.

**Applicable scenarios**: Gym-like environments such as Sokoban, FrozenLake, WebShop. The environments are lightweight Python objects with fast interactions and no network communication needed.

**Fine-grained control capabilities**: Since EnvManager fully controls every step of the interaction loop, TrajEnvManager is particularly suitable for scenarios requiring fine-grained control over the interaction process:
- **Rapid multi-agent prototyping**: Freely orchestrate the interaction order and communication logic of multiple agents in `run_rollout_loop`, without being constrained by black-box interfaces
- **Fine-grained `response_mask` management**: Control at the token level which parts participate in loss computation in `formulate_rollouts` (e.g., only training responses from specific turns, masking tool outputs, etc.)
- **Custom prompt concatenation strategies**: Flexibly decide how to retain, truncate, or summarize historical information in `format_messages`

### 1.4 StepEnvManager

`StepEnvManager` inherits from `TrajEnvManager` and overrides `format_messages` and `formulate_rollouts`, reducing the granularity from "trajectory-level" to "step-level":

- **Independent step-level prompts**: Each step constructs a self-contained prompt (including system instructions, history summary, current observation), rather than concatenating all historical tokens
- **Step-level training samples**: Each step generates an independent training sample, with each sample containing that step's prompt + response and corresponding step reward
- **`state_hash` mechanism**: Computes a hash value for each step's state, used for cross-rollout same-state grouping advantage estimation in the GiGPO algorithm

**Applicable scenarios**: Algorithms requiring step-level advantage estimation (such as GiGPO), or scenarios needing finer-grained credit assignment.

### 1.5 Why Keep TrajEnvManager / StepEnvManager

The TrajEnvManager / StepEnvManager system coexists in parallel with ProxyEnvManager + AgentRunner, each with its own advantages:

| Dimension | TrajEnvManager / StepEnvManager | ProxyEnvManager + AgentRunner |
|-----------|-------------------------------|-------------------------------|
| **Interaction overhead** | Direct function calls, zero serialization overhead | HTTP proxy, with network + serialization overhead |
| **Control granularity** | EnvManager fully controls interaction details (multi-agent orchestration, fine-grained response_mask management, etc.) | Agent executes as black box, fully oriented toward business logic; EnvManager only focuses on sample construction |
| **Prompt construction** | Deeply integrated with `agent_template` / `tokenizer` | Agent constructs prompts independently; ProxyServer transparently intercepts |
| **Applicable environments** | Local gem.Env (Sokoban, FrozenLake, etc.); remote environments require forced alignment with existing abstractions, which is hard to understand | External agent frameworks, remote sandboxes, complex agents |
| **Extending new environments** | Must mix-implement three responsibilities within EnvManager | Only need to implement `AgentRunner.run_job()` |
| **Algorithm support** | Native support for GiGPO step-level decomposition | Supports both step/traj modes via `MessageTracker` |

**In summary**:
- **TrajEnvManager / StepEnvManager** targets **training research scenarios** -- requiring fine-grained control over the interaction process, custom response_mask construction, rapid multi-agent prototype validation, etc., providing the lowest overhead and tightest control
- **ProxyEnvManager + AgentRunner** targets **business logic scenarios** -- agent implementers focus entirely on agent behavior (how to call tools, how to interact with the environment), without needing to understand any training-side details

The two serve different scenario requirements and are not substitutes for each other.

### 1.6 Selection Guide

Choose the appropriate combination based on your scenario:

| Scenario | EnvManager | AgentRunner | Description |
|----------|-----------|-------------|-------------|
| Need fine-grained interaction control | `TrajEnvManager` | Not needed | Direct function calls, customizable response_mask, multi-agent orchestration |
| GiGPO step-level training | `StepEnvManager` | Not needed | Built-in `state_hash` grouping, step-level advantage estimation |
| Local gym environment (business logic priority) | `ProxyEnvManager` | `GEMRunner` | Agent only focuses on behavior; ProxyServer transparently collects trajectories |
| Local environment + function-calling | `ProxyEnvManager` | `ToolCallRunner` | Supports OpenAI tool_calls protocol |
| Remote sandbox + agent callback | `ProxyEnvManager` | `PushModeRunner` | Agent calls back via ALB/ingress |
| Remote sandbox + Roll polling | `ProxyEnvManager` | `PullModeRunner` | Roll drives inference via ModelService |
| Custom agent framework | `ProxyEnvManager` | Custom Runner | Only need to implement `run_job(seed)` |

## 2. AgentRunner Abstraction Design

### 2.1 Motivation

In the original architecture, the following issues arose when integrating new types of agents:

1. **Responsibility coupling**: TrajEnvManager subclasses simultaneously handle interaction loop, message formatting, and training sample construction -- adding new environments requires mixing all logic within a single class
2. **System fragmentation**: ProxyEnvManager uses `MessageTracker` for trajectory collection, which is completely different from TrajEnvManager's `RolloutCache` mechanism. Rock SDK sandbox scenarios are implemented via `HarborRunner`, but its interface (`submit_job(data_item, env_id)`) is too specific
3. **Agent implementers forced to deal with training details**: Agents need to understand training-side concepts like tokenizer, trajectory collection, and sample construction

The core idea of the AgentRunner abstraction is: **agent implementers focus entirely on agent business behavior (how to interact with the environment, how to call tools, when to terminate), without caring about how trajectories are collected or how training samples are constructed**. This allows business teams to write agent logic in the most natural way, while training infrastructure is handled transparently by the framework. Furthermore, since AgentRunner only calls LLMs through standard OpenAI-compatible interfaces, agents implemented based on AgentRunner can be used in both ROLL training environments and directly migrated to production environments -- simply switch `base_url` from ProxyServer to the actual inference service address, sharing the same agent code between training and production.

### 2.2 Separation of Concerns

```
┌─────────────────────────────────────────────────────────┐
│                    ProxyEnvManager                       │
│  ┌───────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ Episode       │  │ Trajectory   │  │ Sample       │  │
│  │ Scheduling    │  │ Collection   │  │ Construction │  │
│  │ run_rollout_  │  │ MessageTracker│  │ formulate_   │  │
│  │ loop()       │  │ ProxyServer   │  │ rollouts()   │  │
│  └───────┬───────┘  └──────┬───────┘  └──────────────┘  │
│          │                 │                             │
│          │  HTTP Intercept │                             │
│          ▼                 ▼                             │
│  ┌─────────────────────────────────────┐                │
│  │         AgentRunner                  │                │
│  │  ┌─────────┐  ┌──────────────────┐  │                │
│  │  │ run_job │  │ _llm_request     │  │                │
│  │  │ (seed)  │  │ (OpenAI compat)  │  │                │
│  │  └─────────┘  └──────────────────┘  │                │
│  └─────────────────────────────────────┘                │
└─────────────────────────────────────────────────────────┘
```

- **AgentRunner** is responsible for: executing a complete episode (loading data, env.reset → interaction loop → returning results)
- **ProxyServer** is responsible for: intercepting AgentRunner's LLM requests, transparently completing inference and trajectory recording
- **ProxyEnvManager** is responsible for: episode scheduling, training sample construction

### 2.3 Request Interception Mechanism

AgentRunner sends LLM inference requests to `base_url` through the standard OpenAI-compatible interface (`/v1/chat/completions`). `base_url` points to a local ProxyServer, which acts as a transparent middleware:

```
AgentRunner ──HTTP POST──▶ ProxyServer ──Intercept──▶ Record trajectory
                               │
                               └──▶ Forward to actual LLM inference backend
                               │
                               ◀──────────────────────────
                               │
AgentRunner ◀──HTTP Response── ProxyServer ◀─────── Return inference result
```

1. ProxyServer routes to the corresponding EnvManager via `Authorization: Bearer {env_id}`
2. EnvManager's `process_request` method completes tokenization, inference, and trajectory recording
3. Completely transparent to AgentRunner -- AgentRunner neither knows nor cares that the request was intercepted

## 3. AgentRunner Base Class

### 3.1 Core Interface

```python
# roll/pipeline/agentic/agent_runner/base.py

class EpisodeResult:
    """Structured result of an episode."""
    def __init__(
        self,
        status: str,                          # "Finished" | "Failed" | "NoData" | "Timeout"
        score: float,                         # Episode-level reward
        step_scores: List[float] = None,      # Per-step reward (optional)
        agent_exit_reason: str = "",
        metrics: Dict[str, Any] = None,       # Additional metrics (latency, pass rate, etc.)
    ): ...


class AgentRunner(ABC):
    """Abstract base class for agent interaction loops.

    Subclasses only need to implement the run_job(seed) method: given a seed,
    load data, execute a complete episode, and return an EpisodeResult.
    """

    def __init__(self, base_url: str, env_id: int, env_config: DictConfig, **kwargs):
        self.base_url = base_url    # ProxyServer address
        self.env_id = env_id        # Environment instance ID
        self.env_config = env_config

    @abstractmethod
    def run_job(self, seed: int) -> EpisodeResult:
        """Execute a complete episode."""
        ...

    def setup(self) -> None:
        """One-time initialization (create env instances, etc.)."""

    def teardown(self) -> None:
        """Clean up resources."""

    def _llm_request(self, client, messages, tools=None, tool_choice="auto") -> Dict:
        """Send an OpenAI-compatible chat completion request to base_url."""
```

### 3.2 Design Highlights

- **Clear `run_job(seed)` semantics**: The runner's sole responsibility is "run an episode." `seed` is the only required external input; data loading is handled internally by the runner
- **`env_id` bound to instance**: One runner instance corresponds to one env slot; `_llm_request` automatically uses `self.env_id` for routing
- **`_llm_request` is a protected method**: Encapsulates HTTP details; subclasses call it directly
- **Does not hold tokenizer / pipeline_config**: These are training-side concerns; AgentRunner only focuses on "executing episodes"

## 4. Built-in Runner Implementations

### 4.1 GEMRunner -- Local Gym Environments

`GEMRunner` is designed for local environments defined through the [GEM](https://github.com/axon-rl/gem) interface (Sokoban, FrozenLake, Math, etc.).

```python
# roll/pipeline/agentic/agent_runner/gem_runner.py

class GEMRunner(AgentRunner):
    """AgentRunner for local gem.Env environments.

    Interaction loop: env.reset(seed) → [construct messages → LLM request → env.step(action)] × N → EpisodeResult
    """

    def setup(self) -> None:
        self.env = gem.make(env_id=env_type, **env_params)
        # Optional: tool_wrapper wrapping

    def run_job(self, seed: int) -> EpisodeResult:
        obs_text, info = self.env.reset(seed=seed)
        messages = [{"role": "system", "content": system_template}, ...]

        for turn in range(max_steps):
            resp = self._llm_request(client, messages)       # → ProxyServer → actual inference
            action = resp["choices"][0]["message"]["content"]
            obs_text, reward, terminated, truncated, _ = self.env.step(action)
            if terminated or truncated:
                break

        return EpisodeResult(status="Finished", score=sum(rewards), step_scores=rewards)
```

#### ToolCallRunner

`ToolCallRunner` inherits from `GEMRunner` and supports the OpenAI function-calling protocol. When the LLM returns `tool_calls`, it passes the complete message dict to the environment for execution:

```python
class ToolCallRunner(GEMRunner):
    """GEM environment Runner using OpenAI function-calling protocol."""

    def run_job(self, seed: int) -> EpisodeResult:
        messages, info = self.env.reset(seed=seed)
        tools = info.get("tools")

        for _ in range(max_steps):
            resp = self._llm_request(client, messages, tools=tools)
            message = resp["choices"][0]["message"]
            action = message if message.get("tool_calls") else message["content"]
            messages, reward, terminated, truncated, _ = self.env.step(action)
            ...
```

### 4.2 RockAgentRunner -- Remote Sandbox Environments

`RockAgentRunner` is designed for Rock SDK sandbox environments (SWE-bench, TerminalBench, etc.), where agent code runs in remote sandbox containers.

```
RockAgentRunner (base class: job configuration, metrics extraction, data loading)
├── PushModeRunner    # Push mode: agent calls back to Roll's ProxyServer
└── PullModeRunner    # Pull mode: Roll actively polls the sandbox
```

#### Push Mode

The agent in the sandbox calls back to Roll's ProxyServer via ALB/ingress to obtain LLM inference. Flow:

```
PushModeRunner.run_job(seed)
  → Load data_item
  → Build JobConfig
  → Job.submit() → agent runs in sandbox
  → agent calls back to ProxyServer via HTTP for LLM inference
  → Job.wait() → extract metrics
  → Return EpisodeResult
```

#### Pull Mode

Roll actively polls the agent in the sandbox via ModelService, driving each inference step. Flow:

```
PullModeRunner.run_job(seed)
  → Load data_item
  → Build JobConfig + ModelServiceOperator
  → Job.submit()
  → _inference_loop():
      while True:
        request = model_service.anti_call_llm(index, response)  # Get agent's LLM request
        response = self._llm_request(client, request)           # Forward to ProxyServer
        # Repeat until SESSION_END
  → Job.wait() → extract metrics
  → Return EpisodeResult
```

### 4.3 Runner and ProxyEnvManager Collaboration

```python
# ProxyEnvManager.run_rollout_loop (simplified)
def run_rollout_loop(self, data):
    self.group_seed = data.meta_info['seed'] + self.env_config['group_seed']

    while True:
        self.episode_id = output_queue.get_episode_id(group_id, env_id)
        if self.episode_id is None:
            break

        self.message_tracker = MessageTracker(tokenizer, ...)
        seed = self.group_seed + self.episode_id

        # Core: only pass seed, entire episode executed by AgentRunner
        episode_result = self.agent_runner.run_job(seed)

        rollout = self.formulate_rollouts(episode_result.to_dict())
        output_queue.put(group_id, episode_id, rollout)
```

**Unchanged parts of ProxyEnvManager**:
- `process_request()` -- HTTP handler, responsible for LLM inference + trajectory recording
- `MessageTracker` -- incremental tokenization, trajectory collection, fork detection
- `formulate_rollouts()` -- training sample construction (supports both `step` and `traj` modes)
- ProxyServer routing mechanism -- routes to the corresponding handler via `env_id`

For detailed explanation of `MessageTracker`'s incremental tokenization, prefix aggregation, and fork detection mechanisms, see [Prefix Aggregation](prefix_aggregation.md).

## 5. Configuration and Usage

### 5.1 Configuration Method

Specify the Runner's fully qualified class path via `agent_runner_cls`, consistent with the `env_manager_cls` pattern, dynamically loaded by the framework via `safe_import_class`:

```yaml
# General configuration pattern
env_manager_cls: roll.pipeline.agentic.env_manager.proxy_env_manager.ProxyEnvManager
agent_runner_cls: <fully qualified class path of AgentRunner subclass>
```

### 5.2 GEMRunner Configuration Example

Suitable for local GEM environments like Sokoban, FrozenLake, using `ProxyEnvManager` + `GEMRunner`.

For the complete example, see `examples/qwen2.5-0.5B-agentic/agentic_val_sokoban_agent_runner.yaml`. Below are the key configuration snippets:

```yaml
# Global specification of EnvManager and AgentRunner
env_manager_cls: roll.pipeline.agentic.env_manager.proxy_env_manager.ProxyEnvManager
agent_runner_cls: roll.pipeline.agentic.agent_runner.gem_runner.GEMRunner

# Training parameters
rollout_batch_size: 1024
sequence_length: 8192
max_tokens_per_step: 64
adv_estimator: "grpo"

# Training environment group configuration
train_env_manager:
  max_env_num_per_worker: 16
  num_env_groups: 128
  group_size: 8                    # Sample 8 trajectories per prompt
  tags: [SimpleSokoban]
  num_groups_partition: [128]

# Validation environment group configuration (mixed evaluation across multiple environments)
val_env_manager:
  max_env_num_per_worker: 32
  num_env_groups: 1024
  group_size: 1
  tags: [SimpleSokoban, LargerSokoban, SokobanDifferentGridVocab, FrozenLake]
  num_groups_partition: [256, 256, 256, 256]

# Custom environments: each environment inherits global env_manager_cls and agent_runner_cls
custom_envs:
  SimpleSokoban:
    ${custom_env.SimpleSokoban}    # Reference environment definition from traj_envs.yaml
  LargerSokoban:
    ${custom_env.LargerSokoban}
  FrozenLake:
    ${custom_env.FrozenLake}
```

Key fields in environment definitions (from `examples/config/traj_envs.yaml`):

```yaml
custom_env:
  SimpleSokoban:
    env_type: sokoban
    max_steps: ${max_actions_per_traj}       # Maximum interaction steps
    max_tokens_per_step: ${max_tokens_per_step}
    env_manager_cls: ${env_manager_cls}      # Inherit global settings
    agent_runner_cls: ${agent_runner_cls}     # Inherit global settings
    agent_system_template: ${agent_system_template}
    agent_template: ${agent_template}        # Template used by GEMRunner to render observations
    env_config:                              # Environment parameters passed to gem.make()
      dim_room: [6, 6]
      num_boxes: 1
```

If the environment uses the function-calling protocol, replace `agent_runner_cls` with `ToolCallRunner`:

```yaml
agent_runner_cls: roll.pipeline.agentic.agent_runner.gem_runner.ToolCallRunner
```

### 5.3 RockAgentRunner Configuration Example

Suitable for remote sandbox environments like SWE-bench:

```yaml
# Push mode
env_manager_cls: roll.pipeline.agentic.env_manager.proxy_env_manager.ProxyEnvManager
agent_runner_cls: roll.pipeline.agentic.agent_runner.rock.push_runner.PushModeRunner

custom_envs:
  RockNativeEnv:
    env_type: rocknative
    max_steps: 60
    max_tokens_per_step: 4096
    env_manager_cls: ${env_manager_cls}
    agent_runner_cls: ${agent_runner_cls}
    env_config:
      dataset_name: data/swe_bench_test.jsonl
      max_iterations: 100
      tool_call_parser: qwen25
      trajectory_mode: traj        # "traj" or "step"
      reward_granularity: pass_rate # "pass_rate" or "binary"
```

```yaml
# Pull mode
agent_runner_cls: roll.pipeline.agentic.agent_runner.rock.pull_runner.PullModeRunner

custom_envs:
  RockNativeEnv:
    ...
    env_config:
      model_service_port: 28080    # ModelService listening port
      ...
```

### 5.4 TrajEnvManager Direct Mode

For simple scenarios that don't need AgentRunner, you can still use TrajEnvManager direct mode:

```yaml
env_manager_cls: roll.pipeline.agentic.env_manager.traj_env_manager.TrajEnvManager
agent_runner_cls: null  # AgentRunner not used

custom_envs:
  SimpleSokoban:
    env_type: sokoban
    max_steps: 10
    env_manager_cls: ${env_manager_cls}
    ...
```

## 6. Custom AgentRunner Development

### 6.1 Implementation Steps

Developing a custom AgentRunner requires only three steps:

**Step 1**: Inherit the `AgentRunner` base class

```python
from roll.pipeline.agentic.agent_runner.base import AgentRunner, EpisodeResult

class MyCustomRunner(AgentRunner):
    def __init__(self, base_url, env_id, env_config, **kwargs):
        super().__init__(base_url, env_id, env_config, **kwargs)
        # Custom initialization
```

**Step 2**: Implement the `run_job(seed)` method

```python
    def run_job(self, seed: int) -> EpisodeResult:
        # 1. Load data (if needed)
        data = self._load_data(seed)

        # 2. Initialize environment
        obs = self.env.reset(seed=seed)

        # 3. Interaction loop
        rewards = []
        with httpx.Client(timeout=3600.0) as client:
            for step in range(self.max_steps):
                messages = self._build_messages(obs)
                resp = self._llm_request(client, messages)  # Inference via ProxyServer
                action = resp["choices"][0]["message"]["content"]
                obs, reward, done, _, _ = self.env.step(action)
                rewards.append(reward)
                if done:
                    break

        # 4. Return result
        return EpisodeResult(
            status="Finished",
            score=sum(rewards),
            step_scores=rewards,
        )
```

**Step 3**: Specify in configuration

```yaml
agent_runner_cls: my_package.my_module.MyCustomRunner
```

### 6.2 Development Constraints

1. **Use `_llm_request` for LLM calls**: Ensure requests go through ProxyServer so that trajectories can be transparently collected
2. **Don't concern yourself with training-side details**: Don't hold tokenizer, don't construct training samples, don't manage trajectory data
3. **Return `EpisodeResult`**: Use the structured return value to make the runner's output contract explicit
4. **`seed` is the sole input**: Data loading and environment initialization should all be driven by seed to ensure reproducibility

## 7. Module Structure

```
roll/pipeline/agentic/agent_runner/
├── __init__.py                  # Exports AgentRunner, EpisodeResult
├── base.py                      # AgentRunner base class + EpisodeResult
├── gem_runner.py                # GEMRunner (text mode) + ToolCallRunner (function-calling mode)
└── rock/
    ├── __init__.py
    ├── rock_agent_runner.py     # RockAgentRunner base class (data loading, job config, metrics extraction)
    ├── push_runner.py           # PushModeRunner (agent callback mode)
    ├── pull_runner.py           # PullModeRunner + ModelServiceHarborTrial + ModelServiceOperator
    └── sandbox_tool_runner.py   # Sandbox tool runner extension
```

## Reference Examples

- GEMRunner + Sokoban: `examples/qwen2.5-0.5B-agentic/agentic_val_sokoban_agent_runner.yaml`
- PushModeRunner + SWE-bench: `examples/agentic_rock/rock_agent_runner_4b.yaml`
- PushModeRunner + 30A3: `examples/agentic_rock/rock_agent_runner_30a3.yaml`

## Related Documentation

- [Prefix Aggregation](prefix_aggregation.md) -- MessageTracker's incremental tokenization, prefix aggregation, and fork detection mechanisms
- [Agentic Engineering Practice](agentic_engineer_practice.md) -- EnvManager development protocol, GlobalDataset usage, trajectory synthesis, and other practical guides

## References

- [Getting Started with AgenticRL Using ROLL](https://zhuanlan.zhihu.com/p/2023433301961049206)
- [GEM Environment Library](https://github.com/axon-rl/gem)
