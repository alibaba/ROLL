# AgentRunner -- Agent 交互循环抽象

## 概述

ROLL 的 Agentic 训练框架支持多种类型的 agent 与环境交互：从本地 gym-like 游戏环境（Sokoban、FrozenLake）到远程沙箱中运行的 SWE-agent。在 ROLL 中，整个 agent rollout 过程是完全交由用户自定义的——框架不预设固定的交互模式，而是提供灵活的扩展点。但框架也不应该直接把 "如何编写 rollout loop" 这个难题丢给用户，因此 ROLL 基于最容易上手的多轮交互场景提供了常见的 rollout loop 实现（`TrajEnvManager` / `StepEnvManager`），业务方可以基于这些示例快速理解框架协议，再做业务自定义。

随着场景的丰富，原有的 `EnvManager` 体系在职责边界和扩展性上暴露出瓶颈。`AgentRunner` 抽象的引入，将 "agent 如何与环境交互" 的逻辑从 `EnvManager` 中解耦出来，使得不同类型的 agent 共享统一接口，同时保留 `EnvManager` 在训练样本构造上的核心能力。

AgentRunner 抽象的设计来源于对众多真实业务场景的应用实践梳理。感谢博客文章 [如何用ROLL快速上手AgenticRL](https://zhuanlan.zhihu.com/p/2023433301961049206) 中关于 Agentic 训练架构的讨论，为本设计提供了有价值的参考。

## 1. 架构背景：EnvManager 体系

在理解 AgentRunner 之前，需要先了解 ROLL 中 EnvManager 的职责和现有架构。

### 1.1 EnvManager 的核心职责

EnvManager 是 Agentic 训练的核心组件，负责三件事：

1. **交互循环**：驱动 agent 与环境的多轮交互（`reset` / `step` / `make_decision`）
2. **消息格式化**：将环境观测构造为 LLM 可理解的 prompt（`format_messages`）
3. **训练样本构造**：将交互轨迹转化为训练所需的 `DataProto`（`formulate_rollouts`）

### 1.2 现有继承体系

```
BaseEnvManager
├── TrajEnvManager                     # 轨迹级拼接，适用于 gym-like 环境
│   ├── StepEnvManager                 # 步级分解，适用于 GiGPO 算法
│   │   └── StepConcatEnvManager       # 步级 + 观测拼接变体
│   ├── AgentNativeStepEnvManager      # SWE/TerminalBench 原生环境
│   └── VLTrajEnvManager               # 视觉语言模型变体
```

### 1.3 TrajEnvManager

`TrajEnvManager` 是最基础的实现。它在一个紧凑的循环中完成所有工作：

```python
# TrajEnvManager 的核心循环（简化）
def run_rollout_loop(self, data):
    while self.running:
        rollout_cache = self.reset()              # env.reset(seed)
        while not done:
            lm_output = self.make_decision(cache)  # format_messages → LLM 推理
            rollout_cache = self.step(lm_output)   # env.step(action)
        rollout = self.formulate_rollouts(cache)   # 拼接完整轨迹 → DataProto
        output_queue.put(rollout)
```

**轨迹级样本构造**：将一个 episode 内所有步的 `prompt_ids + response_ids` 拼接成一条训练序列，episode reward 放置在最后一个 token 上。整个轨迹是一个训练样本。

**适用场景**：Sokoban、FrozenLake、WebShop 等 gym-like 环境。环境是轻量的 Python 对象，交互速度快，无需网络通信。

**精细控制能力**：由于 EnvManager 完全掌控交互循环的每一步，TrajEnvManager 特别适合需要精细控制交互过程的场景：
- **快速实现 multi-agent 原型**：可以在 `run_rollout_loop` 中自由编排多个 agent 的交互顺序和通信逻辑，不受黑盒接口限制
- **精细管理 `response_mask` 构造**：在 `formulate_rollouts` 中可以逐 token 控制哪些部分参与 loss 计算（例如只训练特定 turn 的 response、屏蔽 tool 输出等）
- **自定义 prompt 拼接策略**：在 `format_messages` 中可以灵活决定历史信息的保留、截断、摘要方式

### 1.4 StepEnvManager

`StepEnvManager` 继承自 `TrajEnvManager`，重写了 `format_messages` 和 `formulate_rollouts`，将粒度从"轨迹级"降到"步级"：

- **独立的步级 prompt**：每一步构造一个自包含的 prompt（包含系统指令、历史摘要、当前观测），而非拼接所有历史 token
- **步级训练样本**：每一步生成一个独立的训练样本，每个样本包含该步的 prompt + response 及对应的 step reward
- **`state_hash` 机制**：为每个 step 的状态计算哈希值，用于 GiGPO 算法中跨 rollout 的同状态分组优势估计

**适用场景**：需要步级优势估计的算法（如 GiGPO），或需要更细粒度信用分配的场景。

### 1.5 为什么保留 TrajEnvManager / StepEnvManager

TrajEnvManager / StepEnvManager 体系与 ProxyEnvManager + AgentRunner 并行存在，各有优势：

| 维度 | TrajEnvManager / StepEnvManager | ProxyEnvManager + AgentRunner |
|------|-------------------------------|-------------------------------|
| **交互开销** | 直接函数调用，零序列化开销 | HTTP 代理，有网络 + 序列化开销 |
| **控制粒度** | EnvManager 完全控制交互细节（multi-agent 编排、response_mask 精细管理等） | Agent 黑盒执行，完全面向业务逻辑，EnvManager 只关注样本构造 |
| **prompt 构造** | 深度集成 `agent_template` / `tokenizer` | Agent 自行构造 prompt，ProxyServer 透明拦截 |
| **适用环境** | 本地 gem.Env（Sokoban、FrozenLake 等），远程环境需要强行对齐现有抽象，难以理解 | 外部 agent 框架、远程沙箱、复杂 agent |
| **扩展新环境** | 需在 EnvManager 内混合实现三个职责 | 只需实现 `AgentRunner.run_job()` |
| **算法支持** | 原生支持 GiGPO 步级分解 | 通过 `MessageTracker` 支持 step/traj 两种模式 |

**简单总结**：
- **TrajEnvManager / StepEnvManager** 面向**训练研究场景**——需要精细控制交互过程、自定义 response_mask 构造、快速验证 multi-agent 原型等需求，提供最低开销和最紧密的控制
- **ProxyEnvManager + AgentRunner** 面向**业务逻辑场景**——agent 实现者完全只关注 agent 的行为（如何调用工具、如何与环境交互），不需要了解训练侧的任何细节

两者服务于不同的场景需求，并非替代关系。

### 1.6 选型指南

根据你的场景选择合适的组合：

| 场景 | EnvManager | AgentRunner | 说明 |
|------|-----------|-------------|------|
| 需要精细控制交互过程 | `TrajEnvManager` | 不需要 | 直接函数调用，可自定义 response_mask、multi-agent 编排 |
| GiGPO 步级训练 | `StepEnvManager` | 不需要 | 内置 `state_hash` 分组，步级优势估计 |
| 本地 gym 环境（业务逻辑优先） | `ProxyEnvManager` | `GEMRunner` | agent 只关注行为，ProxyServer 透明收集轨迹 |
| 本地环境 + function-calling | `ProxyEnvManager` | `ToolCallRunner` | 支持 OpenAI tool_calls 协议 |
| 远程沙箱 + agent 回调 | `ProxyEnvManager` | `PushModeRunner` | agent 通过 ALB/ingress 回调 |
| 远程沙箱 + Roll 轮询 | `ProxyEnvManager` | `PullModeRunner` | Roll 通过 ModelService 驱动推理 |
| 自定义 agent 框架 | `ProxyEnvManager` | 自定义 Runner | 只需实现 `run_job(seed)` |

## 2. AgentRunner 抽象设计

### 2.1 引入动机

原有架构中，当需要接入新类型的 agent 时，存在以下问题：

1. **职责耦合**：TrajEnvManager 子类同时承担交互循环、消息格式化、训练样本构造三个职责，新增环境需要在一个类中混合实现所有逻辑
2. **体系割裂**：ProxyEnvManager 使用 `MessageTracker` 收集轨迹，与 TrajEnvManager 的 `RolloutCache` 机制完全不同。Rock SDK 沙箱场景通过 `HarborRunner` 实现，但其接口（`submit_job(data_item, env_id)`）过于具体
3. **agent 实现者被迫关注训练细节**：agent 需要了解 tokenizer、轨迹收集、样本构造等训练侧概念

AgentRunner 抽象的核心思路是：**agent 实现者完全只关注 agent 的业务行为（如何与环境交互、如何调用工具、何时终止），不关心轨迹如何被收集、训练样本如何被构造**。这使得业务团队可以用最自然的方式编写 agent 逻辑，而训练基础设施由框架透明处理。更进一步，由于 AgentRunner 仅通过标准 OpenAI 兼容接口调用 LLM，基于 AgentRunner 实现的 agent 既可以在 ROLL 训练环境中使用，也可以直接迁移到生产环境——只需将 `base_url` 从 ProxyServer 切换为实际的推理服务地址即可，训练与生产共享同一套 agent 代码。

### 2.2 关注点分离

```
┌─────────────────────────────────────────────────────────┐
│                    ProxyEnvManager                       │
│  ┌───────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ Episode 调度  │  │ 轨迹收集     │  │ 样本构造     │  │
│  │ run_rollout_  │  │ MessageTracker│  │ formulate_   │  │
│  │ loop()       │  │ ProxyServer   │  │ rollouts()   │  │
│  └───────┬───────┘  └──────┬───────┘  └──────────────┘  │
│          │                 │                             │
│          │    HTTP 拦截    │                             │
│          ▼                 ▼                             │
│  ┌─────────────────────────────────────┐                │
│  │         AgentRunner                  │                │
│  │  ┌─────────┐  ┌──────────────────┐  │                │
│  │  │ run_job │  │ _llm_request     │  │                │
│  │  │ (seed)  │  │ (OpenAI 兼容)    │  │                │
│  │  └─────────┘  └──────────────────┘  │                │
│  └─────────────────────────────────────┘                │
└─────────────────────────────────────────────────────────┘
```

- **AgentRunner** 负责：执行完整的 episode（加载数据、env.reset → 交互循环 → 返回结果）
- **ProxyServer** 负责：拦截 AgentRunner 的 LLM 请求，透明地完成推理和轨迹记录
- **ProxyEnvManager** 负责：episode 调度、训练样本构造

### 2.3 请求拦截机制

AgentRunner 通过标准 OpenAI 兼容接口（`/v1/chat/completions`）向 `base_url` 发起 LLM 推理请求。`base_url` 指向本地的 ProxyServer，后者作为透明中间层：

```
AgentRunner ──HTTP POST──▶ ProxyServer ──拦截──▶ 记录轨迹
                               │
                               └──▶ 转发到实际 LLM 推理后端
                               │
                               ◀──────────────────────────
                               │
AgentRunner ◀──HTTP Response── ProxyServer ◀─────── 返回推理结果
```

1. ProxyServer 通过 `Authorization: Bearer {env_id}` 路由到对应的 EnvManager
2. EnvManager 的 `process_request` 方法完成 tokenize、推理、轨迹记录
3. 对 AgentRunner 完全透明——AgentRunner 不知道也不关心请求被拦截

## 3. AgentRunner 基类

### 3.1 核心接口

```python
# roll/pipeline/agentic/agent_runner/base.py

class EpisodeResult:
    """一个 episode 的结构化结果。"""
    def __init__(
        self,
        status: str,                          # "Finished" | "Failed" | "NoData" | "Timeout"
        score: float,                         # episode 级别的 reward
        step_scores: List[float] = None,      # 每步 reward（可选）
        agent_exit_reason: str = "",
        metrics: Dict[str, Any] = None,       # 额外指标（耗时、通过率等）
    ): ...


class AgentRunner(ABC):
    """Agent 交互循环的抽象基类。

    子类只需实现 run_job(seed) 方法：给定一个 seed，加载数据、执行完整的 episode，
    返回 EpisodeResult。
    """

    def __init__(self, base_url: str, env_id: int, env_config: DictConfig, **kwargs):
        self.base_url = base_url    # ProxyServer 地址
        self.env_id = env_id        # 环境实例 ID
        self.env_config = env_config

    @abstractmethod
    def run_job(self, seed: int) -> EpisodeResult:
        """执行一个完整的 episode。"""
        ...

    def setup(self) -> None:
        """一次性初始化（创建 env 实例等）。"""

    def teardown(self) -> None:
        """清理资源。"""

    def _llm_request(self, client, messages, tools=None, tool_choice="auto") -> Dict:
        """向 base_url 发起 OpenAI 兼容的 chat completion 请求。"""
```

### 3.2 设计要点

- **`run_job(seed)` 语义清晰**：runner 的唯一职责是 "运行一个 episode"。`seed` 是唯一必需的外部输入，数据加载由 runner 内部完成
- **`env_id` 绑定到实例**：一个 runner 实例对应一个 env slot，`_llm_request` 自动使用 `self.env_id` 进行路由
- **`_llm_request` 是 protected 方法**：封装 HTTP 细节，子类直接调用即可
- **不持有 tokenizer / pipeline_config**：这些是训练侧关注点，AgentRunner 只面向 "执行 episode"

## 4. 内置 Runner 实现

### 4.1 GEMRunner -- 本地 gym 环境

`GEMRunner` 适用于通过 [GEM](https://github.com/axon-rl/gem) 接口定义的本地环境（Sokoban、FrozenLake、Math 等）。

```python
# roll/pipeline/agentic/agent_runner/gem_runner.py

class GEMRunner(AgentRunner):
    """本地 gem.Env 环境的 AgentRunner。

    交互循环: env.reset(seed) → [构造 messages → LLM request → env.step(action)] × N → EpisodeResult
    """

    def setup(self) -> None:
        self.env = gem.make(env_id=env_type, **env_params)
        # 可选：tool_wrapper 包装

    def run_job(self, seed: int) -> EpisodeResult:
        obs_text, info = self.env.reset(seed=seed)
        messages = [{"role": "system", "content": system_template}, ...]

        for turn in range(max_steps):
            resp = self._llm_request(client, messages)       # → ProxyServer → 实际推理
            action = resp["choices"][0]["message"]["content"]
            obs_text, reward, terminated, truncated, _ = self.env.step(action)
            if terminated or truncated:
                break

        return EpisodeResult(status="Finished", score=sum(rewards), step_scores=rewards)
```

#### ToolCallRunner

`ToolCallRunner` 继承自 `GEMRunner`，支持 OpenAI function-calling 协议。当 LLM 返回 `tool_calls` 时，将完整的 message dict 传给环境执行：

```python
class ToolCallRunner(GEMRunner):
    """使用 OpenAI function-calling 协议的 GEM 环境 Runner。"""

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

### 4.2 RockAgentRunner -- 远程沙箱环境

`RockAgentRunner` 适用于 Rock SDK 沙箱环境（SWE-bench、TerminalBench 等），agent 代码运行在远程沙箱容器中。

```
RockAgentRunner (基类：job 配置、metrics 提取、数据加载)
├── PushModeRunner    # Push 模式：agent 回调 Roll 的 ProxyServer
└── PullModeRunner    # Pull 模式：Roll 主动轮询 sandbox
```

#### Push 模式

agent 在沙箱内通过 ALB/ingress 回调 Roll 的 ProxyServer 来获取 LLM 推理。流程：

```
PushModeRunner.run_job(seed)
  → 加载 data_item
  → 构建 JobConfig
  → Job.submit() → agent 在沙箱中运行
  → agent 通过 HTTP 回调 ProxyServer 获取 LLM 推理
  → Job.wait() → 提取 metrics
  → 返回 EpisodeResult
```

#### Pull 模式

Roll 通过 ModelService 主动轮询沙箱中的 agent，驱动每一步推理。流程：

```
PullModeRunner.run_job(seed)
  → 加载 data_item
  → 构建 JobConfig + ModelServiceOperator
  → Job.submit()
  → _inference_loop():
      while True:
        request = model_service.anti_call_llm(index, response)  # 获取 agent 的 LLM 请求
        response = self._llm_request(client, request)           # 转发到 ProxyServer
        # 重复直到 SESSION_END
  → Job.wait() → 提取 metrics
  → 返回 EpisodeResult
```

### 4.3 Runner 与 ProxyEnvManager 的协作

```python
# ProxyEnvManager.run_rollout_loop (简化)
def run_rollout_loop(self, data):
    self.group_seed = data.meta_info['seed'] + self.env_config['group_seed']

    while True:
        self.episode_id = output_queue.get_episode_id(group_id, env_id)
        if self.episode_id is None:
            break

        self.message_tracker = MessageTracker(tokenizer, ...)
        seed = self.group_seed + self.episode_id

        # 核心：只传 seed，整个 episode 由 AgentRunner 执行
        episode_result = self.agent_runner.run_job(seed)

        rollout = self.formulate_rollouts(episode_result.to_dict())
        output_queue.put(group_id, episode_id, rollout)
```

**ProxyEnvManager 不变的部分**：
- `process_request()` — HTTP handler，负责 LLM 推理 + 轨迹记录
- `MessageTracker` — 增量 tokenize、轨迹收集、fork 检测
- `formulate_rollouts()` — 训练样本构造（支持 `step` 和 `traj` 两种模式）
- ProxyServer 路由机制 — 通过 `env_id` 路由到对应的 handler

关于 `MessageTracker` 的增量 tokenization、前缀聚合和分叉检测机制的详细说明，请参阅 [前缀聚合 (Prefix Aggregation)](prefix_aggregation.md)。

## 5. 配置与使用

### 5.1 配置方式

通过 `agent_runner_cls` 指定 Runner 的全限定类路径，与 `env_manager_cls` 模式一致，由框架通过 `safe_import_class` 动态加载：

```yaml
# 通用配置模式
env_manager_cls: roll.pipeline.agentic.env_manager.proxy_env_manager.ProxyEnvManager
agent_runner_cls: <AgentRunner 子类的全限定类路径>
```

### 5.2 GEMRunner 配置示例

适用于 Sokoban、FrozenLake 等本地 GEM 环境，使用 `ProxyEnvManager` + `GEMRunner`。

完整示例参见 `examples/qwen2.5-0.5B-agentic/agentic_val_sokoban_agent_runner.yaml`，以下是关键配置片段：

```yaml
# 全局指定 EnvManager 和 AgentRunner
env_manager_cls: roll.pipeline.agentic.env_manager.proxy_env_manager.ProxyEnvManager
agent_runner_cls: roll.pipeline.agentic.agent_runner.gem_runner.GEMRunner

# 训练参数
rollout_batch_size: 1024
sequence_length: 8192
max_tokens_per_step: 64
adv_estimator: "grpo"

# 训练环境组配置
train_env_manager:
  max_env_num_per_worker: 16
  num_env_groups: 128
  group_size: 8                    # 同一 prompt 采样 8 条轨迹
  tags: [SimpleSokoban]
  num_groups_partition: [128]

# 验证环境组配置（多种环境混合评估）
val_env_manager:
  max_env_num_per_worker: 32
  num_env_groups: 1024
  group_size: 1
  tags: [SimpleSokoban, LargerSokoban, SokobanDifferentGridVocab, FrozenLake]
  num_groups_partition: [256, 256, 256, 256]

# 自定义环境：每个环境继承全局的 env_manager_cls 和 agent_runner_cls
custom_envs:
  SimpleSokoban:
    ${custom_env.SimpleSokoban}    # 引用 traj_envs.yaml 中的环境定义
  LargerSokoban:
    ${custom_env.LargerSokoban}
  FrozenLake:
    ${custom_env.FrozenLake}
```

其中环境定义（来自 `examples/config/traj_envs.yaml`）的关键字段：

```yaml
custom_env:
  SimpleSokoban:
    env_type: sokoban
    max_steps: ${max_actions_per_traj}       # 最大交互步数
    max_tokens_per_step: ${max_tokens_per_step}
    env_manager_cls: ${env_manager_cls}      # 继承全局设置
    agent_runner_cls: ${agent_runner_cls}     # 继承全局设置
    agent_system_template: ${agent_system_template}
    agent_template: ${agent_template}        # GEMRunner 用于渲染观测的模板
    env_config:                              # 传给 gem.make() 的环境参数
      dim_room: [6, 6]
      num_boxes: 1
```

如果环境使用 function-calling 协议，将 `agent_runner_cls` 替换为 `ToolCallRunner`：

```yaml
agent_runner_cls: roll.pipeline.agentic.agent_runner.gem_runner.ToolCallRunner
```

### 5.3 RockAgentRunner 配置示例

适用于 SWE-bench 等远程沙箱环境：

```yaml
# Push 模式
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
      trajectory_mode: traj        # "traj" 或 "step"
      reward_granularity: pass_rate # "pass_rate" 或 "binary"
```

```yaml
# Pull 模式
agent_runner_cls: roll.pipeline.agentic.agent_runner.rock.pull_runner.PullModeRunner

custom_envs:
  RockNativeEnv:
    ...
    env_config:
      model_service_port: 28080    # ModelService 监听端口
      ...
```

### 5.4 TrajEnvManager 直接模式

对于不需要 AgentRunner 的简单场景，仍然可以使用 TrajEnvManager 直接模式：

```yaml
env_manager_cls: roll.pipeline.agentic.env_manager.traj_env_manager.TrajEnvManager
agent_runner_cls: null  # 不使用 AgentRunner

custom_envs:
  SimpleSokoban:
    env_type: sokoban
    max_steps: 10
    env_manager_cls: ${env_manager_cls}
    ...
```

## 6. 自定义 AgentRunner 开发

### 6.1 实现步骤

开发自定义 AgentRunner 只需三步：

**Step 1**：继承 `AgentRunner` 基类

```python
from roll.pipeline.agentic.agent_runner.base import AgentRunner, EpisodeResult

class MyCustomRunner(AgentRunner):
    def __init__(self, base_url, env_id, env_config, **kwargs):
        super().__init__(base_url, env_id, env_config, **kwargs)
        # 自定义初始化
```

**Step 2**：实现 `run_job(seed)` 方法

```python
    def run_job(self, seed: int) -> EpisodeResult:
        # 1. 加载数据（如果需要）
        data = self._load_data(seed)

        # 2. 初始化环境
        obs = self.env.reset(seed=seed)

        # 3. 交互循环
        rewards = []
        with httpx.Client(timeout=3600.0) as client:
            for step in range(self.max_steps):
                messages = self._build_messages(obs)
                resp = self._llm_request(client, messages)  # 通过 ProxyServer 推理
                action = resp["choices"][0]["message"]["content"]
                obs, reward, done, _, _ = self.env.step(action)
                rewards.append(reward)
                if done:
                    break

        # 4. 返回结果
        return EpisodeResult(
            status="Finished",
            score=sum(rewards),
            step_scores=rewards,
        )
```

**Step 3**：在配置中指定

```yaml
agent_runner_cls: my_package.my_module.MyCustomRunner
```

### 6.2 开发约束

1. **使用 `_llm_request` 发起 LLM 调用**：确保请求经过 ProxyServer，使轨迹能被透明收集
2. **不关心训练侧细节**：不持有 tokenizer、不构造训练样本、不管理轨迹数据
3. **返回 `EpisodeResult`**：使用结构化返回值，明确 runner 的输出契约
4. **`seed` 是唯一输入**：数据加载、环境初始化都应通过 seed 驱动，保证可复现性

## 7. 模块结构

```
roll/pipeline/agentic/agent_runner/
├── __init__.py                  # 导出 AgentRunner, EpisodeResult
├── base.py                      # AgentRunner 基类 + EpisodeResult
├── gem_runner.py                # GEMRunner（文本模式）+ ToolCallRunner（function-calling 模式）
└── rock/
    ├── __init__.py
    ├── rock_agent_runner.py     # RockAgentRunner 基类（数据加载、job 配置、metrics 提取）
    ├── push_runner.py           # PushModeRunner（agent 回调模式）
    ├── pull_runner.py           # PullModeRunner + ModelServiceHarborTrial + ModelServiceOperator
    └── sandbox_tool_runner.py   # 沙箱 tool runner 扩展
```

## 参考示例

- GEMRunner + Sokoban: `examples/qwen2.5-0.5B-agentic/agentic_val_sokoban_agent_runner.yaml`
- PushModeRunner + SWE-bench: `examples/agentic_rock/rock_agent_runner_4b.yaml`
- PushModeRunner + 30A3: `examples/agentic_rock/rock_agent_runner_30a3.yaml`

## 相关文档

- [前缀聚合 (Prefix Aggregation)](prefix_aggregation.md) — MessageTracker 的增量 tokenization、前缀聚合和分叉检测机制
- [Agentic 工程实践文档](agentic_engineer_practice.md) — EnvManager 开发协议、GlobalDataset 使用、轨迹合成等实践指南

## 参考文献

- [如何用ROLL快速上手AgenticRL](https://zhuanlan.zhihu.com/p/2023433301961049206)
- [GEM 环境库](https://github.com/axon-rl/gem)
