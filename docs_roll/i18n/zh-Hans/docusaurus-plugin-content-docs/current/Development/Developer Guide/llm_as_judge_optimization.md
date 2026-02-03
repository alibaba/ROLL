# LLM as Judge 在 Agentic 环境中的优化实现

本文档介绍 ROLL 框架中 LLM as Judge 在 Agentic 环境中的优化实现方案，包括系统架构、调用链路、配置方法和最佳实践。

## 概览

LLM as Judge 是一种使用大语言模型作为评判器来评估智能体响应质量的方法。在 Agentic 训练场景中，大规模环境实例并发执行 rollout 时，使用 LLM as Judge 计算 reward 会产生大量并发 LLM 请求，这对外部 LLM 服务的稳定性和吞吐量提出了巨大挑战。

为解决这一问题，ROLL 框架通过**独立的 Reward Cluster** 和**高效的调度机制**，实现了可扩展的本地化并行评估系统，避免了对外部服务的依赖，确保了训练过程的稳定性和可控性。

:::info 文档说明
本文档以 **DeepEyes 环境**的 LLM as Judge 实现为例进行说明。对于其他需要使用 LLM as Judge 的环境，可以参考 `env_manager` 和 `env` 内的调用方式自定义实现。
:::

### 核心优势

- **独立资源管理**：Reward 模型与 Policy 模型分离，可独立分配 GPU 资源，避免资源竞争
- **本地化部署**：通过本地 Reward Cluster 避免外部 API 依赖，保证服务稳定性和数据安全
- **高并发支持**：通过 RequestScheduler 实现多环境并行的高效 reward 评估，支持环境并发扩展
- **统一接口设计**：提供 `generate_by_proxy` 统一工具函数，简化 LLM 调用逻辑，支持文本和多模态
- **灵活配置**：支持多种推理后端（vLLM、SGLang）和自定义生成参数

### 应用场景

典型的 Agentic 训练场景：
- **环境规模**：256个环境组，每组 4 个环境，共 1024个并发环境实例
- **Rollout 频率**：每个环境完成 episode 后调用 LLM Judge
- **并发压力**：在 rollout 高峰期可能有 500+ 个环境同时请求 reward 评估
- **稳定性要求**：训练过程不能因为外部 API 限流或超时而中断

通过本文档介绍的优化实现，可以有效应对上述挑战。

## 系统架构

### 整体架构

```
AgenticPipeline
    ├── Reward Cluster (可选，独立GPU资源)
    │   ├── InferWorker (默认)
    │   └── 支持 vLLM/SGLang 后端
    │
    ├── Reward Scheduler (Ray Named Actor)
    │   ├── 请求路由与负载均衡
    │   ├── 并发控制
    │   └── 请求追踪与清理
    │
    └── Environment Manager
        ├── llm_proxy: 用于 policy 推理
        ├── reward_proxy: 用于 LLM as Judge
        └── env实例
            └── 在 obtain_outcome_reward 中调用 reward_proxy
```

### 关键组件

#### 1. Reward Cluster

**位置**: `roll/pipeline/agentic/agentic_pipeline.py:88-98`

Reward Cluster 是可选组件，仅在配置了 `device_mapping` 时创建：

```python
self.reward = None
if (self.pipeline_config.reward is not None and
    len(self.pipeline_config.reward.device_mapping) > 0):
    self.reward = Cluster(
        name=self.pipeline_config.reward.name,
        worker_cls=self.pipeline_config.reward.worker_cls,  # 默认 InferWorker
        resource_manager=self.resource_manager,
        worker_config=self.pipeline_config.reward,
    )
```

**Worker Class 默认配置**: `roll/pipeline/agentic/agentic_config.py:287`
- 默认使用 `InferWorker` 作为推理引擎，复用ActorInfer Worker实现
- 支持 vLLM、SGLang等多种后端

#### 2. Reward Scheduler (Ray Named Actor)

**位置**: `roll/pipeline/agentic/agentic_pipeline.py:112-125`

Reward Scheduler 作为 Ray Named Actor 创建，供所有环境管理器共享访问：

```python
self.reward_scheduler = RequestScheduler.options(
    name=f"RewardScheduler-{self.pipeline_config.reward.name}",
    get_if_exists=True,
    namespace=RAY_NAMESPACE,
    scheduling_strategy=NodeAffinitySchedulingStrategy(...)
).remote(
    infer_cluster=self.reward,
    pipeline_config=self.pipeline_config,
    resource_manager=self.resource_manager,
)
```

**核心功能**:

- **智能路由**: 使用最少负载路由算法分配请求到不同的 DP rank
- **粘性路由**: 同一环境的请求固定路由到同一 worker（利于 KV cache）
- **请求追踪**: 维护 `request_id` 到 worker 的映射关系

#### 3. Reward Proxy

**位置**: `roll/pipeline/agentic/env_manager/vl_traj_env_manager.py:85-109`

环境管理器通过 Ray 获取 Reward Scheduler 并创建 Reward Proxy：

```python
# 从 Ray 获取 reward scheduler (Named Actor)
if self.pipeline_config.reward:
    self.reward_scheduler = ray.get_actor(
        name=f"RewardScheduler-{pipeline_config.reward.name}",
        namespace=RAY_NAMESPACE
    )

    # 创建 reward proxy
    self.reward_proxy = create_llm_proxy(
        generate_scheduler=self.reward_scheduler,
        llm_proxy_config=pipeline_config.reward.llm_proxy,
        tokenizer=self.reward_tokenizer,
        env=None,
    )
```

**Proxy 工厂函数**: `roll/pipeline/agentic/llm_proxy/__init__.py:11`
- 支持多种 proxy 类型：`policy`、`openai`、`random`
- 通过注册机制实现可扩展性
- 训练验证过policy设置功能正常，基于外部部署的大模型服务可使用openai proxy，注意对并发的挑战

#### 4. 统一工具函数 `generate_by_proxy`

**位置**: `roll/pipeline/agentic/llm_proxy/proxy_utils.py:18-170`

这是env调用的核心组件，提供统一的 LLM 调用接口：

```python
def generate_by_proxy(
    messages: List[Dict[str, Any]],
    tokenizer: PreTrainedTokenizer,
    proxy: BaseLLMProxy,
    enable_thinking: bool = False,
    generation_config: Optional[Dict[str, Any]] = None,
    collator: Optional[Any] = None,
    mm_data: Optional[Dict[str, Any]] = None,
    src_rank: Optional[int] = None,
) -> Optional[str]
```

**核心特性**:

- **统一接口**: 无论文本还是多模态，都使用相同的调用方式
- **自动格式化**: 使用 `tokenizer.apply_chat_template` 格式化消息
- **多模态支持**: 通过 `collator` 参数支持图像/视频输入
- **thinking 机制**: 支持 DeepSeek、Qwen 等模型的思考链
- **路由控制**: 通过 `src_rank` 参数实现粘性路由
- **错误处理**: 返回 `None` 表示推理失败，由调用方处理

## 调用链路

### 完整调用流程

```
1. DeepEyesEnv.step() (env/deepeyes/env.py:182-197)
   当 done=True 时触发 obtain_outcome_reward
   ↓
2. DeepEyesEnv.obtain_outcome_reward() (env/deepeyes/env.py:199-254)
   构建 judge prompt，调用 reward model
   ↓
3. generate_by_proxy() (llm_proxy/proxy_utils.py:18)
   统一的 LLM 调用工具函数
   ↓
4. reward_proxy.generate() (llm_proxy/policy_proxy.py:15)
   通过 Ray 调用 scheduler
   ↓
5. reward_scheduler.generate_one_request() (scheduler/generate_scheduler.py:1296)
   请求路由与负载均衡
   ↓
6. infer_cluster.workers[dp_rank].generate_request()
   实际的模型推理
   ↓
7. 返回 LLM 判断结果
```

## 配置说明

### 完整配置示例

```yaml
# Reward 配置 (LLM as Judge for AgenticPipeline)
reward:
  name: "reward"
  worker_cls: "roll.pipeline.base_worker.InferWorker"  # 默认值，可省略
  model_args:
    model_name_or_path: Qwen/Qwen2.5-72B-Instruct
    dtype: bf16
  generating_args:
    max_new_tokens: 2048
    temperature: 0.2      # 较低温度提高判断稳定性
    top_p: 0.95
    top_k: 20
  strategy_args:
    strategy_name: vllm   # 或 sglang
    strategy_config:
      gpu_memory_utilization: 0.8
      tensor_parallel_size: 4
      load_format: auto
  # 关键：必须非空才会创建 reward cluster
  device_mapping: list(range(8, 16))  # GPUs 8-15
  llm_proxy:
    proxy_type: policy  # 使用 policy proxy
```

### 配置关键点

#### 1. device_mapping（必须配置）

```yaml
# 推荐配置：Policy 和 Reward 使用独立 GPU
actor_infer:
  device_mapping: list(range(0, 8))   # GPUs 0-7

reward:
  device_mapping: list(range(8, 16))  # GPUs 8-15，独立资源
```

- **空或 None**: 不创建 reward cluster，环境无法使用 LLM as Judge
- **非空**: 创建独立的 reward cluster，支持 LLM as Judge
- **独立部署**: 与 actor_infer 使用不同的 GPU 资源，Policy 推理和 Reward 评估并行执行，actor_infer与reward必须得独立部署

#### 2. strategy_name（推理后端选择）

```yaml
strategy_args:
  strategy_name: vllm   # 或 sglang
  strategy_config:
    gpu_memory_utilization: 0.8
    tensor_parallel_size: 4
    load_format: auto	# 必须配置auto, vllm/sglang strategy里默认使用dummy load，会随机初始化参数
```

#### 3. generating_args（生成参数）

```yaml
generating_args:
  max_new_tokens: 2048    # 根据 judge 输出长度调整
  temperature: 0.2        # 较低温度提高稳定性
  top_p: 0.95
  top_k: 20
```

## 总结

LLM as Judge 在 Agentic 环境中的优化实现通过以下关键设计实现高效可扩展：

1. **独立 Reward Cluster**: 资源隔离，避免与 Policy 推理竞争
2. **Ray Named Actor**: Reward Scheduler 作为共享服务，供所有环境访问
3. **统一工具函数**: `generate_by_proxy` 简化调用，支持文本和多模态
4. **智能路由**: 粘性路由和负载均衡，提高缓存利用率

通过合理配置和使用这些组件，可以构建高效、可靠的 LLM as Judge 评估系统。
