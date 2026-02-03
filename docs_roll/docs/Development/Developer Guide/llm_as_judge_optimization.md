# LLM as Judge Optimization in Agentic Environments

This document describes the optimized implementation of LLM as Judge in Agentic environments within the ROLL framework, including system architecture, call chains, configuration methods, and best practices.

## Overview

LLM as Judge is a method that uses large language models as evaluators to assess agent response quality. In Agentic training scenarios, when large-scale environment instances perform concurrent rollouts, using LLM as Judge to compute rewards generates massive concurrent LLM requests, which poses significant challenges to the stability and throughput of external LLM services.

To address this challenge, the ROLL framework implements a scalable **localized parallel evaluation system** through an **independent Reward Cluster** and **efficient scheduling mechanisms**, avoiding dependency on external services and ensuring the stability and controllability of the training process.

:::info Documentation Scope
This document uses the **DeepEyes environment's** LLM as Judge implementation as an example. For other environments that need LLM as Judge, you can refer to the calling patterns in `env_manager` and `env` to implement your own custom solutions.
:::

### Key Advantages

- **Independent Resource Management**: Reward model is separated from Policy model, allowing independent GPU resource allocation and avoiding resource contention
- **Localized Deployment**: Avoid external API dependencies through local Reward Cluster, ensuring service stability and data security
- **High Concurrency Support**: Efficient parallel reward evaluation through RequestScheduler, supporting scalable environment concurrency
- **Unified Interface Design**: Provides `generate_by_proxy` unified utility function, simplifying LLM calls and supporting both text and multimodal inputs
- **Flexible Configuration**: Supports multiple inference backends (vLLM, SGLang) and custom generation parameters

### Application Scenarios

Typical Agentic training scenarios:
- **Environment Scale**: 256 environment groups with 4 environments each, totaling 1024 concurrent environment instances
- **Rollout Frequency**: Each environment calls LLM Judge after completing an episode
- **Concurrency Pressure**: During rollout peaks, 500+ environments may simultaneously request reward evaluation
- **Stability Requirements**: Training process cannot be interrupted by external API rate limiting or timeouts

The optimized implementation described in this document effectively addresses these challenges.

## System Architecture

### Overall Architecture

```
AgenticPipeline
    ├── Reward Cluster (optional, independent GPU resources)
    │   ├── InferWorker (default)
    │   └── Supports vLLM/SGLang backends
    │
    ├── Reward Scheduler (Ray Named Actor)
    │   ├── Request routing and load balancing
    │   ├── Concurrency control
    │   └── Request tracking and cleanup
    │
    └── Environment Manager
        ├── llm_proxy: for policy inference
        ├── reward_proxy: for LLM as Judge
        └── env instances
            └── Call reward_proxy in obtain_outcome_reward
```

### Key Components

#### 1. Reward Cluster

**Location**: `roll/pipeline/agentic/agentic_pipeline.py:88-98`

Reward Cluster is an optional component, created only when `device_mapping` is configured:

```python
self.reward = None
if (self.pipeline_config.reward is not None and
    len(self.pipeline_config.reward.device_mapping) > 0):
    self.reward = Cluster(
        name=self.pipeline_config.reward.name,
        worker_cls=self.pipeline_config.reward.worker_cls,  # Default: InferWorker
        resource_manager=self.resource_manager,
        worker_config=self.pipeline_config.reward,
    )
```

**Worker Class Default Configuration**: `roll/pipeline/agentic/agentic_config.py:287`
- Defaults to `InferWorker` as inference engine, reusing ActorInfer Worker implementation
- Supports multiple backends including vLLM and SGLang

#### 2. Reward Scheduler (Ray Named Actor)

**Location**: `roll/pipeline/agentic/agentic_pipeline.py:112-125`

Reward Scheduler is created as a Ray Named Actor for shared access by all environment managers:

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

**Core Functionality**:

- **Smart Routing**: Uses least-loaded routing algorithm to distribute requests to different DP ranks
- **Sticky Routing**: Requests from the same environment are routed to the same worker (beneficial for KV cache)
- **Request Tracking**: Maintains mapping from `request_id` to workers

#### 3. Reward Proxy

**Location**: `roll/pipeline/agentic/env_manager/vl_traj_env_manager.py:85-109`

Environment manager retrieves Reward Scheduler via Ray and creates Reward Proxy:

```python
# Get reward scheduler from Ray (Named Actor)
if self.pipeline_config.reward:
    self.reward_scheduler = ray.get_actor(
        name=f"RewardScheduler-{pipeline_config.reward.name}",
        namespace=RAY_NAMESPACE
    )

    # Create reward proxy
    self.reward_proxy = create_llm_proxy(
        generate_scheduler=self.reward_scheduler,
        llm_proxy_config=pipeline_config.reward.llm_proxy,
        tokenizer=self.reward_tokenizer,
        env=None,
    )
```

**Proxy Factory Function**: `roll/pipeline/agentic/llm_proxy/__init__.py:11`
- Supports multiple proxy types: `policy`, `openai`, `random`
- Extensible through registration mechanism
- Policy proxy has been validated in training; for externally deployed LLM services, use openai proxy (note concurrency challenges)

#### 4. Unified Utility Function `generate_by_proxy`

**Location**: `roll/pipeline/agentic/llm_proxy/proxy_utils.py:18-170`

This is the core component called by environments, providing a unified LLM calling interface:

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

**Core Features**:

- **Unified Interface**: Same calling pattern for both text and multimodal inputs
- **Automatic Formatting**: Uses `tokenizer.apply_chat_template` to format messages
- **Multimodal Support**: Supports image/video inputs through `collator` parameter
- **Thinking Mechanism**: Supports chain-of-thought for models like DeepSeek and Qwen
- **Routing Control**: Implements sticky routing through `src_rank` parameter
- **Error Handling**: Returns `None` to indicate inference failure, handled by caller

## Call Chain

### Complete Call Flow

```
1. DeepEyesEnv.step() (env/deepeyes/env.py:182-197)
   Triggers obtain_outcome_reward when done=True
   ↓
2. DeepEyesEnv.obtain_outcome_reward() (env/deepeyes/env.py:199-254)
   Builds judge prompt, calls reward model
   ↓
3. generate_by_proxy() (llm_proxy/proxy_utils.py:18)
   Unified LLM calling utility function
   ↓
4. reward_proxy.generate() (llm_proxy/policy_proxy.py:15)
   Calls scheduler via Ray
   ↓
5. reward_scheduler.generate_one_request() (scheduler/generate_scheduler.py:1296)
   Request routing and load balancing
   ↓
6. infer_cluster.workers[dp_rank].generate_request()
   Actual model inference
   ↓
7. Returns LLM judgment result
```

## Configuration Guide

### Complete Configuration Example

```yaml
# Reward Configuration (LLM as Judge for AgenticPipeline)
reward:
  name: "reward"
  worker_cls: "roll.pipeline.base_worker.InferWorker"  # Default value, can be omitted
  model_args:
    model_name_or_path: Qwen/Qwen2.5-72B-Instruct
    dtype: bf16
  generating_args:
    max_new_tokens: 2048
    temperature: 0.2      # Lower temperature for stable judgments
    top_p: 0.95
    top_k: 20
  strategy_args:
    strategy_name: vllm   # or sglang
    strategy_config:
      gpu_memory_utilization: 0.8
      tensor_parallel_size: 4
      load_format: auto
  # Critical: Must be non-empty to create reward cluster
  device_mapping: list(range(8, 16))  # GPUs 8-15
  llm_proxy:
    proxy_type: policy  # Use policy proxy
```

### Configuration Key Points

#### 1. device_mapping (Required)

```yaml
# Recommended: Policy and Reward use independent GPUs
actor_infer:
  device_mapping: list(range(0, 8))   # GPUs 0-7

reward:
  device_mapping: list(range(8, 16))  # GPUs 8-15, independent resources
```

- **Empty or None**: Reward cluster not created, environments cannot use LLM as Judge
- **Non-empty**: Creates independent reward cluster, enables LLM as Judge
- **Independent Deployment**: Use different GPU resources from actor_infer. Policy inference and Reward evaluation run in parallel. actor_infer and reward must be deployed independently

#### 2. strategy_name (Inference Backend Selection)

```yaml
strategy_args:
  strategy_name: vllm   # or sglang
  strategy_config:
    gpu_memory_utilization: 0.8
    tensor_parallel_size: 4
    load_format: auto  # Must configure auto; vllm/sglang strategies default to dummy load which randomly initializes parameters
```

#### 3. generating_args (Generation Parameters)

```yaml
generating_args:
  max_new_tokens: 2048    # Adjust based on judge output length
  temperature: 0.2        # Lower temperature for stability
  top_p: 0.95
  top_k: 20
```

## Summary

The optimized LLM as Judge implementation in Agentic environments achieves efficient scalability through the following key designs:

1. **Independent Reward Cluster**: Resource isolation, avoiding competition with Policy inference
2. **Ray Named Actor**: Reward Scheduler as a shared service, accessible by all environments
3. **Unified Utility Function**: `generate_by_proxy` simplifies calls, supports text and multimodal
4. **Smart Routing**: Sticky routing and load balancing, improving cache utilization

By properly configuring and using these components, you can build an efficient and reliable LLM as Judge evaluation system.
