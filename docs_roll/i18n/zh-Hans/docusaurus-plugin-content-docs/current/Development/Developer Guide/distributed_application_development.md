# ROLL 分布式编程模型与应用开发

ROLL 将分布式应用拆分为若干职责清晰的抽象层。对于模型训练和强化学习任务，最常见的调用关系是：

```text
Pipeline
  └─ Cluster
       └─ Worker（Ray Actor）
            └─ Strategy（可选）
```

其中：

- **Pipeline** 编排完整业务流程以及不同角色之间的数据流。
- **Cluster** 管理同一角色的一组 Worker，并负责调用分发和结果收集。
- **Worker** 定义一个分布式角色提供的业务能力。
- **Strategy** 以统一接口封装不同模型训练或推理后端。该层是可选的。

此外，`ResourceManager` 负责将 Cluster 放置到具体节点和设备，`DataProto` 负责在各层之间传递结构化数据。

本文介绍如何使用这些抽象开发新的 Pipeline，以及应该在哪一层实现不同类型的扩展。

## Strategy 是可选抽象

ROLL 中的大语言模型训练和强化学习 Pipeline 具有相对统一的计算范式：加载模型、前向计算、生成、反向传播、保存检查点、卸载状态以及同步权重。FSDP2、Megatron、vLLM、SGLang 和 Hugging Face 等后端虽然实现不同，但可以自然地抽象为一组稳定接口，因此 ROLL 提供了 Strategy 层。

例如，同一个 Worker 可以通过如下接口执行训练：

```python
metrics = self.strategy.train_step(
    batch=data,
    loss_func=self.loss_func,
)
```

具体使用 FSDP2 还是 Megatron，由 `strategy_args.strategy_name` 决定，而 Pipeline 的算法流程和 Worker 的业务接口可以保持稳定。

但 Strategy 并不是 ROLL 分布式编程模型的强制层级。如果要开发的全新业务应用不具备统一的模型训练或推理范式，例如纯 CPU 规则计算、环境服务、数据处理、外部 API 编排或自定义分布式系统，Worker 可以直接实现业务逻辑，无需创建 Strategy：

```python
class CustomWorker(Worker):
    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE)
    def compute(self, data: DataProto) -> DataProto:
        result = custom_business_logic(data)
        return result
```

因此可以将各层关系理解为：

```text
通用 ROLL 分布式应用：Pipeline -> Cluster -> Worker

模型训练/推理类应用：Pipeline -> Cluster -> Worker -> Strategy
```

是否引入 Strategy，应由业务中是否存在需要跨后端统一的稳定计算接口决定，而不应为了符合固定模板强行增加该层。

## 核心抽象概览

| 抽象 | 所在位置 | 主要职责 | 不应负责 |
| --- | --- | --- | --- |
| Pipeline | Driver 进程 | 编排角色、数据流、训练循环、验证、权重同步和检查点 | 具体模型并行实现、Ray Actor 放置 |
| Cluster | Driver 进程 | 创建和管理同类 Worker、数据分发、远程调用和结果收集 | 算法 loss、任务语义 |
| Worker | Ray Actor | 定义角色能力、维护角色状态、执行单次业务操作 | 全局 Pipeline 调度 |
| Strategy | Worker 内部，可选 | 统一模型后端的初始化、训练、推理、生成、权重同步和状态管理 | Pipeline 业务流程 |
| ResourceManager | Driver 进程 | 节点与设备发现、Placement Group 和 Worker 资源放置 | 业务计算 |
| DataProto | 跨层数据协议 | 携带 Tensor、非 Tensor 数据和元信息 | 决定业务执行顺序 |

## Pipeline：编排完整业务流程

Pipeline 是 Driver 侧的顶层编排器。所有内置训练 Pipeline 都继承自：

```text
roll/pipeline/base_pipeline.py::BasePipeline
```

`BasePipeline` 提供以下通用能力：

- 初始化随机种子和 `ResourceManager`；
- 初始化 checkpoint、resume 和实验 tracker；
- 创建一个或多个 Cluster；
- 配置模型权重同步关系；
- 配置参与 checkpoint 的 Cluster；
- 保存 Pipeline 自身的 `WorkerState`；
- 下载模型以及初始化传输后端；
- 管理 tracing 和通用运行状态。

自定义 Pipeline 通常需要实现：

```python
from roll.pipeline.base_pipeline import BasePipeline


class CustomPipeline(BasePipeline):
    def __init__(self, pipeline_config):
        super().__init__(pipeline_config)
        # 创建 Cluster、数据集和调度组件

    def run(self):
        # 编排完整业务循环
        ...
```

Pipeline 的代码应体现业务算法，而不是某个分布式后端的实现细节。例如：

```python
teacher_output = self.teacher.generate(batch)
train_metrics = self.student.train_step(teacher_output)
```

不建议在 Pipeline 中根据 Strategy 名称分支：

```python
# 不推荐
if self.pipeline_config.student.strategy_args.strategy_name == "megatron_train":
    ...
elif self.pipeline_config.student.strategy_args.strategy_name == "fsdp2_train":
    ...
```

如果两种后端无法通过相同的 Worker 接口运行，应优先检查差异是否属于 Strategy 的职责。

## Worker：定义分布式角色

Worker 是 ROLL 中真正运行在 Ray Actor 内的业务执行单元，基类位于：

```text
roll/distributed/executor/worker.py::Worker
```

Worker 初始化时获得对应的 `WorkerConfig`，并维护：

- `rank`、`world_size` 和 `local_rank`；
- `MASTER_ADDR` 和 `MASTER_PORT`；
- `RankInfo` 中的 DP、TP、PP 和 CP 信息；
- 当前 Cluster、Worker 和设备信息；
- Pipeline 配置以及角色自身的长期状态。

一个 Worker 应围绕单一角色提供稳定的业务接口。例如 SFT Worker 提供 `initialize()`、`train_step()`、`val_step()` 和 `do_checkpoint()`；Reward Worker 可以只提供 `initialize()` 和 `compute_rewards()`。

### 使用 Strategy 的 Worker

模型训练 Worker 通常在 `initialize()` 中创建 Strategy：

```python
from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.decorator import Dispatch, register
from roll.distributed.strategy.factory import create_strategy


class CustomTrainWorker(Worker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self, pipeline_config):
        super().initialize(pipeline_config)
        self.strategy = create_strategy(worker=self)
        self.strategy.initialize(model_provider=my_model_provider)

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE, clear_cache=False, prefetch=True)
    def train_step(self, data: DataProto):
        data = data.to(current_platform.device_type)
        metrics = self.strategy.train_step(data, self.loss_func)
        return DataProto(meta_info={"metrics": metrics}).to("cpu")

    def loss_func(self, data, output_tensor):
        # 定义业务相关的 loss；分布式执行由 Strategy 负责
        ...
```

Worker 定义“计算什么”，Strategy 负责“如何在指定后端完成计算”。

### 不使用 Strategy 的 Worker

当业务不需要模型后端抽象时，可以直接实现 Worker：

```python
class DataProcessWorker(Worker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self, pipeline_config):
        super().initialize(pipeline_config)
        self.processor = build_processor(self.worker_config)

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE, clear_cache=False)
    def process(self, data: DataProto):
        return self.processor(data)
```

此时 Worker 仍然可以使用 Cluster 的资源放置、数据分发、Ray RPC 和结果收集能力。

## Cluster：管理 Worker 集合

Cluster 位于：

```text
roll/distributed/executor/cluster.py::Cluster
```

一个 Cluster 表示一个逻辑角色，例如：

- `actor_train`；
- `actor_infer`；
- `reference`；
- `critic`；
- `reward`；
- `student`；
- `teacher`。

Cluster 根据 `WorkerConfig.world_size` 创建一组 Worker，并负责：

- 请求 `ResourceManager` 分配 Placement Group；
- 将 Worker 包装为 Ray Actor；
- 为各 Worker 设置 rank 和可见设备；
- 建立分布式通信所需的 master 地址和端口；
- 收集 Worker 的 `RankInfo`；
- 将 Worker 中通过 `@register` 声明的方法动态绑定到 Cluster；
- 根据 Dispatch 规则分发参数并收集结果。

创建 Cluster：

```python
self.student = Cluster(
    name=self.pipeline_config.student.name,
    worker_cls=self.pipeline_config.student.worker_cls,
    resource_manager=self.resource_manager,
    worker_config=self.pipeline_config.student,
)
```

初始化后，Pipeline 可以像调用普通对象一样调用 Worker 方法：

```python
ray.get(
    self.student.initialize(
        pipeline_config=self.pipeline_config,
        blocking=False,
    )
)

result_refs = self.student.train_step(batch, blocking=False)
result = DataProto.materialize_concat(result_refs)
```

这里的 `student.train_step()` 是 Cluster 根据 Worker 上的 `@register` 动态生成的代理方法。

当 Pipeline 包含多个角色时，可以使用 `BasePipeline.create_clusters_parallel()` 并行创建：

```python
clusters = self.create_clusters_parallel(
    [
        (
            "student",
            self.pipeline_config.student.name,
            self.pipeline_config.student.worker_cls,
            self.pipeline_config.student,
        ),
        (
            "teacher",
            self.pipeline_config.teacher.name,
            self.pipeline_config.teacher.worker_cls,
            self.pipeline_config.teacher,
        ),
    ]
)

self.student = clusters["student"]
self.teacher = clusters["teacher"]
```

### 推理 Cluster 的层次关系示例

下面以 `actor_infer` 使用 `dp=2、tp=2` 为例，展示 ROLL 对象与推理后端内部执行单元之间的关系：

```text
                         actor_infer Cluster
                    （1 个逻辑角色，world_size=2）
                                  │
                    1 ────────────┴──────────── n
                    │                           │
             Worker, dp_rank=0          Worker, dp_rank=1
             （Ray Actor）               （Ray Actor）
                    │ 1                         │ 1
                    │                           │
               Strategy 1                 Strategy 1
             （vLLM/SGLang）             （vLLM/SGLang）

────────────────────── ROLL 应用抽象与后端实现的边界 ──────────────────────

                    │                           │
             inference engine            inference engine
                    │                           │
             TP rank 0  TP rank 1         TP rank 0  TP rank 1
                 │          │                 │          │
             executor   executor          executor   executor
               GPU 0      GPU 1             GPU 2      GPU 3
```

这张图中的数量关系是：

```text
1 个 Cluster
  -> dp_size 个 Worker
  -> 每个 Worker 持有 1 个 Strategy
  -> 每个 Strategy 创建 1 个独立推理引擎
  -> 每个推理引擎在 tp_size 个设备上执行
```

对应配置可以写成：

```yaml
actor_infer:
  device_mapping: [0, 1, 2, 3]
  strategy_args:
    strategy_name: vllm
    strategy_config:
      tensor_parallel_size: 2
      pipeline_parallel_size: 1
```

`WorkerConfig` 会得到：

```text
num_gpus_per_worker = tensor_parallel_size * pipeline_parallel_size = 2
world_size = len(device_mapping) / num_gpus_per_worker = 2
```

因此 Cluster 创建两个 Worker。两个 Worker 是两个数据并行副本，分别处理不同的请求或 batch 分片；每个 Worker 内部的推理 Strategy 使用两张设备建立一个 TP=2 的推理引擎。

这里需要区分两种 rank：

- Worker 的 `rank` / `dp_rank` 属于 ROLL Cluster，用于跨推理副本的数据分发和结果收集；
- 推理引擎内部的 TP rank 属于 vLLM、SGLang 等后端，由 Strategy 和后端自行管理。

后端 executor 是概念上的执行单元，具体以进程、线程还是后端自有 Worker 的形式运行，由相应推理引擎决定。ROLL 的 Pipeline 和 Cluster 不依赖这些内部实现，只通过 Worker 和 Strategy 的统一接口调用推理能力。

### 训练 Cluster 的层次关系示例

训练 Cluster 的对象关系与推理 Cluster 相似，但训练后端通常由所有 Worker 中的 Strategy 共同组成一个分布式训练作业。下面以 `actor_train.world_size=2` 为例：

```text
                          actor_train Cluster
                       （world_size=2）
                                  │
                    1 ────────────┴──────────── n
                    │                           │
               Worker, rank=0             Worker, rank=1
                （Ray Actor）               （Ray Actor）
                    │ 1                         │ 1
                    │                           │
               Strategy 1                 Strategy 1
            （训练后端适配层）           （训练后端适配层）

────────────────────── ROLL 应用抽象与后端实现的边界 ──────────────────────

                    │                           │
          FSDP2/Megatron backend       FSDP2/Megatron backend
             backend rank=0               backend rank=1
                    │                           │
                    └────── 分布式通信组 ────────┘
```

这里并不是两个互不相关的训练后端实例。两个 Ray Worker 分别承载一个全局 rank；每个 Worker 内的 Strategy 初始化当前 rank 的模型、优化器、通信组和后端状态，所有 Strategy 共同完成一次分布式 `train_step()`。

典型调用链为：

```text
Pipeline
  -> actor_train.train_step(global_batch)
  -> Cluster 根据 Dispatch 和 RankInfo 分发数据
  -> 每个 ActorWorker.train_step(local_data)
  -> 每个 TrainStrategy.train_step(local_data, loss_func)
  -> FSDP2/Megatron 完成 forward、backward、梯度同步和 optimizer step
  -> Cluster 收集各 DP 副本的有效结果
  -> Pipeline 聚合 metrics
```

与推理引擎常见的“一个 ROLL Worker 管理多个内部 TP executor”不同，当前 FSDP2/Megatron 训练配置通常令一个 ROLL Worker 对应一个后端全局 rank 和一张设备。`WorkerConfig` 对这些 Strategy 会将 `num_gpus_per_worker` 规范为 1，因此：

```text
world_size = len(device_mapping)
```

例如：

```yaml
actor_train:
  device_mapping: [0, 1]
  strategy_args:
    strategy_name: fsdp2_train
    strategy_config:
      fsdp_size: 2
```

这会创建两个 Worker，并形成两个 FSDP rank。若使用 Megatron：

```yaml
actor_train:
  device_mapping: [0, 1]
  strategy_args:
    strategy_name: megatron_train
    strategy_config:
      tensor_model_parallel_size: 2
      pipeline_model_parallel_size: 1
```

则同样创建两个 Worker，但两个后端 rank 组成的是一个 TP=2、PP=1、DP=1 的 Megatron 拓扑。

因此，`world_size=2` 只能说明 Cluster 中有两个 Worker，不能单独确定数据并行和模型并行关系。实际的 `dp_size`、`tp_size`、`pp_size` 和 `cp_size` 由 Strategy 根据后端配置计算，并写回每个 Worker 的 `RankInfo`：

```text
FSDP2 示例：world_size=2, dp_size=2, tp_size=1, pp_size=1
Megatron 示例：world_size=2, dp_size=1, tp_size=2, pp_size=1
```

Cluster 随后依据这些 `RankInfo` 完成 DP batch 切分、模型并行 rank 输入复制以及有效输出收集。这也是 Cluster 不直接理解 FSDP2/Megatron 配置，而由 Strategy 提供并行拓扑信息的原因。

## Dispatch：声明数据如何分发

Worker 方法使用 `@register` 声明执行范围、数据分发方式和结果收集方式。相关代码位于：

```text
roll/distributed/scheduler/decorator.py
```

常用 Dispatch 模式如下：

| Dispatch | 输入分发 | 结果收集 | 常见场景 |
| --- | --- | --- | --- |
| `ONE_TO_ALL` | 同一份参数发送给全部 Worker | 收集全部结果 | 初始化、加载、卸载、checkpoint |
| `ONE_TO_ALL_ONE` | 同一份参数发送给全部 Worker | 只保留一个结果 | 全 rank 执行但结果等价的操作 |
| `ALL_TO_ALL` | 输入列表与 Worker 一一对应 | 收集全部结果 | 每个 rank 接收独立控制数据 |
| `DP_MP_COMPUTE` | 按 DP 切分，模型并行组共享同一分片 | 收集每个 DP 副本的有效输出 | Forward、Reward、train step 等计算 |
| `DP_MP_DISPATCH_FIRST` | 只向模型并行组首 rank 发送完整数据 | 收集每个 DP 副本的有效输出 | 遗留框架 API；`get_data_input` 移除后已无内置使用方 |

对于 `DP_MP_COMPUTE`，Cluster 会先按 `dp_size` 切分 batch：

```text
global batch
  ├─ shard 0 -> DP rank 0 对应的 TP/PP/CP ranks
  ├─ shard 1 -> DP rank 1 对应的 TP/PP/CP ranks
  └─ ...
```

结果通常只从满足以下条件的 rank 收集：

```text
tp_rank == 0
cp_rank == 0
pp_rank == pipeline_last_stage
```

因此 Worker 必须正确设置 `RankInfo`。使用现有 Strategy 时，Strategy 初始化过程通常会根据实际并行拓扑更新它；纯数据并行 Worker 可以沿用基类默认的 `dp_rank=rank`、`dp_size=world_size`。

## Strategy：统一模型后端

Strategy 基类位于：

```text
roll/distributed/strategy/strategy.py
```

工厂位于：

```text
roll/distributed/strategy/factory.py::create_strategy
```

当前 Strategy 覆盖的后端包括：

- Hugging Face inference；
- FSDP2 training/inference；
- Megatron training/inference；
- vLLM inference；
- SGLang inference；
- Mock inference。

### InferenceStrategy 接口

`InferenceStrategy` 主要定义：

```python
initialize(...)
forward_step(batch, forward_func)
generate(...)
save_checkpoint(...)
load_checkpoint(...)
load_states(...)
offload_states(...)
setup_model_update(...)
update_parameter_in_bucket(...)
process_weights_after_loading(...)
```

### TrainStrategy 接口

`TrainStrategy` 在推理接口基础上增加：

```python
train_step(batch, loss_func)
model_update(...)
```

Strategy 还提供一组可复用的分布式计算操作，例如：

```python
op_compute_log_probs(...)
op_compute_entropy(...)
op_compute_language_loss(...)
op_compute_gather_by_teacher_indices(...)
```

这些操作处理 TP、PP 或 CP 分片下的计算差异，使 Worker 的算法实现保持一致。

### 选择已有 Strategy

Strategy 由 `WorkerConfig.strategy_args` 配置：

```yaml
student:
  strategy_args:
    strategy_name: fsdp2_train
    strategy_config:
      fsdp_size: 8
```

切换到 Megatron 时，Pipeline 和 Worker 的业务接口通常不变：

```yaml
student:
  strategy_args:
    strategy_name: megatron_train
    strategy_config:
      tensor_model_parallel_size: 2
      pipeline_model_parallel_size: 2
```

### 何时新增 Strategy

以下需求通常适合新增或扩展 Strategy：

- 接入新的模型训练或推理引擎；
- 使用新的模型并行或数据并行方式；
- 需要不同的 forward/backward 调度；
- 需要新的模型状态加载、卸载或 checkpoint 机制；
- 需要不同的训练模型到推理模型权重同步方式；
- 同一种 Worker 业务接口需要在多个后端保持一致。

以下需求通常不应新增 Strategy：

- 仅新增一种 loss；
- 仅修改 batch 字段；
- 仅调整 Pipeline 中角色执行顺序；
- 仅实现规则计算或外部 API 调用；
- 业务不存在多后端统一需求。

这些逻辑应分别放到 Worker、Pipeline 或普通业务组件中。

## ResourceManager 与设备映射

`BasePipeline` 会创建全局 ResourceManager：

```python
self.resource_manager = ResourceManager(
    num_nodes=self.pipeline_config.num_nodes,
    num_gpus_per_node=self.pipeline_config.num_gpus_per_node,
)
```

每个 WorkerConfig 通过 `device_mapping` 选择设备：

```yaml
student:
  device_mapping: list(range(0, 8))
  num_gpus_per_worker: 1
```

Worker 数量由以下关系决定：

```text
world_size = len(device_mapping) / num_gpus_per_worker
```

例如，8 张设备、每个 Worker 使用 1 张设备，会创建 8 个 Worker。对于 vLLM，`num_gpus_per_worker` 会根据 tensor parallel size 和 pipeline parallel size 推导；8 张设备、每个实例使用 4 张设备时，会创建 2 个 Worker。

因此，ROLL 中的 Worker 不等同于单张 GPU。Worker 是一个可以独立提供某项业务能力的资源单元，它可以占用一张、多张或零张 GPU。

CPU Worker 不设置 `device_mapping`，ResourceManager 会按 `world_size` 将它们尽量分散到各节点。

## DataProto：统一数据协议

`DataProto` 位于：

```text
roll/distributed/scheduler/protocol.py
```

它包含三类数据：

```text
batch             TensorDict，存放 Tensor 数据
non_tensor_batch  NumPy/Object 数据
meta_info         控制信息和指标
```

创建数据：

```python
batch = DataProto.from_single_dict(batch_dict)
batch.meta_info = {
    "global_step": global_step,
    "loss_mask_keys": ["labels"],
}
```

异步 Cluster 调用返回的 Ray ObjectRef 可以通过以下方式物化并合并：

```python
result_refs = self.student.train_step(batch, blocking=False)
result = DataProto.materialize_concat(result_refs)
```

设计新 Pipeline 时，应将 DataProto 当作跨层公共契约。修改字段时需要同时检查：

- Pipeline 的生产和消费逻辑；
- Dispatch 如何切分 batch；
- Worker 和 Strategy 是否读取该字段；
- reward、advantage、loss 或 metric 是否依赖该字段；
- Tensor 与 `non_tensor_batch` 的 batch size 是否一致；
- 数据是否应随 Worker 移动到设备或返回 CPU。

## PipelineConfig 与 WorkerConfig

自定义 Pipeline 应定义对应的配置类，并将各逻辑角色声明为 WorkerConfig：

```python
from dataclasses import dataclass, field

from roll.configs.base_config import BaseConfig
from roll.configs.worker_config import WorkerConfig


@dataclass
class CustomPipelineConfig(BaseConfig):
    student: WorkerConfig = field(default_factory=WorkerConfig)
    teacher: WorkerConfig = field(default_factory=WorkerConfig)

    custom_option: int = 1

    def __post_init__(self):
        super().__post_init__()
        if self.student.worker_cls is None:
            self.student.worker_cls = "my_package.worker.StudentWorker"
        if self.teacher.worker_cls is None:
            self.teacher.worker_cls = "my_package.worker.TeacherWorker"
```

YAML 顶层字段映射到 PipelineConfig，角色配置映射到 WorkerConfig：

```yaml
pipeline_cls: my_package.pipeline.CustomPipeline

student:
  name: student
  worker_cls: my_package.worker.StudentWorker
  device_mapping: list(range(0, 4))
  strategy_args:
    strategy_name: fsdp2_train
    strategy_config:
      fsdp_size: 4

teacher:
  name: teacher
  worker_cls: my_package.worker.TeacherWorker
  device_mapping: list(range(4, 8))
  strategy_args:
    strategy_name: vllm
    strategy_config:
      tensor_parallel_size: 4
```

启动入口需要动态加载 `pipeline_cls`，并使用对应配置类解析 YAML。可以参考：

```text
examples/start_agentic_pipeline.py
examples/start_onpolicy_distill_pipeline.py
```

如果现有启动入口固定了配置类型，新 Pipeline 还需要新增或扩展启动入口；仅设置 `pipeline_cls` 并不会自动选择新的 PipelineConfig。

## 权重同步、状态卸载与 Checkpoint

模型训练 Pipeline 经常需要将训练权重同步到推理 Cluster。`BasePipeline` 提供：

```python
self.set_model_update_pair(
    src_cluster=self.student,
    tgt_cluster=self.teacher,
    frequency=1,
)
```

训练循环中调用：

```python
model_update_metrics = self.model_update(global_step)
```

底层由 `ModelUpdateGroup` 和两个 Strategy 的权重同步接口实现。自定义模型应用需要验证源端与目标端的 DP、TP、PP 拓扑以及参数映射关系。

指定参与 checkpoint 的 Cluster：

```python
self.set_checkpoint_clusters(self.student)
```

训练循环中调用：

```python
self.do_checkpoint(global_step)
```

`BasePipeline` 会协调各 Worker 的 checkpoint、Pipeline 自身状态、异步上传以及旧 checkpoint 清理。

对于共享设备的训练和推理角色，应复用现有 `load_states()`、`offload_states()` 和上下文管理器，不要在 Pipeline 中直接销毁模型对象或手工清理显存。

## 一个最小 Pipeline 骨架

下面的示例展示 Pipeline、Cluster 和不使用 Strategy 的 Worker 如何协作：

```python
import ray

from roll.distributed.executor.cluster import Cluster
from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.decorator import Dispatch, register
from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.base_pipeline import BasePipeline


class TransformWorker(Worker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self, pipeline_config):
        super().initialize(pipeline_config)

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE, clear_cache=False)
    def transform(self, data: DataProto):
        data.batch["output"] = data.batch["input"] * 2
        return data


class TransformPipeline(BasePipeline):
    def __init__(self, pipeline_config):
        super().__init__(pipeline_config)

        self.transform_cluster = Cluster(
            name=pipeline_config.transform.name,
            worker_cls=pipeline_config.transform.worker_cls,
            resource_manager=self.resource_manager,
            worker_config=pipeline_config.transform,
        )
        ray.get(
            self.transform_cluster.initialize(
                pipeline_config=pipeline_config,
                blocking=False,
            )
        )

    def run(self):
        for batch in self.build_batches():
            output = self.transform_cluster.transform(batch)
            self.consume(output)
```

这个例子仍然获得了以下 ROLL 能力：

- Ray Actor 部署；
- 多节点设备放置；
- DP 数据切分；
- 远程调用和结果收集；
- DataProto 协议；
- Pipeline 状态与通用基础设施。

但由于业务不涉及多模型后端统一，因此没有引入 Strategy。

## 如何判断应该修改哪一层

| 需求 | 推荐修改层 |
| --- | --- |
| 修改角色执行顺序或整体算法数据流 | Pipeline |
| 新增 Actor、Teacher、Reward 或数据处理角色 | Worker + WorkerConfig |
| 修改数据在 DP/TP/PP rank 之间的分发方式 | 优先选择 Dispatch，必要时扩展 Cluster/Dispatch |
| 接入新的训练或推理后端 | Strategy |
| 修改 loss 或角色内部业务计算 | Worker |
| 修改节点、设备或 Placement Group 分配 | ResourceManager / WorkerConfig |
| 修改跨角色公共数据字段 | DataProto 生产方及所有消费者 |
| 新增纯 CPU、规则或外部服务应用 | Pipeline + Cluster + Worker，通常无需 Strategy |

## 开发步骤

建议按以下顺序实现新的 Pipeline：

1. 明确完整业务流程、输入输出以及需要长期存活的逻辑角色。
2. 检查现有 Pipeline 是否可以通过子类或新增 Worker 满足需求。
3. 为每个逻辑角色定义 Worker 接口和 WorkerConfig。
4. 判断每个 Worker 是否需要 Strategy；优先复用已有 Strategy。
5. 设计 DataProto schema，并追踪每个字段的生产方和消费者。
6. 为 Worker 方法选择正确的 Dispatch。
7. 在 Pipeline 中创建和初始化 Cluster。
8. 实现 `run()`，只表达业务步骤和角色协作。
9. 根据需要配置权重同步、offload/reload、validation 和 checkpoint。
10. 添加单 Worker、单节点、多 Worker、多节点和 resume 验证。

## 验证清单

### 配置和资源

- PipelineConfig 能正确解析最终合并后的 YAML。
- `pipeline_cls` 和各 `worker_cls` 能动态加载。
- `device_mapping`、`num_gpus_per_worker` 和 `world_size` 一致。
- 各 Cluster 的设备共享或隔离关系符合设计。

### Worker 和 Dispatch

- 每个 `@register` 方法使用正确的 Dispatch 和执行范围。
- DP 切分后每个样本只被处理一次或按设计复制。
- TP、PP、CP rank 获得正确输入，并仅收集有效 rank 的输出。
- blocking 与 non-blocking 调用均正确物化结果。

### Strategy（如果使用）

- 至少验证一个目标训练或推理后端。
- forward、backward、generate 和 checkpoint 契约一致。
- Worker 中没有不必要的后端名称分支。
- 多后端支持需要分别验证数值、shape、权重同步和恢复。

### Pipeline

- 训练、验证、异常、最后一步和空数据路径均能结束。
- DataProto 的 key、shape、dtype、batch size 和设备正确。
- 全局 step、随机数状态和 checkpoint 能正确恢复。
- 异步 Ray ObjectRef 不会泄漏或造成死锁。
- Metrics 在正确 step 上报且聚合语义一致。
- 多 Cluster 权重同步和状态卸载顺序正确。

## 参考实现

建议根据业务类型选择最接近的实现：

- 最小单角色训练流程：`roll/pipeline/sft/sft_pipeline.py`
- 双角色偏好训练：`roll/pipeline/dpo/dpo_pipeline.py`
- 多任务RLVR：`roll/pipeline/rlvr/rlvr_pipeline.py`
- Agent 与环境交互：`roll/pipeline/agentic/agentic_pipeline.py`
- Teacher/Student 蒸馏：`roll/pipeline/distill/distill_pipeline.py`
- 仅 Rollout 流程：`roll/pipeline/rlvr/rlvr_rollout_pipeline.py`

实现新 Pipeline 时，应同时阅读目标 Pipeline、对应 Config、Worker 和 YAML，避免只复制 Pipeline 类而遗漏资源、Strategy、Dispatch 或 DataProto 的隐含契约。
