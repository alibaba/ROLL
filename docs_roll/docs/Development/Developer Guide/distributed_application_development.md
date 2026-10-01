# ROLL Distributed Programming Model and Application Development

ROLL decomposes a distributed application into abstractions with clear responsibilities. For model training and reinforcement learning, the most common call hierarchy is:

```text
Pipeline
  └─ Cluster
       └─ Worker (Ray Actor)
            └─ Strategy (optional)
```

- **Pipeline** orchestrates the complete application flow and data dependencies between roles.
- **Cluster** manages a group of Workers for one role and handles call dispatch and result collection.
- **Worker** defines the capabilities of a distributed role.
- **Strategy** exposes a stable interface over model training and inference backends. This layer is optional.

`ResourceManager` maps Clusters to nodes and devices, while `DataProto` carries structured data between layers.

This guide explains how to use these abstractions to build a new Pipeline and how to decide where each extension belongs.

## Strategy Is an Optional Abstraction

Large-language-model training and reinforcement learning share a relatively uniform execution pattern: initialize a model, run forward or generation, perform backward and optimizer steps, save checkpoints, offload state, and synchronize weights. Although FSDP2, Megatron, vLLM, SGLang, and Hugging Face implement these operations differently, they can naturally be represented through a stable Strategy interface.

For example, a Worker can always train through:

```python
metrics = self.strategy.train_step(
    batch=data,
    loss_func=self.loss_func,
)
```

`strategy_args.strategy_name` selects FSDP2 or Megatron without changing the Pipeline algorithm or the Worker's business API.

Strategy is not mandatory in ROLL's distributed programming model. A new application that has no interchangeable model-training or inference backend—such as CPU rule evaluation, an environment service, data processing, external API orchestration, or another distributed system—can implement its logic directly in Worker:

```python
class CustomWorker(Worker):
    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE)
    def compute(self, data: DataProto) -> DataProto:
        return custom_business_logic(data)
```

The two common forms are therefore:

```text
General ROLL distributed application:
Pipeline -> Cluster -> Worker

Model training/inference application:
Pipeline -> Cluster -> Worker -> Strategy
```

Introduce Strategy only when the application has a stable computation interface that should be shared across multiple backends. Do not add it merely to conform to a fixed template.

## Core Abstractions

| Abstraction | Location | Responsibility | Should not own |
| --- | --- | --- | --- |
| Pipeline | Driver process | Roles, data flow, global loop, validation, weight updates, checkpoints | Model-parallel implementation, Ray Actor placement |
| Cluster | Driver process | Worker creation, RPC proxying, dispatch, execution, result collection | Algorithm loss or task semantics |
| Worker | Ray Actor | Role capability, role-local state, one distributed operation | Global Pipeline scheduling |
| Strategy | Inside Worker, optional | Model backend initialization, training, inference, generation, synchronization, state | Pipeline business flow |
| ResourceManager | Driver process | Nodes, devices, placement groups, Worker placement | Business computation |
| DataProto | Cross-layer protocol | Tensor, non-tensor, and metadata transport | Execution ordering |

## Pipeline: Orchestrating the Application

Pipeline is the Driver-side algorithm orchestrator. Built-in training Pipelines inherit from:

```text
roll/pipeline/base_pipeline.py::BasePipeline
```

`BasePipeline` provides:

- random seed and `ResourceManager` initialization;
- checkpoint, resume, and experiment tracking;
- parallel Cluster creation;
- model-update relationships between Clusters;
- checkpoint Cluster registration;
- Pipeline `WorkerState` persistence;
- model download and transfer-backend initialization;
- telemetry and common runtime state.

A custom Pipeline normally implements:

```python
from roll.pipeline.base_pipeline import BasePipeline


class CustomPipeline(BasePipeline):
    def __init__(self, pipeline_config):
        super().__init__(pipeline_config)
        # Create Clusters, datasets, and schedulers.

    def run(self):
        # Orchestrate the application loop.
        ...
```

Pipeline code should describe the algorithm:

```python
teacher_output = self.teacher.generate(batch)
train_metrics = self.student.train_step(teacher_output)
```

Avoid branching on the backend in Pipeline:

```python
# Avoid this.
if self.pipeline_config.student.strategy_args.strategy_name == "megatron_train":
    ...
elif self.pipeline_config.student.strategy_args.strategy_name == "fsdp2_train":
    ...
```

If two backends cannot execute through the same Worker API, first determine whether the difference belongs in Strategy.

## Worker: Defining a Distributed Role

Worker is the business execution unit running inside a Ray Actor. Its base class is:

```text
roll/distributed/executor/worker.py::Worker
```

A Worker receives its `WorkerConfig` and maintains:

- `rank`, `world_size`, and `local_rank`;
- `MASTER_ADDR` and `MASTER_PORT`;
- DP, TP, PP, and CP values in `RankInfo`;
- Cluster, Worker, node, and device information;
- Pipeline configuration and persistent role-local state.

A Worker should expose a focused API for one role. An SFT Worker, for example, provides `initialize()`, `train_step()`, `val_step()`, and `do_checkpoint()`. A rule-based Reward Worker may only need `initialize()` and `compute_rewards()`.

### Worker with Strategy

A model Worker usually creates Strategy during initialization:

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
        # Define algorithm semantics here; Strategy handles backend execution.
        ...
```

Worker defines what to compute; Strategy defines how the selected backend computes it.

### Worker without Strategy

When no model-backend abstraction is needed, implement the operation directly:

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

This Worker still benefits from Cluster placement, Ray RPC, dispatch, collection, logging, and DataProto.

## Cluster: Managing a Worker Group

Cluster is defined at:

```text
roll/distributed/executor/cluster.py::Cluster
```

A Cluster represents one logical role, such as `actor_train`, `actor_infer`, `reference`, `critic`, `reward`, `student`, or `teacher`.

Based on `WorkerConfig.world_size`, Cluster:

- requests placement groups from `ResourceManager`;
- wraps Worker classes as Ray Actors;
- assigns rank and visible devices;
- establishes the master address and port;
- collects `RankInfo` from Workers;
- binds Worker methods marked with `@register` onto Cluster;
- dispatches arguments and collects results.

Create a Cluster with:

```python
self.student = Cluster(
    name=self.pipeline_config.student.name,
    worker_cls=self.pipeline_config.student.worker_cls,
    resource_manager=self.resource_manager,
    worker_config=self.pipeline_config.student,
)
```

After initialization, Pipeline calls Cluster as if it were the role itself:

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

`student.train_step()` is a proxy generated from the registered Worker method.

Create several roles concurrently with `BasePipeline.create_clusters_parallel()`:

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

### Inference Cluster Topology

The following example shows `actor_infer` with `dp=2` and `tp=2`:

```text
                         actor_infer Cluster
                    (one role, world_size=2)
                                  │
                    1 ────────────┴──────────── n
                    │                           │
             Worker, dp_rank=0          Worker, dp_rank=1
               (Ray Actor)                (Ray Actor)
                    │ 1                         │ 1
                    │                           │
               Strategy 1                 Strategy 1
             (vLLM/SGLang)              (vLLM/SGLang)

──────────────── ROLL application / backend implementation boundary ────────────────

                    │                           │
             inference engine            inference engine
                    │                           │
             TP rank 0  TP rank 1         TP rank 0  TP rank 1
                 │          │                 │          │
             executor   executor          executor   executor
               GPU 0      GPU 1             GPU 2      GPU 3
```

The cardinality is:

```text
1 Cluster
  -> dp_size Workers
  -> 1 Strategy per Worker
  -> 1 independent inference engine per Strategy
  -> tp_size devices per inference engine
```

For vLLM:

```yaml
actor_infer:
  device_mapping: [0, 1, 2, 3]
  strategy_args:
    strategy_name: vllm
    strategy_config:
      tensor_parallel_size: 2
      pipeline_parallel_size: 1
```

`WorkerConfig` derives:

```text
num_gpus_per_worker = tensor_parallel_size * pipeline_parallel_size = 2
world_size = len(device_mapping) / num_gpus_per_worker = 2
```

Cluster creates two data-parallel Worker replicas. Each Worker owns one Strategy whose inference engine uses two devices for TP=2.

ROLL Worker `rank/dp_rank` belongs to Cluster and is used for request or batch distribution. Internal TP ranks belong to vLLM or SGLang and are managed by Strategy and the backend. Backend executors are conceptual units; whether they are processes, threads, or backend-specific Workers is an engine implementation detail.

### Training Cluster Topology

Training uses a similar object hierarchy, but Strategy instances normally form one distributed training job together. For `actor_train.world_size=2`:

```text
                          actor_train Cluster
                            (world_size=2)
                                  │
                    1 ────────────┴──────────── n
                    │                           │
               Worker, rank=0             Worker, rank=1
                (Ray Actor)                 (Ray Actor)
                    │ 1                         │ 1
                    │                           │
               Strategy 1                 Strategy 1
            (training adapter)          (training adapter)

──────────────── ROLL application / backend implementation boundary ────────────────

                    │                           │
          FSDP2/Megatron backend       FSDP2/Megatron backend
             backend rank=0               backend rank=1
                    │                           │
                    └──── distributed group ────┘
```

These are not two unrelated training instances. Each Ray Worker holds one backend rank. Its Strategy initializes the model shard, optimizer, communication groups, and backend state; all Strategies participate in each distributed `train_step()`.

```text
Pipeline
  -> actor_train.train_step(global_batch)
  -> Cluster dispatches data using Dispatch and RankInfo
  -> each ActorWorker.train_step(local_data)
  -> each TrainStrategy.train_step(local_data, loss_func)
  -> FSDP2/Megatron forward, backward, synchronization, optimizer step
  -> Cluster collects valid outputs from DP replicas
  -> Pipeline aggregates metrics
```

Current FSDP2/Megatron training configurations normally use one ROLL Worker per backend global rank and one device per Worker. `WorkerConfig` normalizes `num_gpus_per_worker` to one for these Strategies:

```text
world_size = len(device_mapping)
```

FSDP2 example:

```yaml
actor_train:
  device_mapping: [0, 1]
  strategy_args:
    strategy_name: fsdp2_train
    strategy_config:
      fsdp_size: 2
```

Megatron example:

```yaml
actor_train:
  device_mapping: [0, 1]
  strategy_args:
    strategy_name: megatron_train
    strategy_config:
      tensor_model_parallel_size: 2
      pipeline_model_parallel_size: 1
```

Both create two Workers, but the first can form `DP=2, TP=1, PP=1` while the second forms `DP=1, TP=2, PP=1`. `world_size=2` alone does not determine the parallel topology. Strategy computes DP/TP/PP/CP ranks and sizes, writes them to Worker `RankInfo`, and Cluster subsequently uses that information for dispatch and collection.

## Dispatch: Declaring Argument Distribution

Worker methods use `@register` to declare their execution and collection behavior. The implementation is in:

```text
roll/distributed/scheduler/decorator.py
```

| Dispatch | Input distribution | Collection | Typical use |
| --- | --- | --- | --- |
| `ONE_TO_ALL` | Broadcast one value to all Workers | All results | Initialization, load/offload, checkpoint |
| `ONE_TO_ALL_ONE` | Broadcast to all Workers | One representative result | All ranks execute equivalent result |
| `ALL_TO_ALL` | One input per global Worker | All results | Rank-specific control data |
| `DP_MP_COMPUTE` | Split by DP and replicate within model-parallel group | Valid output per DP replica | Forward, reward computation, train step |
| `DP_MP_DISPATCH_FIRST` | Full shard only to first rank in each model-parallel group | Valid output per DP replica | Legacy framework API; no built-in consumer since `get_data_input` was removed |

For `DP_MP_COMPUTE`, Cluster first splits the global batch by `dp_size`:

```text
global batch
  ├─ shard 0 -> TP/PP/CP ranks for DP rank 0
  ├─ shard 1 -> TP/PP/CP ranks for DP rank 1
  └─ ...
```

It normally collects only ranks satisfying:

```text
tp_rank == 0
cp_rank == 0
pp_rank == pipeline_last_stage
```

Worker `RankInfo` must therefore be accurate. Existing Strategies update it from their real topology; a pure data-parallel Worker may use the base defaults `dp_rank=rank` and `dp_size=world_size`.

## Strategy: Normalizing Model Backends

Strategy bases are defined in:

```text
roll/distributed/strategy/strategy.py
```

The factory is:

```text
roll/distributed/strategy/factory.py::create_strategy
```

Current implementations include Hugging Face inference, FSDP2 train/infer, Megatron train/infer, vLLM, SGLang, and mock inference.

### InferenceStrategy

The inference interface includes:

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

### TrainStrategy

`TrainStrategy` extends the model lifecycle with:

```python
train_step(batch, loss_func)
model_update(...)
```

Strategy also provides reusable distributed operations such as:

```python
op_compute_log_probs(...)
op_compute_entropy(...)
op_compute_language_loss(...)
op_compute_gather_by_teacher_indices(...)
```

These helpers encapsulate TP/CP-aware computation so the Worker algorithm remains stable.

### Selecting an Existing Strategy

Configure the backend through `WorkerConfig.strategy_args`:

```yaml
student:
  strategy_args:
    strategy_name: fsdp2_train
    strategy_config:
      fsdp_size: 8
```

Switching to Megatron should normally leave Pipeline and Worker APIs unchanged:

```yaml
student:
  strategy_args:
    strategy_name: megatron_train
    strategy_config:
      tensor_model_parallel_size: 2
      pipeline_model_parallel_size: 2
```

### When to Add Strategy

Add or extend Strategy when integrating:

- a new model training or inference engine;
- a new data/model parallel implementation;
- different forward/backward scheduling;
- a new checkpoint or model-state lifecycle;
- a different training-to-inference weight synchronization mechanism;
- multiple backends behind one stable Worker API.

Do not add Strategy only for:

- a new loss;
- a renamed batch field;
- a different role order in Pipeline;
- rule computation or one external API;
- an application with no multi-backend requirement.

Place those changes in Worker, Pipeline, or an ordinary business component.

## ResourceManager and Device Mapping

`BasePipeline` creates one ResourceManager:

```python
self.resource_manager = ResourceManager(
    num_nodes=self.pipeline_config.num_nodes,
    num_gpus_per_node=self.pipeline_config.num_gpus_per_node,
)
```

Each WorkerConfig selects global devices through `device_mapping`:

```yaml
student:
  device_mapping: list(range(0, 8))
  num_gpus_per_worker: 1
```

Worker count is derived from:

```text
world_size = len(device_mapping) / num_gpus_per_worker
```

Eight devices with one device per Worker create eight Workers. A vLLM instance with TP=4 uses four devices per Worker; eight devices create two Workers.

A ROLL Worker is therefore not synonymous with one GPU. It is an independent logical resource owner and may use zero, one, or several devices. CPU Workers omit `device_mapping`; ResourceManager spreads them across available nodes according to `world_size`.

## DataProto: Cross-Layer Data Contract

`DataProto` is defined in:

```text
roll/distributed/scheduler/protocol.py
```

It carries:

```text
batch             TensorDict tensor data
non_tensor_batch  NumPy/object data
meta_info         control data and metrics
```

Create it with:

```python
batch = DataProto.from_single_dict(batch_dict)
batch.meta_info = {
    "global_step": global_step,
    "loss_mask_keys": ["labels"],
}
```

Materialize asynchronous Cluster results with:

```python
result_refs = self.student.train_step(batch, blocking=False)
result = DataProto.materialize_concat(result_refs)
```

Treat DataProto as a public contract. When changing a field, inspect:

- every Pipeline producer and consumer;
- Dispatch chunking and ordering;
- Worker and Strategy readers;
- reward, advantage, loss, and metric consumers;
- tensor and non-tensor batch-size agreement;
- device movement and CPU return boundaries.

## PipelineConfig and WorkerConfig

Define a config class and represent each persistent distributed role with WorkerConfig:

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

Top-level YAML fields map to PipelineConfig; role sections map to WorkerConfig:

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

The launcher must dynamically load `pipeline_cls` and parse YAML with the matching config class. See:

```text
examples/start_agentic_pipeline.py
examples/start_onpolicy_distill_pipeline.py
```

If a launcher hard-codes a config type, a new Pipeline also requires a new or extended launcher. Setting `pipeline_cls` alone does not select a new PipelineConfig.

## Weight Updates, State Offload, and Checkpoints

Register training-to-inference weight synchronization with:

```python
self.set_model_update_pair(
    src_cluster=self.student,
    tgt_cluster=self.teacher,
    frequency=1,
)
```

Then call:

```python
model_update_metrics = self.model_update(global_step)
```

`ModelUpdateGroup` and source/target Strategies implement the communication. Validate source and target DP/TP/PP topology and parameter mapping.

Register checkpoint roles with:

```python
self.set_checkpoint_clusters(self.student)
```

Call checkpointing from the loop:

```python
self.do_checkpoint(global_step)
```

`BasePipeline` coordinates Worker checkpoints, Pipeline state, RNG persistence, upload, and retention. For shared devices, use the existing `load_states()` and `offload_states()` lifecycle instead of manually deleting models.

## Minimal Strategy-Free Pipeline

This example shows Pipeline, Cluster, and Worker without Strategy:

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

It still uses Ray deployment, multi-node placement, DP dispatch, RPC and collection, DataProto, and common Pipeline state—without introducing Strategy where no backend normalization is needed.

## Choosing the Extension Layer

| Requirement | Recommended layer |
| --- | --- |
| Change role ordering or global data flow | Pipeline |
| Add Actor, Teacher, Reward, or data-processing role | Worker + WorkerConfig |
| Change DP/TP/PP rank distribution | Existing Dispatch first; extend Dispatch/Cluster only if required |
| Add a model training or inference backend | Strategy |
| Change loss or role-local business computation | Worker |
| Change nodes, devices, or placement groups | ResourceManager / WorkerConfig |
| Change a shared field | DataProto producer and every consumer |
| Add a CPU, rule, or external-service application | Pipeline + Cluster + Worker, normally without Strategy |

## Development Workflow

1. Define the complete application flow, inputs, outputs, and persistent roles.
2. Check whether an existing Pipeline can be subclassed or extended with one Worker.
3. Define Worker APIs and WorkerConfig for every role.
4. Decide independently whether each Worker needs Strategy; reuse an existing Strategy when possible.
5. Define the DataProto schema and trace each field's producer and consumers.
6. Select Dispatch for every registered method.
7. Create and initialize Clusters in Pipeline.
8. Implement `run()` using only business steps and role interactions.
9. Configure weight updates, offload/reload, validation, and checkpointing as needed.
10. Validate one Worker, one node, multiple Workers, multiple nodes, and resume.

## Validation Checklist

### Configuration and resources

- The final merged YAML constructs the correct PipelineConfig.
- `pipeline_cls` and every `worker_cls` load dynamically.
- `device_mapping`, `num_gpus_per_worker`, and `world_size` agree.
- Shared or disjoint Cluster device relationships match the design.

### Worker and Dispatch

- Every registered method uses the intended Dispatch and execution scope.
- DP splitting processes each sample exactly as designed.
- TP/PP/CP ranks receive correct inputs and collection selects only valid outputs.
- Blocking and non-blocking calls materialize correctly.

### Strategy, when used

- At least one target training or inference backend is exercised.
- Forward, backward, generate, and checkpoint contracts agree.
- Worker contains no unnecessary backend-name branches.
- Each advertised backend is validated independently for values, shapes, synchronization, and restore.

### Pipeline

- Training, validation, error, empty-input, final-step, and shutdown paths terminate.
- DataProto keys, shapes, dtypes, batch sizes, and devices are correct.
- Global step, RNG, metrics history, and checkpoint resume correctly.
- Asynchronous Ray ObjectRefs do not leak or deadlock.
- Metrics are logged at the correct step with correct aggregation semantics.
- Multi-Cluster weight updates and state offload occur in the correct order.

## Reference Implementations

Choose the closest complete implementation:

- Minimal single-role training: `roll/pipeline/sft/sft_pipeline.py`
- Two-role preference training: `roll/pipeline/dpo/dpo_pipeline.py`
- Multi-role online RL: `roll/pipeline/rlvr/rlvr_pipeline.py`
- Agent-environment interaction: `roll/pipeline/agentic/agentic_pipeline.py`
- Teacher/student distillation: `roll/pipeline/distill/distill_pipeline.py`
- Rollout-only execution: `roll/pipeline/rlvr/rlvr_rollout_pipeline.py`

Read the target Pipeline, Config, Workers, Strategies, launcher, and YAML together. Copying only the Pipeline class can omit resource, Dispatch, DataProto, checkpoint, or backend contracts.
