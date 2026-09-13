# OpenTelemetry 分布式追踪

ROLL 支持 **OpenTelemetry (OTel)** 分布式追踪，用于 Pipeline 可观测性。启用后，系统会自动为每个 Pipeline 步骤、Scheduler 操作和 Worker 执行创建 Span，提供训练循环的端到端追踪视图。

## 简介

在大规模 RL 训练流水线中，理解跨多个 Ray Actor 的执行流程非常困难。仅靠日志难以回答"时间花在哪里？"、"关键路径是什么？"、"哪个步骤是瓶颈？"等问题。

ROLL 的 OpenTelemetry 集成提供：

- **端到端追踪可视化**：将完整的 Pipeline 步骤展示为追踪树（generate → train → validate 等）
- **跨 Actor 上下文传播**：追踪上下文通过 W3C TraceContext 标头自动经由 `DataProto.meta_info` 传播
- **零性能影响**：基准测试表明启用追踪无可测量的性能开销
- **自动 Collector 管理**：Driver 自动启动 OTLP Collector（或内置 Python Receiver 作为回退），退出时自动关闭

### 工作原理

1. **Driver 初始化**：当 `ROLL_OTEL_ENABLED=1` 时，Driver 自动选择空闲端口，启动 OTLP Collector，并为所有 Worker 设置 `OTEL_EXPORTER_OTLP_ENDPOINT`。
2. **Span 创建**：Driver 为每个 Pipeline 步骤创建根 Span，Scheduler 和 Worker 的 Span 嵌套在其下。
3. **上下文传播**：在将数据分发到远程 Actor 之前，追踪上下文被注入到 `DataProto.meta_info` 中。Worker 提取上下文并创建子 Span。
4. **导出**：所有 Span 通过 OTLP gRPC 导出到 Collector，Collector 将其写入 JSONL 文件供离线分析（如导入 Jaeger）。

### Collector 选择

系统自动选择最佳可用的 Collector：

1. **`otelcol-contrib` / `otelcol`**（首选）：生产级系统二进制文件。如果在 `PATH` 中找到，会自动生成配置文件并作为子进程启动。
2. **Python OTLP Receiver**（回退）：ROLL 内置的轻量级纯 Python gRPC 接收器。无需系统依赖 — 仅使用 `grpcio` 和 `protobuf`（已由 OTel SDK 引入）。

## 配置

通过在 `system_envs` 中设置 `ROLL_OTEL_ENABLED` 启用追踪，可选指定输出目录：

```yaml
system_envs:
  ROLL_OTEL_ENABLED: "1"

otlp_output_dir: /path/to/otlp_traces
```

- `system_envs.ROLL_OTEL_ENABLED`：设置为 `"1"` 启用 OpenTelemetry 追踪。Driver 会自动解析空闲端口并为所有 Worker 设置 `OTEL_EXPORTER_OTLP_ENDPOINT`。
- `otlp_output_dir`：OTLP Collector 写入追踪数据的目录。默认为 `./output/otlp`。追踪数据保存为 `<otlp_output_dir>/traces/traces.jsonl`。

无需其他配置 — 端点解析、Collector 启动和 SDK 初始化均自动处理。

### 查看追踪

输出的 `traces.jsonl` 文件可导入 [Jaeger](https://www.jaegertracing.io/) 进行可视化：

1. 启动支持 OTLP 的本地 Jaeger 实例
2. 通过 Jaeger 的 OTLP JSON 导入功能导入 `traces.jsonl` 文件

每个 Trace 展示完整的 Pipeline 步骤层级：driver → scheduler → workers，包含每个操作的耗时。

## 已插桩组件

以下组件已自动插桩：

| 组件 | Span |
|------|------|
| Pipeline Driver | `pipeline_step`、`stop_server`、`model_update`、`validation`、`generate`、`ref_log_probs`、`train` 等 |
| RolloutScheduler | `get_batch`、环境步骤编排 |
| Agentic Pipeline | 轨迹收集、环境管理器操作 |
| Worker | 通过 `@register(trace=True)` 装饰器创建执行 Span |

### 二次开发

在自定义 Pipeline 代码中，可以直接使用追踪工具：

```python
from roll.utils.telemetry import get_tracer, inject_trace_context, attach_trace_context

# 在 Driver/发送端 — 创建 Span 并传播上下文
tracer = get_tracer("driver")
with tracer.start_as_current_span("my_operation"):
    inject_trace_context(data.meta_info)
    # 分发数据到 Worker...

# 在 Worker/接收端 — 附加父上下文
with attach_trace_context(data.meta_info):
    with get_tracer("worker").start_as_current_span("worker_op"):
        # 执行工作...
```

核心 API：

- `get_tracer(name)`：返回 Tracer（追踪禁用/被抑制时返回 no-op）。
- `inject_trace_context(meta_info)`：将当前 Span 上下文注入字典，用于跨 Actor 传播。
- `attach_trace_context(meta_info)`：上下文管理器，从字典中提取并附加父上下文。
- `init_telemetry(service_name)`：初始化 OTel SDK（框架自动调用）。
- `shutdown_telemetry()`：刷新 Span 并关闭 Collector（退出时自动调用）。
