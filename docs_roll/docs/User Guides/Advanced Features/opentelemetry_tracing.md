# OpenTelemetry Distributed Tracing

ROLL supports **OpenTelemetry (OTel)** distributed tracing for pipeline observability. When enabled, spans are automatically created for each pipeline step, scheduler operation, and worker execution, providing an end-to-end trace view of the training loop.

## Introduction

In large-scale RL training pipelines, understanding execution flow across multiple Ray actors is challenging. Key questions like "where is the time spent?", "what is the critical path?", and "which step is the bottleneck?" are difficult to answer with logs alone.

ROLL's OpenTelemetry integration provides:

- **End-to-end trace visualization**: View the full pipeline step as a trace tree (generate → train → validate, etc.)
- **Cross-actor context propagation**: Trace context is automatically propagated through `DataProto.meta_info` via W3C TraceContext headers
- **Zero performance impact**: Benchmarks show no measurable overhead when tracing is enabled
- **Automatic collector management**: The driver automatically starts an OTLP collector (or a built-in Python receiver as fallback) and shuts it down on exit

### How It Works

1. **Driver initialization**: When `ROLL_OTEL_ENABLED=1`, the driver auto-selects a free port, starts an OTLP collector, and sets `OTEL_EXPORTER_OTLP_ENDPOINT` for all workers.
2. **Span creation**: The driver creates root spans for each pipeline step. Scheduler and worker spans are nested underneath.
3. **Context propagation**: Before dispatching data to remote actors, trace context is injected into `DataProto.meta_info`. Workers extract it and create child spans.
4. **Export**: All spans are exported via OTLP gRPC to the collector, which writes them to a JSONL file for offline analysis (e.g., import into Jaeger).

### Collector Selection

The system automatically selects the best available collector:

1. **`otelcol-contrib` / `otelcol`** (preferred): Production-ready system binary. If found on `PATH`, a config file is generated and the binary is launched as a subprocess.
2. **Python OTLP Receiver** (fallback): A lightweight pure-Python gRPC receiver built into ROLL. No system dependencies required — uses only `grpcio` and `protobuf` (already pulled in by the OTel SDK).

## Configuration

Enable tracing by setting `ROLL_OTEL_ENABLED` in `system_envs` and optionally specifying the output directory:

```yaml
system_envs:
  ROLL_OTEL_ENABLED: "1"

otlp_output_dir: /path/to/otlp_traces
```

- `system_envs.ROLL_OTEL_ENABLED`: Set to `"1"` to enable OpenTelemetry tracing. The driver will auto-resolve a free port and set `OTEL_EXPORTER_OTLP_ENDPOINT` for all workers.
- `otlp_output_dir`: Directory where the OTLP collector writes trace data. Defaults to `./output/otlp`. Traces are saved as `<otlp_output_dir>/traces/traces.jsonl`.

No other configuration is needed — endpoint resolution, collector startup, and SDK initialization are all handled automatically.

### Viewing Traces

The output `traces.jsonl` file can be imported into [Jaeger](https://www.jaegertracing.io/) for visualization:

1. Start a local Jaeger instance with OTLP support
2. Import the `traces.jsonl` file via Jaeger's OTLP JSON import

Each trace shows the full pipeline step hierarchy: driver → scheduler → workers, with timing for each operation.

## Instrumented Components

The following components are automatically instrumented:

| Component | Spans |
|-----------|-------|
| Pipeline Driver | `pipeline_step`, `stop_server`, `model_update`, `validation`, `generate`, `ref_log_probs`, `train`, etc. |
| RolloutScheduler | `get_batch`, environment step orchestration |
| Agentic Pipeline | Trajectory collection, environment manager operations |
| Workers | Execution spans via `@register(trace=True)` decorator |

### Custom Instrumentation

For custom pipeline code, use the tracing utilities directly:

```python
from roll.utils.telemetry import get_tracer, inject_trace_context, attach_trace_context

# On the driver/sender side — create spans and propagate context
tracer = get_tracer("driver")
with tracer.start_as_current_span("my_operation"):
    inject_trace_context(data.meta_info)
    # dispatch data to workers...

# On the worker/receiver side — attach parent context
with attach_trace_context(data.meta_info):
    with get_tracer("worker").start_as_current_span("worker_op"):
        # work happens here...
```

Key APIs:

- `get_tracer(name)`: Returns a tracer (or no-op if tracing is disabled/suppressed).
- `inject_trace_context(meta_info)`: Injects the current span context into a dict for cross-actor propagation.
- `attach_trace_context(meta_info)`: Context manager that extracts and attaches parent context from a dict.
- `init_telemetry(service_name)`: Initializes the OTel SDK (called automatically by the framework).
- `shutdown_telemetry()`: Flushes spans and shuts down the collector (called automatically at exit).
