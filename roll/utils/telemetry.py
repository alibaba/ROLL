"""OpenTelemetry tracing utilities for ROLL pipelines.

Provides cross-actor trace context propagation via W3C TraceContext.
The driver auto-starts an OpenTelemetry Collector and all processes
(driver and actors) export spans via OTLP gRPC to it.

**Collector Selection** (automatic):
1. **otelcol-contrib/otelcol** (preferred) - Production-ready system binary.
2. **Python OTLP Receiver** (fallback) - Pure Python, no system dependencies.

Configuration is done via ``system_envs`` in the pipeline config::

    system_envs:
      ROLL_OTEL_ENABLED: "1"

The ``OTEL_EXPORTER_OTLP_ENDPOINT`` is auto-generated in
``BaseConfig.__post_init__`` (picks a free port on the driver node).

Usage::

    # Propagate context to remote actors:
    inject_trace_context(data_proto.meta_info)

    # Attach parent context in a receiver:
    with attach_trace_context(data_proto.meta_info):
        with get_tracer("worker").start_as_current_span("my_span"):
            ...

    # At shutdown:
    shutdown_telemetry()
"""

import atexit
import os
import shutil
import subprocess
import textwrap
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Optional
from urllib.parse import urlparse

from opentelemetry import trace
from opentelemetry.context import Context, attach, detach
from opentelemetry.propagate import inject, extract
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.id_generator import IdGenerator

from roll.utils.otel_receiver import OTLPReceiver
from roll.utils.network_utils import get_node_ip, collect_free_port
from roll.utils.logging import get_logger

logger = get_logger()

_telemetry_initialized = False
_otel_collector_process = None
_otel_receiver = None

_OTEL_CTX_KEY = "_otel_ctx"
_trace_suppressed: ContextVar[bool] = ContextVar("_trace_suppressed", default=False)
_noop_tracer_provider = trace.NoOpTracerProvider()


class _OsUrandomIdGenerator(IdGenerator):
    """Span/trace ID generator immune to ``random.seed()``.

    The default ``RandomIdGenerator`` uses ``random.getrandbits()`` which
    is affected by ML training code calling ``random.seed(42)`` for
    reproducibility.  When every worker shares the same seed, they all
    produce **identical** span IDs, corrupting the trace tree.

    This generator uses ``os.urandom()`` (kernel CSPRNG) which is never
    affected by ``random.seed()``.
    """

    def generate_span_id(self) -> int:
        return int.from_bytes(os.urandom(8), byteorder="big") & ((1 << 64) - 1)

    def generate_trace_id(self) -> int:
        return int.from_bytes(os.urandom(16), byteorder="big") & ((1 << 128) - 1)


def inject_trace_context(meta_info: dict, context: Optional[Context] = None) -> dict:
    """Inject current span context into a *meta_info* dict.

    The context is stored under the ``_otel_ctx`` key as a plain ``dict``
    containing W3C ``traceparent`` / ``tracestate`` headers.  This dict is
    safe to pass through Ray serialisation and ``DataProto.meta_info``.
    """
    carrier: dict = {}
    inject(carrier, context)  # writes traceparent/tracestate into carrier
    meta_info[_OTEL_CTX_KEY] = carrier
    return meta_info


def extract_trace_context(meta_info: Optional[dict]) -> Context:
    """Extract parent span context from a *meta_info* dict.

    Returns a :class:`~opentelemetry.context.Context` that can be passed as
    the ``context`` argument of ``tracer.start_as_current_span()``.
    If *meta_info* is ``None`` or contains no ``_otel_ctx`` key, an empty
    context is returned.
    """
    if meta_info is None:
        return Context()
    carrier = meta_info.get(_OTEL_CTX_KEY, {})
    return extract(carrier)


@contextmanager
def attach_trace_context(meta_info: Optional[dict]):
    """Context manager: extract parent context from *meta_info* and attach it.

    If *meta_info* is ``None`` or contains no trace context (no ``_otel_ctx``
    key), tracing is suppressed for the duration of the block — all
    :func:`get_tracer` calls will return a no-op tracer.

    Automatically detaches when the block exits.  Usage::

        with attach_trace_context(data.meta_info):
            with get_tracer("worker").start_as_current_span("my_span"):
                ...
    """
    ctx = extract_trace_context(meta_info)
    token = attach(ctx)
    suppressed = meta_info is None or _OTEL_CTX_KEY not in meta_info
    suppress_token = _trace_suppressed.set(suppressed)
    try:
        yield ctx
    finally:
        _trace_suppressed.reset(suppress_token)
        detach(token)


def resolve_otel_endpoint() -> str:
    """Auto-detect driver IP and pick a free port for the OTLP gRPC endpoint."""
    ip = get_node_ip()
    port = collect_free_port()
    endpoint = f"http://{ip}:{port}"
    logger.info(f"Resolved OTel endpoint: {endpoint}")
    return endpoint


def start_otel_collector(endpoint: str, output_dir: str) -> None:
    """Start an OpenTelemetry Collector subprocess on the driver.

    Tries ``otelcol-contrib`` / ``otelcol`` first (production-ready).
    Falls back to a pure-Python OTLP receiver if no system binary is available.

    The collector listens for OTLP gRPC on the port embedded in *endpoint*
    and writes traces to ``<output_dir>/traces/traces.jsonl``.
    """
    global _otel_collector_process, _otel_receiver
    if _otel_collector_process is not None or _otel_receiver is not None:
        logger.warning("OTel Collector already running, skipping.")
        return

    parsed = urlparse(endpoint)
    port = parsed.port or 4317

    traces_dir = os.path.join(output_dir, "traces")
    os.makedirs(traces_dir, exist_ok=True)

    # Try system otelcol binary first (production-ready)
    binary = shutil.which("otelcol-contrib") or shutil.which("otelcol")
    if binary is not None:
        config_content = textwrap.dedent(f"""\
            receivers:
              otlp:
                protocols:
                  grpc:
                    endpoint: "0.0.0.0:{port}"
            exporters:
              file:
                path: "{traces_dir}/traces.jsonl"
            service:
              telemetry:
                metrics:
                  level: none
              pipelines:
                traces:
                  receivers: [otlp]
                  exporters: [file]
        """)

        config_path = os.path.join(traces_dir, "otelcol-config.yaml")
        with open(config_path, "w") as f:
            f.write(config_content)

        stdout_f = open(os.path.join(traces_dir, "otelcol.stdout.log"), "w")
        stderr_f = open(os.path.join(traces_dir, "otelcol.stderr.log"), "w")
        p = subprocess.Popen(
            [binary, "--config", config_path],
            stdout=stdout_f,
            stderr=stderr_f,
        )
        time.sleep(1)
        if p.poll() is not None:
            raise RuntimeError(
                f"OTel Collector exited immediately with code {p.returncode}. "
                f"Check {traces_dir}/otelcol.stderr.log for details."
            )
        _otel_collector_process = p
        logger.info(f"OTel Collector started (PID {_otel_collector_process.pid}) on port {port}")
    else:
        # Fallback to Python-based receiver (lightweight, no system dependency)
        output_path = os.path.join(traces_dir, "traces.jsonl")
        receiver = OTLPReceiver(port=port, output_path=output_path)
        receiver.start()
        _otel_receiver = receiver
        logger.info(f"Python OTLPReceiver started on port {port}")

    atexit.register(shutdown_telemetry)


def stop_otel_collector() -> None:
    """Terminate the collector subprocess started by :func:`start_otel_collector`."""
    global _otel_collector_process, _otel_receiver
    if _otel_collector_process is not None:
        _otel_collector_process.terminate()
        try:
            _otel_collector_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _otel_collector_process.kill()
        logger.info("OTel Collector stopped.")
        _otel_collector_process = None
    if _otel_receiver is not None:
        _otel_receiver.stop()
        logger.info("Python OTLPReceiver stopped.")
        _otel_receiver = None

def init_telemetry(
    service_name: str,
    otlp_endpoint: Optional[str] = None,
    instance_id: Optional[str] = None,
) -> None:
    """Initialize OpenTelemetry tracing.

    Requires an OTLP endpoint (provided directly or via the
    ``OTEL_EXPORTER_OTLP_ENDPOINT`` environment variable, typically set
    through ``system_envs`` in the pipeline config).

    Args:
        service_name: The service name for trace identification.
        otlp_endpoint: Optional OTLP gRPC endpoint (e.g. ``http://localhost:4317``).
            If not set, checks ``OTEL_EXPORTER_OTLP_ENDPOINT``.
        instance_id: Optional identifier (e.g. worker name) added as
            the ``service.instance.id`` resource attribute for Jaeger.
    """
    global _telemetry_initialized
    if _telemetry_initialized:
        logger.warning("Telemetry already initialized, skipping.")
        return

    endpoint = otlp_endpoint or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")

    if not endpoint:
        logger.info("Telemetry: no OTLP endpoint available, skipping init.")
        return

    resource_attrs = {"service.name": service_name}
    if instance_id:
        resource_attrs["service.instance.id"] = instance_id
    resource = Resource.create(resource_attrs)
    provider = TracerProvider(resource=resource, id_generator=_OsUrandomIdGenerator())

    otlp_exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
    provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

    trace.set_tracer_provider(provider)
    _telemetry_initialized = True
    log_msg = f"Telemetry initialized: service={service_name}, endpoint={endpoint}"
    if instance_id:
        log_msg += f", instance={instance_id}"
    logger.info(log_msg)


def get_tracer(name: str = "roll") -> trace.Tracer:
    """Return a tracer from the global TracerProvider.

    Returns a no-op tracer if telemetry has not been initialized or if
    the current context has tracing suppressed (i.e. inside an
    ``attach_trace_context({})`` block with no trace context).
    """
    if _trace_suppressed.get():
        return _noop_tracer_provider.get_tracer(name)
    return trace.get_tracer(name)


def shutdown_telemetry() -> None:
    """Flush pending spans, shut down the tracer provider, and stop the collector."""
    global _telemetry_initialized
    if not _telemetry_initialized:
        return
    provider = trace.get_tracer_provider()
    provider.force_flush()
    provider.shutdown()
    _telemetry_initialized = False
    logger.info("Telemetry shut down.")
    stop_otel_collector()
