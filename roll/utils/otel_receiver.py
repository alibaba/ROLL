"""Lightweight Python-based OTLP gRPC receiver for traces.

A minimal replacement for ``otelcol-contrib`` that receives OTLP spans
via gRPC and writes them to a JSONL file (same format as otelcol).
Requires only ``grpcio`` and ``protobuf`` (already pulled in by the OTel SDK).

Usage::

    from roll.utils.otel_receiver import OTLPReceiver

    receiver = OTLPReceiver(port=4317, output_path="traces/traces.jsonl")
    receiver.start()
    # ... application runs ...
    receiver.stop()
"""

import base64
import json
import os
import threading
from concurrent import futures

import grpc
from google.protobuf.json_format import MessageToJson
from opentelemetry.proto.collector.trace.v1 import (
    trace_service_pb2,
    trace_service_pb2_grpc,
)

from roll.utils.logging import get_logger

logger = get_logger()

# Jaeger requires numeric span kind values (protobuf enum numbers),
# not the string enum names that MessageToJson produces.
# See: opentelemetry/proto/trace/v1/trace.proto SpanKind enum
_SPAN_KIND_MAP = {
    "SPAN_KIND_UNSPECIFIED": 0,
    "SPAN_KIND_INTERNAL": 1,
    "SPAN_KIND_SERVER": 2,
    "SPAN_KIND_CLIENT": 3,
    "SPAN_KIND_PRODUCER": 4,
    "SPAN_KIND_CONSUMER": 5,
}


def _normalize_spans(data):
    """Recursively normalize span fields for Jaeger compatibility.

    Performs two transformations required for Jaeger to parse OTLP JSONL:

    1. **Span kind**: Convert string enum (e.g. ``"SPAN_KIND_INTERNAL"``) to the
       corresponding protobuf integer (e.g. ``1``).  Jaeger's OTLP-JSON parser
       requires the numeric form; string enums cause a conversion error for
       large span batches.

    2. **Byte fields**: ``MessageToJson`` encodes protobuf ``bytes`` fields as
       base64 strings (e.g. ``"a+eBrRzJ2u8rRVD7TpsVAg=="``), but the OTLP JSON
       spec (and Jaeger) require ``traceId``, ``spanId``, and ``parentSpanId``
       as lowercase hex strings (e.g. ``"a7e781ad1cc9daefc...""``).
    """
    if isinstance(data, dict):
        result = {}
        for key, value in data.items():
            if key == "kind" and isinstance(value, str) and value in _SPAN_KIND_MAP:
                result[key] = _SPAN_KIND_MAP[value]
            elif key in ("traceId", "spanId", "parentSpanId") and isinstance(value, str):
                try:
                    result[key] = base64.b64decode(value).hex()
                except Exception:
                    result[key] = value  # already hex or unknown format — pass through
            else:
                result[key] = _normalize_spans(value)
        return result
    elif isinstance(data, list):
        return [_normalize_spans(item) for item in data]
    else:
        return data


class _TraceServicer:
    """gRPC servicer that receives spans and writes them to JSONL (matching otelcol format)."""

    def __init__(self, output_path: str, max_lines: int = 500):
        self.output_path = output_path
        self.max_lines = max_lines
        self._lock = threading.Lock()
        self._line_count = 0
        self._file_index = 0
        self._current_path = self._make_path()
        os.makedirs(os.path.dirname(self._current_path) if os.path.dirname(self._current_path) else ".", exist_ok=True)
        self._file = open(self._current_path, "a")
        logger.info(f"OTLPReceiver: writing traces to {self._current_path}")

    def _make_path(self) -> str:
        """Generate file path with index suffix."""
        if self._file_index == 0:
            return self.output_path
        base, ext = os.path.splitext(self.output_path)
        return f"{base}.{self._file_index}{ext}"

    def _rotate_file(self):
        """Close current file and open a new one if line limit reached."""
        self._file.close()
        self._file_index += 1
        self._line_count = 0
        self._current_path = self._make_path()
        self._file = open(self._current_path, "a")
        logger.info(f"OTLPReceiver: rotated to {self._current_path}")

    def Export(self, request, context):
        """Handle Export RPC call from OTLP client."""
        try:
            # Convert to canonical protobuf JSON (camelCase field names, matches otelcol).
            # indent=None produces compact single-line output required for JSONL.
            json_str = MessageToJson(
                request,
                preserving_proto_field_name=False,
                indent=None,
                sort_keys=False,
            )

            # Normalize: hex-encode byte fields and convert span kind to numeric
            data_dict = _normalize_spans(json.loads(json_str))
            json_str = json.dumps(data_dict, separators=(",", ":"))

            with self._lock:
                self._file.write(json_str + "\n")
                self._line_count += 1
                if self._line_count >= self.max_lines:
                    self._rotate_file()
                self._file.flush()
        except Exception as exc:
            logger.warning(f"OTLPReceiver: failed to write spans: {exc}")

        return trace_service_pb2.ExportTraceServiceResponse(
            partial_success=trace_service_pb2.ExportTracePartialSuccess(
                rejected_spans=0,
                error_message="",
            )
        )

    def close(self):
        """Close the output file."""
        with self._lock:
            if self._file:
                self._file.close()
                self._file = None


class OTLPReceiver:
    """Lightweight OTLP gRPC receiver for traces.

    Receives spans via OTLP gRPC protocol and writes them to a JSONL file
    (same format as otelcol's file exporter). Can be used as a drop-in 
    replacement for ``otelcol-contrib`` in development/testing scenarios.
    """

    def __init__(self, port: int, output_path: str, max_lines: int = 500):
        """
        Args:
            port: gRPC port to listen on.
            output_path: Path to write JSONL trace data.
            max_lines: Maximum number of lines per file before rotating (default: 500).
        """
        self.port = port
        self.output_path = output_path
        self.max_lines = max_lines
        self._server = None
        self._servicer = None

    def start(self) -> None:
        """Start the OTLP gRPC receiver."""
        self._servicer = _TraceServicer(self.output_path, self.max_lines)
        self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        trace_service_pb2_grpc.add_TraceServiceServicer_to_server(
            self._servicer, self._server
        )
        self._server.add_insecure_port(f"[::]:{self.port}")
        self._server.start()
        logger.info(f"Python OTLPReceiver started on port {self.port}")

    def stop(self, grace: float = 2.0) -> None:
        """Stop the OTLP gRPC receiver.

        Args:
            grace: Grace period in seconds for pending RPCs to complete.
        """
        if self._server:
            self._server.stop(grace)
            self._server = None
        if self._servicer:
            self._servicer.close()
            self._servicer = None
        logger.info("Python OTLPReceiver stopped.")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
