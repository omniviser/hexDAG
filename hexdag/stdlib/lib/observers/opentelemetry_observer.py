"""OpenTelemetry exporter observer — hexDAG events → OTel spans.

Reference observability exporter.  Maps the hexDAG event taxonomy to an
OpenTelemetry trace tree::

    pipeline:<name>              (root span, one per run)
    ├── node:<node_name>         (one span per node execution)
    ├── node:<node_name>
    └── llm.aresponse            (one span per port call, with token usage)

Requires the ``otel`` extra::

    pip install hexdag[otel]

Usage::

    from hexdag.drivers.observer_manager import LocalObserverManager
    from hexdag.stdlib.lib.observers import OpenTelemetryObserver

    observer_manager = LocalObserverManager()
    observer_manager.register(OpenTelemetryObserver().handle)
    # pass observer_manager as the "observer_manager" port

Notes
-----
Correlation is by pipeline name (hexDAG events do not carry a run_id),
so one observer instance should monitor one pipeline run at a time.
Register a fresh instance per run when executing pipelines concurrently.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hexdag.kernel.logging import get_logger
from hexdag.kernel.orchestration.events.events import (
    Event,
    NodeCompleted,
    NodeFailed,
    NodeStarted,
    PipelineCompleted,
    PipelineStarted,
    PortCallEvent,
)
from hexdag.kernel.ports.llm import LLMPortCall

if TYPE_CHECKING:
    from datetime import datetime

logger = get_logger(__name__)

_OTEL_IMPORT_ERROR = (
    "OpenTelemetryObserver requires the OpenTelemetry SDK. "
    "Install it with: pip install hexdag[otel]"
)


def _to_ns(timestamp: datetime) -> int:
    """Convert an event timestamp to integer nanoseconds since epoch."""
    return int(timestamp.timestamp() * 1_000_000_000)


class OpenTelemetryObserver:
    """Observer that exports pipeline execution as OpenTelemetry spans.

    Parameters
    ----------
    tracer_provider : Any | None
        OpenTelemetry ``TracerProvider``.  If ``None``, the globally
        configured provider is used (``trace.get_tracer_provider()``).
    instrumentation_name : str
        Name passed to ``get_tracer`` (shows as the instrumentation
        scope in exported traces).

    Examples
    --------
    Export to console (development)::

        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            ConsoleSpanExporter,
            SimpleSpanProcessor,
        )

        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
        observer = OpenTelemetryObserver(tracer_provider=provider)

    Export to an OTLP collector (production)::

        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    """

    def __init__(
        self,
        tracer_provider: Any | None = None,
        instrumentation_name: str = "hexdag",
    ) -> None:
        """Build the observer, resolving an OTel tracer from *tracer_provider*."""
        try:
            from opentelemetry import trace  # lazy: optional opentelemetry dependency
        except ImportError as e:
            raise ImportError(_OTEL_IMPORT_ERROR) from e

        self._trace = trace
        provider = tracer_provider or trace.get_tracer_provider()
        self._tracer = provider.get_tracer(instrumentation_name)

        self._pipeline_span: Any | None = None
        self._node_spans: dict[str, Any] = {}

    async def handle(self, event: Event) -> None:
        """Map a hexDAG event onto the OTel span tree (read-only)."""
        try:
            if isinstance(event, PipelineStarted):
                self._on_pipeline_started(event)
            elif isinstance(event, PipelineCompleted):
                self._on_pipeline_completed(event)
            elif isinstance(event, NodeStarted):
                self._on_node_started(event)
            elif isinstance(event, NodeCompleted):
                self._on_node_finished(event.name, event.timestamp, error=None)
            elif isinstance(event, NodeFailed):
                self._on_node_finished(event.name, event.timestamp, error=event.error)
            elif isinstance(event, PortCallEvent):
                self._on_port_call(event)
        except Exception as e:
            # Observers must never affect execution
            logger.warning("OpenTelemetryObserver failed to handle {}: {}", type(event).__name__, e)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_pipeline_started(self, event: PipelineStarted) -> None:
        """Open the root pipeline span."""
        span = self._tracer.start_span(
            f"pipeline:{event.name}",
            start_time=_to_ns(event.timestamp),
            attributes={
                "hexdag.pipeline.name": event.name,
                "hexdag.pipeline.total_nodes": event.total_nodes,
                "hexdag.pipeline.total_waves": event.total_waves,
            },
        )
        self._pipeline_span = span

    def _on_pipeline_completed(self, event: PipelineCompleted) -> None:
        """Close the pipeline span and any orphaned node spans."""
        span = self._pipeline_span
        if span is None:
            return
        span.set_attribute("hexdag.pipeline.status", event.status)
        span.set_attribute("hexdag.pipeline.duration_ms", event.duration_ms)
        if event.status != "completed":
            span.set_status(self._trace.StatusCode.ERROR, event.reason or event.status)
        span.end(end_time=_to_ns(event.timestamp))
        self._pipeline_span = None
        # Close any orphaned node spans (e.g. cancelled mid-wave)
        for node_span in self._node_spans.values():
            node_span.end()
        self._node_spans.clear()

    def _on_node_started(self, event: NodeStarted) -> None:
        """Open a node span as a child of the pipeline span."""
        span = self._tracer.start_span(
            f"node:{event.name}",
            context=self._parent_context(),
            start_time=_to_ns(event.timestamp),
            attributes={
                "hexdag.node.name": event.name,
                "hexdag.node.wave_index": event.wave_index,
                "hexdag.node.dependencies": list(event.dependencies),
            },
        )
        self._node_spans[event.name] = span

    def _on_node_finished(
        self,
        node_name: str,
        timestamp: datetime,
        error: Exception | None,
    ) -> None:
        """Close a node span, recording *error* when the node failed."""
        span = self._node_spans.pop(node_name, None)
        if span is None:
            return
        if error is not None:
            span.record_exception(error)
            span.set_status(self._trace.StatusCode.ERROR, str(error))
        span.end(end_time=_to_ns(timestamp))

    def _on_port_call(self, event: PortCallEvent) -> None:
        """Emit a child span for a port call (LLM, datastore, …)."""
        end_ns = _to_ns(event.timestamp)
        start_ns = end_ns - int(event.duration_ms * 1_000_000)

        attributes: dict[str, Any] = {
            "hexdag.port.type": event.port_type,
            "hexdag.port.method": event.method,
            "hexdag.node.name": event.node_name,
            "hexdag.port.duration_ms": event.duration_ms,
        }
        if isinstance(event, LLMPortCall):
            if event.model:
                attributes["gen_ai.request.model"] = event.model
            if event.usage:
                if (input_tokens := event.usage.get("input_tokens")) is not None:
                    attributes["gen_ai.usage.input_tokens"] = input_tokens
                if (output_tokens := event.usage.get("output_tokens")) is not None:
                    attributes["gen_ai.usage.output_tokens"] = output_tokens

        # Parent under the originating node span when still open
        parent = self._node_spans.get(event.node_name)
        context = (
            self._trace.set_span_in_context(parent)
            if parent is not None
            else self._parent_context()
        )

        span = self._tracer.start_span(
            f"{event.port_type}.{event.method}",
            context=context,
            start_time=start_ns,
            attributes=attributes,
        )
        span.end(end_time=end_ns)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _parent_context(self) -> Any | None:
        """Context that parents new spans under the pipeline span."""
        if self._pipeline_span is None:
            return None
        return self._trace.set_span_in_context(self._pipeline_span)
