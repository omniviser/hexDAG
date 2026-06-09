"""Langfuse tracing observer for hexDAG pipelines.

Maps hexDAG's event taxonomy to Langfuse's trace/span/generation model:

    SystemStarted        -> langfuse.trace(name=system)
      ProcessStarted     ->   trace.span(name=process)
        PipelineStarted  ->     process_span.span(name=pipeline)
          NodeStarted    ->       span.span(name=node)
            LLMPortCall  ->         span.generation(model, input, output, usage)
            ToolRouterCall ->   span.span(name="tool:X")
          NodeCompleted  ->       node_span.end()
        PipelineCompleted ->    pipeline_span.end()
      ProcessCompleted   ->   process_span.end()
    SystemCompleted      ->   trace.update(output=results)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hexdag.kernel.orchestration.events.events import (
    Event,
    NodeCompleted,
    NodeFailed,
    NodeStarted,
    PipelineCompleted,
    PipelineStarted,
    ProcessCompleted,
    ProcessStarted,
    SystemCompleted,
    SystemStarted,
)
from hexdag.kernel.ports.llm import LLMPortCall
from hexdag.kernel.tool_router import ToolRouterCall

if TYPE_CHECKING:
    from langfuse import Langfuse

# Maximum characters to store in span output (avoid sending huge payloads).
_OUTPUT_TRUNCATE = 1000


class LangfuseObserver:
    """Observer that maps hexDAG events to Langfuse traces.

    Implements the ``Observer`` protocol — register with an
    ``ObserverManager`` and it will receive events automatically.

    Parameters
    ----------
    client : Langfuse
        Pre-configured Langfuse client instance.
    session_id : str | None
        Optional session ID for grouping traces in Langfuse.
    trace_metadata : dict[str, Any] | None
        Extra metadata attached to every trace root.
    flush_on_complete : bool
        Call ``client.flush()`` when the outermost scope ends.
    """

    def __init__(
        self,
        client: Langfuse,
        *,
        session_id: str | None = None,
        trace_metadata: dict[str, Any] | None = None,
        flush_on_complete: bool = True,
    ) -> None:
        self._client = client
        self._session_id = session_id
        self._trace_metadata = trace_metadata or {}
        self._flush_on_complete = flush_on_complete

        # Active Langfuse objects, keyed for nesting.
        self._trace: Any | None = None
        self._process_spans: dict[str, Any] = {}
        self._pipeline_spans: dict[str, Any] = {}
        self._node_spans: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Observer protocol
    # ------------------------------------------------------------------

    async def handle(self, event: Event) -> None:
        """Route hexDAG events to the appropriate Langfuse SDK call."""
        match event:
            case SystemStarted():
                self._on_system_started(event)
            case ProcessStarted():
                self._on_process_started(event)
            case PipelineStarted():
                self._on_pipeline_started(event)
            case NodeStarted():
                self._on_node_started(event)
            case LLMPortCall():
                self._on_llm_port_call(event)
            case ToolRouterCall():
                self._on_tool_call(event)
            case NodeCompleted():
                self._on_node_completed(event)
            case NodeFailed():
                self._on_node_failed(event)
            case PipelineCompleted():
                self._on_pipeline_completed(event)
            case ProcessCompleted():
                self._on_process_completed(event)
            case SystemCompleted():
                self._on_system_completed(event)

    # ------------------------------------------------------------------
    # Private handlers
    # ------------------------------------------------------------------

    def _on_system_started(self, event: SystemStarted) -> None:
        self._trace = self._client.trace(
            name=event.name,
            session_id=self._session_id,
            metadata={
                **self._trace_metadata,
                "total_processes": event.total_processes,
                "execution_order": event.execution_order,
            },
        )

    def _on_process_started(self, event: ProcessStarted) -> None:
        parent = self._trace
        if parent is None:
            return
        span = parent.span(
            name=event.process_name,
            metadata={
                "system_name": event.system_name,
                "index": event.index,
            },
        )
        self._process_spans[event.process_name] = span

    def _on_pipeline_started(self, event: PipelineStarted) -> None:
        # Resolve parent: process span > trace > create standalone trace.
        parent = (
            next(
                (self._process_spans[k] for k in reversed(self._process_spans)),
                None,
            )
            or self._trace
        )
        if parent is None:
            # Standalone pipeline (no System/Process wrapper) — fresh trace.
            self._trace = self._client.trace(
                name=event.name,
                session_id=self._session_id,
                metadata={
                    **self._trace_metadata,
                    "total_nodes": event.total_nodes,
                    "total_waves": event.total_waves,
                },
            )
            parent = self._trace

        span = parent.span(
            name=event.name,
            metadata={
                "total_nodes": event.total_nodes,
                "total_waves": event.total_waves,
            },
        )
        self._pipeline_spans[event.name] = span

    def _on_node_started(self, event: NodeStarted) -> None:
        parent = (
            self._pipeline_spans.get(
                # Pick the most-recently added pipeline span as parent.
                next(reversed(self._pipeline_spans), ""),
            )
            or self._trace
        )
        if parent is None:
            return
        span = parent.span(
            name=event.name,
            metadata={
                "wave_index": event.wave_index,
                "dependencies": list(event.dependencies),
            },
        )
        self._node_spans[event.name] = span

    def _on_llm_port_call(self, event: LLMPortCall) -> None:
        parent = self._node_spans.get(event.node_name) or self._trace
        if parent is None:
            return

        usage = None
        if event.usage:
            usage = {
                "input": event.usage.get("input_tokens", 0),
                "output": event.usage.get("output_tokens", 0),
                "total": event.usage.get("total_tokens", 0),
            }

        parent.generation(
            name=event.method,
            model=event.model,
            input=event.messages,
            output=event.response[:_OUTPUT_TRUNCATE] if event.response else "",
            usage=usage,
            metadata={
                "duration_ms": event.duration_ms,
                "tool_calls": event.tool_calls,
            },
        )

    def _on_tool_call(self, event: ToolRouterCall) -> None:
        parent = self._node_spans.get(event.node_name) or self._trace
        if parent is None:
            return
        span = parent.span(
            name=f"tool:{event.tool_name}",
            input=event.params,
            output=str(event.result)[:_OUTPUT_TRUNCATE] if event.result else None,
            metadata={"duration_ms": event.duration_ms},
        )
        span.end()

    def _on_node_completed(self, event: NodeCompleted) -> None:
        if span := self._node_spans.pop(event.name, None):
            output = str(event.result)[:_OUTPUT_TRUNCATE] if event.result else None
            span.end(
                output=output,
                metadata={"duration_ms": event.duration_ms},
            )

    def _on_node_failed(self, event: NodeFailed) -> None:
        if span := self._node_spans.pop(event.name, None):
            span.end(
                level="ERROR",
                status_message=str(event.error),
            )

    def _on_pipeline_completed(self, event: PipelineCompleted) -> None:
        if span := self._pipeline_spans.pop(event.name, None):
            level = "ERROR" if event.status == "failed" else "DEFAULT"
            span.end(
                metadata={
                    "duration_ms": event.duration_ms,
                    "status": event.status,
                },
                level=level,
            )
        # Flush for standalone pipelines (no process/system wrapping them).
        if self._flush_on_complete and not self._pipeline_spans and not self._process_spans:
            self._client.flush()

    def _on_process_completed(self, event: ProcessCompleted) -> None:
        if span := self._process_spans.pop(event.process_name, None):
            level = "ERROR" if event.status == "failed" else "DEFAULT"
            span.end(
                metadata={
                    "duration_ms": event.duration_ms,
                    "status": event.status,
                    "error": event.error,
                },
                level=level,
            )

    def _on_system_completed(self, event: SystemCompleted) -> None:
        if self._trace:
            self._trace.update(
                output=event.process_results,
                metadata={
                    "duration_ms": event.duration_ms,
                    "status": event.status,
                },
            )
        if self._flush_on_complete:
            self._client.flush()
        self._trace = None
