"""Tests for OpenTelemetryObserver — hexDAG events → OTel spans."""

from __future__ import annotations

import pytest

pytest.importorskip("opentelemetry.sdk", reason="requires hexdag[otel]")

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from hexdag.kernel.orchestration.events.events import (
    NodeCompleted,
    NodeFailed,
    NodeStarted,
    PipelineCompleted,
    PipelineStarted,
)
from hexdag.kernel.ports.llm import LLMPortCall
from hexdag.stdlib.lib.observers import OpenTelemetryObserver


@pytest.fixture()
def exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture()
def observer(exporter: InMemorySpanExporter) -> OpenTelemetryObserver:
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return OpenTelemetryObserver(tracer_provider=provider)


async def _run_simple_pipeline(observer: OpenTelemetryObserver) -> None:
    """Feed a canonical event sequence through the observer."""
    await observer.handle(PipelineStarted(name="demo", total_waves=1, total_nodes=1))
    await observer.handle(NodeStarted(name="analyze", wave_index=1))
    await observer.handle(
        LLMPortCall(
            port_type="llm",
            method="aresponse",
            node_name="analyze",
            duration_ms=12.5,
            model="gpt-4o-mini",
            usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            response="hi",
        )
    )
    await observer.handle(
        NodeCompleted(name="analyze", wave_index=1, result="hi", duration_ms=15.0)
    )
    await observer.handle(PipelineCompleted(name="demo", duration_ms=20.0))


class TestOpenTelemetryObserver:
    @pytest.mark.asyncio()
    async def test_span_tree_exported(
        self, observer: OpenTelemetryObserver, exporter: InMemorySpanExporter
    ) -> None:
        await _run_simple_pipeline(observer)

        spans = {s.name: s for s in exporter.get_finished_spans()}
        assert set(spans) == {"pipeline:demo", "node:analyze", "llm.aresponse"}

        pipeline = spans["pipeline:demo"]
        node = spans["node:analyze"]
        llm = spans["llm.aresponse"]

        # Hierarchy: node under pipeline, llm call under node
        assert node.parent is not None
        assert node.parent.span_id == pipeline.context.span_id
        assert llm.parent is not None
        assert llm.parent.span_id == node.context.span_id

        # Attributes
        assert pipeline.attributes["hexdag.pipeline.total_nodes"] == 1
        assert node.attributes["hexdag.node.wave_index"] == 1
        assert llm.attributes["gen_ai.request.model"] == "gpt-4o-mini"
        assert llm.attributes["gen_ai.usage.input_tokens"] == 10
        assert llm.attributes["gen_ai.usage.output_tokens"] == 5

    @pytest.mark.asyncio()
    async def test_node_failure_sets_error_status(
        self, observer: OpenTelemetryObserver, exporter: InMemorySpanExporter
    ) -> None:
        await observer.handle(PipelineStarted(name="demo", total_waves=1, total_nodes=1))
        await observer.handle(NodeStarted(name="broken", wave_index=1))
        await observer.handle(NodeFailed(name="broken", wave_index=1, error=ValueError("boom")))
        await observer.handle(
            PipelineCompleted(name="demo", duration_ms=5.0, status="failed", reason="boom")
        )

        spans = {s.name: s for s in exporter.get_finished_spans()}
        assert spans["node:broken"].status.status_code == StatusCode.ERROR
        assert spans["pipeline:demo"].status.status_code == StatusCode.ERROR
        events = spans["node:broken"].events
        assert any(e.name == "exception" for e in events)

    @pytest.mark.asyncio()
    async def test_unmatched_events_are_ignored(
        self, observer: OpenTelemetryObserver, exporter: InMemorySpanExporter
    ) -> None:
        """Events without a started pipeline/node never raise."""
        await observer.handle(NodeCompleted(name="ghost", wave_index=1, result=None, duration_ms=1))
        await observer.handle(PipelineCompleted(name="ghost", duration_ms=1.0))
        assert exporter.get_finished_spans() == ()
