"""Tests for LangfuseObserver — uses mock Langfuse client, no real SDK needed."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from hexdag.kernel.orchestration.events.events import (
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
from hexdag.kernel.ports.tool_router import ToolRouterPortCall

from hexdag_plugins.langfuse.observer import LangfuseObserver


def _make_mock_client() -> MagicMock:
    """Create a mock Langfuse client with 4-level chaining.

    trace -> process_span -> pipeline_span -> node_span
    Each level returns a fresh MagicMock with its own .span()/.generation().
    """
    client = MagicMock()
    trace = MagicMock(name="trace")
    process_span = MagicMock(name="process_span")
    pipeline_span = MagicMock(name="pipeline_span")
    node_span = MagicMock(name="node_span")

    client.trace.return_value = trace
    trace.span.return_value = process_span
    process_span.span.return_value = pipeline_span
    pipeline_span.span.return_value = node_span
    node_span.span.return_value = MagicMock(name="tool_span")
    return client


@pytest.fixture
def mock_client() -> MagicMock:
    return _make_mock_client()


@pytest.fixture
def observer(mock_client: MagicMock) -> LangfuseObserver:
    return LangfuseObserver(
        mock_client,
        session_id="test-session",
        trace_metadata={"env": "test"},
    )


# Helper: emit the full System → Process → Pipeline → Node setup sequence.
async def _setup_full_hierarchy(
    obs: LangfuseObserver,
) -> None:
    await obs.handle(SystemStarted(name="s", total_processes=1, execution_order=["p"]))
    await obs.handle(ProcessStarted(system_name="s", process_name="p", index=0))
    await obs.handle(PipelineStarted(name="p", total_waves=1, total_nodes=1))


# ------------------------------------------------------------------
# System-level flow
# ------------------------------------------------------------------


async def test_system_started_creates_trace(
    observer: LangfuseObserver, mock_client: MagicMock
) -> None:
    event = SystemStarted(name="my-system", total_processes=2, execution_order=["a", "b"])
    await observer.handle(event)

    mock_client.trace.assert_called_once_with(
        name="my-system",
        session_id="test-session",
        metadata={
            "env": "test",
            "total_processes": 2,
            "execution_order": ["a", "b"],
        },
    )


async def test_system_completed_updates_trace_and_flushes(
    observer: LangfuseObserver, mock_client: MagicMock
) -> None:
    await observer.handle(SystemStarted(name="sys", total_processes=1, execution_order=["p"]))
    trace = mock_client.trace.return_value

    await observer.handle(
        SystemCompleted(
            name="sys",
            duration_ms=1500.0,
            process_results={"p": {"ok": True}},
            status="completed",
        )
    )

    trace.update.assert_called_once_with(
        output={"p": {"ok": True}},
        metadata={"duration_ms": 1500.0, "status": "completed"},
    )
    mock_client.flush.assert_called_once()


# ------------------------------------------------------------------
# Process-level flow
# ------------------------------------------------------------------


async def test_process_started_creates_span_under_trace(
    observer: LangfuseObserver, mock_client: MagicMock
) -> None:
    await observer.handle(SystemStarted(name="s", total_processes=1, execution_order=["my_proc"]))
    trace = mock_client.trace.return_value

    await observer.handle(ProcessStarted(system_name="s", process_name="my_proc", index=0))

    trace.span.assert_called_once_with(
        name="my_proc",
        metadata={"system_name": "s", "index": 0},
    )


async def test_process_completed_ends_span(
    observer: LangfuseObserver, mock_client: MagicMock
) -> None:
    await observer.handle(SystemStarted(name="s", total_processes=1, execution_order=["p"]))
    await observer.handle(ProcessStarted(system_name="s", process_name="p", index=0))
    process_span = mock_client.trace.return_value.span.return_value

    await observer.handle(
        ProcessCompleted(
            system_name="s",
            process_name="p",
            index=0,
            duration_ms=500.0,
            status="completed",
        )
    )

    process_span.end.assert_called_once()
    end_kwargs = process_span.end.call_args.kwargs
    assert end_kwargs["metadata"]["duration_ms"] == 500.0
    assert end_kwargs["level"] == "DEFAULT"


async def test_process_failed_sets_error_level(
    observer: LangfuseObserver, mock_client: MagicMock
) -> None:
    await observer.handle(SystemStarted(name="s", total_processes=1, execution_order=["p"]))
    await observer.handle(ProcessStarted(system_name="s", process_name="p", index=0))
    process_span = mock_client.trace.return_value.span.return_value

    await observer.handle(
        ProcessCompleted(
            system_name="s",
            process_name="p",
            index=0,
            duration_ms=100.0,
            status="failed",
            error="kaboom",
        )
    )

    end_kwargs = process_span.end.call_args.kwargs
    assert end_kwargs["level"] == "ERROR"
    assert end_kwargs["metadata"]["error"] == "kaboom"


# ------------------------------------------------------------------
# Pipeline nested under Process
# ------------------------------------------------------------------


async def test_pipeline_nests_under_process_span(
    observer: LangfuseObserver, mock_client: MagicMock
) -> None:
    await observer.handle(SystemStarted(name="s", total_processes=1, execution_order=["p"]))
    await observer.handle(ProcessStarted(system_name="s", process_name="p", index=0))
    process_span = mock_client.trace.return_value.span.return_value

    await observer.handle(PipelineStarted(name="p", total_waves=2, total_nodes=5))

    process_span.span.assert_called_once_with(
        name="p",
        metadata={"total_nodes": 5, "total_waves": 2},
    )


# ------------------------------------------------------------------
# Standalone pipeline (no System/Process)
# ------------------------------------------------------------------


async def test_standalone_pipeline_creates_trace(
    observer: LangfuseObserver, mock_client: MagicMock
) -> None:
    await observer.handle(PipelineStarted(name="my-pipe", total_waves=2, total_nodes=5))

    # Should create a trace (no prior SystemStarted)
    mock_client.trace.assert_called_once()
    # And also create a pipeline span under that trace
    mock_client.trace.return_value.span.assert_called_once()


# ------------------------------------------------------------------
# Full hierarchy: System → Process → Pipeline → Node
# ------------------------------------------------------------------


async def test_node_started_creates_span_under_pipeline(
    observer: LangfuseObserver, mock_client: MagicMock
) -> None:
    await _setup_full_hierarchy(observer)
    # trace.span() -> process_span, process_span.span() -> pipeline_span
    pipeline_span = mock_client.trace.return_value.span.return_value.span.return_value

    await observer.handle(NodeStarted(name="analyzer", wave_index=0, dependencies=["input"]))

    pipeline_span.span.assert_called_once_with(
        name="analyzer",
        metadata={"wave_index": 0, "dependencies": ["input"]},
    )


async def test_node_completed_ends_span(observer: LangfuseObserver, mock_client: MagicMock) -> None:
    await _setup_full_hierarchy(observer)
    pipeline_span = mock_client.trace.return_value.span.return_value.span.return_value
    node_span = pipeline_span.span.return_value

    await observer.handle(NodeStarted(name="n", wave_index=0))
    await observer.handle(NodeCompleted(name="n", wave_index=0, result={"x": 1}, duration_ms=42.0))

    node_span.end.assert_called_once()
    assert node_span.end.call_args.kwargs["metadata"]["duration_ms"] == 42.0


async def test_node_failed_ends_span_with_error(
    observer: LangfuseObserver, mock_client: MagicMock
) -> None:
    await _setup_full_hierarchy(observer)
    pipeline_span = mock_client.trace.return_value.span.return_value.span.return_value
    node_span = pipeline_span.span.return_value

    await observer.handle(NodeStarted(name="n", wave_index=0))
    await observer.handle(NodeFailed(name="n", wave_index=0, error=ValueError("boom")))

    node_span.end.assert_called_once_with(level="ERROR", status_message="boom")


# ------------------------------------------------------------------
# LLM generation
# ------------------------------------------------------------------


async def test_llm_port_call_creates_generation(
    observer: LangfuseObserver, mock_client: MagicMock
) -> None:
    await _setup_full_hierarchy(observer)
    pipeline_span = mock_client.trace.return_value.span.return_value.span.return_value
    node_span = pipeline_span.span.return_value

    await observer.handle(NodeStarted(name="llm_node", wave_index=0))
    await observer.handle(
        LLMPortCall(
            port_type="llm",
            method="aresponse",
            node_name="llm_node",
            duration_ms=200.0,
            usage={"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
            model="gpt-4",
            messages=[{"role": "user", "content": "hello"}],
            response="world",
        )
    )

    node_span.generation.assert_called_once_with(
        name="aresponse",
        model="gpt-4",
        input=[{"role": "user", "content": "hello"}],
        output="world",
        usage={"input": 10, "output": 20, "total": 30},
        metadata={"duration_ms": 200.0, "tool_calls": None},
    )


# ------------------------------------------------------------------
# Tool call
# ------------------------------------------------------------------


async def test_tool_call_creates_and_ends_span(
    observer: LangfuseObserver, mock_client: MagicMock
) -> None:
    await _setup_full_hierarchy(observer)
    pipeline_span = mock_client.trace.return_value.span.return_value.span.return_value
    node_span = pipeline_span.span.return_value
    tool_span = node_span.span.return_value

    await observer.handle(NodeStarted(name="agent", wave_index=0))
    await observer.handle(
        ToolRouterPortCall(
            port_type="tool_router",
            method="acall_tool",
            node_name="agent",
            duration_ms=50.0,
            tool_name="search",
            params={"q": "test"},
            result={"items": [1, 2]},
        )
    )

    node_span.span.assert_called_once_with(
        name="tool:search",
        input={"q": "test"},
        output="{'items': [1, 2]}",
        metadata={"duration_ms": 50.0},
    )
    tool_span.end.assert_called_once()


# ------------------------------------------------------------------
# Pipeline completion
# ------------------------------------------------------------------


async def test_standalone_pipeline_completed_flushes(
    observer: LangfuseObserver, mock_client: MagicMock
) -> None:
    """Standalone pipeline (no process) should flush on completion."""
    await observer.handle(PipelineStarted(name="p", total_waves=1, total_nodes=1))
    pipeline_span = mock_client.trace.return_value.span.return_value

    await observer.handle(PipelineCompleted(name="p", duration_ms=300.0, status="completed"))

    pipeline_span.end.assert_called_once()
    mock_client.flush.assert_called_once()


async def test_pipeline_in_system_does_not_flush_early(
    observer: LangfuseObserver, mock_client: MagicMock
) -> None:
    """Pipeline inside a process should NOT flush — wait for SystemCompleted."""
    await _setup_full_hierarchy(observer)

    await observer.handle(PipelineCompleted(name="p", duration_ms=300.0, status="completed"))

    mock_client.flush.assert_not_called()


async def test_pipeline_failed_sets_error_level(
    observer: LangfuseObserver, mock_client: MagicMock
) -> None:
    await observer.handle(PipelineStarted(name="p", total_waves=1, total_nodes=1))
    pipeline_span = mock_client.trace.return_value.span.return_value

    await observer.handle(PipelineCompleted(name="p", duration_ms=100.0, status="failed"))

    end_kwargs = pipeline_span.end.call_args.kwargs
    assert end_kwargs["level"] == "ERROR"


# ------------------------------------------------------------------
# Flush control
# ------------------------------------------------------------------


async def test_no_flush_when_disabled(mock_client: MagicMock) -> None:
    obs = LangfuseObserver(mock_client, flush_on_complete=False)
    await obs.handle(PipelineStarted(name="p", total_waves=1, total_nodes=1))
    await obs.handle(PipelineCompleted(name="p", duration_ms=10.0))

    mock_client.flush.assert_not_called()
