import pytest

from hexai.core.application.events.events import NodeFailed, PipelineStarted
from hexai.core.application.events.taxonomy import (
    EventContext,
    build_envelope,
    validate_envelope,
    validate_event_type,
)


def test_pipeline_started_envelope_contains_required_fields():
    ctx = EventContext(pipeline="doc-index", pipeline_run_id="run-123")
    event = PipelineStarted(name="doc-index", total_waves=3, total_nodes=14)
    event.event_id = "test-pipeline-id"

    envelope = build_envelope(event, ctx)

    validate_envelope(envelope)
    assert envelope["event_type"] == "pipeline:started"
    assert envelope["severity"] == "info"
    assert envelope["attrs"]["total_waves"] == 3
    assert envelope["attrs"]["total_nodes"] == 14


def test_node_failed_envelope_infers_context():
    ctx = EventContext(pipeline="doc-index", pipeline_run_id="run-123")
    event = NodeFailed(name="extract_customers", wave_index=2, error=RuntimeError("Timeout"))
    event.event_id = "test-node"

    envelope = build_envelope(event, ctx)

    validate_envelope(envelope)
    assert envelope["event_type"] == "node:failed"
    assert envelope["severity"] == "error"
    assert envelope["node"] == "extract_customers"
    assert envelope["wave"] == 2
    assert envelope["attrs"]["error"] == "Timeout"


def test_pipeline_run_id_is_stable_across_events():
    ctx = EventContext(pipeline="doc-index", pipeline_run_id="doc-index#run-001")

    started = PipelineStarted(name="doc-index", total_waves=1, total_nodes=2)
    started.event_id = "id-1"
    failed = NodeFailed(name="n1", wave_index=0, error=RuntimeError("boom"))
    failed.event_id = "id-2"

    env1 = build_envelope(started, ctx)
    env2 = build_envelope(failed, ctx)

    assert env1["pipeline_run_id"] == env2["pipeline_run_id"] == "doc-index#run-001"


def test_validate_envelope_requires_pipeline_run_id():
    ctx = EventContext(pipeline="doc-index", pipeline_run_id="run-123")
    event = PipelineStarted(name="doc-index", total_waves=3, total_nodes=14)
    event.event_id = "id-3"
    envelope = build_envelope(event, ctx)
    envelope.pop("pipeline_run_id")

    with pytest.raises(ValueError):
        validate_envelope(envelope)


@pytest.mark.parametrize(
    ("event", "expected"),
    [
        (PipelineStarted(name="doc-index", total_waves=1, total_nodes=1), "info"),
        (NodeFailed(name="n", wave_index=0, error=RuntimeError("boom")), "error"),
    ],
)
def test_severity_inference(event, expected):
    ctx = EventContext(pipeline="doc-index", pipeline_run_id="run-999")
    event.event_id = "severity-test"

    envelope = build_envelope(event, ctx)

    assert envelope["severity"] == expected


@pytest.mark.parametrize(
    "event_type",
    [
        "pipeline:started",
        "node:skipped",
        "observer:error",
        "llm:prompt",
        "tool:completed",
    ],
)
def test_validate_event_type_accepts_approved_formats(event_type):
    validate_event_type(event_type)


@pytest.mark.parametrize(
    "event_type",
    [
        "PIPELINE:STARTED",
        "pipeline-started",
        "unknown:started",
        "pipeline:unknown",
    ],
)
def test_validate_event_type_rejects_invalid_formats(event_type):
    with pytest.raises(ValueError):
        validate_event_type(event_type)
