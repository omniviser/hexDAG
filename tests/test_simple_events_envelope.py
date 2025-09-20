import pytest

from hexai.core.application.events.events import NodeFailed, PipelineStarted
from hexai.core.application.events.taxonomy import (
    EventContext,
    build_envelope,
    validate_envelope,
)


def test_pipeline_started_event_builds_valid_envelope():
    ctx = EventContext(pipeline="doc-index", pipeline_run_id="run-123")
    event = PipelineStarted(name="doc-index", total_waves=3, total_nodes=14)
    event.event_id = "test-pipeline-id"
    env = build_envelope(event, ctx)
    validate_envelope(env)
    assert env["event_type"] == "pipeline:started"
    assert env["severity"] == "info"
    assert env["attrs"]["total_waves"] == 3
    assert env["attrs"]["total_nodes"] == 14


def test_node_failed_event_sets_severity_and_context_fields():
    ctx = EventContext(pipeline="doc-index", pipeline_run_id="run-123")
    event = NodeFailed(name="extract_customers", wave_index=2, error=RuntimeError("TimeoutError"))
    event.event_id = "test-node-id"
    env = build_envelope(event, ctx)
    validate_envelope(env)
    assert env["event_type"] == "node:failed"
    assert env["severity"] == "error"
    assert env["node"] == "extract_customers"
    assert env["wave"] == 2
    assert env["attrs"]["error"] == "TimeoutError"


def test_pipeline_run_id_is_stable_across_events_in_one_run():
    ctx = EventContext(pipeline="doc-index", pipeline_run_id="doc-index#run-001")

    e1 = PipelineStarted(name="doc-index", total_waves=1, total_nodes=2)
    e1.event_id = "id-1"
    env1 = build_envelope(e1, ctx)

    e2 = NodeFailed(name="n1", wave_index=0, error=RuntimeError("boom"))
    e2.event_id = "id-2"
    env2 = build_envelope(e2, ctx)

    assert env1["pipeline_run_id"] == env2["pipeline_run_id"] == "doc-index#run-001"


def test_validate_envelope_requires_pipeline_run_id():
    ctx = EventContext(pipeline="doc-index", pipeline_run_id="run-123")
    e = PipelineStarted(name="doc-index", total_waves=3, total_nodes=14)
    e.event_id = "id-3"
    env = build_envelope(e, ctx)
    env.pop("pipeline_run_id")


    with pytest.raises(ValueError):
        validate_envelope(env)


@pytest.mark.parametrize(
    "event_cls,expected",
    [
        (
            PipelineStarted(name="doc-index", total_waves=1, total_nodes=1),
            "info",
        ),
        (NodeFailed(name="n", wave_index=0, error=RuntimeError("boom")), "error"),
    ],
)
def test_severity_assignment(event_cls, expected):
    ctx = EventContext(pipeline="doc-index", pipeline_run_id="run-999")
    event_cls.event_id = "severity-test"
    env = build_envelope(event_cls, ctx)
    assert env["severity"] == expected
