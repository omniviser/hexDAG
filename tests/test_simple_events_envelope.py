import pytest

from hexai.core.application.events.events import (
    NodeFailed,
    PipelineStarted,
)
from hexai.simple_events.envelope import SimpleContext, to_simple_event
from hexai.simple_events.validators import validate_envelope


def test_pipeline_started_event_builds_valid_envelope():
    ctx = SimpleContext(pipeline="doc-index", pipeline_run_id="run-123")
    event = PipelineStarted(name="doc-index", total_waves=3, total_nodes=14)
    event.event_id = "test-pipeline-id"
    env = to_simple_event(event, ctx)
    validate_envelope(env)
    assert env["event_type"] == "pipeline:started"
    assert env["severity"] == "info"
    assert env["attrs"]["total_waves"] == 3
    assert env["attrs"]["total_nodes"] == 14


def test_node_failed_event_sets_severity_and_context_fields():
    ctx = SimpleContext(pipeline="doc-index", pipeline_run_id="run-123")
    event = NodeFailed(name="extract_customers", wave_index=2, error=RuntimeError("TimeoutError"))
    event.event_id = "test-node-id"
    env = to_simple_event(event, ctx)
    validate_envelope(env)
    assert env["event_type"] == "node:failed"
    assert env["severity"] == "error"
    assert env["node"] == "extract_customers"
    assert env["wave"] == 2
    assert env["attrs"]["error"] == "TimeoutError"


def test_pipeline_run_id_is_stable_across_events_in_one_run():
    ctx = SimpleContext(pipeline="doc-index", pipeline_run_id="doc-index#run-001")

    e1 = PipelineStarted(name="doc-index", total_waves=1, total_nodes=2)
    e1.event_id = "id-1"
    env1 = to_simple_event(e1, ctx)

    e2 = NodeFailed(name="n1", wave_index=0, error=RuntimeError("boom"))
    e2.event_id = "id-2"
    env2 = to_simple_event(e2, ctx)

    assert env1["pipeline_run_id"] == env2["pipeline_run_id"] == "doc-index#run-001"


def test_validate_envelope_requires_pipeline_run_id():
    ctx = SimpleContext(pipeline="doc-index", pipeline_run_id="run-123")
    e = PipelineStarted(name="doc-index", total_waves=3, total_nodes=14)
    e.event_id = "id-3"
    env = to_simple_event(e, ctx)
    env.pop("pipeline_run_id")


    from hexai.simple_events.validators import validate_envelope

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
    ctx = SimpleContext(pipeline="doc-index", pipeline_run_id="run-999")
    event_cls.event_id = "severity-test"
    env = to_simple_event(event_cls, ctx)
    assert env["severity"] == expected
