from dataclasses import dataclass
from typing import Optional

import pytest

from hexai.simple_events.envelope import SimpleContext, to_simple_event
from hexai.simple_events.validators import validate_envelope


@dataclass
class PipelineStartedEvent:
    pipeline: str
    total_waves: int = 0
    total_nodes: int = 0
    event_id: Optional[str] = "test-pipeline-id"


@dataclass
class NodeFailedEvent:
    pipeline: str
    node_name: str
    wave_index: int
    error_type: str = "TimeoutError"
    retryable: bool = True
    event_id: Optional[str] = "test-node-id"


def test_pipeline_started_event_builds_valid_envelope():
    ctx = SimpleContext(pipeline="doc-index", pipeline_run_id="run-123")
    event = PipelineStartedEvent("doc-index", 3, 14)
    env = to_simple_event(event, ctx)
    validate_envelope(env)
    assert env["event_type"] == "pipeline:started"
    assert env["severity"] == "info"
    assert env["attrs"]["total_waves"] == 3
    assert env["attrs"]["total_nodes"] == 14


def test_node_failed_event_sets_severity_and_context_fields():
    ctx = SimpleContext(pipeline="doc-index", pipeline_run_id="run-123")
    event = NodeFailedEvent("doc-index", "extract_customers", 2)
    env = to_simple_event(event, ctx)
    validate_envelope(env)
    assert env["event_type"] == "node:failed"
    assert env["severity"] == "error"
    assert env["node"] == "extract_customers"
    assert env["wave"] == 2
    assert env["attrs"]["error_type"] == "TimeoutError"
    assert env["attrs"]["retryable"] is True


def test_pipeline_run_id_is_stable_across_events_in_one_run():
    ctx = SimpleContext(pipeline="doc-index", pipeline_run_id="doc-index#run-001")

    e1 = PipelineStartedEvent("doc-index", 1, 2, event_id="id-1")
    env1 = to_simple_event(e1, ctx)

    e2 = NodeFailedEvent("doc-index", "n1", 0, event_id="id-2")
    env2 = to_simple_event(e2, ctx)

    assert env1["pipeline_run_id"] == env2["pipeline_run_id"] == "doc-index#run-001"


def test_validate_envelope_requires_pipeline_run_id():
    ctx = SimpleContext(pipeline="doc-index", pipeline_run_id="run-123")
    e = PipelineStartedEvent("doc-index", 3, 14, event_id="id-3")
    env = to_simple_event(e, ctx)
    env.pop("pipeline_run_id")

    import pytest

    from hexai.simple_events.validators import validate_envelope

    with pytest.raises(ValueError):
        validate_envelope(env)


@pytest.mark.parametrize(
    "event_cls,expected",
    [
        (PipelineStartedEvent("doc-index", 1, 1, event_id="s1"), "info"),
        (NodeFailedEvent("doc-index", "n", 0, event_id="s2"), "error"),
    ],
)
def test_severity_assignment(event_cls, expected):
    ctx = SimpleContext(pipeline="doc-index", pipeline_run_id="run-999")
    env = to_simple_event(event_cls, ctx)
    assert env["severity"] == expected
