import pytest

from hexai.core.application.events.taxonomy import validate_event_type


@pytest.mark.parametrize(
    "et",
    [
        "pipeline:started",
        "node:skipped",
        "observer:error",
        "llm:prompt",
        "tool:completed",
    ],
)
def test_event_type_valid(et):
    validate_event_type(et)


@pytest.mark.parametrize(
    "et",
    [
        "PIPELINE:STARTED",
        "pipeline-started",
        "unknown:started",
        "pipeline:unknown",
    ],
)
def test_event_type_invalid(et):
    with pytest.raises(ValueError):
        validate_event_type(et)
