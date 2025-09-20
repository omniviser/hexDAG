# Event Taxonomy

The HexDAG event system emits envelopes that follow a canonical taxonomy. This
page documents the approved namespaces/actions, the envelope schema and the
mapping between internal event classes and emitted event types.

## Namespaces & Actions

| Namespace | Actions                          |
|-----------|----------------------------------|
| `pipeline` | `started`, `completed`, `failed` |
| `dag`      | `started`, `completed`, `failed` |
| `wave`     | `started`, `completed`           |
| `node`     | `started`, `completed`, `failed`, `skipped` |
| `policy`   | `decision`                       |
| `observer` | `timeout`, `error`               |
| `registry` | `resolved`, `missing`            |
| `tool`     | `called`, `completed`            |
| `llm`      | `prompt`, `response`             |
| `port`     | _reserved_                       |
| `memory`   | _reserved_                       |

## Envelope Shape

```jsonc
{
  "event_type": "node:failed",
  "event_id": "01J8YF0H1E3V06TP0G3K1FN9VQ",
  "timestamp": "2025-09-09T10:15:27.099Z",
  "pipeline": "doc-index",
  "pipeline_run_id": "doc-index#2025-09-09T10:15:20.001Z",
  "node": "extract_customers",
  "wave": 2,
  "severity": "error",
  "attrs": { "error": "TimeoutError" }
}
```

Optional context fields (`tenant`, `project`, `environment`, `correlation_id`)
are included when present in the `EventContext`.

## Event Mapping

| Event class            | Event type         | Envelope field overrides |
|------------------------|--------------------|---------------------------|
| `PipelineStarted`      | `pipeline:started` | `pipeline ← name`         |
| `PipelineCompleted`    | `pipeline:completed` | `pipeline ← name`       |
| `NodeStarted`          | `node:started`     | `node ← name`, `wave ← wave_index` |
| `NodeCompleted`        | `node:completed`   | `node ← name`, `wave ← wave_index` |
| `NodeFailed`           | `node:failed`      | `node ← name`, `wave ← wave_index` |
| `WaveStarted`          | `wave:started`     | `wave ← wave_index`       |
| `WaveCompleted`        | `wave:completed`   | `wave ← wave_index`       |
| `LLMPromptSent`        | `llm:prompt`       | `node ← node_name`        |
| `LLMResponseReceived`  | `llm:response`     | `node ← node_name`        |
| `ToolCalled`           | `tool:called`      | `node ← node_name`        |
| `ToolCompleted`        | `tool:completed`   | `node ← node_name`        |

`build_envelope()` in `hexai.core.application.events.taxonomy` uses this table
(`EVENT_REGISTRY`) to produce deterministic envelopes. Validation helpers in the
same module enforce type conformity, timestamp parsing and JSON-serializable
payloads.
