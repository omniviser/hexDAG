# 🌐 Simple Event Taxonomy

A minimal, human-friendly event type format for all emitted events in `<namespace>:<action>` form.
Standardized envelope ensures consistency, routing and future batching.

---

## 🏷️ Canonical Namespaces & Actions

- **pipeline**: started, completed, failed
- **dag**: started, completed, failed
- **wave**: started, completed
- **node**: started, completed, failed, skipped
- **policy**: decision
- **observer**: timeout, error
- **registry**: resolved, missing
*(reserved: port, tool, memory)*

---

## 📦 Event Envelope

**Required fields**
- `event_type` → e.g. `"node:failed"`
- `event_id` → ULID / UUIDv7 string
- `timestamp` → RFC3339 UTC (ms precision, with `Z`)
- `pipeline` → pipeline name
- `pipeline_run_id` → stable across the run
- `severity` → info | warn | error
- `attrs` → JSON-serializable object

**Optional fields**
- `node`, `wave`, `tenant`, `project`, `environment`, `correlation_id`

---

## 📝 Examples

### ✅ pipeline:started
```json
{
  "event_type": "pipeline:started",
  "event_id": "01J8YF0CCHB3S2S0YC9D5K1VZY",
  "timestamp": "2025-09-09T10:15:23.412Z",
  "pipeline": "doc-index",
  "pipeline_run_id": "doc-index#2025-09-09T10:15:20.001Z",
  "severity": "info",
  "attrs": { "total_waves": 3, "total_nodes": 14 }
}
```

### ❌ node:failed
```json
{
  "event_type": "node:failed",
  "event_id": "01J8YF0H1E3V06TP0G3K1FN9VQ",
  "timestamp": "2025-09-09T10:15:27.099Z",
  "pipeline": "doc-index",
  "pipeline_run_id": "doc-index#2025-09-09T10:15:20.001Z",
  "node": "extract_customers",
  "wave": 2,
  "severity": "error",
  "attrs": { "error_type": "TimeoutError", "retryable": true }
}
```

---

## 🔄 Mapping Table

- PipelineStartedEvent → pipeline:started
- PipelineCompletedEvent → pipeline:completed
- PipelineFailedEvent → pipeline:failed
- DagStartedEvent → dag:started
- DagCompletedEvent → dag:completed
- DagFailedEvent → dag:failed
- WaveStartedEvent → wave:started
- WaveCompletedEvent → wave:completed
- NodeStartedEvent → node:started
- NodeCompletedEvent → node:completed
- NodeFailedEvent → node:failed
- NodeSkippedEvent → node:skipped
- PolicyDecisionEvent / ControlDecisionEvent → policy:decision
- ObserverTimeoutEvent → observer:timeout
- ObserverErrorEvent → observer:error
- RegistryResolvedEvent → registry:resolved
- RegistryMissingEvent → registry:missing

---

## ✅ Validation Rules

- `event_type` must match `^[a-z]+:[a-z]+$`
- must be in approved namespace/action sets
- `event_id` required and unique
- `timestamp` must be RFC3339 with `Z`
- `pipeline_run_id` required for pipeline|dag|wave|node|policy
- `attrs` must be JSON-serializable

---

## 🎯 Acceptance Criteria

- All emitted events follow `<namespace>:<action>`
- Envelope includes required fields, optional when available
- Mapping table covers all Tier-1 events, no runtime guessing
- JSON-serializable payloads
- Unit tests: type format, mapping, severity, required fields, run ID stability

---

✨ This taxonomy ensures consistent, routable events across hexDAG.
