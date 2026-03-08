# hexDAG Gap Tracking

**Last updated:** 2026-03-07 | **Version:** 0.7.0.dev9

This document tracks known gaps between the roadmap, documentation, and implementation.

---

## Architecture Debt

### Port Wrappers → Port Probes (Milestone 2 blocker)

`hexdag/kernel/orchestration/port_wrappers.py` wraps adapters for observability but breaks `isinstance()` protocol checks. The planned replacement (`port_probes.py` with `instrument_ports()`) does not exist yet.

**Blocks:** Resource Limits, Caps, DataStore events, error handling unification.

### Deprecated Ports Still Active

| Deprecated | Replacement | Status |
|---|---|---|
| `Memory` | `SupportsKeyValue` | `AzureCosmosAdapter` still implements `Memory` |
| `Database` | `SupportsQuery` | Migration incomplete |
| `HexDAGLib` | `Service` | `lib_base.py` kept as deprecated shim |

---

## Unimplemented Protocols

| Protocol | Implementations |
|---|---|
| `SupportsTTL` | 0 (defined in `kernel/ports/data_store.py`) |
| `SupportsTransactions` | 0 (defined in `kernel/ports/data_store.py`) |

---

## Unstarted Milestones

### Milestone 2: Observability & Resource Management
- Port Probes (`instrument_ports()`)
- Resource Limits & Caps
- `kind: Adapter` manifest
- `kind: Policy` manifest

### Milestone 4: Entity-Bound Pipelines
- `StateTransitionObserver` in orchestrator
- `spec.entities` in pipeline manifests
- Entity-triggered pipeline spawning

### Milestone 5: Production Runtime
- `SystemRunner` modes (standalone, daemon, cluster)
- `EventBus` port & adapters
- `GovernancePort` for audit/compliance

---

## Missing Adapter Parity

| Adapter | Gap |
|---|---|
| `AnthropicAdapter` | Missing `SupportsVision`, `SupportsEmbedding` |
| Ollama / local LLM | No adapter (stdlib or plugin) |
| AWS Secrets Manager | No adapter |
| GCP Secret Manager | No adapter |
| HashiCorp Vault | No adapter |

**Note:** Azure adapters exist in `hexdag_plugins/azure/` (KeyVault, Cosmos, Blob, OpenAI).

---

## Feature Examples Needed

The following features lack runnable examples:

- `SystemRunner` / `kind: System`
- `Service` + `@tool` + `@step` pattern
- `CheckpointNode` save/restore
- `CompositeNode` (while, for-each, times, if-else, switch)
- `PipelineSpawner` fork/exec
- `VFSTools` virtual filesystem

---

## MCP Server Gaps

- `/proc/*` VFS providers are defined but not wired (missing lib instances in `mcp_server.py`)
- 9 `api/processes.py` tools (process management) are not exposed as MCP tools
- `mcp_server.py` currently exposes: VFS, docs, execution, validation, and pipeline tools

---

## Documentation Auto-Generation

- `docs/nodes.md` and `docs/adapters.md` are stale initial-release snapshots (5 nodes, 2 adapters)
- Auto-generated docs go to `docs/generated/mcp/` via `scripts/generate_mcp_docs.py`
- No script currently regenerates the root-level `docs/nodes.md` and `docs/adapters.md`
- **Action needed:** Create a script that uses `DocExtractor`/`GuideGenerator` to regenerate these files

---

## Orphaned / Stale Code

| Path | Issue |
|---|---|
| `tests/hexai/` | Orphaned namespace from project rename |
| `tests/benchmarks/` | Empty directory |
| `tests/hexdag/kernel/orchestration/policies/` | Stub for non-existent source module |
| `tests/hexdag/kernel/application/routes/` | Stub for non-existent source module |
