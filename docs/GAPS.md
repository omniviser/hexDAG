# hexDAG Gap Tracking

**Last updated:** 2026-03-25 | **Version:** 0.7.0.dev9

This document tracks known gaps between the roadmap, documentation, and implementation.

---

## 1. Architecture Debt (Critical Blockers)

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

## 2. Unimplemented Protocols

| Protocol | Implementations | Notes |
|---|---|---|
| `SupportsTTL` | 1 (Redis only) | Needs SQLite, in-memory, and other adapters |
| `SupportsTransactions` | 0 | Defined and exported but no adapter implements it |

**Location:** `kernel/ports/data_store.py`

---

## 3. Missing Adapter Parity

### LLM Adapters

| Adapter | Gap |
|---|---|
| `AnthropicAdapter` | Missing `SupportsVision`, `SupportsEmbedding` protocols |
| `OpenAIAdapter` | Raises `NotImplementedError` for image embeddings (lines 787, 805) |
| Ollama / local LLM | No adapter (stdlib or plugin) |

### Secret Management Adapters

| Adapter | Status |
|---|---|
| AWS Secrets Manager | No adapter |
| GCP Secret Manager | No adapter |
| HashiCorp Vault | No adapter |

**Note:** Azure adapters exist in `hexdag_plugins/azure/` (KeyVault, Cosmos, Blob, OpenAI).

### Database Adapters

| Adapter | Gap |
|---|---|
| `CSVAdapter` | Raises `NotImplementedError` for SQL queries (line 134); limited to CRUD |

---

## 4. YAML Pipeline & Compiler Gaps

### Documentation vs. Implementation Mismatches

- **CLAUDE.md claims** `loop_node` and `conditional_node` aliases exist. In reality, these are handled by `composite_node` with `mode: while | for-each | times | if-else | switch`. The aliases don't resolve.
- **`kind: Adapter`** — Compiler plugin exists (`compiler/plugins/adapter_definition.py`) but standalone YAML adapter definitions are not wired into `spec.ports.<name>.adapter:`.
- **`kind: Middleware`** — Plugin exists (`compiler/plugins/middleware_definition.py`) but middleware stack is hardcoded in `prepare_ports()` rather than fully data-driven.
- **`kind: Policy`** — Not implemented (no source or test directories exist).
- **`kind: Manifest`** — Not implemented. Planned for adapter-declared caps, mounts, and profiles.

### Incomplete Features

- **Node-level port support:** `orchestrator_factory.py` line 446: `TODO: Add node-level port support when available in YAML builder`
- **DataNode** is deprecated in favor of `ExpressionNode` but this isn't documented in CLAUDE.md.
- **PortCallNode** is deprecated in favor of `ServiceCallNode`/`ToolCallNode` but not documented.

---

## 5. VAS (Virtual Agent System) Implementation Gaps

VAS is ~30% implemented. Here's the breakdown by roadmap phase:

### Phase A: Core VAS Tree (~20% complete)

**Implemented:**
- VAS port protocol and LocalVFS driver (`kernel/ports/vfs.py`, `drivers/vfs/local.py`)
- 4 providers: `LibProvider`, `ProcRunsProvider`, `ProcScheduledProvider`, `ProcEntitiesProvider`
- VASTools service and API layer (`stdlib/lib/vfs_tools.py`, `api/vfs.py`)
- 3 MCP tools: `vas_read`, `vas_list`, `vas_stat`

**Missing:**
- `EtcPipelinesProvider` (`/etc/pipelines/`) — pipeline discovery
- `PipelineRegistry` — scans directories for pipeline YAML files
- `DevPortsProvider` (`/dev/ports/`) — bound adapter introspection
- `SysProvider` (`/sys/`) — version, config, cgroups, caps
- `ProcSelfProvider` (`/proc/self/`) — agent self-awareness
- `MemProvider` (`/mem/`) — agent memory
- Wire `/proc/*` providers into MCP server

### Phase B: Security & Caps (0% complete)

- `CapSet` domain model and security enforcement
- `kind: Config` security extensions (caps.profiles, mcp.permissions)
- `kind: Manifest` loader for protocol extraction
- VAS permission checking in LocalVAS
- Child process cap inheritance

### Phase C3: Resource Accounting (0% complete)

- `ResourceAccounting` service
- `ResourceUsage` and `ResourceLimits` models incomplete
- `/sys/cgroup/` sub-provider for resource limits/usage
- Cost budget enforcement middleware

### Phase D-E: Advanced Operations (0% complete)

- `awrite()`, `aexec()`, `awatch()`, `apipe()` VAS methods
- Injection into running pipelines
- Event-driven piping between agents

---

## 6. MCP Server Gaps

### Process Management Not Wired

9 functions in `api/processes.py` (lines 37-277) are defined but not exposed as MCP tools:

| Function | Blocked By |
|---|---|
| `list_pipeline_runs()` | ProcessRegistry instance not injected |
| `get_pipeline_run()` | ProcessRegistry instance not injected |
| `spawn_pipeline()` | PipelineSpawner instance not injected |
| `schedule_pipeline()` | Scheduler instance not injected |
| `cancel_scheduled()` | Scheduler instance not injected |
| `list_scheduled()` | Scheduler instance not injected |
| `get_entity_state()` | EntityState instance not injected |
| `transition_entity()` | EntityState instance not injected |
| `get_entity_history()` | EntityState instance not injected |

### `/proc/*` VFS Providers

Providers are defined but not wired — `mcp_server.py` line 78 comment: "note: not yet wired into `mcp_server.py`".

**MCP currently exposes:** VFS, docs, execution, validation, and pipeline tools only.

---

## 7. SystemRunner & Production Runtime Gaps

### SystemRunner Modes (Milestone 4/5)

| Mode | Status |
|---|---|
| `manual` | Implemented |
| `schedule` | Not implemented |
| `continuous` | Not implemented (restart policies missing) |
| `event` | Not implemented (requires EventBus) |

### Missing Infrastructure

- **EventBus** port & adapters — not implemented
- **GovernancePort** for audit/compliance — not implemented
- **Studio System Manager** — daemon host with process supervision not started
  - Missing: REST API `/api/system/status`, `/api/system/start`, `/api/system/stop`

---

## 8. Middleware Gaps

| Middleware | Status |
|---|---|
| `ObservableLLM` | Implemented |
| `ObservableToolRouter` | Implemented |
| `CircuitBreaker` | Not implemented (ROADMAP Phase C2) |
| `CostTracker` | Not implemented (ROADMAP Phase C2) |
| Observable DataStore | Not implemented (Phase C1.5: "Future: DataStorePortCall, FileStoragePortCall") |
| Observable FileStorage | Not implemented |

---

## 9. Test Infrastructure Gaps

### Empty Test Directories

| Path | Status |
|---|---|
| `tests/hexdag/kernel/orchestration/policies/` | Contains only `__init__.py` (0 tests) |
| `tests/hexdag/kernel/application/routes/` | Contains only `__init__.py` (0 tests) |
| `tests/benchmarks/` | Contains only `__init__.py` — empty benchmark suite |

### Orphaned Code

| Path | Issue |
|---|---|
| `tests/hexai/` | Orphaned namespace from project rename; 3 test files remain |
| `tests/hexai/adapters/database/csv/test_csv_adapter.py` | Tests features that raise `NotImplementedError` in production |
| `tests/hexai/adapters/database/sqlalchemy/` | SQLAlchemy adapter test in wrong namespace |

---

## 10. Feature Examples Needed

The following features lack runnable examples:

- `SystemRunner` / `kind: System` — multi-pipeline orchestration
- `Service` + `@tool` + `@step` pattern — no user-facing examples
- `CheckpointNode` save/restore — tests exist but no narrative example
- `CompositeNode` (while, for-each, times, if-else, switch) — tests exist but no narrative example
- `PipelineSpawner` fork/exec — not used in examples
- `VFSTools` virtual filesystem — limited examples

---

## 11. Documentation Gaps

### Stale Auto-Generated Docs

- `docs/nodes.md` — initial 5-node snapshot (stale; actual node count is 12+)
- `docs/adapters.md` — initial 2-adapter snapshot (stale)
- Auto-generated docs go to `docs/generated/mcp/` via `scripts/generate_mcp_docs.py`
- **No script** currently regenerates root-level `docs/nodes.md` and `docs/adapters.md`

### Missing Documentation

- No docs for `CheckpointNode` save/restore patterns
- No docs for `CompositeNode` patterns (if/else, while, for-each, switch)
- No comprehensive docs for `SystemRunner` modes
- No docs for `kind: System` YAML manifests
- No docs for VAS beyond roadmap notes

### CLAUDE.md Inaccuracies

| Claim | Reality | Fix |
|---|---|---|
| `loop_node` alias available | Use `composite_node` with `mode: while` | Update alias list |
| `conditional_node` alias available | Use `composite_node` with `mode: if-else` | Update alias list |
| Node types: "agent, llm, function, conditional, loop" | Actual: agent, llm, function, composite, checkpoint, expression, mapped_input, service_call, tool_call, api_call | Update node types list |
| No mention of DataNode deprecation | DataNode deprecated → ExpressionNode | Add deprecation note |
| No mention of PortCallNode deprecation | Replaced by ServiceCallNode/ToolCallNode | Add deprecation note |

---

## 12. Minor Code Issues

| File | Line | Issue |
|---|---|---|
| `hexdag/cli/commands/create_cmd.py` | 73 | `"description": "TODO: Add description"` |
| `hexdag/cli/commands/plugin_dev_cmd.py` | 117 | `# TODO: Implement the {port} port interface methods` |
| `hexdag/cli/commands/plugin_dev_cmd.py` | 232 | `# TODO: Add more tests for your adapter functionality` |
| `hexdag/kernel/orchestration/orchestrator_factory.py` | 446 | `# TODO: Add node-level port support when available in YAML builder` |

---

## 13. Notebook Coverage

### Existing (8 notebooks)

- `01_introduction.ipynb`
- `02_yaml_pipelines.ipynb`
- `03_practical_workflow.ipynb`
- `03_yaml_includes_and_composition.ipynb`
- `06_dynamic_reasoning_agent.ipynb`
- `advanced_fewshot_and_retry.ipynb`
- `02_real_world_use_cases/code_review_security_audit.ipynb`
- `02_real_world_use_cases/investment_research_assistant.ipynb`

### Missing per CLAUDE.md Structure

The recommended `01_getting_started/`, `02_real_world_use_cases/`, `03_advanced_patterns/` structure is partially populated. Missing topics include:
- Services and `@tool`/`@step` patterns
- CompositeNode control flow
- CheckpointNode usage
- SystemRunner multi-pipeline orchestration
- VAS agent introspection
- Production deployment patterns

---

## Priority Summary

### Critical (Blocks other work)

1. **Port Probes** implementation — blocks resource limits, caps, observable events
2. **`SupportsTransactions`** — zero implementations for a defined protocol
3. **MCP server wiring** — 9 process management tools + `/proc/*` providers defined but not exposed

### High (Affects users)

4. **CLAUDE.md inaccuracies** — `loop_node`/`conditional_node` aliases don't exist; node type list outdated
5. **Missing adapters** — Ollama/local LLM, secret managers (AWS/GCP/Vault)
6. **Missing examples** — SystemRunner, Services, CheckpointNode, CompositeNode
7. **Stale docs** — `nodes.md` and `adapters.md` significantly out of date

### Medium (Framework completeness)

8. **VAS Phases A-E** — only ~20-30% implemented
9. **SystemRunner modes** — only `manual` mode works
10. **Middleware gaps** — CircuitBreaker, CostTracker, observable DataStore/FileStorage
11. **Orphaned test namespace** — `tests/hexai/` from project rename

### Low (Cosmetic / minor)

12. Empty test directories (benchmarks, policies, routes)
13. Minor TODOs in CLI scaffolding templates
14. Deprecated import warnings in `__init__.py`
