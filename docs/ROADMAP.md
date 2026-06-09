# Development Roadmap: hexDAG Framework

> **Strategic development plan for the hexDAG workflow engine**

For the current architecture, see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Architecture: Core + Brain

```
hexdag (core)           pip install hexdag
├── kernel/             — engine: orchestrator, domain models, ports, System, LifecycleRunner
├── compiler/           — YAML compilation: kind: Pipeline, System, Config, Adapter, Middleware, Macro
├── stdlib/             — built-in nodes, adapters, middleware
├── drivers/            — executors, observer manager, pipeline spawner
├── api/                — build-time functions: components, validation, pipeline editing, docs, export
├── mcp_server.py       — build-time MCP: component discovery, validation, YAML editing, explain, docs
├── docs/               — documentation extraction + guide generation
└── cli/                — hexdag init, lint, validate, run, explain

hexdag-brain            pip install hexdag-brain
├── runtime api/        — execution, systems, processes, logs
├── mcp_server.py       — runtime MCP: run pipeline, transition entity, process management
└── sdk                 — HexDAGBrain class for programmatic agent control
```

**Dependency:** `hexdag-brain → hexdag`

### What Each Layer Owns

| Concern | hexdag (core) | hexdag-brain |
|---|---|---|
| Compile YAML to DAG | x | |
| Execute pipelines (PipelineRunner) | x | |
| Port contracts + middleware | x | |
| System, LifecycleRunner | x | |
| Component discovery (MCP) | x | |
| Build-time MCP (validate, edit, explain, docs) | x | |
| Pipeline YAML manipulation | x | |
| Documentation generation | x | |
| Runtime MCP (run, transition, process mgmt) | | x |
| Run process / transition entity (api) | | x |
| Process management (spawn, schedule, list) | | x |
| Log querying | | x |
| HexDAGBrain SDK class | | x |

---

## Philosophy

- **Library-first** -- `pip install hexdag`, use `PipelineRunner`/`SystemRunner` in your own app. No server required. Core includes a build-time MCP server for AI-assisted YAML authoring (component discovery, validation, pipeline editing, docs).
- **Brain as runtime agent SDK** -- `pip install hexdag-brain` adds runtime MCP tools (run pipelines, transition entities, manage processes) + `HexDAGBrain` SDK class. Thin wrapper over core for agents that need to operate a running system.
- **Good ports, not many adapters** -- hexDAG ships clean port contracts and mock adapters. Users write their own or use `hexdag-plugins`. The framework's value is in the port logic, not adapter count.
- **External observability** -- cost tracking, tracing, and monitoring are handled by integrations (Langfuse, OpenTelemetry) via the observer pattern, not by framework-internal middleware.

---

## Completed Work

### Core Engine
- [x] DAG orchestration -- topological sort, wave-based parallel execution
- [x] Node system -- FunctionNode, LLMNode, AgentNode, CompositeNode (while/for-each/times/if-else/switch), TransitionNode, ServiceCallNode, ApiCallNode, WaitNode
- [x] Validation framework -- multi-strategy (Pydantic, type checking, custom)
- [x] Event system -- NodeStarted/Completed/Failed, PipelineStarted/Completed, PortCallEvent, StateTransitionEvent, BodyStarted/Completed/Failed
- [x] Hexagonal architecture -- ports (contracts) / adapters (implementations) / drivers (infrastructure)
- [x] n8n-like data flow -- upstream node outputs auto-available, `input_mapping` optional, `MISSING` sentinel, build-time naming validation

### YAML Compiler
- [x] YAML -> DirectedGraph compilation with plugin system
- [x] Macros (`kind: Macro`) -- reusable node templates with subgraph expansion
- [x] Environment management -- multi-document YAML with `metadata.namespace`
- [x] Preprocessing -- env var substitution, `!include` tags, Jinja2 templates, `!py` expressions
- [x] `kind: Config`, `kind: Adapter`, `kind: Middleware` manifests

### Ports & Middleware
- [x] LLM, DataStore, PipelineSpawner, ToolRouter, Executor, ObserverManager, FileStorage, SecretStore
- [x] `SupportsStructuredOutput` protocol with native OpenAI + Anthropic implementations
- [x] Composable port middleware -- `prepare_ports()` auto-stacks based on adapter capabilities
- [x] Declarative middleware stacking -- `spec.ports.<name>.middleware:` in YAML
- [x] Concrete middleware -- RetryWithBackoff, RateLimiter, ResponseCache, Timeout, RoundRobin
- [x] Unified port events -- `PortCallEvent` base with `LLMPortCall`, `ToolRouterPortCall`
- [x] Port call observers -- `PortCallStoreObserver`, `PortCallLogObserver`

### Entity Lifecycle & Systems
- [x] `spec.state_machines` on Pipeline and System configs
- [x] TransitionNode, transition handlers, guards, cascade depth limits
- [x] LifecycleRunner -- event-driven runner for lifecycle-aware Systems
- [x] `kind: System` + SystemBuilder + SystemRunner + pipes
- [x] PipelineMemory, `ctx` pipeline context, graph-level routing

### Pipeline Resume & Extraction
- [x] `pre_seeded_results`, `PipelineRunner.resume()`, CheckpointManager
- [x] ExtractionJob service -- multi-round extraction state tracking
- [x] WaitNode + EventCorrelationRegistry -- async suspend/resume

### API & Integration
- [x] Build-time MCP server -- component discovery, pipeline building/validation, docs (stays in core)
- [x] Build-time api/ -- components, validation, pipeline editing, documentation, export (stays in core)
- [x] Runtime api/ -- execution, systems, processes, logs (moving to brain)
- [x] CLI -- `hexdag init`, `hexdag lint`, `hexdag validate`, `hexdag run`, `hexdag studio`
- [x] Plugin system -- `hexdag-plugins` namespace package

---

## Next: v1.0 Release (core)

### Remaining Middleware
- [x] **CircuitBreaker** -- failure threshold -> open/half-open/closed, prevents cascade failures

### Resource Accounting
- [x] **ResourceAccounting** (`kernel/domain/resource_accounting.py`) -- `ResourceUsage` model, `ResourceLimits`, `LimitCheck`. Events: `ResourceWarning`, `ResourceLimitExceeded`
- [x] **ResourceAccounting observer** (`stdlib/middleware/resource_accounting.py`) -- `ResourceAccountingObserver` enforces per-pipeline limits with warning/exceeded events

### Mental Model & Docs
- [x] **"How to Build" mental model** -- decision tree in README (Pipeline vs System, which node type, which port). Clear path from business problem to running YAML.
- [x] **Simplify GUIDE.md** -- remove architecture sections (already in ARCHITECTURE.md) and node reference (auto-generated). GUIDE becomes a practical builder's guide: YAML syntax, data flow, services, entity lifecycle. Add mental model at top.
- [x] **Drop VAS from core MCP** -- replaced `vas_read`/`vas_list`/`vas_stat` with direct typed tools (`list_nodes`, `list_adapters`, `list_tools`, `list_macros`, `list_tags`, `get_component_schema`). Removed execution tools (brain territory).

### Developer Experience
- [x] **`hexdag explain`** -- CLI command (like `kubectl explain`) for YAML field docs
- [x] **JSON Schema export** -- per-kind schemas in `schemas/` + `hexdag generate-types` CLI

### Cleanup
- [x] **Fix stale completed items** -- Scheduler, DatabaseTools, CheckpointNode references cleaned up
- [x] **Unified error handling** -- collapsed 3 duplicated `except` blocks into 1 in `node_executor.py`

### Release
- [x] **API freeze** -- backward compatibility commitment. See [PUBLIC_API.md](PUBLIC_API.md)
- [ ] **PyPI publication** -- `pip install hexdag` with semantic versioning

---

## Next: hexdag-brain v0.1

### Package Extraction

**Moves to brain** (runtime / execution concerns):

| Current location | What it does | Why brain |
|---|---|---|
| `hexdag/api/execution.py` | Run pipelines | Runtime execution, not authoring |
| `hexdag/api/systems.py` | Run system, run process, transition entity | Runtime system operations |
| `hexdag/api/processes.py` | Spawn, schedule, list runs, cancel | Runtime process management |
| `hexdag/api/logs.py` | Query execution logs | Runtime introspection |

**Stays in core** (engine + build-time authoring):

| Current location | Why core |
|---|---|
| `hexdag/kernel/` (all) | Engine: orchestrator, domain models, ports, System, LifecycleRunner, agent_tools.py |
| `hexdag/compiler/` | YAML compilation |
| `hexdag/stdlib/` (nodes, adapters, middleware, entity_state, process_registry, pipeline_memory) | Built-in components + kernel dependencies |
| `hexdag/drivers/` (executors, observer manager, spawner, vfs/local.py) | Infrastructure |
| `hexdag/api/components.py` | Component discovery — build-time |
| `hexdag/api/validation.py` | Pipeline validation — build-time |
| `hexdag/api/pipeline.py` | YAML manipulation — build-time |
| `hexdag/api/documentation.py` | Doc serving — build-time |
| `hexdag/api/export.py` | Project export — build-time |
| `hexdag/mcp_server.py` | Build-time MCP: components, validation, editing, explain, docs. VAS tools removed — direct typed tools instead. |
| `hexdag/docs/` (extractors, generators, models) | Documentation generation for MCP |
| `hexdag/cli/` | CLI: init, lint, validate, run, explain |

### HexDAGBrain SDK

Thin wrapper class for runtime agent control:

```python
from hexdag_brain import HexDAGBrain

brain = HexDAGBrain(system="./system.yaml")

# Runtime
result = await brain.run_pipeline("pipeline.yaml", {"topic": "AI trends"})
result = await brain.run_process("order_processing", {"order_id": "123"})
await brain.transition("order", "ORD-123", "SHIPPED")
runs = await brain.list_runs(status="completed", limit=10)
```

### Brain MCP Server

Runtime MCP tools for agents that need to operate a running system:

- Run pipeline, run process, transition entity
- List runs, system status, process management

Build-time MCP stays in core (component discovery, validation, YAML editing, explain, docs).

### v1.0 Cleanup: Drop VAS from Core MCP

Before brain extraction, replace the 3 VAS MCP tools (`vas_read`, `vas_list`, `vas_stat`) in `mcp_server.py` with direct typed tools from `api/components.py`. This removes the dependency on `api/vfs.py` and `api/execution.py` from core MCP, making brain extraction clean.

---

## After v1.0: v1.1 (core)

- [ ] **Graph optimizer passes** -- dead-node elimination, constant-folding
- [ ] **Pipeline memory expressions** -- `memory(key, default)` read + `memory_set(key, value)` in expressions
- [ ] **Saga compensation** -- pipeline-level `on_failure.compensate`
- [ ] **Lifecycle gating** -- `on_exit` pipelines per state + `await_pipeline: true`

---

## Parked

Items not planned for the foreseeable future. Existing code remains but receives no further investment.

| Item | Why parked |
|---|---|
| **VAS / VFS** (entire abstraction) | Filesystem abstraction for things that aren't files. Component discovery is `resolver.get_builtin_aliases()`. Process state is `process_registry.get_run()`. VAS adds indirection for no benefit. Existing code (`drivers/vfs/`, `kernel/ports/vfs.py`, `api/vfs.py`, `stdlib/lib/vfs_tools.py`, VFS providers) stays in the codebase but is not used by MCP and receives no investment. Core MCP uses direct typed tools instead. |
| **CapSet enforcement / YAML security** | `CapSet` domain model exists. Enforcement is a platform concern — host apps control access. |
| **Daemon host modes** (schedule, continuous, event) | Core is a library, not a process supervisor. Host apps own scheduling and supervision. |
| **EventBus port** | App-level concern. Users integrate their own message bus. |
| **GovernancePort** | Authorization/audit belongs in the host application. |
| **CostTracker middleware** | External observability (Langfuse, OpenTelemetry) via observer pattern. Not framework-internal. |
| **Application recipes** (Text-to-SQL, Research, AutoRAG) | Examples, not framework features. Ship as notebooks. |
| **Adapter ecosystem** | Good port contracts > many adapters. Community/plugins extend. |
| **Statecharts** | Flat state machines cover 90% of cases. |
| **Distributed lifecycle runner** | Single-process sufficient for library usage. |

---

## Success Metrics

- `pip install hexdag` — engine with zero opinions about how you host it
- `pip install hexdag-brain` — agent SDK with MCP server in < 5 min setup
- `hexdag explain` for instant YAML documentation
- IDE autocomplete via JSON Schema
- < 100ms overhead per node
- Clean port contracts implementable in < 50 lines
