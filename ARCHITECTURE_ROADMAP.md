# hexDAG Architecture Roadmap: Kernel Extensions

**Status:** Active
**Last Updated:** 2026-02

hexDAG is an operating system for AI agents. This roadmap describes the planned
kernel-level extensions that complete the OS analogy -- the YAML compiler for
multi-pipeline orchestration, resource limits and capabilities for safety,
a virtual filesystem for uniform introspection (VFS), IPC (EventBus),
permissions (GovernancePort), and a CPU scheduler for multi-pipeline
coordination (CentralAgent).

Each follows the uniform entity pattern: kernel defines Protocol, stdlib ships default
implementation, users write their own.

For the current architecture, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
For the detailed YAML compiler plan, see [docs/YAML_COMPILER_PLAN.md](docs/YAML_COMPILER_PLAN.md).

---

## YAML Compiler

**Status:** In Progress (Step 0a complete)
**Location:** `hexdag/compiler/`

### Architecture: YAML as Source Language

YAML is to hexDAG what C source is to GCC -- the **source language** that gets
compiled into kernel objects. The compiler (`compiler/`) translates manifests into
domain models (`DirectedGraph`, `PipelineConfig`, `ResourceLimits`, etc.). The
**kernel never touches YAML** -- it only knows domain models and execution.

```
YAML Manifest  →  Compiler (hexdag/compiler/)  →  Domain Models  →  Kernel Execution
```

This matches Linux: the kernel defines structs, userspace tools parse config files,
and the kernel only accepts structured data via syscalls.

### Directory Structure

```
hexdag/
├── kernel/              ← Execution engine only (no YAML, no TOML)
│   ├── config/
│   │   ├── models.py    ← Pure domain: HexDAGConfig, LoggingConfig, ManifestEntry
│   │   └── __init__.py  ← Re-exports (backward compat)
│   ├── domain/          ← Pure domain models (ResourceLimits, CapSet, Pipe, Policy)
│   ├── orchestration/   ← Orchestrator, events, models (OrchestratorConfig)
│   └── ports/           ← Port interfaces
│
├── compiler/            ← YAML/TOML compilation pipeline
│   ├── yaml_builder.py          ← Pipeline YAML → DirectedGraph + PipelineConfig
│   ├── yaml_validator.py        ← Schema validation
│   ├── component_instantiator.py ← Port/adapter instantiation
│   ├── pipeline_config.py       ← PipelineConfig model
│   ├── config_loader.py         ← kind: Config + hexdag.toml (deprecated) loading
│   ├── system_builder.py        ← kind: System → SystemConfig (planned)
│   ├── system_config.py         ← SystemConfig model (planned)
│   ├── system_validator.py      ← System manifest validation (planned)
│   ├── plugins/                 ← Entity plugins per kind
│   │   ├── macro_definition.py  ← kind: Macro (existing)
│   │   ├── node_entity.py       ← Node spec parsing (existing)
│   │   ├── config_definition.py ← kind: Config (planned)
│   │   ├── adapter_definition.py ← kind: Adapter (planned)
│   │   └── policy_definition.py ← kind: Policy (planned)
│   ├── preprocessing/           ← Env vars, includes, templates
│   └── tags/                    ← Custom YAML tags (!py, !include)
```

### Manifest Taxonomy (6 Kinds)

| Kind | Purpose | Linux Analogy | Status |
|---|---|---|---|
| `Pipeline` | Single DAG workflow | Process binary | **Exists** |
| `Macro` | Reusable node template (expands into subgraph) | Shared library | **Exists** |
| `Config` | Kernel defaults: limits, caps, orchestrator tuning | `/etc/sysctl.conf` | **Planned** |
| `System` | Multi-process topology + pipes + budgets | `docker-compose.yml` / systemd | **Planned** |
| `Adapter` | Reusable port adapter configuration | `/etc/` config | **Planned** |
| `Policy` | Execution policy (retry, timeout, rate-limit) | iptables / cgroup | **Planned** |

### `kind: Config` — Kernel Settings

Replaces `hexdag.toml` with a YAML manifest. Everything from TOML moves to YAML,
plus new typed sections:

```yaml
kind: Config
metadata:
  name: production-defaults
spec:
  modules: [hexdag.kernel.ports, hexdag.stdlib.nodes]
  logging: { level: INFO, format: structured }
  kernel:
    max_concurrent_nodes: 10
    default_node_timeout: 300.0
  limits:
    max_llm_calls: 100
    max_cost_usd: 10.00
  caps:
    deny: [secret]
```

**Config inheritance:** `kind: Config` defaults → `kind: System` overrides → `kind: Pipeline` overrides → runtime

### `kind: System` — Multi-Process Orchestration

Orchestrates multiple pipelines as one deployable unit with resource budgets,
capability boundaries, and directed data flow:

```yaml
kind: System
metadata:
  name: customer-support
spec:
  ports:
    llm: { ref: prod-llm }
  processes:
    research-agent:
      exec: ./pipelines/research.yaml
      limits: { max_llm_calls: 30, max_cost_usd: 2.00 }
      caps: [llm, memory, tool:web_search]
    analysis-agent:
      exec: ./pipelines/analysis.yaml
      caps: [llm, memory, datastore.read]
  pipes:
    - from: research-agent
      to: analysis-agent
      mapping:
        research_results: "{{ research-agent.researcher.output }}"
  init:
    type: manual
```

### Runtime Architecture: Library-First

hexDAG is a library first, a server second. Two layers:

```
Layer 1: Kernel Runtime Primitives (library — no server needed)
├── SystemRunner          ← Execute kind: System
├── PipelineRunner        ← Execute kind: Pipeline (existing)
├── Scheduler             ← Delayed/recurring execution (existing lib)
├── ProcessRegistry       ← Track running processes (existing lib)
├── ResourceAccounting    ← Enforce limits at runtime (planned)
└── CapSet                ← Enforce caps at runtime (planned)

Layer 2: Hosts (optional — provide HTTP/UI around Layer 1)
├── hexdag studio         ← Built-in host: FastAPI + React UI
├── External FastAPI app  ← User's own app importing hexdag
├── hexdag CLI            ← One-shot commands
└── MCP Server            ← AI agent interface
```

Any Python application with an async event loop can use Layer 1 directly.

### Phased Delivery

| Phase | What | Status |
|---|---|---|
| **Phase 1** | `manual` mode — one-shot `SystemRunner.run()`, library-first | In Progress |
| **Phase 2** | `schedule` mode + Studio System Manager + status API | Planned |
| **Phase 3** | `continuous` mode + process supervision + restart policies | Planned |
| **Phase 4** | `event` mode + EventBus integration | Planned |
| **Phase 5** | MCP tools for System management + `hexdag explain` CLI | Planned |

---

## Resource Limits & Caps

**Status:** Planned

### Resource Limits

Hard ceilings on resource consumption per pipeline/process. Enforced at runtime
via port wrappers.

**Domain model:** `kernel/domain/resource_limits.py`

```python
@dataclass(frozen=True, slots=True)
class ResourceLimits:
    max_total_tokens: int | None = None
    max_llm_calls: int | None = None
    max_tool_calls: int | None = None
    max_cost_usd: float | None = None
    warning_threshold: float = 0.8
    on_limit: Literal["raise", "skip"] = "raise"
```

**Enforcement:** `ResourceAccounting` in `kernel/orchestration/resource_accounting.py`.
Pre-call checks in `ObservableLLMWrapper.aresponse()` and
`ObservableToolRouterWrapper.acall_tool()`.

**Events:** `ResourceWarning`, `ResourceLimitExceeded`

### Caps (Capabilities)

Process-scoped permission boundaries. An allowlist of what ports/tools a process
can access. Enforced at the same port wrapper layer as limits.

**Domain model:** `kernel/domain/caps.py`

```python
@dataclass(frozen=True, slots=True)
class CapSet:
    caps: frozenset[str]  # {"llm", "datastore.read", "tool:search"}

    def allows(self, cap: str) -> bool: ...
    def intersect(self, child: CapSet) -> CapSet: ...
```

**Cap grammar:** `llm`, `memory`, `datastore.read`, `datastore.write`,
`tool:<name>`, `spawner`, `secret`. No caps field = unrestricted (backward compatible).

**Child inheritance:** Child caps = `parent.intersect(child_declared)`. Children
can never have MORE caps than their parent.

**Events:** `CapDenied`

### YAML Integration

Both work in standalone Pipeline and System manifests:

```yaml
# Standalone pipeline
kind: Pipeline
metadata:
  name: research-agent
spec:
  limits: { max_llm_calls: 30, max_cost_usd: 2.00 }
  caps: [llm, memory, tool:web_search]
  nodes: [...]

# In System manifest
spec:
  processes:
    research-agent:
      exec: ./research.yaml
      limits: { max_llm_calls: 30 }
      caps: [llm, memory]
```

---

## Planned Ports

### VFS (Virtual Filesystem)

**Location:** `kernel/ports/vfs.py`

Uniform path-based interface for all hexDAG entities. Every pipeline run,
component, port, and configuration is addressable by path -- the core of
"everything is a file."

#### Two interfaces, one system

hexDAG exposes system state through two complementary interfaces, exactly
as Linux does:

```
                        ┌──────────────────────────────────────────────┐
                        │          Agent / MCP / Studio                │
                        └─────────┬────────────────────┬───────────────┘
                                  │                    │
                    ┌─────────────▼──────────┐  ┌──────▼───────────────┐
                    │   VFS  (namespace)      │  │  Syscalls (api/)     │
                    │                         │  │                      │
                    │  aread   /proc/runs/42  │  │  list_pipeline_runs( │
                    │  alist   /lib/nodes/    │  │    status="done",    │
                    │  astat   /proc/ent/...  │  │    limit=50)         │
                    │  aexec   /etc/pipe/... ²│  │  spawn_pipeline(...) │
                    │  awatch  /proc/runs/.. ³│  │  schedule_pipeline() │
                    │                         │  │  transition_entity() │
                    │  Path-based.            │  │  Parameterized.      │
                    │  One entity at a time.  │  │  Filtering, sorting, │
                    │  Uniform addressing.    │  │  branching logic.    │
                    └─────────┬───────────────┘  └──────┬───────────────┘
                              │                         │
                              ▼                         ▼
                    ┌──────────────────────────────────────────────────┐
                    │           System Libs  (stdlib/lib/)             │
                    │  ProcessRegistry · EntityState · Scheduler       │
                    └──────────────────────────────────────────────────┘
```

**VFS = namespace.** Like `/proc` in Linux. Given a path, return the entity
at that path. No query parameters, no filtering -- just hierarchical
name-to-content resolution. Its power is **uniform addressing**: agents
can browse the entire system through one interface (`aread`, `alist`,
`astat`), and in Phase 2 invoke any entity through `aexec`.

**Syscalls = typed operations.** Like `ps(1)`, `kill(2)`, `waitpid(2)`.
The `api/processes.py` functions take typed parameters (`status`, `ref_id`,
`limit`), do smart dispatch (e.g. `ref_id` -> `alist_by_ref`, otherwise ->
`alist`), and return structured results. These are the query/action interface
for consumers that need filtering, pagination, or multi-parameter operations
that don't map to a single path.

Both interfaces delegate to the same underlying libs. They coexist --
the VFS doesn't replace the syscalls, and the syscalls don't make the
VFS redundant. Each serves a different access pattern.

| Linux                          | hexDAG VFS                                  | hexDAG Syscall                                      |
|--------------------------------|---------------------------------------------|-----------------------------------------------------|
| `cat /proc/1234/status`       | `vfs.aread("/proc/runs/<id>")`              | `get_pipeline_run(registry, run_id)`                |
| `ls /proc/`                   | `vfs.alist("/proc/runs/")`                  | --                                                  |
| `ps aux --sort=-pcpu \| head` | --                                          | `list_pipeline_runs(status=..., limit=50)`          |
| `kill -9 1234`                | `vfs.aexec("/proc/runs/<id>/cancel", {})` 2 | `cancel_scheduled(scheduler, task_id)`              |

2 = Phase 2 &emsp; 3 = Phase 3

#### VFS Phases

**Phase 1 -- Introspection:**

```python
class VFS(Protocol):
    async def aread(self, path: str) -> str: ...
    async def alist(self, path: str) -> list[DirEntry]: ...
    async def astat(self, path: str) -> StatResult: ...
```

**Phase 2 -- Execution:**

```python
    async def aexec(self, path: str, args: dict) -> Any: ...
```

Invoke entities by path. Spawn pipelines via `exec("/etc/pipelines/order-processing", {...})`,
transition entity state via `exec("/proc/entities/order/123/transition", {"to": "shipped"})`,
call tools via `exec("/lib/tools/search", {"query": "..."})`.

**Phase 3 -- Reactivity:**

```python
    async def awatch(self, path: str) -> AsyncIterator[Event]: ...
```

Subscribe to changes on a path (like inotify). Bridges to the existing event/observer
system. Watch `/proc/runs/<id>/status` for state changes, `/lib/nodes/` for new plugins.

#### Namespace tree

```
/proc/runs/<run_id>/status|info       ProcessRegistry
/proc/scheduled/<task_id>/status      Scheduler
/proc/entities/<type>/<id>/state      EntityState
/dev/ports/<name>                     Bound adapter info
/lib/nodes|adapters|macros|tools|libs|prompts|tags/
/etc/pipelines/<name>                 Pipeline definitions
/sys/version|config                   System metadata
```

#### Mount system

Providers register at path prefixes. Longest-prefix match resolves paths.
Each subsystem implements `VFSProvider`:

```python
class VFSProvider(Protocol):
    async def read(self, relative_path: str) -> str: ...
    async def write(self, relative_path: str, data: str) -> int: ...
    async def stat(self, relative_path: str) -> StatResult: ...
    async def readdir(self, relative_path: str) -> list[DirEntry]: ...
```

#### Impact on existing code

- `kernel/discovery.py` -- unchanged (scanning engine, VFS providers call it)
- `kernel/resolver.py` -- unchanged (module path resolution)
- `api/components.py` -- read-only functions become thin VFS wrappers (backwards compatible)
- `api/processes.py` -- mutation functions (`spawn_pipeline`, `transition_entity`) route through `vfs.aexec()` in Phase 2; parameterized query functions (`list_pipeline_runs`, `list_scheduled`) remain as syscall-style APIs (VFS provides uniform introspection, not query semantics)
- `ToolRouter` -- unchanged in Phase 1; auto-populated from VFS in Phase 4 (deferred)

**Stdlib default:** `LocalVFS` in `drivers/vfs/local.py` -- in-process, mount-based dispatch.
**User implementations:** Distributed VFS (etcd/consul), REST-backed VFS.

---

### EventBus

**Location:** `kernel/ports/event_bus.py`

Cross-pipeline pub/sub signals. Pipelines emit and subscribe to named events across
runs, enabling reactive multi-pipeline coordination.

**Use cases:**
- Pipeline A completes -> triggers Pipeline B automatically
- Order state change in Pipeline A -> notifies monitoring Pipeline C
- Error in any pipeline -> triggers alert pipeline
- `kind: System` with `init: { type: event }` -- EventBus triggers system execution

**Protocol sketch:**

```python
class EventBus(Protocol):
    async def apublish(self, topic: str, payload: dict) -> None: ...
    async def asubscribe(self, topic: str, handler: Callable) -> str: ...
    async def aunsubscribe(self, subscription_id: str) -> None: ...
```

**Stdlib default:** In-memory pub/sub with asyncio queues.
**User implementations:** Redis Pub/Sub, Kafka, NATS, cloud event buses.

---

### GovernancePort

**Location:** `kernel/ports/governance.py`

Authorization and audit. Controls who can spawn pipelines, transition entity states,
and access data. Complements Caps (process-scoped permissions) with user-level auth.

**Use cases:**
- Only authorized users can trigger production pipelines
- Audit log of all pipeline executions and state transitions
- Entity state transitions require approval based on role

**Protocol sketch:**

```python
class Governance(Protocol):
    async def aauthorize(self, action: str, principal: str, resource: str) -> bool: ...
    async def aaudit(self, action: str, principal: str, resource: str, result: str) -> None: ...
    async def alist_permissions(self, principal: str) -> list[str]: ...
```

**Stdlib default:** Allow-all policy (no restrictions).
**User implementations:** RBAC, ABAC, OPA integration, cloud IAM.

---

## Planned Libs

### VFSTools

**Location:** `stdlib/lib/vfs_tools.py`

Agent-callable VFS operations. Extends `HexDAGLib` so methods auto-become agent
tools. Gives any agent the ability to introspect the running system through
a single lib rather than separate tools per subsystem.

**Phase 1 methods:** `aread_path(path)`, `alist_directory(path)`, `astat_path(path)`
**Phase 2 methods:** `aexec_path(path, args)`, `awatch_path(path)`

**Dependencies:** VFS port.

---

### CentralAgent

**Location:** `stdlib/lib/central_agent.py`

Meta-orchestrator ("CPU"). An LLM-powered agent that assigns tasks across multiple
pipelines based on goals and available capabilities.

**How it works:**
1. Receives a high-level goal (e.g., "Process this customer complaint")
2. Uses ProcessRegistry to see what pipelines are available and running
3. Uses PipelineSpawner to launch appropriate pipelines
4. Monitors progress via ProcessRegistry events
5. Makes routing decisions via LLM port

**Dependencies:** PipelineSpawner, ProcessRegistry, LLM port, Scheduler.

**Protocol:** Extends `HexDAGLib` so its methods auto-become agent tools.

---

## Deferred Items

### LockPort

**Status:** Deferred -- premature for single-process execution. Only needed for
distributed multi-process scenarios (multiple machines).

**Location:** `kernel/ports/lock.py` (when needed)

Distributed locking and coordination. Prevents concurrent pipeline runs from
conflicting on shared resources.

**Stdlib default:** In-memory asyncio locks (single-process).
**User implementations:** Redis locks (redlock), etcd leases, Consul sessions.

### ArtifactStore

**Status:** Deferred -- FileStorage + DataStore are sufficient for current needs.

**Location:** `kernel/ports/artifact_store.py` (when needed)

Pipeline artifact storage with semantic metadata.

**Stdlib default:** Local filesystem storage.
**User implementations:** S3, Azure Blob, GCS, MLflow artifact store.

### VFS Phase 3 (watch) & Phase 4 (tool integration)

**Status:** Deferred -- EventBus covers reactive subscriptions better (Phase 3).
ToolRouter already handles tool discovery (Phase 4).

---

## Planned Kernel Internals

### RunContext (rename ExecutionContext)

Rename `ExecutionContext` to `RunContext` for clarity. The context object carries run
metadata, ports, results, and configuration through the entire execution stack.

**Changes:**
- `ExecutionContext` -> `RunContext` (with deprecation alias)
- Add `run_id`, `pipeline_name`, `parent_run_id` as first-class fields
- Add typed accessors for common ports (`.llm`, `.memory`, `.database`)

### NodeHook / PortHook

Per-node and per-port lifecycle hooks that extend the existing `hooks.py` system.

**NodeHook:** Before/after node execution.
- Use cases: logging, input/output transformation, caching, authorization

**PortHook:** Before/after port method calls.
- Use cases: request/response logging, retry decoration, circuit breaking

**Design:** Hooks are registered per-node or globally, and receive the execution
context plus the arguments/return value of the call.

---

## Existing vs Planned Overview

```
kernel/ports/ (existing)              kernel/ports/ (planned)
  LLM                                   VFS
  DataStore                              EventBus
  Database (deprecated)                  Governance
  ToolRouter
  ObserverManager
  Executor
  PipelineSpawner
  FileStorage
  SecretStore
  Memory (deprecated)

kernel/domain/ (planned)
  ResourceLimits
  CapSet
  Pipe
  Policy

compiler/ (existing + planned)
  yaml_builder.py (existing)           system_builder.py (planned)
  pipeline_config.py (existing)        system_config.py (planned)
  config_loader.py (planned move)      system_validator.py (planned)
  plugins/macro_definition.py (existing)
  plugins/config_definition.py (planned)
  plugins/adapter_definition.py (planned)
  plugins/policy_definition.py (planned)

stdlib/lib/ (existing)               stdlib/lib/ (planned)
  ProcessRegistry                       VFSTools
  EntityState                           CentralAgent
  Scheduler
  DatabaseTools

kernel/ internals (planned)
  RunContext (rename ExecutionContext)
  NodeHook / PortHook
  SystemRunner
  ResourceAccounting
```

---

## Implementation Priority

1. **YAML Compiler refactor** -- Move `pipeline_builder/` -> `compiler/`, config loader. Structural foundation. **(Step 0a complete)**
2. **`kind: Config` + Resource Limits + Caps** -- Organization-wide safety defaults. Budget enforcement. Process-scoped permissions.
3. **`kind: System` + SystemRunner** -- Multi-process orchestration with pipes and topological execution.
4. **`kind: Adapter` + `kind: Policy`** -- Reusable adapter configs and execution policies.
5. **VFS Phase 1** -- Uniform introspection. Primary query interface. Simplifies api/ layer.
6. **EventBus** -- Cross-pipeline IPC. Required by CentralAgent and `init: { type: event }`.
7. **RunContext rename** -- Low risk, high clarity improvement. Can be done anytime.
8. **VFS Phase 2 (exec)** -- Entity invocation by path.
9. **NodeHook / PortHook** -- Enables middleware-style extensibility.
10. **GovernancePort** -- Required for production multi-tenant deployments.
11. **CentralAgent** -- Depends on EventBus, VFSTools, and existing libs.
