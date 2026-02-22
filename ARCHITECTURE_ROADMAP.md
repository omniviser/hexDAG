# hexDAG Architecture Roadmap: Kernel Extensions

**Status:** Planning
**Last Updated:** 2026-02

hexDAG is an operating system for AI agents. This roadmap describes the planned
kernel-level extensions that complete the OS analogy -- adding a virtual filesystem
for uniform introspection (VFS), IPC (EventBus), locks (LockPort), permissions
(GovernancePort), a filesystem for artifacts (ArtifactStore), and a CPU scheduler
for multi-pipeline coordination (CentralAgent).

Each follows the uniform entity pattern: kernel defines Protocol, stdlib ships default
implementation, users write their own.

For the current architecture, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## Planned Ports

### VFS (Virtual Filesystem)

**Location:** `kernel/ports/vfs.py`

Uniform path-based interface for all hexDAG entities. Every pipeline run,
component, port, and configuration is addressable by path — the core of
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
at that path. No query parameters, no filtering — just hierarchical
name-to-content resolution. Its power is **uniform addressing**: agents
can browse the entire system through one interface (`aread`, `alist`,
`astat`), and in Phase 2 invoke any entity through `aexec`.

**Syscalls = typed operations.** Like `ps(1)`, `kill(2)`, `waitpid(2)`.
The `api/processes.py` functions take typed parameters (`status`, `ref_id`,
`limit`), do smart dispatch (e.g. `ref_id` → `alist_by_ref`, otherwise →
`alist`), and return structured results. These are the query/action interface
for consumers that need filtering, pagination, or multi-parameter operations
that don't map to a single path.

Both interfaces delegate to the same underlying libs. They coexist —
the VFS doesn't replace the syscalls, and the syscalls don't make the
VFS redundant. Each serves a different access pattern.

| Linux                          | hexDAG VFS                                  | hexDAG Syscall                                      |
|--------------------------------|---------------------------------------------|-----------------------------------------------------|
| `cat /proc/1234/status`       | `vfs.aread("/proc/runs/<id>")`              | `get_pipeline_run(registry, run_id)`                |
| `ls /proc/`                   | `vfs.alist("/proc/runs/")`                  | —                                                   |
| `ps aux --sort=-pcpu \| head` | —                                           | `list_pipeline_runs(status=..., limit=50)`          |
| `kill -9 1234`                | `vfs.aexec("/proc/runs/<id>/cancel", {})` ² | `cancel_scheduled(scheduler, task_id)`              |

² = Phase 2 &emsp; ³ = Phase 3

#### VFS Phases

**Phase 1 — Introspection (implemented):**

```python
class VFS(Protocol):
    async def aread(self, path: str) -> str: ...
    async def alist(self, path: str) -> list[DirEntry]: ...
    async def astat(self, path: str) -> StatResult: ...
```

**Phase 2 — Execution:**

```python
    async def aexec(self, path: str, args: dict) -> Any: ...
```

Invoke entities by path. Spawn pipelines via `exec("/etc/pipelines/order-processing", {...})`,
transition entity state via `exec("/proc/entities/order/123/transition", {"to": "shipped"})`,
call tools via `exec("/lib/tools/search", {"query": "..."})`.

**Phase 3 — Reactivity:**

```python
    async def awatch(self, path: str) -> AsyncIterator[Event]: ...
```

Subscribe to changes on a path (like inotify). Bridges to the existing event/observer
system. Watch `/proc/runs/<id>/status` for state changes, `/lib/nodes/` for new plugins.

**Phase 4 — Tool Integration:**

VFS auto-registers executable paths as ToolRouter tools at pipeline startup. Agents
still see individual typed tool schemas (e.g., `adatabase_query(sql, params)`), but
each tool is backed by a VFS `exec` call. New tools appear automatically when a
provider is mounted.

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

- `kernel/discovery.py` — unchanged (scanning engine, VFS providers call it)
- `kernel/resolver.py` — unchanged (module path resolution)
- `api/components.py` — read-only functions become thin VFS wrappers (backwards compatible)
- `api/processes.py` — mutation functions (`spawn_pipeline`, `transition_entity`) route through `vfs.aexec()` in Phase 2; parameterized query functions (`list_pipeline_runs`, `list_scheduled`) remain as syscall-style APIs (VFS provides uniform introspection, not query semantics)
- `ToolRouter` — unchanged in Phase 1; auto-populated from VFS in Phase 4

**Stdlib default:** `LocalVFS` in `drivers/vfs/local.py` — in-process, mount-based dispatch.
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

### LockPort

**Location:** `kernel/ports/lock.py`

Distributed locking and coordination. Prevents concurrent pipeline runs from
conflicting on shared resources.

**Use cases:**
- Only one pipeline can write to a shared database table at a time
- Rate limiting across pipeline runs
- Leader election for CentralAgent

**Protocol sketch:**

```python
class Lock(Protocol):
    async def aacquire(self, name: str, timeout: float | None = None) -> bool: ...
    async def arelease(self, name: str) -> None: ...
    async def ais_locked(self, name: str) -> bool: ...
```

**Stdlib default:** In-memory asyncio locks (single-process).
**User implementations:** Redis locks (redlock), etcd leases, Consul sessions.

---

### GovernancePort

**Location:** `kernel/ports/governance.py`

Authorization and audit. Controls who can spawn pipelines, transition entity states,
and access data.

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

### ArtifactStore

**Location:** `kernel/ports/artifact_store.py`

Pipeline artifact storage with semantic metadata. Extends the existing FileStorage
concept with a semantic layer for pipeline inputs, outputs, and intermediate results.

**Use cases:**
- Store and retrieve pipeline run artifacts (inputs, outputs, logs)
- Compare artifacts across pipeline versions
- Pipeline result caching (skip re-execution if inputs haven't changed)

**Protocol sketch:**

```python
class ArtifactStore(Protocol):
    async def astore(self, run_id: str, name: str, data: bytes, metadata: dict) -> str: ...
    async def aretrieve(self, artifact_id: str) -> tuple[bytes, dict]: ...
    async def alist(self, run_id: str) -> list[dict]: ...
    async def adelete(self, artifact_id: str) -> None: ...
```

**Stdlib default:** Local filesystem storage.
**User implementations:** S3, Azure Blob, GCS, MLflow artifact store.

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
  Database (deprecated)                  Lock
  ToolRouter                             Governance
  ObserverManager                        ArtifactStore
  Executor
  PipelineSpawner
  FileStorage
  SecretStore
  Memory (deprecated)

stdlib/lib/ (existing)               stdlib/lib/ (planned)
  ProcessRegistry                       VFSTools
  EntityState                           CentralAgent
  Scheduler
  DatabaseTools

kernel/ internals (planned)
  RunContext (rename ExecutionContext)
  NodeHook / PortHook
```

---

## Implementation Priority

1. **VFS Phase 1** -- Uniform introspection. Primary query interface. Simplifies api/ layer.
2. **EventBus** -- Cross-pipeline IPC. Required by CentralAgent.
3. **LockPort** -- Required for safe concurrent pipeline execution.
4. **RunContext rename** -- Low risk, high clarity improvement. Can be done anytime.
5. **VFS Phase 2 (exec)** -- Entity invocation by path. Spawn pipelines, transition states, call tools.
6. **NodeHook / PortHook** -- Enables middleware-style extensibility.
7. **GovernancePort** -- Required for production multi-tenant deployments.
8. **VFS Phase 3 (watch)** -- Reactive path subscriptions. Bridges to event/observer system.
9. **ArtifactStore** -- Required for pipeline result caching and audit trails.
10. **VFS Phase 4 (tool integration)** -- Auto-register VFS executables as ToolRouter tools.
11. **CentralAgent** -- Depends on EventBus, LockPort, VFSTools, and existing libs.
