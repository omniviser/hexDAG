# hexDAG Architecture Roadmap: Kernel Extensions

**Status:** Planning
**Last Updated:** 2026-02

hexDAG is an operating system for AI agents. This roadmap describes the planned
kernel-level extensions that complete the OS analogy -- adding IPC (EventBus),
locks (LockPort), permissions (GovernancePort), a filesystem for artifacts
(ArtifactStore), and a CPU scheduler for multi-pipeline coordination (CentralAgent).

Each follows the uniform entity pattern: kernel defines Protocol, stdlib ships default
implementation, users write their own.

For the current architecture, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## Planned Ports

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
class SupportsEventBus(Protocol):
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
class SupportsLock(Protocol):
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
class SupportsGovernance(Protocol):
    async def aauthorize(self, action: str, principal: str, resource: str) -> bool: ...
    async def aaudit(self, action: str, principal: str, resource: str, result: str) -> None: ...
    async def alist_permissions(self, principal: str) -> list[str]: ...
```

**Stdlib default:** Allow-all policy (no restrictions).
**User implementations:** RBAC, ABAC, OPA integration, cloud IAM.

---

### ArtifactStore

**Location:** `kernel/ports/artifact_store.py`

Pipeline artifact storage with semantic metadata. Extends the existing FileStoragePort
concept with a semantic layer for pipeline inputs, outputs, and intermediate results.

**Use cases:**
- Store and retrieve pipeline run artifacts (inputs, outputs, logs)
- Compare artifacts across pipeline versions
- Pipeline result caching (skip re-execution if inputs haven't changed)

**Protocol sketch:**

```python
class SupportsArtifactStore(Protocol):
    async def astore(self, run_id: str, name: str, data: bytes, metadata: dict) -> str: ...
    async def aretrieve(self, artifact_id: str) -> tuple[bytes, dict]: ...
    async def alist(self, run_id: str) -> list[dict]: ...
    async def adelete(self, artifact_id: str) -> None: ...
```

**Stdlib default:** Local filesystem storage.
**User implementations:** S3, Azure Blob, GCS, MLflow artifact store.

---

## Planned Libs

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
  LLM                                   EventBus
  Memory / DataStore                     LockPort
  Database                               GovernancePort
  ToolRouter                             ArtifactStore
  ObserverManager
  Executor
  PipelineSpawner
  FileStorage
  Secret
  VectorSearch

stdlib/lib/ (existing)               stdlib/lib/ (planned)
  ProcessRegistry                       CentralAgent
  EntityState
  Scheduler
  DatabaseTools

kernel/ internals (planned)
  RunContext (rename ExecutionContext)
  NodeHook / PortHook
```

---

## Implementation Priority

1. **EventBus** -- Unlocks multi-pipeline coordination. Required by CentralAgent.
2. **LockPort** -- Required for safe concurrent pipeline execution.
3. **RunContext rename** -- Low risk, high clarity improvement. Can be done anytime.
4. **NodeHook / PortHook** -- Enables middleware-style extensibility.
5. **GovernancePort** -- Required for production multi-tenant deployments.
6. **ArtifactStore** -- Required for pipeline result caching and audit trails.
7. **CentralAgent** -- Depends on EventBus, LockPort, and existing libs.
