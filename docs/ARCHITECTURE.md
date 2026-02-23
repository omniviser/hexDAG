# hexDAG System Architecture

hexDAG is an **operating system for AI agents**. Just as Linux provides processes,
syscalls, drivers, and libraries so that programs don't reinvent the wheel, hexDAG
provides pipelines, ports, drivers, and a standard library so that AI agents don't
reinvent orchestration.

The codebase is organized around a Linux-kernel-inspired directory structure. This
document explains what belongs where, what each layer does, and how they interact.

---

## The OS Analogy

| Linux | hexDAG | Purpose |
|-------|--------|---------|
| Kernel | `kernel/` | Core execution engine, system call interfaces (Protocols), domain models |
| System calls | `kernel/ports/` | Contracts for external capabilities (LLM, Memory, Database, etc.) |
| Drivers | `drivers/` | Low-level infrastructure (executor, observer manager, pipeline spawner) |
| `/lib` | `stdlib/` | Standard library -- built-in nodes, adapters, macros, system libs |
| Processes | Pipeline runs | Tracked by `ProcessRegistry` (like `ps`) |
| `fork`/`exec` | `PipelineSpawner` | Launch sub-pipelines from within a running pipeline |
| Process scheduler | `Scheduler` | Delayed and recurring pipeline execution |
| State machines | `EntityState` | Business entity lifecycle management |
| `/usr/bin` | `api/` | User-facing tools (MCP + Studio REST) |
| Shell | `cli/` | Command-line interface (`hexdag init`, `hexdag run`, etc.) |

---

## The Four Layers

```
hexdag/
  kernel/   -- Core primitives. Protocols, domain models, orchestration.   (/kernel)
  stdlib/   -- Standard library. Built-in nodes, adapters, macros, libs.   (/lib)
  drivers/  -- Low-level infrastructure. Executor, observer, spawner.      (/drivers)
  api/      -- Public API. MCP tools + Studio REST endpoints.              (/usr/bin)
  cli/      -- CLI commands (hexdag init, hexdag run, etc.)
  docs/     -- Documentation extraction utilities
```

The kernel defines contracts. The stdlib ships implementations. Drivers provide
infrastructure. The API exposes everything to users.

---

## Decision Rules

When adding something new, ask:

1. **Defines a Protocol or domain model?** --> `kernel/`
2. **Implements a Protocol for users to use or extend?** --> `stdlib/`
3. **Low-level infrastructure the orchestrator needs internally?** --> `drivers/`
4. **User-facing function (MCP tool or REST endpoint)?** --> `api/`

If it doesn't fit any of these, it probably belongs in `cli/` (commands) or `docs/`
(documentation utilities).

---

## Kernel

The kernel is the core of hexDAG. It defines all contracts (Protocols), domain models,
and the execution engine. **The kernel depends on nothing outside the kernel.**

```
hexdag/kernel/
  ports/              Protocol definitions (interfaces)
  domain/             Domain models (data structures)
  orchestration/      Execution engine
    port_wrappers.py    Transparent interceptors (observability, caching, retry)
  pipeline_builder/   YAML-to-DAG compilation
  validation/         Type validation, retry logic
  config/             TOML config loading
  resolver.py         Module path resolution
  lib_base.py         HexDAGLib base class for system libraries
  logging.py          Structured logging
  utils/              Internal utilities (timer, async helpers)
```

### kernel/ports/ -- Protocol Definitions

Each port is a Python Protocol class that defines an interface. Adapters in `stdlib/`
and drivers in `drivers/` implement these protocols.

| Port | File | Purpose |
|------|------|---------|
| LLM | `llm.py` | Language model interactions (chat, streaming, usage tracking) |
| Memory | `memory.py` | Key-value memory for agents (get, set, search) |
| DataStore | `data_store.py` | Unified key-value/TTL/query/schema/transactions storage |
| Database | `database.py` | SQL database access (queries, schema inspection, transactions) |
| ToolRouter | `tool_router.py` | Function calling and tool dispatch |
| ObserverManager | `observer_manager.py` | Event routing to observers |
| Executor | `executor.py` | How nodes get executed (local, distributed) |
| PipelineSpawner | `pipeline_spawner.py` | Fork/exec for sub-pipeline runs |
| FileStorage | `file_storage.py` | File read/write abstraction |
| SecretStore | `secret.py` | Secret/credential management |
| VectorSearch | `vector_search.py` | Vector similarity search |
| Healthcheck | `healthcheck.py` | Health status reporting |
| ApiCall | `api_call.py` | External API call abstraction |

### kernel/domain/ -- Domain Models

| Model | File | Purpose |
|-------|------|---------|
| NodeSpec, DirectedGraph | `dag.py` | DAG structure, node specifications, topological sorting |
| PipelineRun | `pipeline_run.py` | Pipeline execution state and results |
| StateMachineConfig | `entity_state.py` | State machine definitions for business entities |
| ScheduledTask | `scheduled_task.py` | Scheduled/recurring pipeline execution definitions |
| AgentToolRouter | `agent_tools.py` | Built-in agent tool routing (tool_end, change_phase) |

### kernel/orchestration/ -- Execution Engine

The orchestrator walks a DirectedGraph in topological order, executing nodes wave by wave.

```
kernel/orchestration/
  orchestrator.py       Main orchestrator (entry point for pipeline execution)
  node_executor.py      Single-node execution with retry, validation, timeout
  input_mapper.py       Resolves {{dep.field}} template expressions
  execution_context.py  Thread-safe context carrying ports, results, metadata
  hooks.py              Pre-DAG and post-DAG hook execution
  events/               Event definitions and observer infrastructure
    events.py           Event dataclasses (PipelineStarted, NodeCompleted, etc.)
    observers/          Built-in observers (logging, memory, cost profiler)
```

### kernel/pipeline_builder/ -- YAML-to-DAG Compilation

Transforms YAML pipeline definitions into executable DirectedGraphs.

```
kernel/pipeline_builder/
  builder.py            Main YamlPipelineBuilder
  preprocessing/        Runs before graph building
    include.py          !include tag for YAML composition
    env_vars.py         ${VAR} and ${VAR:default} resolution
    template.py         Jinja2 templating (build-time vs runtime)
  plugins/              Entity-level plugins
    macros.py           Macro expansion (macro_invocation -> subgraph)
    nodes.py            Node construction from registry
```

### kernel/lib_base.py -- HexDAGLib Base Class

Base class for system libraries. Public async methods prefixed with `a` auto-become
agent-callable tools. This enables libraries like ProcessRegistry and Scheduler to
expose their functionality to LLM agents.

### kernel/config/ -- Configuration

Loads configuration from `hexdag.toml` or `pyproject.toml [tool.hexdag]`.
Supports `${ENV_VAR}` substitution and LRU caching.

---

## Uniform Entity Pattern

All entities follow the same pattern: the kernel defines the contract, the stdlib
ships built-in implementations, and users write their own.

| Entity | kernel/ contract | stdlib/ builtins | User custom |
|--------|-----------------|-----------------|-------------|
| Nodes | `NodeSpec` + `BaseNodeFactory` | LLMNode, AgentNode, FunctionNode, LoopNode, ConditionalNode | `myapp.nodes.X` |
| Adapters | Protocol in `kernel/ports/` | OpenAI, Anthropic, SQLite, Mock, Memory variants | `myapp.adapters.X` |
| Macros | (convention) | ReasoningAgent, ConversationAgent | `myapp.macros.X` |
| Prompts | (convention) | tool_prompts, error_correction | `myapp.prompts.X` |
| Libs | `HexDAGLib` base class | ProcessRegistry, EntityState, Scheduler, DatabaseTools | `myapp.lib.X` |

All entities are referenced by their full Python module path in YAML:

```yaml
nodes:
  - kind: hexdag.stdlib.nodes.LLMNode        # Full path
  - kind: llm_node                            # Built-in alias
  - kind: myapp.nodes.CustomNode              # User custom
```

---

## Stdlib

The standard library ships all built-in implementations that users interact with directly.

```
hexdag/stdlib/
  adapters/     Port implementations
    openai/       OpenAI LLM adapter
    anthropic/    Anthropic LLM adapter
    memory/       Memory adapters (in-memory, SQLite, file, session, state)
    database/     Database adapters (SQLite, CSV)
    mock/         Mock adapters for testing (LLM, database, tool router, embedding)
    secret/       Secret management (local file-based)
    local/        Re-exports from drivers/ for backward compatibility
    unified_tool_router.py  Combined tool routing
  nodes/        Node factories
    llm_node.py         LLM interaction nodes
    agent_node.py       ReAct agent nodes with tool access
    function_node.py    Execute arbitrary Python functions
    loop_node.py        Iterative processing
    conditional_node.py Conditional execution paths
  macros/       Reusable pipeline templates
    reasoning_agent.py    Multi-step reasoning macro
    conversation_agent.py Conversational agent macro
  prompts/      Prompt templates
    tool_prompts.py       Tool calling format prompts
    error_correction.py   Error correction prompts
  lib/          System libraries (HexDAGLib subclasses)
    process_registry.py   Track pipeline runs (like ps)
    entity_state.py       Declarative state machines for business entities
    scheduler.py          Delayed/recurring pipeline execution
    database_tools.py     Agent-callable SQL query tools
```

---

## Drivers

Drivers are low-level infrastructure that the orchestrator needs internally. Unlike
adapters (which users swap regularly -- OpenAI today, Anthropic tomorrow), drivers are
infrastructure that rarely changes.

Each driver implements a kernel Protocol:

| Driver | Directory | Implements | Purpose |
|--------|-----------|-----------|---------|
| LocalExecutor | `drivers/executors/` | `Executor` | Execute nodes via asyncio in the local process |
| LocalObserverManager | `drivers/observer_manager/` | `ObserverManager` | Route events to registered observers |
| LocalPipelineSpawner | `drivers/pipeline_spawner/` | `PipelineSpawner` | Fork sub-pipeline runs in the local process |

Users swap adapters (OpenAI <-> Anthropic), not drivers. Drivers are the plumbing.

---

## Port Wrappers

Between raw ports and the nodes that call them, the orchestrator can inject
**port wrappers** — transparent interceptors that add infrastructure concerns
without changing the port contract.

```
Node calls port method
  |
  v
Port Wrapper (transparent)         ← observability, caching, retry, rate-limit
  |-- emit event (LLMPromptSent)
  |-- check policy (future: rate-limit, auth)
  |-- delegate to underlying adapter
  |-- emit event (LLMResponseReceived)
  |
  v
Adapter (actual implementation)
```

**Location:** `kernel/orchestration/port_wrappers.py`

| Wrapper | Wraps | Adds |
|---------|-------|------|
| `ObservableLLMWrapper` | `SupportsGeneration` | Emits `LLMPromptSent`/`LLMResponseReceived`, tracks duration and token usage |
| `ObservableToolRouterWrapper` | `ToolRouter` | Emits `ToolCalled`/`ToolCompleted`, tracks duration, logs errors |

Wrappers are applied automatically by the orchestrator via `wrap_ports_with_observability()`
at pipeline startup. Nodes don't know they're talking to a wrapper — the interface is
identical to the raw port. This is the same pattern as Linux's VFS caching layer
sitting between `read(2)` and the actual filesystem driver.

### Port wrappers vs Libs

Both can wrap ports, but they serve different purposes:

```
                    Transparent to caller?    Exposed as agent tools?    Owns state?
                    ─────────────────────     ──────────────────────     ──────────
Port Wrappers       Yes (same interface)      No                         No
Libs                No (different API)        Yes (auto-tool)            Often yes
```

- **Port wrapper** = kernel-level interception. Like Linux's block I/O scheduler
  sitting between `write(2)` and the disk driver. Adds caching, retry, metrics.
  The caller doesn't know it's there.

- **Lib** = userspace library. Like `libpq` providing `PQexec()` on top of the
  raw PostgreSQL wire protocol. Adds business-domain convenience (e.g. `alist_tables()`,
  `adescribe_table()`). Agents call libs explicitly by name.

An ORM-like abstraction over the Database port is a **Lib** — it wraps a
`SupportsQuery` port with higher-level operations and exposes them as agent
tools. `DatabaseTools` in `stdlib/lib/` is exactly this pattern.

---

## API Layer

The API layer exposes hexDAG functionality to external consumers. Both the MCP server
and Studio REST API consume the same implementation functions.

```
hexdag/api/
  vfs.py             VFS factory and path wrappers (namespace interface)
  processes.py       Parameterized process queries and actions (syscall interface)
  components.py      Component listing and schema generation
  validation.py      Pipeline YAML validation
  execution.py       Pipeline execution (build + run)
  pipeline.py        Pipeline CRUD operations
  documentation.py   MCP guide content generation
  export.py          Pipeline export (JSON, Python code)
```

### VFS vs Syscalls

hexDAG exposes system state through two complementary interfaces — like
Linux's `/proc` filesystem alongside `ps(1)` and `kill(2)`:

```
  VFS  (namespace — uniform addressing)     Syscalls  (api/ — typed operations)
  ──────────────────────────────────────     ──────────────────────────────────────
  vfs.aread("/proc/runs/<id>")              list_pipeline_runs(status=..., limit=50)
  vfs.alist("/lib/nodes/")                  spawn_pipeline(name, input, ref_id=...)
  vfs.astat("/proc/entities/order/123")     transition_entity(type, id, to_state)
```

**VFS** answers "what's at this path?" — one path, one entity. Agents use it
to browse the system uniformly.

**Syscalls** answer "find/do something with parameters" — filtering, sorting,
branching logic. The `api/processes.py` functions take typed parameters and
delegate to the appropriate lib method.

Both delegate to the same underlying libs (`ProcessRegistry`, `EntityState`,
`Scheduler`). They coexist — VFS doesn't replace syscalls, and syscalls
don't make VFS redundant.

---

## Config and Discovery

How the framework bootstraps and discovers components:

### 1. Default Config

`get_default_config()` returns a hardcoded list of built-in modules:

```python
modules=[
    "hexdag.kernel.ports",
    "hexdag.stdlib.nodes",
    "hexdag.stdlib.adapters.mock",
    "hexdag.drivers.executors",
    "hexdag.drivers.observer_manager",
    "hexdag.drivers.pipeline_spawner",
    "hexdag.kernel.domain.agent_tools",
]
```

### 2. User Config

Users add their own modules via `hexdag.toml` or `pyproject.toml`:

```toml
[tool.hexdag]
modules = ["myapp.nodes", "myapp.adapters"]
```

### 3. Plugin Auto-Discovery

Any installed package under the `hexdag_plugins` namespace is auto-discovered:

```python
# In your package's pyproject.toml:
[project.entry-points."hexdag_plugins"]
my_plugin = "my_plugin_package"
```

### 4. Module Resolution

`resolver.resolve("hexdag.stdlib.nodes.LLMNode")` maps a module path string to the
actual Python class. Used by the pipeline builder to instantiate components from YAML.

---

## Execution Flow

```
YAML file
  |
  v
PipelineBuilder
  |-- Preprocessing (include, env vars, Jinja2 templates)
  |-- Entity plugins (macros, nodes)
  |-- Component instantiation via resolver
  v
DirectedGraph + PipelineConfig
  |
  v
Orchestrator
  |-- Initialize ports (adapters)
  |-- Wrap ports (observability, policy — port_wrappers.py)
  |-- Pre-DAG hooks
  |-- Wave execution (topological order)
  |   |-- For each wave: execute nodes in parallel (asyncio.gather)
  |   |-- InputMapper resolves {{dep.field}} references
  |   |-- NodeExecutor handles retry, validation, timeout
  |   |-- Events emitted at each stage
  |-- Post-DAG hooks
  |-- Cleanup (adapter.aclose())
  v
Results (dict of node_name -> output)
```

### Key Execution Concepts

**Waves**: Nodes are grouped into waves based on topological sort. All nodes in a wave
can execute in parallel because their dependencies are in earlier waves.

**Skip Propagation**: A node with `when: "expr"` evaluating to false is skipped. When
ALL dependencies of a node were skipped, the downstream node auto-skips too.

**Events**: Every lifecycle stage emits events (PipelineStarted, WaveStarted,
NodeStarted, NodeCompleted, NodeFailed, NodeSkipped, etc.) that observers receive.

---

## Extension Points

### Custom Node

```python
from hexdag.stdlib.nodes import BaseNodeFactory
from hexdag.kernel.domain.dag import NodeSpec

class MyNode(BaseNodeFactory):
    def __call__(self, name: str, **kwargs) -> NodeSpec:
        async def process(ctx, **inputs):
            return {"result": "processed"}
        return NodeSpec(name=name, fn=process)
```

Use in YAML: `kind: myapp.nodes.MyNode`

### Custom Adapter

Implement a kernel Protocol:

```python
from hexdag.kernel.ports.llm import SupportsLLM

class MyLLMAdapter:
    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.api_key = api_key
        self.model = model

    async def aresponse(self, messages: list[dict], **kwargs) -> str:
        # Your implementation
        ...
```

Use in YAML: `adapter: myapp.adapters.MyLLMAdapter`

### Custom Lib

```python
from hexdag.kernel.lib_base import HexDAGLib

class MyLib(HexDAGLib):
    """Public async a* methods auto-become agent tools."""

    async def ado_something(self, query: str) -> str:
        """This becomes an agent-callable tool."""
        return f"Result for {query}"
```

### Custom Observer

```python
from hexdag.kernel.orchestration.events import NodeCompleted

class MyObserver:
    async def on_event(self, event):
        if isinstance(event, NodeCompleted):
            print(f"Node {event.node_name} completed in {event.duration_ms}ms")
```

### Custom Tool

Plain functions with type hints and docstrings:

```python
def search_database(query: str, limit: int = 10) -> list[dict]:
    """Search the database for matching records."""
    return results
```

---

## Kernel Extensions Roadmap

The following kernel-level primitives are planned. Each follows the uniform entity
pattern: kernel defines Protocol, stdlib ships default implementation, users write
their own.

### Planned Ports (kernel/ports/)

| Port | Purpose |
|------|---------|
| **EventBus** | Cross-pipeline pub/sub signals. Pipelines emit/subscribe to named events across runs. Enables reactive multi-pipeline coordination. |
| **LockPort** | Distributed locking/coordination. Prevents concurrent pipeline runs from conflicting on shared resources. |
| **GovernancePort** | Authorization and audit. Controls who can spawn pipelines, transition entity states, access data. |
| **ArtifactStore** | Pipeline artifact storage (extends FileStorage). Semantic layer for pipeline inputs/outputs/intermediate results. |

### Planned Libs (stdlib/lib/)

| Lib | Purpose |
|-----|---------|
| **CentralAgent** | Meta-orchestrator ("CPU"). LLM-powered task assignment across multiple pipelines. Uses PipelineSpawner + ProcessRegistry + LLM port. |

### Planned Kernel Internals

| Component | Purpose |
|-----------|---------|
| **RunContext** | Rename ExecutionContext for clarity. Carries run metadata, ports, results through the execution stack. |
| **NodeHook / PortHook** | Per-node lifecycle hooks (extends existing hooks.py). Before/after execution, before/after port calls. |

### Existing vs Planned

```
kernel/ports/ (existing)              kernel/ports/ (planned)
  LLM                                   EventBus
  DataStore / Memory                     LockPort
  Database                               GovernancePort
  ToolRouter                             ArtifactStore
  ObserverManager
  Executor
  PipelineSpawner
  FileStorage
  SecretStore
  VectorSearch
  VFS (Phase 1)

stdlib/lib/ (existing)               stdlib/lib/ (planned)
  ProcessRegistry                       CentralAgent
  EntityState
  Scheduler
  DatabaseTools
  VFSTools
```

---

**See Also**:
- [Hexagonal Architecture](HEXAGONAL_ARCHITECTURE.md) -- Why ports and adapters
- [Plugin System](PLUGIN_SYSTEM.md) -- Extending hexDAG with plugins
- [Roadmap](ROADMAP.md) -- Development roadmap and planned extensions
