# YAML as the System Compiler: Multi-Pipeline Orchestration for hexDAG

**Status:** In Progress (Steps 0a, 0b, 0c complete)
**Last Updated:** 2026-02

## Context

hexDAG is an operating system for AI agents. Today, YAML describes **one pipeline** (`kind: Pipeline`). But OS-level features -- resource limits, capability boundaries, pipeline chaining -- operate **across** pipelines. This plan makes YAML the compiler and blueprint for the entire system.

This plan introduces two things:

**1. The YAML Compiler model.** YAML is to hexDAG what C source is to GCC -- the **source language** that gets compiled into kernel objects. The YAML compiler (`compiler/`) translates manifests into domain models (`DirectedGraph`, `PipelineConfig`, `ResourceLimits`, etc.). The **kernel never touches YAML** -- it only knows domain models and execution. Step 0 moves the compiler out of the kernel and relocates config loading.

**2. A complete manifest taxonomy** with three layers of declarative configuration:

- **`kind: Config`** -- Organization-wide defaults: resource limits, caps, orchestrator tuning. Like `/etc/sysctl.conf`.
- **`kind: Pipeline`** (existing, extended) -- Single pipeline with optional `spec.limits` and `spec.caps` that override Config defaults.
- **`kind: System`** (new) -- Multi-process topology: which processes run together, how they connect via pipes, per-process overrides.

**Inheritance chain:** `kind: Config` defaults -> `kind: System` overrides -> `kind: Pipeline` overrides -> runtime

**Architectural principle:** The kernel defines domain models. The YAML compiler (userspace) parses manifests into those models. The kernel runs the compiled objects. YAML is an implementation detail the kernel never sees.

---

## Roadmap Assessment

### Well-Aligned with Agentic Vision
| Roadmap Item | Why it matters for agents |
|---|---|
| **EventBus** | Cross-pipeline IPC -- agents react to each other |
| **CentralAgent** | LLM-powered meta-orchestrator |
| **VFS Phase 2 (exec)** | Invoke entities by path |
| **GovernancePort** | User-level auth (complements Capabilities) |

### Gaps Filled by This Plan
| Gap | Solution |
|---|---|
| No kernel-level defaults | **Kernel Config** -- typed defaults in `hexdag.yaml` for limits, caps, orchestrator |
| No budget enforcement | **Resource Limits** -- hard stops when limits breached |
| No process-scoped permissions | **Caps** -- restrict what a process can access at runtime |
| No declarative multi-process topology | **`kind: System`** -- YAML blueprint for agent systems |
| No process chaining | **Pipes** -- directed data flow between processes |

### Deferred (Over-Engineered for Current Stage)
| Item | Reason |
|---|---|
| **LockPort** | Premature -- only needed for distributed multi-process |
| **ArtifactStore** | FileStorage + DataStore sufficient |
| **VFS Phase 3 (watch)** | EventBus covers reactive subscriptions better |
| **VFS Phase 4 (tool integration)** | ToolRouter already handles tool discovery |

---

## Part 1: `kind: Config` -- Kernel Settings as YAML

### The Problem

Two issues:

1. **No typed kernel defaults.** `HexDAGConfig` (`kernel/config/models.py`) has `modules`, `plugins`, `dev_mode`, `logging`, and a generic `settings: dict[str, Any]`. No typed fields for resource limits, caps, or orchestrator tuning.

2. **Serialization format leaks into the kernel.** `kernel/config/loader.py` imports `tomllib` -- a serialization concern living inside the kernel. In the YAML-as-compiler model, the kernel should only define domain models. Parsing (whether YAML or TOML) is a compiler/userspace concern.

### Design: `kind: Config` as a YAML Manifest

```yaml
kind: Config
metadata:
  name: production-defaults
  namespace: prod
spec:
  # Module loading (same as today's hexdag.toml)
  modules:
    - hexdag.kernel.ports
    - hexdag.stdlib.nodes
    - hexdag.stdlib.adapters.openai
    - hexdag.drivers.executors
  plugins: []

  # Logging
  logging:
    level: INFO
    format: structured

  # Kernel execution defaults (like /etc/sysctl.conf)
  kernel:
    max_concurrent_nodes: 10
    default_node_timeout: 300.0    # 5 min timeout for all nodes
    strict_validation: false

  # Default resource limits -- all pipelines inherit these
  limits:
    max_llm_calls: 100             # Safety net: no pipeline makes >100 LLM calls
    max_cost_usd: 10.00            # Hard ceiling: $10 per pipeline run
    max_tool_calls: 200
    warning_threshold: 0.8         # Warn at 80% of limits

  # Default caps -- all processes inherit these
  caps:
    default_set: [llm, memory, datastore.read]  # Default allowlist
    deny: [secret]                               # Always denied unless explicitly granted
```

### Replacing `hexdag.toml`

`kind: Config` replaces `hexdag.toml` entirely. Everything that was in TOML moves to YAML:

- `modules` / `plugins` -> `spec.modules` / `spec.plugins`
- `[logging]` -> `spec.logging`
- `dev_mode` -> `spec.dev_mode`
- `[settings]` -> `spec.settings`
- NEW: `[kernel]`, `[limits]`, `[caps]` -> `spec.kernel`, `spec.limits`, `spec.caps`

No chicken-and-egg: `yaml.safe_load` is stdlib, always available before any hexDAG modules load.

**Migration path**: Existing `hexdag.toml` files continue to work during a deprecation period. `hexdag init` generates `kind: Config` YAML. A migration script converts TOML -> YAML.

**Config file discovery** (new order):
1. `HEXDAG_CONFIG_PATH` env var
2. `hexdag.yaml` in current directory
3. `hexdag.toml` in current directory (deprecated, with warning)

### Moving Config Loading Out of the Kernel

The YAML compiler model means: **the kernel defines domain models, userspace handles parsing.**

This matches Linux exactly: the kernel defines structs (`task_struct`, `cgroup`), config files (`/etc/sysctl.conf`, `/etc/fstab`) are parsed by userspace tools (`sysctl`, `mount`), and the kernel only accepts structured data via syscalls. The kernel never reads config files.

**What moves (done in Step 0):**
- `kernel/config/loader.py` -> `compiler/config_loader.py` (the "compiler" knows about serialization formats)
- `kernel/config/models.py` -> stays in kernel -- these are pure domain models

**What stays in kernel:**
- `HexDAGConfig`, `OrchestratorConfig`, `DefaultLimits`, `DefaultCaps`, `LoggingConfig` -- pure data models, no parsing
- These are the "compiled objects" that the kernel operates on

**The compiler (`compiler/`) handles:**
- YAML parsing (`yaml.safe_load`)
- TOML parsing (deprecated, for backward compat)
- `kind: Config` document processing
- Env var substitution
- Producing `HexDAGConfig` instances for the kernel

### Data Model Changes

**File:** `kernel/config/models.py` (edit existing)

Add two new typed config dataclasses, reuse existing `OrchestratorConfig`:

```python
@dataclass(frozen=True, slots=True)
class DefaultLimits:
    """Default resource limits for all processes."""
    max_total_tokens: int | None = None
    max_llm_calls: int | None = None
    max_tool_calls: int | None = None
    max_cost_usd: float | None = None
    warning_threshold: float = 0.8

@dataclass(frozen=True, slots=True)
class DefaultCaps:
    """Default capability boundaries for all processes."""
    default_set: list[str] | None = None  # None = unrestricted
    deny: list[str] | None = None         # Always denied (e.g., ["secret", "spawner"])
```

Extend `HexDAGConfig` (reuse `OrchestratorConfig` -- no `KernelConfig` duplication):

```python
from hexdag.kernel.orchestration.models import OrchestratorConfig

@dataclass(slots=True)
class HexDAGConfig:
    modules: list[str] = field(default_factory=list)
    plugins: list[str] = field(default_factory=list)
    dev_mode: bool = False
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    settings: dict[str, Any] = field(default_factory=dict)
    # NEW -- typed config sections (reuse OrchestratorConfig, add limits + caps)
    orchestrator: OrchestratorConfig = field(default_factory=OrchestratorConfig)
    limits: DefaultLimits = field(default_factory=DefaultLimits)
    caps: DefaultCaps = field(default_factory=DefaultCaps)
```

**Note:** `OrchestratorConfig` already exists at `kernel/orchestration/models.py` with the exact fields we need: `max_concurrent_nodes`, `strict_validation`, `default_node_timeout`. No new `KernelConfig` needed.

### Resolution Order

```
kind: Config spec.limits     <- Organization defaults (hexdag.yaml)
  | overridden by
kind: System pipeline.limits <- System-level per-pipeline override
  | overridden by
kind: Pipeline spec.limits   <- Pipeline's own declaration
  | read by
ResourceAccounting           <- Runtime enforcement
```

### Backward Compatibility

- All new fields default to current behavior (no limits, no caps, 10 concurrent nodes)
- Existing `hexdag.toml` continues to work with a deprecation warning
- No `kind: Config` = no defaults applied (fully permissive, as today)
- `hexdag init` generates `hexdag.yaml` with `kind: Config` instead of `hexdag.toml`

---

## Part 2: YAML Style -- OS Terms + K8s Structure

**Decision:** Use K8s manifest structure (`kind`/`metadata`/`spec`) for composability and schema validation, but adopt OS-native terminology inside for hexDAG's identity as an agent OS. Skip `apiVersion` for now -- add API groups and versioning later when the schema stabilizes.

### Terminology Mapping

| K8s-style (before) | OS-native (after) | Rationale |
|---|---|---|
| `pipelines` | `processes` | Pipelines are processes in the OS analogy |
| `capabilities` | `caps` | Linux `capabilities(7)` uses short form |
| `source` | `exec` | Processes are "executed" from a binary (pipeline YAML) |
| `trigger` | `init` | System initialization trigger (like init system) |

### The Chosen Style -- Complete Example

```yaml
---
kind: Config
metadata:
  name: production-defaults
spec:
  kernel:
    concurrency: 10
    timeout: 300.0
  limits:
    max_cost_usd: 10.00
    max_llm_calls: 100
  caps:
    deny: [secret]
---
kind: Adapter
metadata:
  name: prod-llm
spec:
  port: llm
  adapter: hexdag.stdlib.adapters.openai.OpenAIAdapter
  config:
    model: gpt-4
    api_key: ${OPENAI_API_KEY}
---
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
      limits: { max_llm_calls: 10, max_cost_usd: 1.00 }
      caps: [llm, memory, datastore.read]
    response-agent:
      exec: ./pipelines/response.yaml
      limits: { max_llm_calls: 5, max_cost_usd: 0.50 }
      caps: [llm, memory]
  pipes:
    - from: research-agent
      to: analysis-agent
      mapping:
        research_results: "{{ research-agent.researcher.output }}"
    - from: analysis-agent
      to: response-agent
      mapping:
        analysis: "{{ analysis-agent.analyzer.output }}"
  init:
    type: manual
```

---

## Part 3: The Manifest Taxonomy -- All YAML Kinds

### Design Principle

A `kind` earns its place in the taxonomy only if it represents a **standalone, reusable, versionable** artifact. Things that only make sense inline (node specs, port configs) stay inline. The test: "Would someone git-commit this as its own file?"

### Complete Taxonomy

```
kind: Config       -- Kernel defaults: limits, caps, orchestrator (new)
kind: Pipeline     -- Single DAG workflow (existing)
kind: Macro        -- Reusable node template (existing)
kind: System       -- Multi-process topology (new)
kind: Adapter      -- Reusable adapter configuration (new)
kind: Policy       -- Reusable execution policy (new)
```

### `kind: Adapter` -- Reusable Port Configuration

**The problem**: In a System with 5 processes all using the same OpenAI adapter, you copy-paste the adapter config 5 times.

```yaml
kind: Adapter
metadata:
  name: production-openai
  namespace: prod
spec:
  port: llm
  adapter: hexdag.stdlib.adapters.openai.OpenAIAdapter
  config:
    model: gpt-4
    api_key: ${OPENAI_API_KEY}
    temperature: 0.7
    max_tokens: 4096
```

**Referenced in Pipeline or System:**
```yaml
spec:
  ports:
    llm: { ref: production-openai }
```

### `kind: Policy` -- Reusable Execution Policies

```yaml
kind: Policy
metadata:
  name: standard-retry
spec:
  type: retry
  config:
    max_retries: 3
    backoff: exponential
    initial_delay_ms: 1000
    max_delay_ms: 30000
    retryable_errors: [TimeoutError, ConnectionError]
---
kind: Policy
metadata:
  name: cost-guard
spec:
  type: resource_limit
  config:
    max_cost_usd: 5.00
    max_llm_calls: 50
    on_limit: raise
```

### Deliberately NOT Added

| Candidate | Why not a `kind` |
|---|---|
| **Node** | Nodes are Python classes (node factories). Custom nodes are code, not YAML. |
| **Lib** | Libs are Python classes extending `HexDAGLib`. Methods auto-become tools. |
| **Prompt** | Prompt templates are strings/functions, not structured manifests. |
| **Port** | Ports are Protocol interfaces in Python. You implement them as Drivers. |
| **Schedule** | A property of a workload (`kind: System` `spec.init`), not standalone. |
| **Environment** | Handled by `metadata.namespace` + `kind: Config` per namespace. |

---

## Part 4: `kind: System` -- The System Blueprint

### YAML Schema

```yaml
kind: System
metadata:
  name: customer-support-system
  description: Multi-agent customer support with research, analysis, and response
  namespace: prod

spec:
  # Shared ports -- available to all processes (unless caps restrict)
  ports:
    llm:
      adapter: hexdag.stdlib.adapters.openai.OpenAIAdapter
      config:
        model: gpt-4
        api_key: ${OPENAI_API_KEY}
    memory:
      adapter: hexdag.stdlib.adapters.memory.InMemoryMemory
    database:
      adapter: hexdag.stdlib.adapters.sqlite.SQLiteAdapter
      config:
        db_path: ./data/support.db

  # Process declarations
  processes:
    research-agent:
      exec: ./pipelines/research.yaml
      limits:
        max_llm_calls: 30
        max_cost_usd: 2.00
        max_tool_calls: 50
      caps: [llm, memory, tool:web_search, tool:doc_search]

    analysis-agent:
      exec: ./pipelines/analysis.yaml
      limits:
        max_llm_calls: 10
        max_cost_usd: 1.00
      caps: [llm, memory, datastore.read]

    response-agent:
      exec: ./pipelines/response.yaml
      limits:
        max_llm_calls: 5
        max_cost_usd: 0.50
      caps: [llm, memory]
      ports:
        llm:
          adapter: hexdag.stdlib.adapters.openai.OpenAIAdapter
          config:
            model: gpt-4o-mini  # Cheaper model for final response

  # Data flow -- directed pipes between processes
  pipes:
    - from: research-agent
      to: analysis-agent
      mapping:
        research_results: "{{ research-agent.researcher.output }}"
        raw_data: "{{ research-agent.data_collector.output }}"

    - from: analysis-agent
      to: response-agent
      mapping:
        analysis: "{{ analysis-agent.analyzer.output }}"
        sentiment: "{{ analysis-agent.sentiment.output }}"

  # Init (how the system starts)
  init:
    type: manual  # or: schedule, event, continuous
```

### Key Design Decisions

1. **Processes are references, not inline.** `exec: ./pipelines/research.yaml` keeps Pipeline YAML files independent and reusable.
2. **One new `kind`, not many.** `kind: System` handles topology, limits, caps, pipes, and init in one manifest.
3. **Shared ports with per-process overrides.** System-level `spec.ports` are inherited by all processes.
4. **Caps restrict, limits cap.** Caps = allowlist of ports/tools. Limits = numerical ceilings. Both enforced via port wrappers.
5. **Pipes define a DAG of processes.** Topological execution order with Jinja2 template mapping.
6. **Environment selection reuses existing pattern.** Multi-document YAML with `metadata.namespace`.

### What Stays Runtime (Not in YAML)

| Concern | Why runtime, not YAML |
|---|---|
| Dynamic routing decisions | CentralAgent uses LLM reasoning -- can't be predetermined |
| State tracking | ProcessRegistry tracks what's running -- emergent, not declared |
| Error recovery strategy | Depends on runtime failure modes |
| Resource accounting state | Counters that increment during execution |

---

## Part 5: Resource Limits

### Domain Model

**File:** `kernel/domain/resource_limits.py` (new)

```python
@dataclass(frozen=True, slots=True)
class ResourceLimits:
    max_total_tokens: int | None = None
    max_llm_calls: int | None = None
    max_tool_calls: int | None = None
    max_cost_usd: float | None = None
    warning_threshold: float = 0.8  # Emit warning at 80%
    on_limit: Literal["raise", "skip"] = "raise"
```

### Enforcement

**File:** `kernel/orchestration/resource_accounting.py` (new)

- `ResourceAccounting` class holds counters + limits
- `check_and_record_llm_call(tokens, cost)` -- increments, checks, raises `ResourceLimitExceeded`
- `check_and_record_tool_call()` -- same pattern
- Reads from `CostProfilerObserver` data (reuse, don't duplicate)

**Enforcement point:** Existing `ObservableLLMWrapper.aresponse()` and `ObservableToolRouterWrapper.acall_tool()`:

```python
# In ObservableLLMWrapper.aresponse():
if accounting := get_resource_accounting():
    accounting.check_llm_call()  # Raises ResourceLimitExceeded
# ... existing code ...
if accounting := get_resource_accounting():
    accounting.record_llm_call(usage.total_tokens, estimated_cost)
```

### Events

- `ResourceWarning(Event)`: pipeline_name, resource_type, current_value, limit_value, threshold
- `ResourceLimitExceeded(Event)`: pipeline_name, resource_type, current_value, limit_value

---

## Part 6: Caps (Capabilities)

### Domain Model

**File:** `kernel/domain/caps.py` (new)

```python
@dataclass(frozen=True, slots=True)
class CapSet:
    caps: frozenset[str]  # e.g., {"llm", "datastore.read", "tool:search_orders"}

    def allows(self, cap: str) -> bool: ...
    def intersect(self, child: CapSet) -> CapSet: ...  # Child <= parent
```

**Cap grammar:**
- `llm` -- access to LLM port
- `memory` / `datastore.read` / `datastore.write` -- data access
- `tool:<name>` -- specific tool access
- `spawner` -- can spawn child processes
- `secret` -- can read secrets
- No caps field = unrestricted (backward compatible)

### Enforcement

Port wrappers check caps before each call:

```python
# In ObservableLLMWrapper.aresponse():
if cap_set := get_cap_set():
    if not cap_set.allows("llm"):
        raise CapDenied("Process lacks 'llm' capability")
```

For tools: check `tool:<tool_name>` cap in `ObservableToolRouterWrapper.acall_tool()`.

### Child Process Inheritance

When `PipelineSpawner.aspawn()` creates a child, the child's caps = `parent.intersect(child_declared)`. A child can never have MORE caps than its parent.

---

## Part 7: Pipes (Data Flow Between Processes)

### How Pipes Work

Pipes define directed edges between processes in a System. The SystemRunner:

1. Parses the `pipes` section into a DAG of processes
2. Computes topological execution order
3. Executes processes in order
4. After each process completes, evaluates Jinja2 templates in downstream `mapping` to construct input data
5. Passes constructed input to the next process

### Domain Model

**File:** `kernel/domain/pipe.py` (new)

```python
@dataclass(frozen=True, slots=True)
class Pipe:
    from_process: str
    to_process: str
    mapping: dict[str, str]  # key -> Jinja2 template referencing upstream outputs
```

### Parallel Execution

Processes with no pipe dependencies execute in parallel (same principle as node waves within a pipeline). The SystemRunner computes waves of processes just as the Orchestrator computes waves of nodes.

---

## Part 8: Compilation & Execution Model

### New Components

| Component | Location | Role |
|---|---|---|
| `SystemConfig` | `compiler/system_config.py` | Pydantic model for parsed System manifest |
| `SystemBuilder` | `compiler/system_builder.py` | Parses System YAML -> `SystemConfig` |
| `SystemRunner` | `kernel/system_runner.py` | Executes a `SystemConfig` (peers with `PipelineRunner`) |
| `ResourceLimits` | `kernel/domain/resource_limits.py` | Frozen dataclass |
| `CapSet` | `kernel/domain/caps.py` | Frozen dataclass |
| `Pipe` | `kernel/domain/pipe.py` | Frozen dataclass |
| `ResourceAccounting` | `kernel/orchestration/resource_accounting.py` | Runtime counter + enforcement |

### Compilation Flow

```
System YAML
  -> SystemBuilder.build_from_yaml_string()
    -> Parse YAML (reuse yaml.safe_load)
    -> Select environment (reuse namespace pattern)
    -> Validate structure (new SystemValidator)
    -> For each process:
        -> Resolve exec path
        -> YamlPipelineBuilder.build_from_yaml_string()  # REUSE existing builder
        -> Attach limits + caps from System manifest
    -> Parse pipes into DAG
    -> Validate pipe DAG (no cycles, valid process refs)
    -> Return SystemConfig
```

### Execution Flow

```
SystemConfig
  -> SystemRunner.run(system_config, inputs)
    -> Instantiate shared ports (reuse ComponentInstantiator)
    -> Compute process execution waves from pipe DAG
    -> For each wave (sequential):
        -> For each process in wave (parallel):
            -> Create PipelineRunner with:
                - Shared ports (merged with per-process overrides)
                - ResourceAccounting (from process's limits)
                - CapSet (from process's caps)
            -> Evaluate Jinja2 input mapping from upstream results
            -> PipelineRunner.run_from_string(yaml, input_data)
            -> Collect results
    -> Return aggregated results
```

### Backward Compatibility

| Scenario | Behavior |
|---|---|
| Existing `kind: Pipeline` with no limits/caps | Works unchanged -- unlimited resources, all caps |
| `kind: Pipeline` with `spec.limits` (standalone) | PipelineRunner reads limits, creates ResourceAccounting |
| `kind: System` referencing existing Pipeline YAMLs | System-level limits/caps override pipeline-level |
| `PipelineRunner` used directly (Python API) | Optional `resource_limits` and `caps` params |

---

## Part 9: Runtime Architecture -- Library-First with Studio as One Host

### The Problem

hexDAG pipelines are currently **one-shot**: run, execute, exit. For `kind: System` to work with timed triggers, recurring schedules, and long-lived process management, hexDAG needs a persistent runtime -- like `systemd` manages services in Linux.

But critically: **hexDAG is a library first, a server second.** Any Python application (FastAPI, Django, a CLI tool, a Jupyter notebook) should be able to use the Scheduler, SystemRunner, and all kernel primitives directly -- without Studio or any hexDAG daemon running. Studio is just one deployment option, not a prerequisite.

### Design Principle: Two Layers

```
Layer 1: Kernel Runtime Primitives (library -- no server needed)
|-- SystemRunner          <- Execute kind: System as one-shot or long-lived
|-- PipelineRunner        <- Execute kind: Pipeline (existing)
|-- Scheduler             <- asyncio-based delayed/recurring execution (existing lib)
|-- ProcessRegistry       <- Track running processes (existing lib)
|-- ResourceAccounting    <- Enforce limits at runtime (new)
+-- CapSet                <- Enforce caps at runtime (new)

Layer 2: Hosts (optional -- provide HTTP/UI around Layer 1)
|-- hexdag studio         <- Studio: FastAPI + React UI + System dashboard
|-- External FastAPI app  <- User's own app importing hexdag as library
|-- hexdag CLI            <- One-shot CLI commands
+-- MCP Server            <- AI agent interface
```

**Layer 1 is pure Python with asyncio.** No HTTP, no server, no framework dependency. Any application with an async event loop can use it.

**Layer 2 hosts wrap Layer 1** with HTTP routes, UI, or CLI for convenience. Studio is one such host, but it's never required.

### Layer 1: Library Usage (External Application Integration)

An external application integrates with hexDAG the same way it uses any Python library -- import and call:

```python
# External FastAPI application using hexDAG as a library
from fastapi import FastAPI
from contextlib import asynccontextmanager

from hexdag import PipelineRunner
from hexdag.kernel.system_runner import SystemRunner
from hexdag.stdlib.lib.scheduler import Scheduler
from hexdag.stdlib.lib.process_registry import ProcessRegistry
from hexdag.drivers.pipeline_spawner import LocalPipelineSpawner

# Create hexDAG runtime primitives
spawner = LocalPipelineSpawner()
scheduler = Scheduler(spawner=spawner)
registry = ProcessRegistry()

@asynccontextmanager
async def lifespan(app):
    await scheduler.aschedule_recurring("health_check", interval_seconds=300)
    await scheduler.aschedule_recurring("data_sync", interval_seconds=3600)
    yield
    await scheduler.ateardown()

app = FastAPI(lifespan=lifespan)

@app.post("/run-pipeline")
async def run_pipeline(yaml_path: str, inputs: dict):
    runner = PipelineRunner()
    return await runner.run(yaml_path, input_data=inputs)

@app.post("/run-system")
async def run_system(system_yaml: str, inputs: dict):
    system_runner = SystemRunner()
    return await system_runner.run(system_yaml, input_data=inputs)
```

**Lifecycle contract:**
1. **Instantiate** -- Create runtime objects (`Scheduler`, `SystemRunner`, etc.)
2. **Use** -- Call async methods within your event loop
3. **Teardown** -- Call `ateardown()` on libs that hold asyncio tasks

### Layer 2: Studio as One Host

```
hexdag studio (FastAPI/Uvicorn) <- Built-in host for Layer 1
|-- Studio UI (React)           <- Visual dashboard
|-- REST API                    <- HTTP interface around Layer 1
|-- System Manager              <- Loads System manifests, supervises processes
|-- Scheduler instance          <- Timed triggers (Layer 1 primitive)
|-- Pipeline Execution          <- Already exists
+-- MCP Server                  <- AI agent interface
```

### Execution Modes (from `spec.init`)

| Mode | Behavior | Requires Persistent Loop? |
|---|---|---|
| `manual` | Run all processes once in topological order, exit when done | No -- one-shot via `SystemRunner.run()` |
| `schedule` | Re-execute on schedule | Yes -- Scheduler needs event loop (any host) |
| `event` | Listen for EventBus events to trigger | Yes -- EventBus listener (future) |
| `continuous` | Restart processes on completion | Yes -- supervisor loop |

### Phased Delivery

| Phase | What | Depends On |
|---|---|---|
| **Phase 1** (this plan) | `manual` mode only -- one-shot `SystemRunner.run()`, library-first API | Steps 0-10 |
| **Phase 2** (follow-up) | `schedule` mode + Studio System Manager + status API | Phase 1 |
| **Phase 3** (follow-up) | `continuous` mode + process supervision + restart policies | Phase 2 |
| **Phase 4** (follow-up) | `event` mode + EventBus integration | Phase 3 + EventBus |
| **Phase 5** (follow-up) | MCP tools for System management + `hexdag explain` CLI | Any phase |

---

## Implementation Order

Each step has a **kernel side** (domain models, enforcement, runtime) and a **compiler side** (YAML parsing, kind handlers, builder integration). The kernel never touches YAML -- it only knows the compiled domain objects.

### Step 0: Structural Refactor -- Move YAML Compiler Out of Kernel

**0a. Move `pipeline_builder/` -> `compiler/` -- COMPLETED**

Moved 15 source files, updated ~99 import references, created backward-compat wrapper with deprecation warning. All 2368 tests pass.

**0b. Move `kernel/config/loader.py` -> `compiler/config_loader.py` -- COMPLETED**

Moved config loader to `compiler/config_loader.py`, replaced original with deprecation wrapper, updated `kernel/config/__init__.py` (lazy `__getattr__`) and `kernel/discovery.py` imports. All 2440 tests pass.

**0c. Extend `HexDAGConfig` -- reuse `OrchestratorConfig` -- COMPLETED**

Added `DefaultLimits` and `DefaultCaps` frozen dataclasses to `kernel/config/models.py`. Extended `HexDAGConfig` with `orchestrator: OrchestratorConfig`, `limits: DefaultLimits`, `caps: DefaultCaps`. Updated config parser to handle `orchestrator`/`kernel`, `limits`, and `caps` TOML sections. All 2440 tests pass.

### Step 1: `kind: Config`

| File | Action | What |
|---|---|---|
| `kernel/config/models.py` | ~~Edit~~ | ~~Add `DefaultLimits`, `DefaultCaps` dataclasses~~ (done in Step 0c) |
| `compiler/config_loader.py` | Edit | Add YAML parsing for `kind: Config` |
| `compiler/plugins/config_definition.py` | New | `EntityPlugin` for `kind: Config` |

### Step 2: Resource Limits

| File | Action | What |
|---|---|---|
| `kernel/domain/resource_limits.py` | New | `ResourceLimits` frozen dataclass |
| `kernel/orchestration/resource_accounting.py` | New | Runtime counters + enforcement |
| `kernel/orchestration/port_wrappers.py` | Edit | Add pre/post-call checks |
| `kernel/orchestration/events/events.py` | Edit | Add `ResourceWarning`, `ResourceLimitExceeded` |
| `kernel/context/execution_context.py` | Edit | Add `_resource_accounting` ContextVar |
| `kernel/exceptions.py` | Edit | Add `ResourceLimitExceeded` exception |

### Step 3: Caps

| File | Action | What |
|---|---|---|
| `kernel/domain/caps.py` | New | `CapSet` with `allows()`, `intersect()` |
| `kernel/orchestration/port_wrappers.py` | Edit | Add `_check_cap()` |
| `kernel/orchestration/events/events.py` | Edit | Add `CapDenied` event |
| `kernel/context/execution_context.py` | Edit | Add `_cap_set` ContextVar |
| `kernel/exceptions.py` | Edit | Add `CapDenied` exception |

### Step 4: Pipeline Integration

| File | Action | What |
|---|---|---|
| `kernel/pipeline_runner.py` | Edit | Accept `resource_limits` and `caps` params |
| `kernel/orchestration/orchestrator.py` | Edit | Set ContextVars before execution |
| `compiler/pipeline_config.py` | Edit | Add `limits` and `caps` fields |
| `compiler/yaml_builder.py` | Edit | Parse `spec.limits` and `spec.caps` |

### Step 5: `kind: Adapter`

| File | Action | What |
|---|---|---|
| `compiler/plugins/adapter_definition.py` | New | `EntityPlugin` for `kind: Adapter` |
| `compiler/yaml_builder.py` | Edit | Register adapter plugin, resolve `{ ref: name }` |

### Step 6: `kind: Policy`

| File | Action | What |
|---|---|---|
| `kernel/domain/policy.py` | New | Base `Policy` dataclass |
| `compiler/plugins/policy_definition.py` | New | `EntityPlugin` for `kind: Policy` |
| `compiler/yaml_builder.py` | Edit | Register policy plugin |

### Step 7: Pipes Domain Model

| File | Action | What |
|---|---|---|
| `kernel/domain/pipe.py` | New | `Pipe` frozen dataclass |

### Step 8: `kind: System` + SystemBuilder

| File | Action | What |
|---|---|---|
| `compiler/system_config.py` | New | `SystemConfig` Pydantic model |
| `compiler/system_builder.py` | New | Parse System YAML -> `SystemConfig` |
| `compiler/system_validator.py` | New | Validate System manifest |

### Step 9: SystemRunner

| File | Action | What |
|---|---|---|
| `kernel/system_runner.py` | New | Execute `SystemConfig` via `PipelineRunner` per process |

### Step 10: Tests

| Area | What |
|---|---|
| `tests/hexdag/kernel/domain/` | Unit tests for `ResourceLimits`, `CapSet`, `Pipe`, `Policy` |
| `tests/hexdag/kernel/orchestration/` | Tests for `ResourceAccounting`, port wrapper enforcement |
| `tests/hexdag/compiler/` | Tests for new kind plugins, SystemBuilder, ref resolution |
| `tests/hexdag/kernel/` | Integration tests for SystemRunner end-to-end |
| `tests/hexdag/kernel/config/` | Tests for config loader migration |

---

## Verification

1. **Step 0 verification**: `uv run pytest` -- all existing tests pass after moves
2. **Lint** -- `uv run ruff check hexdag/`
3. **Type check** -- `uv run pyright ./hexdag`
4. **Config kind test**: `kind: Config` YAML with `spec.limits` -> verify defaults load and propagate
5. **Integration test**: Pipeline YAML with `spec.limits` -> verify enforcement stops at limit
6. **Integration test**: Pipeline with `spec.caps` -> verify `CapDenied` raised for disallowed ports
7. **Integration test**: System YAML with 3 processes + pipes -> verify topological execution and data flow
8. **Backward compatibility**: Run existing examples and tests unchanged

---

## Auto-Documentation: `hexdag explain`

Users need to know what fields are allowed for each `kind`. Solution: `hexdag explain` (like `kubectl explain`).

```bash
$ hexdag explain System
KIND:     System
DESCRIPTION:
     Multi-process topology: which processes run together,
     how they connect via pipes, per-process overrides.

FIELDS:
   metadata     <ObjectMeta>    Standard metadata (name, namespace, description)
   spec         <SystemSpec>    System specification

$ hexdag explain System.spec.processes
KIND:     System
FIELD:    processes <map[string, ProcessSpec]>
DESCRIPTION:
     Process declarations. Each key is a process name.

FIELDS:
   exec         <string>        Path to Pipeline YAML file
   limits       <Limits>        Resource limits (max_llm_calls, max_cost_usd, etc.)
   caps         <list[string]>  Capability allowlist (llm, memory, tool:*, etc.)
   ports        <map>           Per-process port overrides
```

**Implementation:** Reuse `SchemaGenerator` + Pydantic `model_json_schema()`. Generate JSON Schema files per kind for IDE YAML LSP autocomplete.
