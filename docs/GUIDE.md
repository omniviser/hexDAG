# hexDAG Framework Guide

> **Operations and knowledge as code.** hexDAG is an operating system for AI agents — a platform where companies encode how they operate and what they know into executable, version-controlled, composable code.
>
> Designed to be read by developers joining the team and by AI agents exploring the codebase.

**Related docs:** [ARCHITECTURE.md](ARCHITECTURE.md) (design philosophy) · [ROADMAP.md](ROADMAP.md) (strategic direction) · [quickstart.md](quickstart.md) (first pipeline)

---

## Table of Contents

1. [What hexDAG Is](#1-what-hexdag-is) — Operations and knowledge as code
2. [The Integration Model](#2-the-integration-model) — How all concepts work together
3. [The Layer Model](#3-the-layer-model) — Kernel, compiler, stdlib, drivers
4. [How a Pipeline Runs](#4-how-a-pipeline-runs-end-to-end) — End-to-end execution flow
5. [YAML Syntax](#5-yaml-syntax-the-4-special-syntaxes) — The 4 special syntaxes
6. [The Uniform Entity Pattern](#6-the-uniform-entity-pattern) — Nodes, adapters, macros, services
7. [Ports, Adapters, and Middleware](#7-ports-adapters-and-middleware) — Contracts and implementations
8. [Node Types](#8-node-types) — Processing steps in the DAG
9. [Data Flow Between Nodes](#9-data-flow-between-nodes) — input_mapping, expressions, templates
10. [Services](#10-services) — @tool and @step decorators
11. [Macros vs Nodes](#11-macros-vs-nodes) — Single step vs template expansion
12. [Entity Lifecycle](#12-entity-lifecycle-state-machines) — State machines for business objects
13. [The Compiler](#13-the-compiler-in-detail) — YAML to kernel objects
14. [The Orchestrator](#14-the-orchestrator-in-detail) — Wave-based execution engine
15. [File Index](#15-file-index-where-to-find-things) — Where to find things

---

## 1. What hexDAG Is

hexDAG enables **operations and knowledge as code** — companies encode how they operate (workflows, decisions, agent strategies, process rules) and what they know (entity states, business rules, transition logic, memory) into executable, version-controlled code.

### Operations as Code

- **Processes** — workflows defined in YAML, compiled into DAGs, executed with concurrency
- **Agents** — autonomous reasoning strategies (ReAct, conversation, planner)
- **Scheduling** — delayed and recurring execution (`cron` for operations)
- **System orchestration** — multi-pipeline coordination, lifecycle management

### Knowledge as Code

- **Entities** — business objects with states, transitions, and enforcement rules
- **Memory** — per-run state, conversation history, shared context
- **Expressions** — business rules encoded as safe evaluable logic
- **Configuration** — environment management, secrets, capability narrowing

### The Linux Analogy

The architecture mirrors Linux — this isn't just marketing, it's the actual structure:

| Linux | hexDAG | What it does |
|-------|--------|-------------|
| **Kernel** (scheduler, VFS, syscalls) | [`hexdag/kernel/`](../hexdag/kernel/) | DAG orchestrator, port protocols, event system |
| **Kernel module interfaces** | `agent_base`, `entity_base` | How agents and entities plug into the kernel |
| **Loadable modules** (ext4, e1000) | [`hexdag/stdlib/`](../hexdag/stdlib/) agents, entities, nodes | ReAct agent, state machines, LLM nodes |
| **glibc** (`/lib/`) | [`hexdag/stdlib/nodes/`](../hexdag/stdlib/nodes/) | LLMNode, FunctionNode — every pipeline uses these |
| **Daemons** (systemd, crond) | Services (ProcessRegistry, Scheduler) | Track runs, schedule execution |
| **Device drivers** | [`hexdag/stdlib/adapters/`](../hexdag/stdlib/adapters/) | OpenAI, Postgres, Mock — implement port protocols |
| **I/O schedulers** | [`hexdag/stdlib/middleware/`](../hexdag/stdlib/middleware/) | Retry, timeout, cache — transparent port wrappers |
| **`/drivers/`** | [`hexdag/drivers/`](../hexdag/drivers/) | LocalExecutor, VFS providers, ObserverManager |
| **`gcc`** | [`hexdag/compiler/`](../hexdag/compiler/) | YAML manifests → kernel domain models |
| **`/usr/bin`** | [`hexdag/api/`](../hexdag/api/) | User-facing functions |
| **Shell** | [`hexdag/cli/`](../hexdag/cli/) | Command-line interface |
| **Signals** | Events | NodeStarted, StateTransitionEvent, PipelineCompleted |

---

## 2. The Integration Model

These concepts aren't independent modules — they're ONE tightly integrated system. A real company operation flows through all of them:

```
1. Agent receives a customer request                            → AGENTS
2. Agent reasons about it, decides to create an order           → AGENTS
3. Agent transitions the order to "processing"                  → ENTITIES
4. Transition emits StateTransitionEvent                        → EVENTS
5. Event triggers a fulfillment pipeline                        → PROCESSES
6. Pipeline uses memory to carry order context between nodes    → MEMORY
7. Expressions encode: "if total > 1000, require_approval"      → EXPRESSIONS
8. Capabilities restrict which agents can transition what       → SECURITY
9. Observers track everything for audit trail                   → OBSERVABILITY
10. Fulfillment completes → entity transitions to "shipped"     → ENTITIES
11. "shipped" triggers notification pipeline                    → PROCESSES
```

### The Connective Tissue

- **Events** are the glue — an entity transition emits an event that a scheduler responds to
- **Services (@tool/@step)** are the unified interface — agents interact with entities, memory, and processes through the same mechanism
- **Capabilities (CapSet)** are the security model — they scope what each agent/process can do
- **VAS** is the introspection layer — agents inspect the system through a uniform path-based namespace (`/proc/`, `/lib/`, `/sys/`)

---

## 3. The Layer Model

hexDAG has 6 layers. Data flows top to bottom. Ports sit alongside because every layer can reach them.

```
┌─────────────────────────────────────────┐
│  CLI & API                              │  User-facing surface
│  hexdag validate · hexdag run · API     │  hexdag/cli/, hexdag/api/
└────────────────────┬────────────────────┘
                     │ calls
┌────────────────────▼────────────────────┐
│  Compiler                               │  YAML → kernel objects
│  YamlPipelineBuilder · Validator        │  hexdag/compiler/
│  Resolver · ReferenceResolver           │
└────────────────────┬────────────────────┘
                     │ outputs: DirectedGraph + PipelineConfig
┌────────────────────▼────────────────────┐    ┌─────────────────────┐
│  Kernel                                 │    │  Ports & Adapters   │
│  Orchestrator · PipelineRunner          │◄──►│  LLM · DataStore    │
│  DirectedGraph · NodeSpec               │    │  Database · Memory   │
│  Events · ExecutionContext              │    │  + Middleware        │
└────────────────────┬────────────────────┘    │  + Services         │
                     │ uses                     └─────────────────────┘
┌────────────────────▼────────────────────┐
│  Standard Library                       │  Built-in components
│  Nodes · Adapters · Macros · Middleware  │  hexdag/stdlib/
└────────────────────┬────────────────────┘
                     │ backed by
┌────────────────────▼────────────────────┐
│  Drivers                                │  Infrastructure
│  LocalExecutor · ObserverManager · VFS  │  hexdag/drivers/
└─────────────────────────────────────────┘
```

**Key insight:** The Kernel never imports from Stdlib. Stdlib implements contracts defined by the Kernel. This is the hexagonal architecture — business logic and infrastructure are cleanly separated.

---

## 4. How a Pipeline Runs (End to End)

This is the journey from a YAML file to execution results:

```
1. You write a YAML file (kind: Pipeline)
                │
2. PipelineRunner receives it
                │
3. Compiler: YamlPipelineBuilder processes YAML
   ├── Phase 1: Parse YAML, resolve !include tags
   ├── Phase 2: Select environment (dev/staging/prod)
   ├── Phase 3: Validate schema, types, detect cycles
   ├── Phase 4: Preprocess — substitute ${VAR}, render {{ templates }}
   └── Phase 5: Build — instantiate nodes, expand macros, auto-detect dependencies
                │
4. Output: DirectedGraph (nodes + edges) + PipelineConfig (ports, policies)
                │
5. OrchestratorFactory instantiates ports (adapters) from config
                │
6. Orchestrator walks the DAG:
   ├── Compute waves (groups of nodes with no mutual dependencies)
   ├── For each wave:
   │   ├── Run all nodes in the wave concurrently (asyncio.gather)
   │   ├── For each node:
   │   │   ├── Check `when` condition (skip if false)
   │   │   ├── Prepare inputs (upstream results + input_mapping)
   │   │   ├── Validate input against Pydantic schema
   │   │   ├── Execute async function (with timeout)
   │   │   ├── Validate output against Pydantic schema
   │   │   └── On failure: retry with exponential backoff
   │   └── Emit events: NodeStarted → NodeCompleted/NodeFailed
   └── Emit: WaveCompleted → PipelineCompleted
                │
7. Results: dict of {node_name: output} for every node
```

**In Python:**
```python
from hexdag.kernel.pipeline_runner import PipelineRunner

runner = PipelineRunner(port_overrides={"llm": MockLLM()})
results = await runner.run("my_pipeline.yaml", input_data={"query": "hello"})
# results == {"analyzer": {"result": "..."}, "summarizer": {"summary": "..."}}
```

---

## 5. YAML Syntax — The 4 Special Syntaxes

hexDAG YAML has 4 special syntaxes. They look similar but do different things at different times. This is the most common source of confusion.

### Overview

| Syntax | Purpose | When | Example |
|--------|---------|------|---------|
| `!include` | Merge another YAML file | Build-time, phase 1 (first) | `!include shared/ports.yaml` |
| `${VAR}` | Environment variable | Build-time, phase 4 (secrets deferred to runtime) | `${OPENAI_API_KEY}` |
| `{{ expr }}` | Jinja2 template | Build-time for metadata; **runtime** for node specs | `{{ analyzer.result }}` |
| `node.field` | Value extraction | Runtime, in input_mapping and expressions | `"analyzer.result"` |

### `!include` — File Composition

Merges another YAML file inline. Processed first, before anything else.

```yaml
# main.yaml
spec:
  ports:
    "!include": shared/ports.yaml              # entire file merged here
  nodes:
    - "!include": nodes/analysis.yaml#nodes    # specific anchor
```

- Supports `!include path.yaml#anchor` to pick a section
- Circular references detected and blocked
- Located in: [`hexdag/compiler/preprocessing/include.py`](../hexdag/compiler/preprocessing/include.py)

### `${VAR}` — Environment Variables

Substitutes environment variable values at build time.

```yaml
spec:
  ports:
    llm:
      config:
        api_key: ${OPENAI_API_KEY}    # deferred to runtime (secret pattern)
        base_url: ${LLM_BASE_URL}     # resolved at build time
```

- Secret-pattern names (`*_API_KEY`, `*_SECRET`) are **deferred** to runtime
- Other variables resolved immediately during preprocessing
- Located in: [`hexdag/compiler/yaml_builder.py`](../hexdag/compiler/yaml_builder.py) (EnvironmentVariablePlugin)

### `{{ expr }}` — Jinja2 Templates

Renders data into text. Timing depends on where it appears:

```yaml
metadata:
  name: {{ env }}_pipeline           # build-time (metadata)

spec:
  nodes:
    - kind: llm_node
      spec:
        prompt: "Analyze {{ analyzer.result }}"   # RUNTIME (node spec)
```

- **Build-time:** metadata, top-level fields — uses build-time variables
- **Runtime:** anything inside node `spec:` — uses actual upstream node outputs
- Located in: [`hexdag/compiler/yaml_builder.py`](../hexdag/compiler/yaml_builder.py) (TemplatePlugin) and [`hexdag/kernel/orchestration/prompt/`](../hexdag/kernel/orchestration/prompt/)

### `node.field` — Value Extraction

Extracts a value from an upstream node's output. **Not a template** — this is direct object access.

```yaml
spec:
  input_mapping:
    data: "analyzer.result"         # passes the actual Python object
    query: "$input.user_query"      # from pipeline input
  expressions:
    total: "analyzer.count * 2"     # used in expression evaluation
    ok: "analyzer.status == 'done'" # boolean expression
```

- Used in: `input_mapping` values, `expressions`, `when` conditions
- `$input.field` reads from pipeline input
- `$ctx.run_id` reads execution context
- `state.counter` reads loop iteration state
- `memory('key')` reads PipelineMemory
- Located in: [`hexdag/compiler/reference_resolver.py`](../hexdag/compiler/reference_resolver.py), [`hexdag/kernel/expression_parser.py`](../hexdag/kernel/expression_parser.py)

### The Critical Difference

```yaml
# TEMPLATE — renders text (Jinja2, becomes a string)
prompt: "Analyze {{ analyzer.result }}"

# EXTRACTION — passes the actual object (dict, list, number, etc.)
input_mapping:
  data: "analyzer.result"
```

These look similar but are completely different operations. Templates produce strings. Extraction passes the raw Python object.

---

## 6. The Uniform Entity Pattern

Nodes, adapters, macros, and services are all "entities." They follow the same pattern:

1. **Kernel defines a base class** (the contract)
2. **Stdlib ships built-in implementations**
3. **Users write their own** by subclassing the same base
4. **YAML references by alias** or full module path
5. **Resolver maps alias to class** at build time

| Entity | Base Class | Registers Via | Produces | YAML Reference |
|--------|-----------|---------------|----------|----------------|
| **Node** | `BaseNodeFactory` | `yaml_alias="llm_node"` | 1 NodeSpec (single step) | `kind: llm_node` |
| **Adapter** | `HexDAGAdapter` | `yaml_alias` + `port` | Port implementation | `adapter: openai` |
| **Macro** | `ConfigurableMacro` | `yaml_alias` or YAML `kind: Macro` | Sub-graph (N nodes) | `kind: macro_name` |
| **Service** | `Service` | Full module path (no alias) | @tool and @step methods | `class: myapp.OrderService` |

### How Registration Works

All entities use `__init_subclass__` to auto-register:

```python
# When you write this...
class MyNode(BaseNodeFactory, yaml_alias="my_node"):
    def __call__(self, name, **kwargs):
        return NodeSpec(name=name, fn=my_async_fn)

# ...it auto-registers "my_node" → MyNode in the alias registry.
# Now YAML can say: kind: my_node
```

### How Resolution Works

The Resolver ([`hexdag/kernel/resolver.py`](../hexdag/kernel/resolver.py)) checks in order:

1. Runtime components (YAML-defined macros)
2. User-registered aliases
3. Built-in aliases (auto-discovered from stdlib at bootstrap)
4. `ConfigurableMacro` registry
5. Full module path fallback (`myapp.nodes.MyNode`)

Located in: [`hexdag/kernel/resolver.py`](../hexdag/kernel/resolver.py), [`hexdag/kernel/_alias_registry.py`](../hexdag/kernel/_alias_registry.py)

---

## 7. Ports, Adapters, and Middleware

### The Relationship

**Port** = abstract contract. "I need something that can generate text."
**Supports*** = fine-grained capability. "Can it also call tools?"
**Adapter** = concrete implementation. "Use OpenAI GPT-4."
**Middleware** = transparent wrapper. "Add retry + cache."

```
Node ──requires──► Port (contract)
                     ▲
                     │ implements
                     │
                  Adapter (OpenAI / Anthropic / Mock)
                     ▲
                     │ wraps
                     │
                  Middleware (Retry → Timeout → Cache)
```

### Ports and Supports* Protocols

Each port is split into fine-grained `Supports*` sub-protocols:

| Port | Sub-protocols | Located in |
|------|---------------|-----------|
| **LLM** | `SupportsGeneration`, `SupportsFunctionCalling`, `SupportsEmbedding`, `SupportsVision`, `SupportsStructuredOutput`, `SupportsUsageTracking` | [`kernel/ports/llm.py`](../hexdag/kernel/ports/llm.py) |
| **DataStore** | `SupportsKeyValue`, `SupportsQuery`, `SupportsTTL`, `SupportsTransactions`, `SupportsCollectionStorage` | [`kernel/ports/data_store.py`](../hexdag/kernel/ports/data_store.py) |
| **Database** | `SupportsQuery`, `SupportsRawSQL`, `SupportsTransactions`, `SupportsSchema`, `SupportsStreamingQuery` | [`kernel/ports/database.py`](../hexdag/kernel/ports/database.py) |
| **Memory** | `aappend`, `aget`, `aclear` | [`kernel/ports/memory.py`](../hexdag/kernel/ports/memory.py) |
| **Other** | APICall, FileStorage, SecretStore, ToolRouter, Executor, VFS, PipelineSpawner, ObserverManager, VectorSearch | `kernel/ports/` |

**Naming convention:**
- Plain noun = port type: `LLM`, `DataStore`, `Database`
- `Supports*` = capability check: `SupportsGeneration`, `SupportsKeyValue`
- Rule of thumb: "What kind of port?" → plain noun. "Can this adapter do X?" → `Supports*`

### How Nodes Declare Port Needs

```python
class LLMNode(BaseNodeFactory, yaml_alias="llm_node"):
    _hexdag_port_capabilities: ClassVar[dict[str, list[type]]] = {
        "llm": [SupportsGeneration, SupportsFunctionCalling]
    }
```

The orchestrator validates at startup that the configured adapter implements all required `Supports*` protocols. If OpenAI provides `SupportsGeneration + SupportsFunctionCalling` but your node needs `SupportsEmbedding`, you get a clear error.

### Built-in Adapters

| Adapter | Port | Capabilities | Located in |
|---------|------|-------------|-----------|
| `OpenAILLM` | LLM | Generation, FunctionCalling, Vision, Embedding, StructuredOutput | [`stdlib/adapters/openai/`](../hexdag/stdlib/adapters/openai/) |
| `AnthropicLLM` | LLM | Generation, FunctionCalling, Vision | [`stdlib/adapters/anthropic/`](../hexdag/stdlib/adapters/anthropic/) |
| `MockLLM` | LLM | Generation (predefined responses) | [`stdlib/adapters/mock/`](../hexdag/stdlib/adapters/mock/) |
| `PostgresAdapter` | Database | Query, Transactions, Schema | [`stdlib/adapters/database/`](../hexdag/stdlib/adapters/database/) |
| `SQLiteAdapter` | Database | Query, Schema | [`stdlib/adapters/database/`](../hexdag/stdlib/adapters/database/) |
| `InMemoryMemory` | Memory | Full protocol | [`stdlib/adapters/memory/`](../hexdag/stdlib/adapters/memory/) |
| `RedisMemory` | Memory | Full protocol | [`stdlib/adapters/redis/`](../hexdag/stdlib/adapters/redis/) |
| `MockDatabaseAdapter` | Database | In-memory | [`stdlib/adapters/mock/`](../hexdag/stdlib/adapters/mock/) |

### Port Override Levels (3 tiers)

```yaml
spec:
  # Level 1: Global — applies to ALL nodes
  ports:
    llm:
      adapter: openai
      config: { model: gpt-4 }

  # Level 2: Per-type — overrides global for all llm_nodes
  type_ports:
    llm_node:
      llm:
        config: { temperature: 0.3 }

  # Level 3: Per-node — overrides everything for one specific node
  node_ports:
    creative_writer:
      llm:
        adapter: anthropic
        config: { model: claude-sonnet }
```

### Middleware

Middleware wraps adapters transparently. Nodes don't know middleware exists.

```yaml
# In pipeline config (planned: spec.ports.llm.middleware)
# Currently configured in Python:
from hexdag.stdlib.middleware import compose, RetryMiddleware, TimeoutMiddleware

llm = compose(
    RetryMiddleware(max_retries=3),
    TimeoutMiddleware(timeout=30),
)(OpenAILLM(model="gpt-4"))
```

Available middleware:

| Middleware | Purpose | Located in |
|-----------|---------|-----------|
| `RetryMiddleware` | Exponential backoff retry | [`stdlib/middleware/retry.py`](../hexdag/stdlib/middleware/retry.py) |
| `TimeoutMiddleware` | Per-call timeout | [`stdlib/middleware/timeout.py`](../hexdag/stdlib/middleware/timeout.py) |
| `RateLimiterMiddleware` | Token bucket rate limiting | [`stdlib/middleware/rate_limiter.py`](../hexdag/stdlib/middleware/rate_limiter.py) |
| `ResponseCacheMiddleware` | Response memoization | [`stdlib/middleware/response_cache.py`](../hexdag/stdlib/middleware/response_cache.py) |
| `StructuredOutputMiddleware` | JSON schema enforcement | [`stdlib/middleware/structured_output.py`](../hexdag/stdlib/middleware/structured_output.py) |
| `ObservableMiddleware` | Event tracking for port calls | [`stdlib/middleware/observable.py`](../hexdag/stdlib/middleware/observable.py) |

`compose()` stacks them: `compose(retry, timeout, cache)(adapter)`.

---

## 8. Node Types

Every node is a factory that produces a `NodeSpec`. The `NodeSpec` wraps an async function with metadata (schemas, dependencies, retry config).

### Built-in Nodes

| Node | YAML `kind` | Purpose | Requires Port? |
|------|-------------|---------|---------------|
| **LLMNode** | `llm_node` | Send prompts to LLM, get responses | Yes: LLM (SupportsGeneration) |
| **AgentNode** | `agent_node` / `react_agent` | ReAct reasoning loop with tools | Yes: LLM (Generation + FunctionCalling) |
| **FunctionNode** | `function_node` | Run any Python function | No |
| **ExpressionNode** | `expression_node` | Compute values from upstream data | No |
| **TransitionNode** | `transition` | Move entity through state machine | No (uses EntityState service) |
| **CompositeNode** | `composite` | Embed a sub-DAG | No |
| **DataNode** | `data_node` | Output static values | No |
| **ApiCallNode** | `api_call` | HTTP requests with URL templating | Yes: APICall |
| **ServiceCallNode** | `service_call` | Invoke @step methods on services | No |
| **ToolCallNode** | `tool_call` | Invoke @tool methods on services | No |
| **CheckpointNode** | `checkpoint` | Save execution state | No |
| **MappedInput** | `mapped_input` | Iterate over lists (for_each) | No |

### LLMNode — Talk to AI Models

```yaml
- kind: llm_node
  metadata:
    name: analyzer
  spec:
    system_message: "You are an analyst."
    human_message: "Analyze: {{ $input.topic }}"
    output_schema:           # optional: enforce JSON structure
      sentiment: str
      confidence: float
```

- Prompt templating with `{{ var }}` (Jinja2, rendered at runtime)
- Few-shot examples via `examples` field
- Structured output via `output_schema` (uses SupportsStructuredOutput)
- Conversation history from upstream nodes
- Located in: [`hexdag/stdlib/nodes/llm_node.py`](../hexdag/stdlib/nodes/llm_node.py)

### AgentNode — ReAct Reasoning

```yaml
- kind: agent_node
  metadata:
    name: researcher
  spec:
    initial_prompt: "Research {{ $input.topic }}"
    max_steps: 10
    entities: [ticket]       # opt-in to state machine tools
```

- Multi-step think → act → observe loop
- Calls tools via ToolRouter (including @tool methods from services)
- Phase transitions: main → refinement → response
- `entities` field opts into state machine tools for those entity types
- Located in: [`hexdag/stdlib/nodes/agent_node.py`](../hexdag/stdlib/nodes/agent_node.py)

### FunctionNode — Run Python Code

```yaml
- kind: function_node
  metadata:
    name: process
  spec:
    fn: "myapp.utils.process_data"    # module path — no import needed
    input_mapping:
      raw_data: "analyzer.result"
    unpack_input: true                # spread dict into function params
```

- `fn` is a module path string — fully declarative, no Python imports in YAML
- Input schema inferred from type hints
- `unpack_input: true` spreads the dict into function keyword arguments
- Located in: [`hexdag/stdlib/nodes/function_node.py`](../hexdag/stdlib/nodes/function_node.py)

### ExpressionNode — Compute Values

```yaml
- kind: expression_node
  metadata:
    name: compute
  spec:
    expressions:
      total: "analyzer.count * 2"
      rate: "normalize.rate_low"
      valid: "analyzer.status in ['active', 'pending']"
    output_fields: [total, rate, valid]
```

- Evaluates safe Python-like expressions (AST-validated, no `eval()`)
- References upstream nodes directly — no input_mapping needed
- Located in: [`hexdag/stdlib/nodes/expression_node.py`](../hexdag/stdlib/nodes/expression_node.py), [`hexdag/kernel/expression_parser.py`](../hexdag/kernel/expression_parser.py)

---

## 9. Data Flow Between Nodes

hexDAG uses an n8n-like data flow model. Upstream outputs are automatically available downstream. There are 3 mechanisms:

### Mechanism 1: `input_mapping` — Wire Fields Explicitly

Maps specific upstream output fields to node input parameters.

```yaml
spec:
  input_mapping:
    data: "analyzer.result"          # upstream node output
    query: "$input.user_query"       # pipeline input
```

- Syntax: `"node_name.field.subfield"` (raw path, no wrapping)
- **Required for:** `function_node`, `service_call` (parameter names matter)
- **Optional for:** `llm_node`, `expression_node` (they have their own access patterns)
- Located in: [`hexdag/stdlib/nodes/mapped_input.py`](../hexdag/stdlib/nodes/mapped_input.py)

### Mechanism 2: `expressions` — Compute Values

Evaluate safe Python-like expressions using upstream data.

```yaml
spec:
  expressions:
    total: "analyzer.count * 2"
    ok: "analyzer.status in ['active', 'pending']"
    pay: "coalesce(negotiation.target_pay, load.target_pay)"
```

- AST-validated — no `eval()`, only whitelisted operations
- 100+ built-in functions: `coalesce`, `default`, `isnone`, `isempty`, `len`, `min`, `max`, `sum`, `abs`, `round`, `str`, `int`, `float`, `bool`, `now`, `utcnow`, `upper`, `lower`, `split`, `join`, etc.
- Located in: [`hexdag/kernel/expression_parser.py`](../hexdag/kernel/expression_parser.py)

### Mechanism 3: `{{ templates }}` — Build Text

Jinja2 rendering at runtime, producing strings.

```yaml
spec:
  human_message: "Analyze {{ analyzer.result }} for topic {{ $input.topic }}"
```

- Used for: LLM prompts, messages, any text that includes data
- Rendered **after** upstream nodes complete

### Auto-Dependency Detection

**You do not manually list dependencies.** The compiler (`ReferenceResolver`) scans all three mechanisms — input_mapping, expressions, and templates — and auto-detects which nodes are referenced.

```yaml
nodes:
  - kind: llm_node
    name: analyzer               # no dependencies field needed!

  - kind: expression_node
    name: compute
    spec:
      expressions:
        total: "analyzer.count * 2"    # compiler sees "analyzer" → adds dependency

  - kind: llm_node
    name: summarizer
    spec:
      prompt: "Summarize: {{ compute.total }}"  # compiler sees "compute" → adds dependency
```

Located in: [`hexdag/compiler/reference_resolver.py`](../hexdag/compiler/reference_resolver.py)

### MISSING Sentinel — Typo Safety

- `typo_node.field` (unknown node name) → **BUILD ERROR** with "did you mean?" suggestion
- `real_node.typo_field` (unknown field on known node) → `None` at runtime
- This prevents silent data loss from misspelled node names

### Safe Path Modifiers

```yaml
field | required        # fail at runtime if value is None
field | default('x')    # use 'x' if value is None
```

### Expression Namespaces

| Namespace | Access Pattern | Purpose |
|-----------|---------------|---------|
| Upstream nodes | `node_name.field.subfield` | Output of completed nodes |
| Pipeline input | `$input.field` or `input.field` | Initial input data |
| Execution context | `$ctx.run_id`, `$ctx.pipeline_name` | Runtime metadata |
| Loop state | `state.counter`, `state.item` | Current iteration in LoopNode |
| Pipeline memory | `memory('key')`, `memory('key', default)` | Shared key-value store |

---

## 10. Services

Services are the unified abstraction for port-backed operations. They replace the deprecated `PortCallNode` and `HexDAGLib`.

### Creating a Service

```python
from hexdag.kernel.service import Service, tool, step

class OrderService(Service):
    def __init__(self, store: SupportsKeyValue) -> None:
        self._store = store

    @tool
    async def get_order(self, order_id: str) -> dict:
        """Get order by ID. Agent-callable during ReAct reasoning."""
        return await self._store.aget(f"order:{order_id}")

    @step
    async def save_order(self, order_id: str, data: dict) -> dict:
        """Persist an order. Usable as a DAG node via ServiceCallNode."""
        await self._store.aset(f"order:{order_id}", data)
        return {"saved": True}

    @tool
    @step
    async def validate_order(self, order_id: str) -> dict:
        """Both agent tool and DAG step."""
        ...
```

- `@tool` — agent-callable during ReAct reasoning (auto-generates OpenAI-compatible tool schemas)
- `@step` — deterministic DAG node (invoked by `ServiceCallNode`)
- Both can be stacked on the same method
- `asetup()` / `ateardown()` — lifecycle hooks, called before/after pipeline

### Built-in Services

| Service | Purpose | Linux Analogy | Located in |
|---------|---------|--------------|-----------|
| `ProcessRegistry` | Track pipeline runs | `ps` | [`stdlib/lib/process_registry.py`](../hexdag/stdlib/lib/process_registry.py) |
| `EntityState` | Declarative state machines | — | [`stdlib/lib/entity_state.py`](../hexdag/stdlib/lib/entity_state.py) |
| `Scheduler` | Delayed/recurring execution | `cron` | [`stdlib/lib/scheduler.py`](../hexdag/stdlib/lib/scheduler.py) |
| `PipelineMemory` | Run-scoped key-value store | — | [`stdlib/lib/pipeline_memory.py`](../hexdag/stdlib/lib/pipeline_memory.py) |
| `DatabaseTools` | Agent-callable SQL queries | — | [`stdlib/lib/database_tools.py`](../hexdag/stdlib/lib/database_tools.py) |
| `VFSTools` | Virtual filesystem introspection | — | [`stdlib/lib/vfs_tools.py`](../hexdag/stdlib/lib/vfs_tools.py) |

Located in: [`hexdag/kernel/service.py`](../hexdag/kernel/service.py) (base class)

---

## 11. Macros vs Nodes

This is a common source of confusion.

**A Node** produces **1 NodeSpec** — a single processing step.
**A Macro** expands into **a sub-graph** — multiple nodes with internal dependencies.

| | Node | Macro |
|---|------|-------|
| **YAML** | `kind: llm_node` | `kind: Macro` (definition) or macro invocation |
| **Output** | 1 NodeSpec | DirectedGraph (N nodes) |
| **When expanded** | At build time (single step) | At build time (multiple steps) |
| **Names** | `analyzer` | Auto-prefixed: `macro_instance_step_1` |
| **Nesting** | Nodes don't contain nodes | Macros **cannot** contain other macros |
| **Use when** | Single operation | Reusable multi-step pattern |

### Defining a Macro in YAML

```yaml
kind: Macro
metadata:
  name: retry_pattern
spec:
  parameters:
    - name: fn
      type: str
    - name: max_retries
      type: int
      default: 3
  nodes:
    - kind: function_node
      metadata:
        name: "attempt"
      spec:
        fn: "{{ fn }}"
```

### Invoking a Macro

```yaml
- kind: retry_pattern        # or kind: macro_invocation
  metadata:
    name: fetch_data
  spec:
    macro: retry_pattern
    inputs:
      fn: "myapp.fetch"
      max_retries: 5
```

The compiler expands this into nodes named `fetch_data_attempt` (prefixed with the instance name).

### Defining a Macro in Python

```python
class RetryMacro(ConfigurableMacro, yaml_alias="retry"):
    def expand(self, instance_name, inputs, dependencies):
        graph = DirectedGraph()
        # Build sub-graph with actual NodeSpec objects
        return graph
```

Located in: [`hexdag/stdlib/macros/`](../hexdag/stdlib/macros/), [`hexdag/compiler/plugins/macro_entity.py`](../hexdag/compiler/plugins/macro_entity.py)

---

## 12. Entity Lifecycle (State Machines)

hexDAG supports declarative state machines for business entities (orders, tickets, documents).

### Pipeline-Level (Single Pipeline)

```yaml
kind: Pipeline
spec:
  state_machines:
    document:
      initial: RECEIVED
      transitions:
        RECEIVED: [CLASSIFIED, REJECTED]
        CLASSIFIED: [EXTRACTED]
        EXTRACTED: [VALIDATED]
        VALIDATED: [FILED]

  nodes:
    - kind: transition
      metadata:
        name: mark_classified
      spec:
        entity: document
        entity_id: "$input.doc_id"
        to_state: CLASSIFIED
```

### System-Level (Multi-Pipeline)

```yaml
kind: System
spec:
  state_machines:
    ticket:
      initial: OPEN
      transitions:
        OPEN: [INVESTIGATING, ESCALATED, CLOSED]
        INVESTIGATING: [RESOLVED, ESCALATED]
      handlers:
        on_transition: myapp.hooks.TicketHandler

  states:
    INVESTIGATING:
      on_enter: ticket-investigate    # triggers this pipeline
    CLOSED:
      terminal: true
      requires: [resolution_summary]

  processes:
    - name: ticket-investigate
      pipeline: pipelines/investigate.yaml
```

### Key Concepts

- **TransitionNode** (`kind: transition`): validates transition against state machine, fires handlers, emits `StateTransitionEvent`
- **Handlers are transactional**: handler failure = transition rollback
- **Agent tool scoping**: agents must declare `entities: [ticket]` to access state machine tools
- **Events**: `StateTransitionEvent`, `EntityGarbageCollected`, `EntityObligationFailed`, `EntityCompensationEvent`
- **LifecycleRunner**: event-driven multi-pipeline runner with cascade depth limits and terminal state GC

Located in: [`hexdag/kernel/domain/entity_state.py`](../hexdag/kernel/domain/entity_state.py), [`hexdag/stdlib/nodes/transition_node.py`](../hexdag/stdlib/nodes/transition_node.py), [`hexdag/stdlib/lib/entity_state.py`](../hexdag/stdlib/lib/entity_state.py)

---

## 13. The Compiler in Detail

The compiler (`YamlPipelineBuilder`) transforms YAML into kernel domain models through 5 phases.

### Phase 1: Parse + Include

- Parse YAML into Python dict
- Resolve `!include` directives (recursive, circular reference detection)
- Custom tag discovery (`!py`)
- Located in: [`hexdag/compiler/preprocessing/include.py`](../hexdag/compiler/preprocessing/include.py), [`hexdag/compiler/tag_discovery.py`](../hexdag/compiler/tag_discovery.py)

### Phase 2: Environment Selection

- Select dev/staging/prod configuration
- Merge environment-specific overrides
- Located in: [`hexdag/compiler/yaml_builder.py`](../hexdag/compiler/yaml_builder.py)

### Phase 3: Validate

- `YamlValidator` checks: node types exist, port configs valid, no dependency cycles, schema compliance
- Located in: [`hexdag/compiler/yaml_validator.py`](../hexdag/compiler/yaml_validator.py)

### Phase 4: Preprocess (Plugin Pipeline)

Three preprocessing plugins run in order:

| Plugin | What it does | Located in |
|--------|-------------|-----------|
| `IncludePreprocessPlugin` | `!include` resolution | [`hexdag/compiler/yaml_builder.py`](../hexdag/compiler/yaml_builder.py) |
| `EnvironmentVariablePlugin` | `${VAR}` substitution (defers secrets) | [`hexdag/compiler/yaml_builder.py`](../hexdag/compiler/yaml_builder.py) |
| `TemplatePlugin` | `{{ }}` Jinja2 rendering (non-spec fields only) | [`hexdag/compiler/yaml_builder.py`](../hexdag/compiler/yaml_builder.py) |

### Phase 5: Build Graph (Entity Plugins)

Entity plugins process YAML into kernel objects:

| Plugin | What it does | Located in |
|--------|-------------|-----------|
| `NodeEntityPlugin` | Instantiate node factories → NodeSpec objects | [`hexdag/compiler/plugins/node_entity.py`](../hexdag/compiler/plugins/node_entity.py) |
| `MacroEntityPlugin` | Expand macros → sub-graphs | [`hexdag/compiler/plugins/macro_entity.py`](../hexdag/compiler/plugins/macro_entity.py) |
| `MacroDefinitionPlugin` | Register `kind: Macro` definitions | [`hexdag/compiler/plugins/`](../hexdag/compiler/plugins/) |
| `AdapterDefinitionPlugin` | Process adapter overrides | [`hexdag/compiler/plugins/`](../hexdag/compiler/plugins/) |
| `ConfigDefinitionPlugin` | Process config overrides | [`hexdag/compiler/plugins/`](../hexdag/compiler/plugins/) |
| `MiddlewareDefinitionPlugin` | Process middleware stacking | [`hexdag/compiler/plugins/`](../hexdag/compiler/plugins/) |

After entity plugins, the `ReferenceResolver` scans all input_mapping, expressions, and templates to auto-detect node dependencies. Then `ComponentInstantiator` resolves aliases and handles deferred secret resolution.

**Output:** `DirectedGraph` + `PipelineConfig`

### YAML Manifest Types

| Kind | Purpose | Located in |
|------|---------|-----------|
| `kind: Pipeline` | Single workflow (most common) | [`hexdag/compiler/yaml_builder.py`](../hexdag/compiler/yaml_builder.py) |
| `kind: System` | Multi-pipeline orchestration | [`hexdag/compiler/system_builder.py`](../hexdag/compiler/system_builder.py) |
| `kind: Config` | Shared configuration | [`hexdag/compiler/config_loader.py`](../hexdag/compiler/config_loader.py) |
| `kind: Macro` | Reusable node template | [`hexdag/compiler/plugins/macro_entity.py`](../hexdag/compiler/plugins/macro_entity.py) |
| `kind: Adapter` | Adapter definition | [`hexdag/compiler/plugins/`](../hexdag/compiler/plugins/) |
| `kind: Middleware` | Port wrapper definition | [`hexdag/compiler/plugins/`](../hexdag/compiler/plugins/) |

---

## 14. The Orchestrator in Detail

The Orchestrator ([`hexdag/kernel/orchestration/orchestrator.py`](../hexdag/kernel/orchestration/orchestrator.py)) is the core execution engine.

### Wave-Based Execution

```
DirectedGraph:  A → C
                B → C → D

Waves:
  Wave 1: [A, B]     ← run concurrently
  Wave 2: [C]        ← depends on A and B
  Wave 3: [D]        ← depends on C
```

Nodes in the same wave have no mutual dependencies and run concurrently via `asyncio.gather()`. The `OrchestratorConfig` controls `max_concurrent_nodes` (default: 10).

### Execution Components

| Component | Role | Located in |
|-----------|------|-----------|
| `Orchestrator` | Main loop: wave computation, concurrent execution | [`kernel/orchestration/orchestrator.py`](../hexdag/kernel/orchestration/orchestrator.py) |
| `OrchestratorFactory` | Creates Orchestrator from PipelineConfig | [`kernel/orchestration/orchestrator_factory.py`](../hexdag/kernel/orchestration/orchestrator_factory.py) |
| `NodeExecutor` | Per-node execution: timeout, retry, validation | [`kernel/orchestration/components/node_executor.py`](../hexdag/kernel/orchestration/components/node_executor.py) |
| `ExecutionCoordinator` | Input mapping, `when` evaluation, observer notifications | [`kernel/orchestration/components/execution_coordinator.py`](../hexdag/kernel/orchestration/components/execution_coordinator.py) |
| `BodyExecutor` | Schema validation, port access, service invocation | [`kernel/orchestration/body_executor.py`](../hexdag/kernel/orchestration/body_executor.py) |
| `LifecycleManager` | Port asetup/aclose, service lifecycle, hooks | [`kernel/orchestration/components/lifecycle_manager.py`](../hexdag/kernel/orchestration/components/lifecycle_manager.py) |
| `CheckpointManager` | Save/restore execution state between waves | [`kernel/orchestration/components/checkpoint_manager.py`](../hexdag/kernel/orchestration/components/checkpoint_manager.py) |

### Event System

Every orchestrator action emits events. Observers react to them.

| Event | When | Data |
|-------|------|------|
| `PipelineStarted` | Execution begins | total_waves, total_nodes |
| `PipelineCompleted` | Execution ends | name, duration_ms, node_results |
| `NodeStarted` | Node begins | node_name, wave_index |
| `NodeCompleted` | Node succeeds | node_name, output, duration_ms |
| `NodeFailed` | Node fails | node_name, error, attempt |
| `NodeSkipped` | `when` evaluates false | node_name, reason |
| `WaveCompleted` | All nodes in wave done | wave_index, results |
| `StateTransitionEvent` | Entity state changes | entity, from_state, to_state |
| `PortCallEvent` | Port method called | port_name, method, duration |

**Built-in observers:** SimpleLogging, ExecutionTracer, PerformanceMetrics, CostProfiler, Alerting, DataQuality.

Located in: [`hexdag/kernel/orchestration/events/`](../hexdag/kernel/orchestration/events/)

---

## 15. File Index — Where to Find Things

### "I want to..."

| Goal | Look in |
|------|---------|
| Understand how YAML is compiled | [`hexdag/compiler/yaml_builder.py`](../hexdag/compiler/yaml_builder.py) |
| Add a new node type | [`hexdag/stdlib/nodes/`](../hexdag/stdlib/nodes/) — subclass `BaseNodeFactory` |
| Add a new adapter | [`hexdag/stdlib/adapters/`](../hexdag/stdlib/adapters/) — subclass `HexDAGAdapter` |
| Add a new service | Subclass `hexdag.kernel.service.Service` |
| See how the orchestrator works | [`hexdag/kernel/orchestration/orchestrator.py`](../hexdag/kernel/orchestration/orchestrator.py) |
| See how expressions are evaluated | [`hexdag/kernel/expression_parser.py`](../hexdag/kernel/expression_parser.py) |
| See how dependencies are auto-detected | [`hexdag/compiler/reference_resolver.py`](../hexdag/compiler/reference_resolver.py) |
| See how aliases are resolved | [`hexdag/kernel/resolver.py`](../hexdag/kernel/resolver.py) |
| See all port contracts | [`hexdag/kernel/ports/`](../hexdag/kernel/ports/) |
| See all events | [`hexdag/kernel/orchestration/events/events.py`](../hexdag/kernel/orchestration/events/events.py) |
| See built-in middleware | [`hexdag/stdlib/middleware/`](../hexdag/stdlib/middleware/) |
| See state machine logic | [`hexdag/stdlib/lib/entity_state.py`](../hexdag/stdlib/lib/entity_state.py) |
| Run a pipeline from Python | [`hexdag/kernel/pipeline_runner.py`](../hexdag/kernel/pipeline_runner.py) |
| Run a multi-pipeline system | [`hexdag/kernel/system_runner.py`](../hexdag/kernel/system_runner.py) |

### Directory Map

```
hexdag/
├── kernel/                          # Core execution engine
│   ├── domain/                      #   Domain models: NodeSpec, DirectedGraph, PipelineConfig
│   ├── orchestration/               #   Orchestrator, events, observers, components
│   │   ├── components/              #     NodeExecutor, ExecutionCoordinator, etc.
│   │   ├── events/                  #     Event types and observer base
│   │   └── prompt/                  #     PromptTemplate, FewShotPromptTemplate
│   ├── ports/                       #   Port protocols: LLM, DataStore, Database, Memory, etc.
│   ├── context/                     #   ExecutionContext (async-safe via contextvars)
│   ├── validation/                  #   Sanitized types, secure JSON
│   ├── schema/                      #   SchemaGenerator (JSON Schema from types)
│   ├── config/                      #   HexDAGConfig model and loader
│   ├── linting/                     #   Pipeline lint rules
│   ├── utils/                       #   Caching, serialization, timers
│   ├── service.py                   #   Service base, @tool, @step decorators
│   ├── resolver.py                  #   Component alias → class resolution
│   ├── expression_parser.py         #   Safe expression evaluation (AST-based)
│   ├── pipeline_runner.py           #   One-liner YAML → results
│   ├── system_runner.py             #   Multi-pipeline orchestration
│   └── discovery.py                 #   Plugin/adapter/tool auto-discovery
│
├── compiler/                        # YAML → kernel domain models
│   ├── plugins/                     #   Entity plugins: node, macro, adapter, config
│   ├── preprocessing/               #   Include, env var, template plugins
│   ├── yaml_builder.py              #   Main 5-phase compiler
│   ├── yaml_validator.py            #   Schema validation
│   ├── reference_resolver.py        #   Auto-dependency detection
│   ├── component_instantiator.py    #   Alias resolution + deferred secrets
│   ├── system_builder.py            #   kind: System builder
│   ├── config_loader.py             #   kind: Config loader
│   └── py_tag.py                    #   !py YAML tag
│
├── stdlib/                          # Built-in components
│   ├── nodes/                       #   Node factories: LLMNode, AgentNode, FunctionNode, etc.
│   ├── adapters/                    #   Port adapters: openai, anthropic, mock, database, memory
│   │   ├── openai/
│   │   ├── anthropic/
│   │   ├── mock/
│   │   ├── database/
│   │   ├── memory/
│   │   └── secret/
│   ├── macros/                      #   Reusable DAG templates: reasoning_agent, conversation
│   ├── middleware/                   #   Port wrappers: retry, timeout, rate_limiter, cache
│   ├── lib/                         #   System services: ProcessRegistry, EntityState, Scheduler
│   │   └── observers/               #     Built-in observers
│   └── prompts/                     #   Prompt templates: chat, tool, few-shot, error correction
│
├── drivers/                         # Infrastructure implementations
│   ├── executors/                   #   LocalExecutor
│   ├── observer_manager/            #   LocalObserverManager
│   ├── pipeline_spawner/            #   LocalPipelineSpawner
│   ├── http_client/                 #   HttpClientDriver
│   └── vfs/                         #   VFS providers: /lib, /sys, /proc
│
├── api/                             # User-facing API layer
│   ├── execution.py                 #   execute(), execute_streaming()
│   ├── pipeline.py                  #   init_pipeline(), add_node(), remove_node()
│   ├── components.py                #   list_nodes(), list_adapters(), get_node_schema()
│   ├── validation.py                #   validate_pipeline()
│   ├── processes.py                 #   list_runs(), get_run(), cancel_run()
│   ├── documentation.py             #   get_quick_start(), get_node_guide()
│   ├── vfs.py                       #   list_path(), read_path()
│   └── logs.py                      #   query_logs(), tail_logs()
│
└── cli/                             # Command-line interface
    └── commands/                    #   validate, build, init, lint, docs, studio, plugins
```
