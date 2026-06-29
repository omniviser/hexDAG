# hexDAG Framework Guide

> **Developer-first workflow engine for AI agents.** Compose n8n-like automations in YAML or Python, run LangGraph-style agent flows as typed DAGs, and ship them with observability, replay, and human approval.
>
> Designed to be read by developers joining the team and by AI agents exploring the codebase.

**Related docs:** [ARCHITECTURE.md](ARCHITECTURE.md) (design philosophy, layers, compiler, orchestrator internals) · [PUBLIC_API.md](PUBLIC_API.md) (all public symbols, version policy, import paths) · [ROADMAP.md](ROADMAP.md) (strategic direction) · [Quick Start](getting-started/quickstart.md) (first pipeline)

---

## Table of Contents

1. [How Everything Connects](#1-how-everything-connects) — Integration model
2. [How a Pipeline Runs](#2-how-a-pipeline-runs) — End-to-end execution flow
3. [YAML Syntax](#3-yaml-syntax) — The 4 special syntaxes
4. [Ports, Adapters, and Middleware](#4-ports-adapters-and-middleware) — Contracts and implementations
5. [Data Flow Between Nodes](#5-data-flow-between-nodes) — input_mapping, expressions, templates
6. [Services](#6-services) — @tool and @step decorators
7. [Macros vs Nodes](#7-macros-vs-nodes) — Single step vs template expansion
8. [Composite Nodes](#8-composite-nodes-control-flow) — if/else, while, for-each, switch
9. [Entity Lifecycle](#9-entity-lifecycle) — State machines for business objects
10. [Suspension & Resume](#10-suspension--resume) — Human-in-the-loop, external events

For architecture internals (layers, compiler phases, orchestrator, file index), see [ARCHITECTURE.md](ARCHITECTURE.md).
For all public symbols and import paths, see [PUBLIC_API.md](PUBLIC_API.md).
For node/adapter/port reference, see the auto-generated [Reference docs](reference/nodes.md).

---

## 1. How Everything Connects

These concepts aren't independent modules — they're ONE tightly integrated system. A real company operation flows through all of them:

```
1. Agent receives a customer request                            → AGENTS
2. Agent reasons about it, decides to create an order           → AGENTS
3. Agent transitions the order to "processing"                  → ENTITIES
4. Transition emits StateTransitionEvent                        → EVENTS
5. Event triggers a fulfillment pipeline                        → PROCESSES
6. Pipeline uses memory to carry order context between nodes    → MEMORY
7. Expressions encode: "if total > 1000, require_approval"      → EXPRESSIONS
8. Observers track everything for audit trail                   → OBSERVABILITY
9. Fulfillment completes → entity transitions to "shipped"      → ENTITIES
10. "shipped" triggers notification pipeline                    → PROCESSES
```

### The Connective Tissue

- **Events** are the glue — an entity transition emits an event that triggers pipelines
- **Services (@tool/@step)** are the unified interface — agents interact with entities, memory, and processes through the same mechanism
- **Expressions** encode business rules inline in YAML — no Python needed for simple logic

---

## 2. How a Pipeline Runs

The journey from a YAML file to execution results:

```
1. You write a YAML file (kind: Pipeline)
                │
2. PipelineRunner receives it
                │
3. Compiler processes YAML
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
   ├── For each wave: run all nodes concurrently (asyncio.gather)
   │   ├── Check `when` condition (skip if false)
   │   ├── Prepare inputs (upstream results + input_mapping)
   │   ├── Execute async function (with timeout)
   │   └── On failure: retry with exponential backoff or route to error handler
   └── Emit events: NodeStarted → NodeCompleted/NodeFailed → PipelineCompleted
                │
7. Results: dict of {node_name: output} for every node
```

**In Python:**
```python
from hexdag import PipelineRunner, MockLLM

runner = PipelineRunner(port_overrides={"llm": MockLLM()})
results = await runner.run("my_pipeline.yaml", input_data={"query": "hello"})
```

For compiler and orchestrator internals, see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## 3. YAML Syntax

hexDAG YAML has 4 special syntaxes. They look similar but do different things at different times. This is the most common source of confusion.

| Syntax | Purpose | When | Example |
|--------|---------|------|---------|
| `!include` | Merge another YAML file | Build-time, phase 1 (first) | `!include shared/ports.yaml` |
| `${VAR}` | Environment variable | Build-time, phase 4 (secrets deferred to runtime) | `${OPENAI_API_KEY}` |
| `{{ expr }}` | Jinja2 template | Build-time for metadata; **runtime** for node specs | `{{ analyzer.result }}` |
| `node.field` | Value extraction (any depth) | Runtime, in input_mapping and expressions | `"analyzer.result.items.0"` |

### `!include` — File Composition

Merges another YAML file inline. Processed first, before anything else.

```yaml
spec:
  ports:
    "!include": shared/ports.yaml              # entire file merged here
  nodes:
    - "!include": nodes/analysis.yaml#nodes    # specific anchor
```

### `${VAR}` — Environment Variables

```yaml
spec:
  ports:
    llm:
      config:
        api_key: ${OPENAI_API_KEY}    # deferred to runtime (secret pattern)
        base_url: ${LLM_BASE_URL}     # resolved at build time
```

Secret-pattern names (`*_API_KEY`, `*_SECRET`) are **deferred** to runtime. Others resolved immediately.

### `{{ expr }}` — Jinja2 Templates

```yaml
metadata:
  name: {{ env }}_pipeline           # build-time (metadata)

spec:
  nodes:
    - kind: llm_node
      spec:
        prompt: "Analyze {{ analyzer.result }}"   # RUNTIME (node spec)
```

### `node.field` — Value Extraction

Extracts the actual Python object (not a string). Used in `input_mapping`, `expressions`, `when` conditions.

```yaml
spec:
  input_mapping:
    data: "analyzer.result"         # passes the actual Python object
    query: "$input.user_query"      # from pipeline input
  expressions:
    total: "analyzer.count * 2"     # expression evaluation
```

**Nested paths walk arbitrary depth.** A dotted path descends as many levels as you
write, for both node and `$input` references — `analyzer.result.items.0` and
`$input.user.profile.name` both resolve at runtime. A missing **deep** segment yields
`None` (treated as an optional field), while a wrong **first** segment (unknown node or
undeclared input field) is a build-time error. The validator's `input_schema` and
sibling-consistency checks key on the **top-level** field by design, so `$input.a.b.c` is
validated as `a`.

### The Critical Difference

```yaml
# TEMPLATE — renders text (Jinja2, becomes a string)
prompt: "Analyze {{ analyzer.result }}"

# EXTRACTION — passes the actual object (dict, list, number, etc.)
input_mapping:
  data: "analyzer.result"
```

---

## 4. Ports, Adapters, and Middleware

**Port** = abstract contract. "I need something that can generate text."
**Supports*** = fine-grained capability. "Can it also call tools?"
**Adapter** = concrete implementation. "Use OpenAI GPT-4."
**Middleware** = transparent wrapper. "Add retry + cache."

```
Node ──requires──► Port (contract)
                     ▲
                     │ implements
                  Adapter (OpenAI / Anthropic / Mock)
                     ▲
                     │ wraps
                  Middleware (Retry → Timeout → Cache)
```

### Ports and Supports* Protocols

| Port | Key sub-protocols |
|------|-------------------|
| **LLM** | `SupportsGeneration`, `SupportsFunctionCalling`, `SupportsStructuredOutput`, `SupportsVision`, `SupportsEmbedding` |
| **DataStore** | `SupportsKeyValue`, `SupportsQuery`, `SupportsTTL`, `SupportsTransactions` |
| **Database** | `SupportsQuery`, `SupportsRawSQL`, `SupportsTransactions`, `SupportsSchema` |
| **Other** | APICall, FileStorage, SecretStore, Executor, PipelineSpawner, ObserverManager |

**Naming:** "What kind of port?" → plain noun (`LLM`, `DataStore`). "Can this adapter do X?" → `Supports*`.

### Port Override Levels

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

Declared per-port in YAML. Nodes don't know middleware exists.

```yaml
spec:
  ports:
    llm:
      adapter: openai
      middleware:
        - hexdag.stdlib.middleware.RetryWithBackoff:
            max_retries: 3
        - hexdag.stdlib.middleware.Timeout:
            timeout_seconds: 30
```

Available: `RetryWithBackoff`, `RateLimiter`, `ResponseCache`, `Timeout`, `RoundRobin`, `CircuitBreaker`, `BatchGeneration`, `DistributedCache`.

**Python equivalent** — use `compose()` to stack middleware programmatically:

```python
from hexdag.stdlib.middleware import compose, ResponseCache, RetryWithBackoff, Timeout
from hexdag import MockLLM

llm = compose(
    MockLLM(),
    ResponseCache,       # cache identical calls
    RetryWithBackoff,    # retry transient failures
    Timeout,             # enforce time limits
)
```

**Full middleware stack order** (inner → outer):

```
adapter → [Cache → RateLimiter → CircuitBreaker → Retry → Timeout]
        → StructuredOutputFallback (if needed) → ObservableLLM
```

Auto-middleware (`ObservableLLM`, `ObservableToolRouter`) is framework-managed and always outermost — you never declare it in YAML.

For all middleware symbols and signatures, see [PUBLIC_API.md > Middleware](PUBLIC_API.md#middleware-from-hexdagstdlibmiddleware-import-).

---

## 5. Data Flow Between Nodes

Upstream node outputs are automatically available downstream. There are 3 mechanisms:

### 1. `input_mapping` — Wire Fields Explicitly

```yaml
spec:
  input_mapping:
    data: "analyzer.result"          # upstream node output
    query: "$input.user_query"       # pipeline input
```

`input_mapping` is **optional for all node types.** The orchestrator auto-infers parameter values from the upstream namespace:

- **LLM and expression nodes** pull what they need via templates (`{{ analyzer.result }}`) or expressions (`analyzer.count * 2`).
- **Function and service_call nodes** auto-match upstream fields to function parameter names. Use `unpack_input: true` on function nodes to enable this.

Explicit `input_mapping` is only needed when upstream field names **differ** from the target parameter names, or when you want to compute derived values.

**Inline expressions** — mapping values can be expressions, not just references:

```yaml
input_mapping:
  total: "order.price * order.quantity"
  label: "'Order: ' + order.name"
  fallback: "coalesce(order.notes, 'none')"
```

**Modifiers** — control what happens when a value is missing:

```yaml
input_mapping:
  data: "analyzer.result | required"       # fail if None
  notes: "analyzer.notes | default('n/a')" # use 'n/a' if None
```

### 2. `expressions` — Compute Values

```yaml
spec:
  expressions:
    total: "analyzer.count * 2"
    ok: "analyzer.status in ['active', 'pending']"
    pay: "coalesce(negotiation.target_pay, load.target_pay)"
```

AST-validated — no `eval()`. Built-in functions: `coalesce`, `default`, `isnone`, `len`, `min`, `max`, `sum`, `round`, `str`, `int`, `float`, `now`, `upper`, `lower`, `split`, `join`, etc.

### 3. `{{ templates }}` — Build Text

```yaml
spec:
  human_message: "Analyze {{ analyzer.result }} for topic {{ $input.topic }}"
```

Jinja2 rendering at runtime, producing strings.

### Dependency Detection

The compiler **auto-infers** dependencies by scanning `input_mapping`, `expressions`, and `{{ templates }}` for upstream node references. This is the recommended approach — no manual wiring needed:

```yaml
nodes:
  - kind: llm_node
    name: analyzer                           # no dependencies field needed

  - kind: expression_node
    name: compute
    spec:
      expressions:
        total: "analyzer.count * 2"          # compiler sees "analyzer" → adds dependency
```

**Three dependency mechanisms exist:**

| Mechanism | Status | Purpose |
|-----------|--------|---------|
| Auto-inferred | **Recommended** | Compiler detects from data references |
| `wait_for: [node_a]` | **Active** | Ordering-only — "run after this, but don't consume its output" |
| `dependencies: [node_a]` | **Deprecated** | Explicit listing (compiler warns if redundant) |

**`wait_for`** is useful for side-effect sequencing — e.g., ensure a logging node runs after a save node even though it doesn't use the save result:

```yaml
- kind: function_node
  metadata: { name: notify }
  spec:
    fn: "myapp.send_notification"
  wait_for: [save_order]             # ordering only, no data dependency
```

**`dependencies`** still works but the compiler will warn you:
- If your explicit deps are a subset of what it already inferred → *"Consider removing the redundant 'dependencies' key"*
- If inferred deps are missing from your explicit list → the compiler auto-adds them

### Safety

- `typo_node.field` (unknown node) → **BUILD ERROR** with "did you mean?"
- `real_node.typo_field` (unknown field) → `None` at runtime
- `field | required` → fail if None
- `field | default('x')` → use 'x' if None

### Expression Namespaces

| Namespace | Pattern | Purpose |
|-----------|---------|---------|
| Upstream nodes | `node_name.field` | Output of completed nodes (any nested sub-path) |
| Pipeline input | `$input.field` | Initial input data (field or any nested sub-path) |
| Execution context | `$ctx.run_id` | Runtime metadata |
| Loop state | `state.counter` | Current iteration |

---

## 6. Services

Services wrap port-backed business logic behind `@tool` and `@step` decorators.

```python
from hexdag.kernel.service import Service, tool, step

class OrderService(Service):
    def __init__(self, store: SupportsKeyValue) -> None:
        self._store = store

    @tool
    async def get_order(self, order_id: str) -> dict:
        """Agent-callable during ReAct reasoning."""
        return await self._store.aget(f"order:{order_id}")

    @step
    async def save_order(self, order_id: str, data: dict) -> dict:
        """Deterministic DAG node via ServiceCallNode."""
        await self._store.aset(f"order:{order_id}", data)
        return {"saved": True}

    @tool
    @step
    async def validate_order(self, order_id: str) -> dict:
        """Both agent tool and DAG step."""
        ...
```

- `@tool` — agent-callable during ReAct reasoning (auto-generates tool schemas)
- `@step` — deterministic DAG node (invoked by `service_call_node`)
- Both can be stacked on the same method

### YAML Usage

Register services and call `@step` methods via `service_call_node`:

```yaml
spec:
  services:
    orders:
      class: myapp.services.OrderService
      config:
        store: { ref: main_store }

  nodes:
    # @step methods — called as deterministic DAG nodes
    - kind: service_call_node
      metadata: { name: save }
      spec:
        service: orders
        method: save_order

    # @tool methods — available to agents during ReAct reasoning
    - kind: agent_node
      metadata: { name: order_agent }
      spec:
        initial_prompt_template: "Handle order {{ $input.order_id }}"
        available_tools: ["orders:get_order", "orders:validate_order"]
        max_steps: 5
```

`@tool` methods use the pattern `service_name:method_name` in `available_tools`. The agent can call these tools during multi-step reasoning.

For all service-related symbols, see [PUBLIC_API.md > Services](PUBLIC_API.md#services--extension-points).

### Built-in Services

| Service | Purpose |
|---------|---------|
| `ProcessRegistry` | Track pipeline runs — status, duration, results |
| `EntityState` | Declarative state machines with validated transitions |
| `PipelineMemory` | Run-scoped key-value store (auto-registered) |

---

## 7. Macros vs Nodes

**A Node** produces **1 NodeSpec** — a single processing step.
**A Macro** expands into **a sub-graph** — multiple nodes with internal dependencies.

| | Node | Macro |
|---|------|-------|
| YAML | `kind: llm_node` | `kind: macro_invocation` |
| Output | 1 NodeSpec | DirectedGraph (N nodes) |
| Names | `analyzer` | Auto-prefixed: `instance_step_1` |
| Use when | Single operation | Reusable multi-step pattern |

### Built-in Macros

| Macro | YAML alias | Purpose |
|---|---|---|
| `ReasoningAgentMacro` | `core:reasoning_agent` | Multi-step ReAct reasoning with tool calling |
| `ConversationMacro` | `core:conversation_agent` | Multi-turn chat with memory |
| `LLMMacro` | `core:llm_macro` | Structured LLM call (consider `llm_node` for simple cases) |

### Invoking a Built-in Macro

From [deep_research_agent.yaml](../examples/mcp/deep_research_agent.yaml):

```yaml
nodes:
  - kind: macro_invocation
    metadata:
      name: research_agent
    spec:
      macro: core:reasoning_agent
      config:
        main_prompt: |
          You are a deep research agent with web search.
          Research Question: {{research_question}}
        max_steps: 10
        allowed_tools:
          - research:tavily_search
          - research:tavily_qna_search
        tool_format: mixed
```

The macro expands at build time into an internal sub-graph (prompt node, tool dispatch, reasoning loop) — you only configure the high-level behavior.

### Defining a Custom Macro

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
      metadata: { name: attempt }
      spec:
        fn: "{{ fn }}"
```

Invoke with:

```yaml
- kind: macro_invocation
  metadata: { name: fetch_data }
  spec:
    macro: retry_pattern
    config:
      fn: "myapp.fetch"
      max_retries: 5
```

---

## 8. Composite Nodes (Control Flow)

`CompositeNode` provides unified control flow: `while`, `for-each`, `times`, `if-else`, and `switch`.

All modes support two execution patterns:
- **Inline body** — when `body` is specified, the composite node executes it internally
- **Yield to downstream** — when no body, the node yields state to downstream nodes

### if-else — Single Condition Branch

```yaml
- kind: composite_node
  metadata: { name: route_by_priority }
  spec:
    mode: if-else
    condition: "$input.priority == 'urgent'"
    body: "myapp.handle_urgent"
    else_body: "myapp.handle_normal"
```

### for-each — Collection Iteration

```yaml
- kind: composite_node
  metadata: { name: process_items }
  spec:
    mode: for-each
    items: "$input.items"
    concurrency: 5
    body:
      - kind: expression_node
        spec:
          expressions:
            result: "$item * 2"
```

### while — Condition Loop

```yaml
- kind: composite_node
  metadata: { name: retry_loop }
  spec:
    mode: while
    condition: "state.attempts < 3 and not state.success"
    initial_state:
      attempts: 0
      success: false
    body: "myapp.attempt_operation"
    collect: last
```

### times — Fixed Count Loop

```yaml
- kind: composite_node
  metadata: { name: generate_variants }
  spec:
    mode: times
    count: 5
    body: "myapp.generate_one"
    collect: list
```

### switch — Multi-Branch Routing

```yaml
- kind: composite_node
  metadata: { name: router }
  spec:
    mode: switch
    branches:
      - condition: "status == 'urgent'"
        action: "urgent_path"
      - condition: "status == 'normal'"
        action: "normal_path"
    else_action: "default_path"
```

### Parameters

| Parameter | Values | Purpose |
|-----------|--------|---------|
| `collect` | `list`, `last`, `first`, `dict`, `reduce` | How to aggregate iteration results |
| `error_handling` | `fail_fast`, `continue`, `collect` | What happens when an iteration fails |
| `concurrency` | integer | Max parallel iterations (for-each) |
| `max_iterations` | integer | Safety limit for while loops |

---

## 9. Entity Lifecycle

hexDAG supports declarative state machines for business entities.

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
      metadata: { name: mark_classified }
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

### Python Usage

```python
from hexdag import System

async with System.from_yaml("ticket-system.yaml") as system:
    await system.transition("ticket", "T-1", "INVESTIGATING")
    result = await system.run_process("extract", {"ticket_id": "T-1"})
```

See [examples/libs/run_order_lifecycle.py](../examples/libs/run_order_lifecycle.py) for a complete runnable example.

### Key Concepts

- **TransitionNode** (`kind: transition`): validates against state machine, fires handlers, emits `StateTransitionEvent`
- **Handlers are transactional**: handler failure = transition rollback
- **Agent tool scoping**: agents declare `entities: [ticket]` to access state machine tools
- **LifecycleRunner**: event-driven multi-pipeline runner with cascade depth limits and terminal state GC

---

## 10. Suspension & Resume

hexDAG supports pipeline suspension for human-in-the-loop and external event patterns. A pipeline can pause at a `wait_node`, persist its state, and resume when external data arrives.

### WaitNode

```yaml
- kind: wait_node
  metadata: { name: await_reply }
  spec:
    event_key: "email_reply:{{$input.conversation_id}}"
    timeout: 7d
    on_timeout: timeout_handler
```

On suspend, the node returns a `Suspended` signal. The pipeline result has `status: SUSPENDED` and includes `run_id` for later resumption.

### Resuming with External Data

```python
from hexdag import PipelineRunner, InMemoryMemory

runner = PipelineRunner(checkpoint_storage=InMemoryMemory())
result = await runner.run("approval_pipeline.yaml", input_data=data)
# result.status == PipelineStatus.SUSPENDED

# Later, when the external event arrives:
result = await runner.resume_with_event(
    "approval_pipeline.yaml",
    run_id=result.run_id,
    event_data={"approved": True, "reviewer": "jane@co.com"},
)
# Downstream nodes execute with original context preserved
```

Requires `checkpoint_storage` to be configured so pipeline state survives between suspend and resume.
