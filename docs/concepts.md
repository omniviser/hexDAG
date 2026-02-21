# Core Concepts

Understanding hexDAG's core concepts will help you build robust AI workflows.

## Directed Acyclic Graphs (DAGs)

A DAG represents your workflow as a graph of nodes and dependencies:

- **Nodes** — Individual processing steps
- **Edges** — Dependencies between nodes
- **Acyclic** — No circular dependencies allowed

### Why DAGs?

1. **Automatic Parallelization** — Independent nodes run concurrently
2. **Clear Dependencies** — Explicit data flow
3. **Deterministic Execution** — Same inputs produce same outputs
4. **Easy Visualization** — Graph structure is intuitive

## YAML Pipelines

hexDAG uses a declarative YAML manifest format:

```yaml
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: my-pipeline
spec:
  ports:
    llm:
      adapter: hexdag.builtin.adapters.openai.OpenAIAdapter
      config:
        model: gpt-4
  nodes:
    - kind: llm_node
      metadata:
        name: summarizer
      spec:
        prompt_template: "Summarize: {{input}}"
      dependencies: [input_node]
```

Key structural elements:
- `apiVersion` / `kind` / `metadata` / `spec` — Kubernetes-style manifest
- `spec.ports` — Adapter configuration (LLM, database, memory)
- `spec.nodes` — Processing steps with `kind`, `metadata.name`, `spec`, and `dependencies`

## Execution

### PipelineRunner

The recommended way to run YAML pipelines. Handles parsing, secret loading, port instantiation, and execution in one call. See [PipelineRunner](PIPELINE_RUNNER.md) for full documentation.

```python
from hexdag import PipelineRunner

runner = PipelineRunner()
results = await runner.run("pipeline.yaml", input_data={"key": "value"})
```

### Orchestrator

The low-level execution engine used internally by `PipelineRunner`. Use directly for advanced scenarios requiring fine-grained control over ports, events, or execution.

```python
from hexdag.core.orchestration.orchestrator import Orchestrator

orchestrator = Orchestrator(ports={"llm": my_llm})
results = await orchestrator.run(graph, inputs={"key": "value"})
```

## Node Types

hexDAG supports multiple node types, referenced by `kind` in YAML:

### function_node

Execute Python functions referenced by module path:

```yaml
- kind: function_node
  metadata:
    name: process_data
  spec:
    fn: "myapp.processing.transform"
  dependencies: []
```

### llm_node

Language model interactions with prompt templating:

```yaml
- kind: llm_node
  metadata:
    name: summarizer
  spec:
    prompt_template: "Summarize: {{input}}"
  dependencies: [loader]
```

### agent_node

ReAct pattern agents with tool access:

```yaml
- kind: agent_node
  metadata:
    name: researcher
  spec:
    initial_prompt_template: "Research {{topic}}"
    max_steps: 10
    tools: [search, calculator]
  dependencies: []
```

### loop_node

Iterative processing with conditions:

```yaml
- kind: loop_node
  metadata:
    name: refiner
  spec:
    max_iterations: 5
  dependencies: [draft]
```

### conditional_node

Conditional execution paths:

```yaml
- kind: conditional_node
  metadata:
    name: route
  spec:
    condition: "{{quality_score > 0.8}}"
  dependencies: [evaluator]
```

## Conditional Execution & Skip Propagation

hexDAG supports conditional node execution via `when` clauses and automatic skip propagation through dependency chains.

### When Clauses

Any node can include a `when` expression. The expression is evaluated against the node's input data. If it evaluates to `False`, the node is skipped.

```yaml
- kind: llm_node
  metadata:
    name: send_notification
  spec:
    prompt_template: "Draft notification for {{input}}"
    when: "requires_notification == True"
  dependencies: [classifier]
```

Skipped nodes produce a result of `{"_skipped": True, "reason": "..."}` and emit a `NodeSkipped` event.

### Automatic Skip Propagation

When **all** dependencies of a node are skipped, the downstream node is automatically skipped without needing its own `when` clause. This prevents cascading errors in branching pipelines.

```
  classifier (when: "False")
       |
       v
  send_email       <-- auto-skipped (_upstream_skipped)
       |
       v
  log_result       <-- auto-skipped (_upstream_skipped)
```

**Partial skip:** If a node has multiple dependencies and only some are skipped, the node still executes with the non-skipped results.

### Skip Result Format

| Field | Type | Meaning |
|-------|------|---------|
| `_skipped` | `bool` | Node was skipped (always `True`) |
| `_upstream_skipped` | `bool` | Skip was propagated from upstream |
| `reason` | `str` | Human-readable explanation |

## Ports & Adapters

hexDAG follows hexagonal architecture. **Ports** define interfaces; **adapters** implement them.

| Port | Interface | Purpose |
|------|-----------|---------|
| `LLM` | `hexdag.core.ports.LLM` | Language model interactions |
| `DatabasePort` | `hexdag.core.ports.DatabasePort` | Data persistence |
| `Memory` | `hexdag.core.ports.Memory` | Agent conversation memory |
| `ToolRouter` | `hexdag.core.ports.ToolRouter` | Function calling / tool execution |
| `SecretPort` | `hexdag.core.ports.SecretPort` | Secret management (KeyVault, etc.) |

Ports are declared in YAML under `spec.ports` and auto-instantiated by `PipelineRunner`:

```yaml
spec:
  ports:
    llm:
      adapter: hexdag.builtin.adapters.openai.OpenAIAdapter
      config:
        api_key: "${OPENAI_API_KEY}"
        model: gpt-4
    database:
      adapter: hexdag.builtin.adapters.sqlite.SQLiteAdapter
      config:
        db_path: "./data.db"
```

For testing, use mock adapters or `port_overrides`:

```python
from hexdag import PipelineRunner, MockLLM

runner = PipelineRunner(port_overrides={"llm": MockLLM(responses="test")})
```

## Event System & Observability

Monitor workflow execution through the observer pattern.

### Built-in Observers

| Observer | Purpose |
|----------|---------|
| `PerformanceMetricsObserver` | Node timing, success rates, execution counts |
| `CostProfilerObserver` | Token usage, cost estimation, bottleneck detection |
| `AlertingObserver` | Threshold-based alerts for slow/failing nodes |
| `ResourceMonitorObserver` | Concurrency tracking, wave-level parallelism |
| `DataQualityObserver` | Null/empty/error detection in node outputs |
| `ExecutionTracerObserver` | Full event trace with timing for debugging |

### Available Events

- `PipelineStarted` / `PipelineCompleted` — Pipeline lifecycle
- `NodeStarted` / `NodeCompleted` / `NodeFailed` / `NodeSkipped` — Node execution
- `WaveStarted` / `WaveCompleted` — Wave-level parallelism
- `LLMPromptSent` / `LLMResponseReceived` — LLM interactions (with token usage)
- `ToolCalled` / `ToolCompleted` — Tool execution

## Validation Framework

Type-safe data flow with Pydantic schemas declared in YAML:

```yaml
- kind: function_node
  metadata:
    name: process
  spec:
    fn: "myapp.process"
    input_schema:
      text: str
      count: int
    output_schema:
      result: list[str]
  dependencies: [upstream]
```

hexDAG automatically validates inputs against schemas, checks compatibility between connected nodes, and provides detailed error messages.

## Next Steps

- [PipelineRunner](PIPELINE_RUNNER.md) — One-liner pipeline execution
- [Node Types](node-types.md) — Detailed node documentation
- [Implementation Guide](IMPLEMENTATION_GUIDE.md) — Production workflows
- [Plugin System](PLUGIN_SYSTEM.md) — Custom adapters and node types
