# Core Concepts

Understanding HexDAG's core concepts will help you build robust AI workflows.

## Directed Acyclic Graphs (DAGs)

A DAG represents your workflow as a graph of nodes and dependencies:

- **Nodes** - Individual processing steps
- **Edges** - Dependencies between nodes
- **Acyclic** - No circular dependencies allowed

### Why DAGs?

1. **Automatic Parallelization** - Independent nodes run concurrently
2. **Clear Dependencies** - Explicit data flow
3. **Deterministic Execution** - Same inputs = same outputs
4. **Easy Visualization** - Graph structure is intuitive

## NodeSpec

`NodeSpec` defines a single processing step:

```python
from hexdag.core.domain import NodeSpec

spec = NodeSpec(
    name="process",              # Unique identifier
    function=my_function,        # Callable or node type
    depends_on=["upstream"],     # Node dependencies
    params={"key": "value"}      # Node-specific parameters
)
```

## Orchestrator

The `Orchestrator` executes your DAG:

```python
from hexdag.builtin import Orchestrator

orchestrator = Orchestrator()
results = await orchestrator.arun(graph, inputs={"key": "value"})
```

Key features:

- **Topological Execution** - Respects dependencies
- **Concurrent Processing** - Parallel execution with asyncio
- **Error Handling** - Graceful failure management
- **Event Emission** - Real-time monitoring

## Node Types

HexDAG supports multiple node types:

### FunctionNode

Execute Python functions:

```python
def my_function(inputs: dict) -> dict:
    return {"result": inputs["data"] * 2}

NodeSpec(name="fn", function=my_function)
```

### LLMNode

Language model interactions:

```yaml
- type: llm
  id: summarizer
  params:
    prompt_template: "Summarize: {{text}}"
    model: "gpt-4"
```

### ReActAgentNode

ReAct pattern agents with tools:

```yaml
- type: agent
  id: researcher
  params:
    initial_prompt_template: "Research {{topic}}"
    max_steps: 10
    tools: [search, calculator]
```

### LoopNode

Iterative processing:

```yaml
- type: loop
  id: process_batch
  params:
    max_iterations: 100
    condition: "{{continue_processing}}"
```

### ConditionalNode

Conditional execution:

```yaml
- type: conditional
  id: check_quality
  params:
    condition: "{{quality_score > 0.8}}"
    true_branch: [publish]
    false_branch: [retry]
```

## Event System & Observability

Monitor workflow execution through the observer pattern:

### Built-in Observers

| Observer | Purpose |
|----------|---------|
| `PerformanceMetricsObserver` | Node timing, success rates, execution counts |
| `CostProfilerObserver` | Token usage, cost estimation, bottleneck detection |
| `AlertingObserver` | Threshold-based alerts for slow/failing nodes |
| `ResourceMonitorObserver` | Concurrency tracking, wave-level parallelism |
| `DataQualityObserver` | Null/empty/error detection in node outputs |
| `ExecutionTracerObserver` | Full event trace with timing for debugging |

### Cost Profiling Example

```python
from hexdag.core.orchestration.events import (
    CostProfilerObserver,
    COST_PROFILING_EVENTS,
)

profiler = CostProfilerObserver(model="gpt-4o-mini")
observer_manager.register(profiler.handle, event_types=COST_PROFILING_EVENTS)

# ... run pipeline ...

print(profiler.format_report())
# Pipeline: customer-support-v2
# Total tokens:  4,230 (est. $0.013)
# Total latency: 3.2s
# Bottleneck:    researcher (2.1s, 3,100 tokens)
```

### Available Events

- `PipelineStarted` / `PipelineCompleted` - Pipeline lifecycle
- `NodeStarted` / `NodeCompleted` / `NodeFailed` - Node execution
- `WaveStarted` / `WaveCompleted` - Wave-level parallelism
- `LLMPromptSent` / `LLMResponseReceived` - LLM interactions (with token usage)
- `ToolCalled` / `ToolCompleted` - Tool execution

## Validation Framework

Type-safe data flow with Pydantic:

```python
from pydantic import BaseModel

class InputSchema(BaseModel):
    text: str
    count: int

class OutputSchema(BaseModel):
    result: list[str]

def validated_function(inputs: InputSchema) -> OutputSchema:
    return OutputSchema(result=[inputs.text] * inputs.count)
```

HexDAG automatically:

- Validates inputs against schemas
- Checks compatibility between connected nodes
- Provides detailed error messages

## Hexagonal Architecture

HexDAG follows hexagonal architecture:

- **Domain** - Core business logic (DAG, NodeSpec)
- **Application** - Use cases (Orchestrator, NodeFactory)
- **Ports** - Interface definitions (LLM, Database, Memory)
- **Adapters** - External implementations (OpenAI, PostgreSQL)

Benefits:

- **Testability** - Mock external services easily
- **Maintainability** - Clear separation of concerns
- **Extensibility** - Add adapters without changing core

## Next Steps

- [Node Types](node-types.md) - Detailed node documentation
- [Implementation Guide](IMPLEMENTATION_GUIDE.md) - Production workflows
- [📓 Notebooks](../notebooks/) - Interactive tutorials
