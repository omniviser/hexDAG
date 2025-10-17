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
from hexdag.core.application import Orchestrator

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

## Event System

Monitor workflow execution through events:

```python
from hexdag.core.application.events import (
    EventBus,
    NodeStarted,
    NodeCompleted,
    NodeFailed,
    DAGStarted,
    DAGCompleted
)

event_bus = EventBus()
event_bus.subscribe(NodeStarted, lambda e: print(f"Started: {e.node_name}"))
```

Available events:

- `DAGStarted` / `DAGCompleted` - Workflow lifecycle
- `NodeStarted` / `NodeCompleted` / `NodeFailed` - Node execution
- `ValidationError` - Schema validation failures

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
