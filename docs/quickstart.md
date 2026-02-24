# Quick Start

This guide will help you create your first hexDAG workflow in minutes.

## YAML Pipeline (Recommended)

The recommended way to build workflows is with declarative YAML:

```yaml
# workflow.yaml
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: data-pipeline
spec:
  nodes:
    - kind: function_node
      metadata:
        name: load
      spec:
        fn: "mymodule.load_data"
      dependencies: []

    - kind: function_node
      metadata:
        name: transform
      spec:
        fn: "mymodule.transform_data"
      dependencies: [load]

    - kind: function_node
      metadata:
        name: save
      spec:
        fn: "mymodule.save_results"
      dependencies: [transform]
```

Run the pipeline with `PipelineRunner`:

```python
import asyncio
from hexdag import PipelineRunner

async def main():
    runner = PipelineRunner()
    results = await runner.run("workflow.yaml")
    print(results)

asyncio.run(main())
```

Or from the command line:

```bash
hexdag pipeline run workflow.yaml
```

## Programmatic DAG (Advanced)

For fine-grained control, build DAGs in Python directly:

```python
import asyncio
from hexdag.kernel.domain import DirectedGraph, NodeSpec
from hexdag.kernel.orchestration.orchestrator import Orchestrator

# Define processing functions
def load_data(inputs: dict) -> dict:
    return {"data": [1, 2, 3, 4, 5]}

def transform_data(inputs: dict) -> dict:
    data = inputs["load"]["data"]
    return {"transformed": [x * 2 for x in data]}

def save_results(inputs: dict) -> dict:
    results = inputs["transform"]["transformed"]
    return {"status": "saved", "count": len(results)}

# Build the DAG
load = NodeSpec(name="load", fn=load_data)
transform = NodeSpec(name="transform", fn=transform_data).after("load")
save = NodeSpec(name="save", fn=save_results).after("transform")

graph = DirectedGraph()
graph.add_many(load, transform, save)

# Execute
async def main():
    orchestrator = Orchestrator()
    results = await orchestrator.run(graph, initial_input={})
    print(f"Results: {results}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Observability

Monitor execution through the observer pattern:

```python
from hexdag.kernel.orchestration.events import NodeStarted, NodeCompleted
from hexdag.kernel.ports.observer_manager import Observer

class LoggingObserver(Observer):
    async def handle(self, event) -> None:
        if isinstance(event, NodeStarted):
            print(f"Started: {event.name}")
        elif isinstance(event, NodeCompleted):
            print(f"Completed: {event.name} ({event.duration_ms}ms)")
```

## Next Steps

- [Architecture](ARCHITECTURE.md) - System architecture and the OS analogy
- [Roadmap](ROADMAP.md) - What's planned and what's been built
- [Node Guide](generated/mcp/node_guide.md) - Creating custom nodes
- [Adapter Guide](generated/mcp/adapter_guide.md) - Creating custom adapters
