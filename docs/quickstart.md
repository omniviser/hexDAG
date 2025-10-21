# Quick Start

This guide will help you create your first HexDAG workflow in minutes.

## Your First DAG

Create a simple data processing workflow:

```python
import asyncio
from hexdag.core.domain import DirectedGraph, NodeSpec
from hexdag.builtin import Orchestrator

# Define processing functions
def load_data(inputs: dict) -> dict:
    """Load data from source."""
    return {"data": [1, 2, 3, 4, 5]}

def transform_data(inputs: dict) -> dict:
    """Transform the data."""
    data = inputs["load"]["data"]
    return {"transformed": [x * 2 for x in data]}

def save_results(inputs: dict) -> dict:
    """Save the results."""
    results = inputs["transform"]["transformed"]
    print(f"Results: {results}")
    return {"status": "saved"}

# Build the DAG
graph = DirectedGraph()
graph.add_node(NodeSpec(name="load", function=load_data))
graph.add_node(NodeSpec(name="transform", function=transform_data, depends_on=["load"]))
graph.add_node(NodeSpec(name="save", function=save_results, depends_on=["transform"]))

# Execute the workflow
async def main():
    orchestrator = Orchestrator()
    results = await orchestrator.arun(graph, inputs={})
    print(f"Final results: {results}")

if __name__ == "__main__":
    asyncio.run(main())
```

## YAML-Based Workflow

For complex workflows, use YAML configuration:

```yaml
# workflow.yaml
name: data_pipeline
description: Simple data processing pipeline

nodes:
  - type: function
    id: load
    function: mymodule.load_data
    depends_on: []

  - type: function
    id: transform
    function: mymodule.transform_data
    depends_on: [load]

  - type: function
    id: save
    function: mymodule.save_results
    depends_on: [transform]
```

Load and run the YAML workflow:

```python
from hexdag.pipeline_builder import YamlPipelineBuilder

# Build from YAML
builder = YamlPipelineBuilder()
graph = builder.build_from_file("workflow.yaml")

# Execute
orchestrator = Orchestrator()
results = await orchestrator.arun(graph, inputs={})
```

## Event Monitoring

Monitor your workflow execution with events:

```python
from hexdag.builtin.events import EventBus, NodeStarted, NodeCompleted

# Subscribe to events
event_bus = EventBus()

def on_node_start(event: NodeStarted):
    print(f"Started: {event.node_name}")

def on_node_complete(event: NodeCompleted):
    print(f"Completed: {event.node_name}")

event_bus.subscribe(NodeStarted, on_node_start)
event_bus.subscribe(NodeCompleted, on_node_complete)

# Run with event monitoring
orchestrator = Orchestrator(event_bus=event_bus)
```

## Next Steps

- [Core Concepts](concepts.md) - Understand DAGs, nodes, and orchestration
- [Node Types](node-types.md) - Explore LLM, Agent, Loop, and Conditional nodes
- [📓 YAML Pipelines Notebook](../notebooks/02_yaml_pipelines.ipynb) - Declarative workflows tutorial
