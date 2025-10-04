# HexDAG

**Enterprise-ready AI agent orchestration framework**

HexDAG transforms complex AI workflows into deterministic, testable, and maintainable systems through declarative YAML configurations and DAG-based orchestration.

## Key Features

- **Async-First Architecture** - Non-blocking execution for maximum performance
- **Event-Driven Observability** - Real-time monitoring via comprehensive event system
- **Pydantic Validation Everywhere** - Type safety at every layer
- **Hexagonal Architecture** - Clean separation of business logic and infrastructure
- **Composable Declarative Files** - Complex workflows from simple YAML components
- **DAG-Based Orchestration** - Intelligent dependency management and parallelization

## Quick Example

```python
from hexai.core.domain import DirectedGraph, NodeSpec
from hexai.core.application import Orchestrator

# Define workflow as DAG
graph = DirectedGraph()
graph.add_node(NodeSpec(
    name="process_data",
    function=process_function,
    depends_on=[]
))

# Execute with async orchestration
orchestrator = Orchestrator()
results = await orchestrator.arun(graph, inputs={"data": "..."})
```

## YAML Pipelines

```yaml
name: ai_workflow
description: AI-powered data processing

nodes:
  - type: agent
    id: researcher
    params:
      initial_prompt_template: "Research: {{topic}}"
      max_steps: 5
    depends_on: []

  - type: llm
    id: analyzer
    params:
      prompt_template: "Analyze: {{researcher.results}}"
    depends_on: [researcher]
```

## Installation

```bash
pip install hexdag
```

## Documentation Structure

- **Getting Started** - Installation, quickstart, and core concepts
- **User Guide** - Detailed guides for DAGs, nodes, YAML pipelines, events
- **Reference** - Auto-generated component documentation from registry
- **Namespaces** - Components organized by namespace
- **Development** - Contributing, plugin development, architecture

## Next Steps

1. [Installation Guide](getting-started/installation.md) - Set up HexDAG
2. [Quick Start](getting-started/quickstart.md) - Build your first workflow
3. [Core Concepts](getting-started/concepts.md) - Understand the architecture
4. [Node Types](guide/node-types.md) - Explore available node types
5. [YAML Pipelines](guide/yaml-pipelines.md) - Declarative workflow configuration
