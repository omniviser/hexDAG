# Node Types

HexDAG provides several built-in node types for different processing patterns.

## Overview

All nodes in HexDAG are registered in the component registry and can be used in both Python code and YAML pipelines.

See the [Nodes Reference](../reference/nodes.md) for complete API documentation.

## Available Node Types

### FunctionNode

Execute arbitrary Python functions with validation.

```python
from hexai.core.domain import NodeSpec

def process_data(inputs: dict) -> dict:
    return {"result": inputs["data"] * 2}

NodeSpec(name="process", function=process_data)
```

### LLMNode

Language model interactions with prompt templating.

```yaml
- type: llm
  id: summarizer
  params:
    prompt_template: "Summarize: {{text}}"
    model: "gpt-4"
```

### AgentNode

Multi-step reasoning agents with tool access.

```yaml
- type: agent
  id: researcher
  params:
    initial_prompt_template: "Research {{topic}}"
    max_steps: 10
    tools: [search, calculator]
```

### LoopNode

Iterative processing with custom conditions.

```yaml
- type: loop
  id: process_batch
  params:
    max_iterations: 100
    condition: "{{continue_processing}}"
```

### ConditionalNode

Conditional routing based on runtime values.

```yaml
- type: conditional
  id: check_quality
  params:
    condition: "{{quality_score > 0.8}}"
```

## Next Steps

- [YAML Pipelines](yaml-pipelines.md) - Build declarative workflows
- [Nodes Reference](../reference/nodes.md) - Complete API documentation
