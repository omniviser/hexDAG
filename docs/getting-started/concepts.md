# Core Concepts

This page covers the key concepts you need to understand hexDAG. For the full deep-dive, see the [Framework Guide](../GUIDE.md).

## Pipelines

A pipeline is a YAML file that defines a workflow as a directed acyclic graph (DAG) of nodes.

```yaml
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: my-workflow
spec:
  ports:
    llm:
      adapter: openai
      config:
        model: gpt-4
  nodes:
    - kind: llm_node
      metadata:
        name: analyzer
      spec:
        human_message: "Analyze: {{ $input.topic }}"
```

## Nodes

Nodes are the processing steps in a pipeline. Each node wraps an async function with type-validated inputs and outputs. See [Node Types](../GUIDE.md#8-node-types) for the full list.

## Ports and Adapters

**Ports** define contracts — "I need something that can generate text." **Adapters** implement them — "Use OpenAI GPT-4." This lets you swap implementations without changing your pipeline logic. See [Ports, Adapters, and Middleware](../GUIDE.md#7-ports-adapters-and-middleware).

## Data Flow

Upstream node outputs are automatically available downstream. You reference them with `node_name.field` syntax, and the compiler auto-detects dependencies. See [Data Flow Between Nodes](../GUIDE.md#9-data-flow-between-nodes).

## The 4 YAML Syntaxes

| Syntax | Purpose | Example |
|--------|---------|---------|
| `!include` | Merge another file | `!include shared/ports.yaml` |
| `${VAR}` | Environment variable | `${OPENAI_API_KEY}` |
| `{{ expr }}` | Jinja2 template | `{{ analyzer.result }}` |
| `node.field` | Value extraction | `"analyzer.result"` |

See [YAML Syntax](../GUIDE.md#5-yaml-syntax-the-4-special-syntaxes) for the full explanation.

## Services

Services wrap ports behind a stable API. Methods decorated with `@tool` are agent-callable; `@step` methods are DAG nodes. See [Services](../GUIDE.md#10-services).

## Further Reading

- [Framework Guide](../GUIDE.md) — comprehensive guide to how everything fits together
- [Architecture](../ARCHITECTURE.md) — design philosophy and decision rules
- [YAML Reference](../generated/mcp/yaml_reference.md) — complete manifest specification
