# Creating Custom Tools for hexDAG Agents

## Overview

Tools are functions that agents can invoke during execution. They enable
agents to interact with external systems, perform calculations, or access data.

## Quick Start

```python
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression.

    Args:
        expression: Math expression like "2 + 2"

    Returns:
        Result as a string
    """
    result = eval(expression)  # Use safe evaluation in production
    return str(result)
```

## Built-in Tools

### change_phase

Change the agent's reasoning phase with typed context.

**Parameters:**
- `phase` (`str`):
- `previous_phase` (`str | None`, optional): Default: `None`
- `reason` (`str | None`, optional): Default: `None`
- `carried_data` (`dict[str, typing.Any] | None`, optional): Default: `None`
- `target_output` (`str | None`, optional): Default: `None`
- `iteration` (`int | None`, optional): Default: `None`
- `metadata` (`dict[str, typing.Any] | None`, optional): Default: `None`

**Returns:** `dict[str, Any]`

### tool_end

End agent execution with structured output.

**Returns:** `dict[str, Any]`

### adatabase_query

Execute a SQL query and return results.

**Returns:** `list[dict[str, Any]]`

*This is an async tool.*

### adescribe_table

Get column information for a database table.

**Returns:** `list[dict[str, Any]]`

*This is an async tool.*

### alist_tables

List all tables in the database.

**Returns:** `list[str]`

*This is an async tool.*

## Using Tools with Agents

```yaml
- kind: agent_node
  metadata:
    name: research_agent
  spec:
    initial_prompt_template: "Research: {{topic}}"
    max_steps: 5
    tools:
      - hexdag.kernel.domain.agent_tools.tool_end
      - mycompany.tools.search
  dependencies: []
```

## Tool Invocation Format

Agents invoke tools using:
```
INVOKE_TOOL: tool_name(param1="value", param2=123)
```

## Tool Reference

### Synchronous Tools

| Tool | Description | Return Type |
|------|-------------|-------------|
| `change_phase` | Change the agent's reasoning phase with ... | `dict[str, Any]` |
| `tool_end` | End agent execution with structured outp... | `dict[str, Any]` |

### Asynchronous Tools

| Tool | Description | Return Type |
|------|-------------|-------------|
| `adatabase_query` | Execute a SQL query and return results. | `list[dict[str, Any]]` |
| `adescribe_table` | Get column information for a database ta... | `list[dict[str, Any]]` |
| `alist_tables` | List all tables in the database. | `list[str]` |

## Best Practices

1. **Type Hints**: Always add parameter and return types
2. **Docstrings**: Write clear descriptions for LLM understanding
3. **Error Handling**: Return error messages, don't raise exceptions
4. **Idempotent**: Tools should be safe to retry
