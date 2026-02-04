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
- `phase` (`str`): The new phase name to transition to
- `previous_phase` (`str | None`, optional): The phase being transitioned from (auto-filled if not provided) Default: `None`
- `reason` (`str | None`, optional): Explanation for why the phase change is occurring Default: `None`
- `carried_data` (`dict[str, typing.Any] | None`, optional): Data to carry forward from the previous phase Default: `None`
- `target_output` (`str | None`, optional): Expected output format or goal for the new phase Default: `None`
- `iteration` (`int | None`, optional): Current iteration number if in a loop or retry scenario Default: `None`
- `metadata` (`dict[str, typing.Any] | None`, optional): Additional metadata about the phase transition Default: `None`

**Returns:** `dict[str, Any]`

### tool_end

End tool execution with structured output.

**Returns:** `dict[str, Any]`

### database_execute

Execute a SQL command (INSERT, UPDATE, DELETE) on the database.

**Parameters:**
- `sql` (`str`): SQL command to execute
- `params` (`dict[str, typing.Any] | None`, optional): Optional parameters for parameterized queries Default: `None`
- `database_port` (`typing.Any | None`, optional): Injected database port (provided by framework) Default: `None`

**Returns:** `None`

*This is an async tool.*

### database_query

Execute a SQL query on the database.

**Parameters:**
- `sql` (`str`): SQL query to execute
- `params` (`dict[str, typing.Any] | None`, optional): Optional parameters for parameterized queries Default: `None`
- `database_port` (`hexdag.core.ports.database.DatabasePort | None`, optional): Injected database port (provided by framework) Default: `None`

**Returns:** `list[dict[str, Any]]`

*This is an async tool.*

### database_query_sync

Execute a database query synchronously.

**Parameters:**
- `sql` (`str`): SQL query to execute
- `params` (`dict[str, typing.Any] | None`, optional): Optional parameters Default: `None`
- `database_port` (`hexdag.core.ports.database.DatabasePort | None`, optional): Injected database port Default: `None`

**Returns:** `list[dict[str, Any]]`

### describe_table

Get schema information for a database table.

**Parameters:**
- `table` (`str`): Name of the table
- `database_port` (`hexdag.core.ports.database.DatabasePort | None`, optional): Injected database port Default: `None`

**Returns:** `list[dict[str, Any]]`

*This is an async tool.*

### list_tables

List all tables in the database.

**Parameters:**
- `database_port` (`hexdag.core.ports.database.DatabasePort | None`, optional): Injected database port Default: `None`

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
      - hexdag.builtin.tools.builtin_tools.tool_end
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
| `tool_end` | End tool execution with structured outpu... | `dict[str, Any]` |
| `database_query_sync` | Execute a database query synchronously. | `list[dict[str, Any]]` |

### Asynchronous Tools

| Tool | Description | Return Type |
|------|-------------|-------------|
| `database_execute` | Execute a SQL command (INSERT, UPDATE, D... | `None` |
| `database_query` | Execute a SQL query on the database. | `list[dict[str, Any]]` |
| `describe_table` | Get schema information for a database ta... | `list[dict[str, Any]]` |
| `list_tables` | List all tables in the database. | `list[str]` |

## Best Practices

1. **Type Hints**: Always add parameter and return types
2. **Docstrings**: Write clear descriptions for LLM understanding
3. **Error Handling**: Return error messages, don't raise exceptions
4. **Idempotent**: Tools should be safe to retry
