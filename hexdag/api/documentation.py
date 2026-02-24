"""Documentation API.

Provides unified functions for hexDAG documentation and guides.
"""

from __future__ import annotations

from pathlib import Path

# Generated documentation directory
_GENERATED_DOCS_DIR = Path(__file__).parent.parent.parent / "docs" / "generated" / "mcp"


def _load_generated_doc(filename: str) -> str | None:
    """Load generated documentation from file.

    Parameters
    ----------
    filename : str
        Name of the documentation file (e.g., "adapter_guide.md")

    Returns
    -------
    str | None
        File contents if exists, None otherwise
    """
    doc_path = _GENERATED_DOCS_DIR / filename
    if doc_path.exists():
        return doc_path.read_text()
    return None


def get_syntax_reference() -> str:
    """Get reference for hexDAG YAML syntax including variable references.

    Returns comprehensive documentation on:
    - $input.field - Reference initial pipeline input
    - {{node.output}} - Jinja2 template for node outputs
    - ${ENV_VAR} - Environment variables
    - input_mapping syntax and usage

    Returns
    -------
    str
        Detailed syntax reference documentation

    Examples
    --------
    >>> ref = get_syntax_reference()
    >>> "$input" in ref
    True
    """
    # Try to load auto-generated documentation first
    generated = _load_generated_doc("syntax_reference.md")
    if generated:
        return generated

    # Fallback to static documentation
    return """# hexDAG Variable Reference Syntax

## 1. Initial Input Reference: $input

Use `$input.field` in `input_mapping` to access the original pipeline input.
This allows passing data from the initial request through multiple pipeline stages.

```yaml
nodes:
  - kind: function_node
    metadata:
      name: processor
    spec:
      fn: myapp.process
      input_mapping:
        load_id: $input.load_id        # Gets initial input's load_id
        carrier: $input.carrier_mc     # Gets initial input's carrier_mc
    dependencies: [extractor]
```

**Key Points:**
- `$input` refers to the ENTIRE initial pipeline input
- `$input.field` extracts a specific field from initial input
- Works regardless of node dependencies
- Useful for passing request context through the pipeline

## 2. Node Output Reference in Prompt Templates: {{node.field}}

Use Jinja2 syntax in prompt templates to reference previous node outputs.

```yaml
- kind: llm_node
  metadata:
    name: analyzer
  spec:
    prompt_template: |
      Analyze this data:
      {{extractor.result}}

      Previous analysis:
      {{validator.summary}}
  dependencies: [extractor, validator]
```

**Key Points:**
- Use `{{node_name.field}}` syntax
- Nodes must be in dependencies to reference them
- Works in prompt_template, initial_prompt_template, etc.

## 3. Environment Variables: ${ENV_VAR}

Reference environment variables in configuration.

```yaml
spec:
  ports:
    llm:
      adapter: openai
      config:
        api_key: ${OPENAI_API_KEY}
        model: ${MODEL:gpt-4}  # With default value
```

**Key Points:**
- `${VAR}` - Required variable
- `${VAR:default}` - With default value
- Resolved at pipeline build time

## 4. Input Mapping

Map data between nodes using input_mapping.

```yaml
- kind: function_node
  metadata:
    name: transformer
  spec:
    fn: myapp.transform
    input_mapping:
      data: previous_node  # Map entire output
      id: $input.request_id  # Map from initial input
  dependencies: [previous_node]
```
"""


def get_type_reference() -> str:
    """Get reference for hexDAG YAML output_schema type system.

    Returns documentation on supported types, nullable types,
    and examples for LLM output parsing.

    Returns
    -------
    str
        Type system reference documentation
    """
    generated = _load_generated_doc("type_reference.md")
    if generated:
        return generated

    return """# hexDAG Output Schema Type Reference

## Supported Types

Use these types in `output_schema` on `llm_node`, `agent_node`, etc.

### Required Types (non-nullable)

- `str`, `int`, `float`, `bool`, `list`, `dict`, `Any`

### Nullable Types (accept null, default to null)

Append `?` to any base type. Nullable fields accept `null` without
validation errors and default to `null` when omitted.

- `str?`, `int?`, `float?`, `bool?`, `list?`, `dict?`

## Example

```yaml
- kind: llm_node
  metadata:
    name: extract_data
  spec:
    prompt_template: "Extract from: {{text}}"
    parse_json: true
    output_schema:
      rate: float        # Required
      order_id: str      # Required
      mc_number: str?    # Nullable — null if not found
      origin: str?       # Nullable
```

## When to Use

- **Required** (`str`): LLM must always provide a value
- **Nullable** (`str?`): Field may legitimately be absent or null
"""


def get_custom_adapter_guide() -> str:
    """Get a comprehensive guide for creating custom adapters.

    Returns documentation on:
    - Creating adapters by implementing port interfaces
    - Using custom adapters in YAML pipelines
    - Testing patterns for adapters

    Returns
    -------
    str
        Detailed guide for creating custom adapters
    """
    generated = _load_generated_doc("adapter_guide.md")
    if generated:
        return generated

    return '''# Creating Custom Adapters in hexDAG

## Overview

Adapters implement "ports" (interfaces) that connect your pipelines to external
services like LLMs, databases, and APIs. They are plain Python classes that
implement the required port interface.

## Quick Start

### LLM Adapter Example

```python
from hexdag.kernel import LLM

class MyLLMAdapter(LLM):
    """Custom LLM adapter."""

    def __init__(
        self,
        model: str = "gpt-4",
        api_key: str | None = None,
        temperature: float = 0.7
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.temperature = temperature

    async def aresponse(self, messages: list[dict]) -> str:
        # Your implementation here
        ...
```

### Cache Adapter Example

```python
class MemoryCacheAdapter:
    """Simple in-memory cache adapter."""

    def __init__(self, max_size: int = 100, ttl: int = 3600):
        self.cache = {}
        self.max_size = max_size
        self.ttl = ttl

    async def aget(self, key: str):
        return self.cache.get(key)

    async def aset(self, key: str, value: any):
        self.cache[key] = value
```

## Using in YAML

Reference adapters by their full module path:

```yaml
spec:
  ports:
    llm:
      adapter: mypackage.adapters.MyLLMAdapter
      config:
        model: gpt-4-turbo
        temperature: 0.5
```
'''


def get_custom_node_guide() -> str:
    """Get a comprehensive guide for creating custom nodes.

    Returns documentation on:
    - Creating nodes with the node factory pattern
    - Input/output schemas
    - Using custom nodes in YAML pipelines

    Returns
    -------
    str
        Detailed guide for creating custom nodes
    """
    generated = _load_generated_doc("node_guide.md")
    if generated:
        return generated

    return '''# Creating Custom Nodes in hexDAG

## Overview

Nodes are the building blocks of hexDAG pipelines. Each node processes data
and passes results to dependent nodes.

## Node Factory Pattern

```python
from hexdag.kernel import NodeSpec
from hexdag.stdlib.nodes import BaseNodeFactory

class MyProcessorNode(BaseNodeFactory):
    """Custom data processor node."""

    def __call__(
        self,
        name: str,
        config: dict,
        timeout: float = 30.0,
        **kwargs
    ) -> NodeSpec:
        async def process(inputs: dict, context):
            # Your processing logic here
            return {"result": processed_data}

        return NodeSpec(
            id=name,
            fn=process,
            input_schema={"data": str},
            output_schema={"result": str},
        )
```

## Using in YAML

Reference nodes by their full module path:

```yaml
nodes:
  - kind: mypackage.nodes.MyProcessorNode
    metadata:
      name: processor1
    spec:
      config:
        key: value
      timeout: 60.0
    dependencies: [input_node]
```
'''


def get_custom_tool_guide() -> str:
    """Get a guide for creating custom tools for agents.

    Returns documentation on:
    - Creating tools as Python functions
    - Tool schemas and descriptions
    - Using tools with agents

    Returns
    -------
    str
        Detailed guide for creating custom tools
    """
    generated = _load_generated_doc("tool_guide.md")
    if generated:
        return generated

    return '''# Creating Custom Tools in hexDAG

## Overview

Tools are functions that agents can use during their reasoning process.
They provide capabilities like web search, database queries, etc.

Tools are plain Python functions with type hints and docstrings.

## Creating a Tool

```python
def search_database(query: str, limit: int = 10) -> list[dict]:
    """Search the database for matching records.

    Args:
        query: Search query string
        limit: Maximum number of results

    Returns:
        List of matching records
    """
    # Your search logic here
    return results
```

## Using Tools with Agents

Reference tools by their full module path in YAML:

```yaml
nodes:
  - kind: agent_node
    metadata:
      name: research_agent
    spec:
      tools:
        - mypackage.tools.search_database
        - mypackage.tools.web_search
      initial_prompt_template: |
        Research the following topic: {{input}}
```

## Tool Requirements

- **Type hints**: Required for automatic schema generation
- **Docstring**: Provides description for the LLM
- **Args section**: Documents each parameter
- **Returns section**: Documents return type
'''


def get_extension_guide(component_type: str | None = None) -> str:
    """Get a guide for extending hexDAG with custom components.

    Parameters
    ----------
    component_type : str | None
        Optional specific component type to get guide for:
        - "adapter" - Custom adapter guide
        - "node" - Custom node guide
        - "tool" - Custom tool guide
        If None, returns overview of all extension types.

    Returns
    -------
    str
        Extension guide documentation
    """
    if component_type == "adapter":
        return get_custom_adapter_guide()
    if component_type == "node":
        return get_custom_node_guide()
    if component_type == "tool":
        return get_custom_tool_guide()
    return """# Extending hexDAG

hexDAG can be extended with custom components:

## 1. Custom Adapters
Connect to external services (LLMs, databases, APIs).
Implement port interfaces (e.g., `LLM`, `Memory`, `Database`).
Use `get_custom_adapter_guide()` for details.

## 2. Custom Nodes
Create new processing logic for pipelines.
Extend `BaseNodeFactory` and implement the `__call__` method.
Use `get_custom_node_guide()` for details.

## 3. Custom Tools
Add capabilities for agents to use.
Create functions with type hints and docstrings.
Use `get_custom_tool_guide()` for details.

## Quick Reference

| Component | Pattern | Purpose |
|-----------|---------|---------|
| Adapter | Implement port interface | External service integration |
| Node | Extend BaseNodeFactory | Pipeline processing step |
| Tool | Function with type hints | Agent capability |

## Using Custom Components in YAML

Reference components by their full module path:

```yaml
spec:
  ports:
    llm:
      adapter: mypackage.adapters.MyLLMAdapter
  nodes:
    - kind: mypackage.nodes.MyNode
      spec:
        tools:
          - mypackage.tools.my_tool
```
"""


def explain_yaml_structure() -> str:
    """Explain the structure of hexDAG YAML pipelines.

    Returns
    -------
    str
        YAML structure documentation
    """
    generated = _load_generated_doc("yaml_structure.md")
    if generated:
        return generated

    return """# hexDAG YAML Pipeline Structure

## Basic Structure

```yaml
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: my-pipeline
  description: Optional description
spec:
  ports:
    llm:
      adapter: openai
      config:
        model: gpt-4
    memory:
      adapter: in_memory
  nodes:
    - kind: llm_node
      metadata:
        name: analyzer
      spec:
        prompt_template: "Analyze: {{input}}"
      dependencies: []
```

## Required Fields

- `apiVersion`: Always "hexdag/v1"
- `kind`: Always "Pipeline"
- `metadata.name`: Unique pipeline identifier
- `spec.nodes`: List of pipeline nodes

## Node Structure

Each node has:
- `kind`: Node type (llm_node, function_node, agent_node, etc.)
- `metadata.name`: Unique node name
- `spec`: Node-specific configuration
- `dependencies`: List of node names this node depends on

## Port Configuration

Ports connect pipelines to external services:
- `llm`: Language model access
- `memory`: Persistent memory
- `database`: Data storage
- `secret`: Secret management
"""
