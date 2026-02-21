"""Guide generators for MCP documentation.

This module generates documentation directly from extracted component
documentation - no external templates needed.
"""

import json
from pathlib import Path
from typing import Any

from hexdag.core.docs.models import AdapterDoc, NodeDoc, ToolDoc
from hexdag.core.logging import get_logger

logger = get_logger(__name__)

# Path to the generated schema
SCHEMA_PATH = Path(__file__).parent.parent.parent.parent / "schemas" / "pipeline-schema.json"


class GuideGenerator:
    """Generate documentation guides from extracted component docs.

    All documentation is generated programmatically from code introspection,
    ensuring it stays in sync with the actual implementation.
    """

    def generate_adapter_guide(self, adapters: list[AdapterDoc]) -> str:
        """Generate adapter creation guide.

        Parameters
        ----------
        adapters : list[AdapterDoc]
            List of adapter documentation objects

        Returns
        -------
        str
            Complete adapter guide as markdown
        """
        # Group adapters by port type
        adapters_by_port: dict[str, list[AdapterDoc]] = {}
        for adapter in adapters:
            port = adapter.port_type
            if port not in adapters_by_port:
                adapters_by_port[port] = []
            adapters_by_port[port].append(adapter)

        lines = [
            "# Creating Custom Adapters in hexDAG",
            "",
            "## Overview",
            "",
            "hexDAG uses adapters to connect pipelines to external services like LLMs,",
            "databases, and APIs. Adapters implement ports (interfaces) with async methods.",
            "",
            "## Quick Start",
            "",
            "### Simple Adapter (No Secrets)",
            "",
            "```python",
            "class MemoryCacheAdapter:",
            '    """Simple in-memory cache adapter."""',
            "",
            "    def __init__(self, max_size: int = 100, ttl: int = 3600):",
            "        self.cache = {}",
            "        self.max_size = max_size",
            "        self.ttl = ttl",
            "",
            "    async def aget(self, key: str):",
            "        return self.cache.get(key)",
            "",
            "    async def aset(self, key: str, value: any):",
            "        self.cache[key] = value",
            "```",
            "",
            "### Adapter with Secrets",
            "",
            "Use `secret()` in defaults to declare secrets that auto-resolve from environment:",
            "",
            "```python",
            "from hexdag.core.secrets import secret",
            "",
            "class OpenAIAdapter:",
            '    """OpenAI LLM adapter with automatic secret resolution."""',
            "",
            "    def __init__(",
            "        self,",
            '        api_key: str = secret(env="OPENAI_API_KEY"),  # Auto-resolved',
            '        model: str = "gpt-4",',
            "        temperature: float = 0.7",
            "    ):",
            "        self.api_key = api_key",
            "        self.model = model",
            "        self.temperature = temperature",
            "",
            "    async def aresponse(self, messages: list) -> str:",
            "        # Your implementation using self.api_key",
            "        ...",
            "```",
            "",
            "## Secret Resolution",
            "",
            "Secrets declared with `secret()` are resolved in this order:",
            "1. **Explicit kwargs** - Values passed directly to `__init__`",
            "2. **Environment variables** - From the `env` parameter",
            "3. **Memory port** - From orchestrator memory (with `secret:` prefix)",
            "4. **Error** - If required and no default",
            "",
            "## Available Adapters",
            "",
        ]

        # Generate adapter tables by port type
        for port_type in sorted(adapters_by_port.keys()):
            port_adapters = adapters_by_port[port_type]
            lines.append(f"### {port_type}")
            lines.append("")
            lines.append("| Adapter | Description |")
            lines.append("|---------|-------------|")
            for adapter in port_adapters:
                desc = adapter.description[:60]
                if len(adapter.description) > 60:
                    desc += "..."
                lines.append(f"| `{adapter.name}` | {desc} |")
            lines.append("")

        # Add YAML usage section
        lines.extend([
            "## Using Adapters in YAML",
            "",
            "```yaml",
            "apiVersion: hexdag/v1",
            "kind: Pipeline",
            "metadata:",
            "  name: my-pipeline",
            "spec:",
            "  ports:",
            "    llm:",
            "      adapter: hexdag.builtin.adapters.openai.OpenAIAdapter",
            "      config:",
            "        api_key: ${OPENAI_API_KEY}",
            "        model: gpt-4",
            "",
            "  nodes:",
            "    - kind: llm_node",
            "      metadata:",
            "        name: analyzer",
            "      spec:",
            '        prompt_template: "Analyze: {{input}}"',
            "      dependencies: []",
            "```",
            "",
            "## Best Practices",
            "",
            "1. **Async First**: Use `async def` for I/O operations",
            "2. **Type Hints**: Add type annotations for better tooling",
            "3. **Docstrings**: Document your adapter's purpose and config",
            "4. **Error Handling**: Wrap external calls in try/except",
            "5. **Secrets**: Use `secret()` - never hardcode secrets",
        ])

        return "\n".join(lines)

    def generate_node_guide(self, nodes: list[NodeDoc]) -> str:
        """Generate node creation guide.

        Parameters
        ----------
        nodes : list[NodeDoc]
            List of node documentation objects

        Returns
        -------
        str
            Complete node guide as markdown
        """
        lines = [
            "# Creating Custom Nodes in hexDAG",
            "",
            "## Overview",
            "",
            "Nodes are the building blocks of hexDAG pipelines. Each node performs a specific",
            "task and can be connected to other nodes via dependencies.",
            "",
            "## Quick Start",
            "",
            "### Using FunctionNode (Simplest)",
            "",
            "Reference any Python function by module path:",
            "",
            "```yaml",
            "- kind: function_node",
            "  metadata:",
            "    name: my_processor",
            "  spec:",
            "    fn: mycompany.processors.process_data",
            "  dependencies: []",
            "```",
            "",
            "```python",
            "# mycompany/processors.py",
            "def process_data(input_data: dict) -> dict:",
            '    """Your processing logic."""',
            '    return {"result": input_data["value"] * 2}',
            "```",
            "",
            "## Available Node Types",
            "",
        ]

        # Generate node documentation
        for node in nodes:
            lines.append(f"### {node.name}")
            lines.append("")
            lines.append(f"{node.description}")
            lines.append("")
            lines.append(f"**Kind**: `{node.kind}`")
            lines.append("")

            if node.parameters:
                lines.append("| Parameter | Type | Required | Description |")
                lines.append("|-----------|------|----------|-------------|")
                for param in node.parameters:
                    req = "Yes" if param.required else "No"
                    desc = param.description[:50]
                    if len(param.description) > 50:
                        desc += "..."
                    lines.append(f"| `{param.name}` | `{param.type_hint}` | {req} | {desc} |")
                lines.append("")

            if node.yaml_example:
                lines.append("**Example:**")
                lines.append("```yaml")
                lines.append(node.yaml_example.strip())
                lines.append("```")
                lines.append("")

        # Add custom node section
        lines.extend([
            "## Creating Custom Nodes",
            "",
            "```python",
            "from hexdag.builtin.nodes import BaseNodeFactory",
            "from hexdag.core.domain.dag import NodeSpec",
            "",
            "class CustomProcessorNode(BaseNodeFactory):",
            '    """Custom node for specialized processing."""',
            "",
            "    def __call__(",
            "        self,",
            "        name: str,",
            "        threshold: float = 0.5,",
            "        **kwargs",
            "    ) -> NodeSpec:",
            "        async def process_fn(input_data: dict) -> dict:",
            '            if input_data.get("score", 0) > threshold:',
            '                return {"status": "pass"}',
            '            return {"status": "fail"}',
            "",
            "        return NodeSpec(",
            "            name=name,",
            "            fn=process_fn,",
            '            deps=frozenset(kwargs.get("deps", [])),',
            "        )",
            "```",
            "",
            "## Best Practices",
            "",
            "1. **Async Functions**: Use `async def` for the node function",
            "2. **Immutable**: Don't modify input_data; return new dict",
            "3. **Type Hints**: Add types for better IDE support",
            "4. **Docstrings**: Document purpose and parameters",
        ])

        return "\n".join(lines)

    def generate_tool_guide(self, tools: list[ToolDoc]) -> str:
        """Generate tool creation guide.

        Parameters
        ----------
        tools : list[ToolDoc]
            List of tool documentation objects

        Returns
        -------
        str
            Complete tool guide as markdown
        """
        # Separate sync and async tools
        async_tools = [t for t in tools if t.is_async]
        sync_tools = [t for t in tools if not t.is_async]

        lines = [
            "# Creating Custom Tools for hexDAG Agents",
            "",
            "## Overview",
            "",
            "Tools are functions that agents can invoke during execution. They enable",
            "agents to interact with external systems, perform calculations, or access data.",
            "",
            "## Quick Start",
            "",
            "```python",
            "def calculate(expression: str) -> str:",
            '    """Evaluate a mathematical expression.',
            "",
            "    Args:",
            '        expression: Math expression like "2 + 2"',
            "",
            "    Returns:",
            "        Result as a string",
            '    """',
            "    result = eval(expression)  # Use safe evaluation in production",
            "    return str(result)",
            "```",
            "",
            "## Built-in Tools",
            "",
        ]

        # Generate tool documentation
        for tool in tools:
            lines.append(f"### {tool.name}")
            lines.append("")
            lines.append(tool.description)
            lines.append("")

            if tool.parameters:
                lines.append("**Parameters:**")
                for param in tool.parameters:
                    opt = "" if param.required else ", optional"
                    default = f" Default: `{param.default}`" if param.default else ""
                    desc_part = f" {param.description}" if param.description else ""
                    lines.append(
                        f"- `{param.name}` (`{param.type_hint}`{opt}):{desc_part}{default}"
                    )
                lines.append("")

            lines.append(f"**Returns:** `{tool.return_type}`")
            if tool.is_async:
                lines.append("")
                lines.append("*This is an async tool.*")
            lines.append("")

        # Add usage section
        lines.extend([
            "## Using Tools with Agents",
            "",
            "```yaml",
            "- kind: agent_node",
            "  metadata:",
            "    name: research_agent",
            "  spec:",
            '    initial_prompt_template: "Research: {{topic}}"',
            "    max_steps: 5",
            "    tools:",
            "      - hexdag.core.domain.agent_tools.tool_end",
            "      - mycompany.tools.search",
            "  dependencies: []",
            "```",
            "",
            "## Tool Invocation Format",
            "",
            "Agents invoke tools using:",
            "```",
            'INVOKE_TOOL: tool_name(param1="value", param2=123)',
            "```",
            "",
            "## Tool Reference",
            "",
        ])

        # Add reference tables
        if sync_tools:
            lines.append("### Synchronous Tools")
            lines.append("")
            lines.append("| Tool | Description | Return Type |")
            lines.append("|------|-------------|-------------|")
            for tool in sync_tools:
                desc = tool.description[:40]
                if len(tool.description) > 40:
                    desc += "..."
                lines.append(f"| `{tool.name}` | {desc} | `{tool.return_type}` |")
            lines.append("")

        if async_tools:
            lines.append("### Asynchronous Tools")
            lines.append("")
            lines.append("| Tool | Description | Return Type |")
            lines.append("|------|-------------|-------------|")
            for tool in async_tools:
                desc = tool.description[:40]
                if len(tool.description) > 40:
                    desc += "..."
                lines.append(f"| `{tool.name}` | {desc} | `{tool.return_type}` |")
            lines.append("")

        lines.extend([
            "## Best Practices",
            "",
            "1. **Type Hints**: Always add parameter and return types",
            "2. **Docstrings**: Write clear descriptions for LLM understanding",
            "3. **Error Handling**: Return error messages, don't raise exceptions",
            "4. **Idempotent**: Tools should be safe to retry",
        ])

        return "\n".join(lines)

    def generate_syntax_reference(self) -> str:
        """Generate syntax reference guide.

        Returns
        -------
        str
            Complete syntax reference as markdown
        """
        return """# hexDAG Variable Reference Syntax

## 1. Initial Input Reference: $input

Use `$input.field` in `input_mapping` to access the original pipeline input.

```yaml
nodes:
  - kind: function_node
    metadata:
      name: processor
    spec:
      fn: myapp.process
      input_mapping:
        load_id: $input.load_id
        carrier: $input.carrier_mc
    dependencies: [extractor]
```

## 2. Node Output Reference: {{node.field}}

Use Jinja2 syntax in prompt templates to reference previous node outputs.

```yaml
- kind: llm_node
  metadata:
    name: analyzer
  spec:
    prompt_template: |
      Analyze this data:
      {{extractor.result}}
```

## 3. Environment Variables: ${VAR}

```yaml
spec:
  ports:
    llm:
      config:
        model: ${MODEL}              # Resolved at build time
        api_key: ${OPENAI_API_KEY}   # Secret - resolved at runtime
```

**Secret Patterns (deferred to runtime):**
- `*_API_KEY`, `*_SECRET`, `*_TOKEN`, `*_PASSWORD`, `*_CREDENTIAL`, `SECRET_*`

## 4. Input Mapping

```yaml
- kind: function_node
  metadata:
    name: merger
  spec:
    fn: myapp.merge_results
    input_mapping:
      request_id: $input.id          # From initial input
      analysis: analyzer.result       # From dependency
  dependencies: [analyzer]
```

## Quick Reference

| Syntax | Location | Purpose |
|--------|----------|---------|
| `$input.field` | input_mapping | Access initial pipeline input |
| `{{node.field}}` | prompt_template | Jinja2 template reference |
| `${VAR}` | Any string | Environment variable |
| `${VAR:default}` | Any string | Env var with default |
| `node.path` | input_mapping | Dependency output extraction |
"""

    def generate_extension_guide(
        self,
        adapters: list[AdapterDoc],
        nodes: list[NodeDoc],
        tools: list[ToolDoc],
    ) -> str:
        """Generate extension overview guide.

        Parameters
        ----------
        adapters : list[AdapterDoc]
            List of adapter documentation objects
        nodes : list[NodeDoc]
            List of node documentation objects
        tools : list[ToolDoc]
            List of tool documentation objects

        Returns
        -------
        str
            Complete extension guide as markdown
        """
        lines = [
            "# Extending hexDAG - Overview",
            "",
            "## Extension Points",
            "",
            "| Component | Purpose | Available |",
            "|-----------|---------|-----------|",
            f"| **Adapter** | Connect to external services | {len(adapters)} |",
            f"| **Node** | Custom processing logic | {len(nodes)} |",
            f"| **Tool** | Agent-callable functions | {len(tools)} |",
            "",
            "## Quick Reference",
            "",
            "### Adapters",
            "Use `get_custom_adapter_guide()` for full documentation.",
            "",
            "### Nodes",
            "Use `get_custom_node_guide()` for full documentation.",
            "",
            "### Tools",
            "Use `get_custom_tool_guide()` for full documentation.",
            "",
            "## MCP Tools for Development",
            "",
            "| Tool | Purpose |",
            "|------|---------|",
            "| `list_nodes()` | See available nodes |",
            "| `list_adapters()` | See available adapters |",
            "| `list_tools()` | See available tools |",
            "| `get_component_schema()` | Get config schema |",
            "| `validate_yaml_pipeline()` | Validate your YAML |",
            "| `get_pipeline_schema()` | Get full JSON schema |",
        ]

        return "\n".join(lines)

    def generate_pipeline_schema_guide(self) -> str:
        """Generate pipeline schema reference guide from JSON schema.

        Reads the auto-generated pipeline-schema.json and produces
        a human-readable markdown reference.

        Returns
        -------
        str
            Complete pipeline schema guide as markdown
        """
        lines = [
            "# hexDAG Pipeline Schema Reference",
            "",
            "This reference is auto-generated from the pipeline JSON schema.",
            "",
            "## Overview",
            "",
            "hexDAG pipelines are defined in YAML using a Kubernetes-like structure.",
            "The schema provides validation and IDE autocompletion support.",
            "",
            "## Pipeline Structure",
            "",
            "```yaml",
            "apiVersion: hexdag/v1",
            "kind: Pipeline",
            "metadata:",
            "  name: my-pipeline",
            "  description: Pipeline description",
            "spec:",
            "  ports: {}     # Adapter configurations",
            "  nodes: []     # Processing nodes",
            "  events: {}    # Event handlers",
            "```",
            "",
        ]

        # Try to load and parse schema
        try:
            schema = self._load_schema()
            if schema:
                lines.extend(self._generate_node_types_section(schema))
                lines.extend(self._generate_ports_section(schema))
                lines.extend(self._generate_events_section(schema))
        except Exception as e:
            logger.warning(f"Could not load pipeline schema: {e}")
            lines.extend([
                "## Node Types",
                "",
                "*Schema not available. Run `scripts/generate_schemas.py` first.*",
                "",
            ])

        # Add IDE setup section
        lines.extend([
            "## IDE Setup",
            "",
            "### VS Code",
            "",
            "Add to `.vscode/settings.json`:",
            "",
            "```json",
            "{",
            '  "yaml.schemas": {',
            '    "./schemas/pipeline-schema.json": ["*.yaml", "pipelines/*.yaml"]',
            "  }",
            "}",
            "```",
            "",
            "### Schema Location",
            "",
            (
                "The schema file is at `schemas/pipeline-schema.json` and is "
                "auto-generated from node `_yaml_schema` attributes."
            ),
        ])

        return "\n".join(lines)

    def _load_schema(self) -> dict[str, Any] | None:
        """Load the pipeline schema JSON file.

        Returns
        -------
        dict[str, Any] | None
            Parsed schema or None if not found
        """
        if not SCHEMA_PATH.exists():
            return None
        return json.loads(SCHEMA_PATH.read_text())

    def _generate_node_types_section(self, schema: dict[str, Any]) -> list[str]:
        """Generate node types documentation from schema.

        Parameters
        ----------
        schema : dict[str, Any]
            Parsed JSON schema

        Returns
        -------
        list[str]
            Lines of markdown documentation
        """
        lines = ["## Node Types", ""]

        defs = schema.get("$defs", {})

        # Find all node specs
        node_specs = [(name, spec) for name, spec in defs.items() if name.endswith("NodeSpec")]

        if not node_specs:
            lines.append("*No node types found in schema.*")
            lines.append("")
            return lines

        # Generate table
        lines.append("| Node Kind | Description |")
        lines.append("|-----------|-------------|")

        for name, spec in sorted(node_specs):
            kind = name.replace("NodeSpec", "").lower()
            # Convert CamelCase to snake_case
            kind = "".join(f"_{c.lower()}" if c.isupper() else c for c in kind).lstrip("_")

            # Get description from spec
            desc = spec.get("description", "")
            if not desc:
                props = spec.get("properties", {})
                spec_prop = props.get("spec", {})
                desc = spec_prop.get("description", "No description")

            # Truncate long descriptions
            if len(desc) > 60:
                desc = desc[:57] + "..."

            lines.append(f"| `{kind}_node` | {desc} |")

        lines.append("")

        # Generate detailed sections for each node
        for name, spec in sorted(node_specs):
            lines.extend(self._generate_node_detail(name, spec))

        return lines

    def _generate_node_detail(self, name: str, spec: dict[str, Any]) -> list[str]:
        """Generate detailed documentation for a single node type.

        Parameters
        ----------
        name : str
            Node spec name (e.g., "FunctionNodeSpec")
        spec : dict[str, Any]
            Node specification from schema

        Returns
        -------
        list[str]
            Lines of markdown documentation
        """
        kind = name.replace("NodeSpec", "").lower()
        kind = "".join(f"_{c.lower()}" if c.isupper() else c for c in kind).lstrip("_")

        lines = [f"### {kind}_node", ""]

        # Get description
        props = spec.get("properties", {})
        spec_prop = props.get("spec", {})
        desc = spec_prop.get("description", spec.get("description", ""))
        if desc:
            lines.append(desc)
            lines.append("")

        # Extract spec properties
        spec_props = spec_prop.get("properties", {})
        required = spec_prop.get("required", [])

        if spec_props:
            lines.append("**Parameters:**")
            lines.append("")
            lines.append("| Parameter | Type | Required | Description |")
            lines.append("|-----------|------|----------|-------------|")

            for param_name, param_spec in sorted(spec_props.items()):
                param_type = self._get_type_from_schema(param_spec)
                is_required = "Yes" if param_name in required else "No"
                param_desc = param_spec.get("description", "")
                if len(param_desc) > 40:
                    param_desc = param_desc[:37] + "..."
                lines.append(f"| `{param_name}` | {param_type} | {is_required} | {param_desc} |")

            lines.append("")

        # Add example
        lines.append("**Example:**")
        lines.append("")
        lines.append("```yaml")
        lines.append(f"- kind: {kind}_node")
        lines.append("  metadata:")
        lines.append(f"    name: my_{kind}")
        lines.append("  spec:")

        # Add required params as example (limit to first 3)
        lines.extend(f"    {param_name}: # required" for param_name in required[:3])

        lines.append("  dependencies: []")
        lines.append("```")
        lines.append("")

        return lines

    def _get_type_from_schema(self, spec: dict[str, Any]) -> str:
        """Extract type string from JSON schema property.

        Parameters
        ----------
        spec : dict[str, Any]
            Property specification

        Returns
        -------
        str
            Human-readable type string
        """
        if "const" in spec:
            return f'`"{spec["const"]}"`'

        if "enum" in spec:
            return " | ".join(f'`"{v}`"' for v in spec["enum"][:3])

        if "anyOf" in spec:
            types = [self._get_type_from_schema(s) for s in spec["anyOf"][:2]]
            return " | ".join(types)

        type_val = spec.get("type")
        if isinstance(type_val, list):
            return " | ".join(type_val)
        if type_val == "array":
            items = spec.get("items", {})
            item_type = items.get("type", "any")
            return f"list[{item_type}]"
        if type_val:
            return type_val

        return "any"

    def _generate_ports_section(self, schema: dict[str, Any]) -> list[str]:
        """Generate ports documentation from schema.

        Parameters
        ----------
        schema : dict[str, Any]
            Parsed JSON schema

        Returns
        -------
        list[str]
            Lines of markdown documentation
        """
        return [
            "## Ports Configuration",
            "",
            "Ports connect pipelines to external services:",
            "",
            "```yaml",
            "spec:",
            "  ports:",
            "    llm:",
            "      adapter: hexdag.builtin.adapters.openai.OpenAIAdapter",
            "      config:",
            "        api_key: ${OPENAI_API_KEY}",
            "        model: gpt-4",
            "    memory:",
            "      adapter: hexdag.builtin.adapters.memory.InMemoryMemory",
            "    database:",
            "      adapter: hexdag.builtin.adapters.database.sqlite.SQLiteAdapter",
            "      config:",
            "        db_path: ./data.db",
            "```",
            "",
            "### Available Port Types",
            "",
            "| Port | Purpose |",
            "|------|---------|",
            "| `llm` | Language model interactions |",
            "| `memory` | Persistent agent memory |",
            "| `database` | Data persistence |",
            "| `secret` | Secret/credential management |",
            "| `tool_router` | Tool invocation routing |",
            "",
        ]

    def _generate_events_section(self, schema: dict[str, Any]) -> list[str]:
        """Generate events documentation from schema.

        Parameters
        ----------
        schema : dict[str, Any]
            Parsed JSON schema

        Returns
        -------
        list[str]
            Lines of markdown documentation
        """
        return [
            "## Events Configuration",
            "",
            "Configure event handlers for observability:",
            "",
            "```yaml",
            "spec:",
            "  events:",
            "    node_failed:",
            "      - type: alert",
            "        target: pagerduty",
            "        severity: high",
            "    pipeline_completed:",
            "      - type: metrics",
            "        target: datadog",
            "```",
            "",
            "### Event Types",
            "",
            "| Event | When Triggered |",
            "|-------|----------------|",
            "| `pipeline_started` | Pipeline execution begins |",
            "| `pipeline_completed` | Pipeline execution finishes |",
            "| `node_started` | Node execution begins |",
            "| `node_completed` | Node execution finishes |",
            "| `node_failed` | Node execution fails |",
            "",
            "### Handler Types",
            "",
            "| Type | Purpose |",
            "|------|---------|",
            "| `alert` | Send alerts (PagerDuty, Slack) |",
            "| `metrics` | Emit metrics (Datadog, Prometheus) |",
            "| `log` | Write to logs |",
            "| `webhook` | Call external webhooks |",
            "| `callback` | Execute Python callbacks |",
            "",
        ]
