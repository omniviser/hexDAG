"""MCP (Model Context Protocol) server for hexDAG.

Exposes hexDAG functionality as MCP tools for LLM-powered editors like Claude Code and Cursor.
This server enables LLMs to:
- Discover available components by scanning builtin modules
- Build YAML pipelines with guided, structured approaches
- Validate pipeline configurations
- Generate pipeline templates

Components are discovered by scanning module contents (no registry needed).

Installation
------------
    uv add "hexdag[mcp]"

Usage
-----
Development mode::

    uv run mcp dev hexdag/mcp_server.py

Install for Claude Desktop/Cursor::

    uv run mcp install hexdag/mcp_server.py --name hexdag

Example Claude Desktop config::

    {
      "mcpServers": {
        "hexdag": {
          "command": "uv",
          "args": ["run", "python", "-m", "hexdag.mcp_server"]
        }
      }
    }
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from mcp.server.fastmcp import FastMCP

from hexdag.core.pipeline_builder import YamlPipelineBuilder
from hexdag.core.resolver import ResolveError, resolve
from hexdag.core.schema import SchemaGenerator

# Generated documentation directory
_GENERATED_DOCS_DIR = Path(__file__).parent.parent / "docs" / "generated" / "mcp"


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


def _create_pipeline_base(
    name: str,
    description: str = "",
    ports: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create base pipeline configuration structure.

    Centralizes pipeline structure creation to avoid duplication.

    Parameters
    ----------
    name : str
        Pipeline name (metadata.name)
    description : str, optional
        Pipeline description
    ports : dict[str, Any] | None, optional
        Port configurations (llm, memory, etc.)

    Returns
    -------
    dict[str, Any]
        Pipeline configuration dict ready for adding nodes
    """
    config: dict[str, Any] = {
        "apiVersion": "hexdag/v1",
        "kind": "Pipeline",
        "metadata": {
            "name": name,
        },
        "spec": {
            "nodes": [],
        },
    }

    if description:
        config["metadata"]["description"] = description

    if ports:
        config["spec"]["ports"] = ports

    return config


# Create MCP server
mcp = FastMCP(
    "hexDAG",
    dependencies=["pydantic>=2.0", "pyyaml>=6.0", "jinja2>=3.1.0"],
)


# ============================================================================
# Component Discovery (Module-based instead of registry)
# ============================================================================


def _discover_components_in_module(
    module: Any, suffix: str | None = None, base_class: type | None = None
) -> list[dict[str, Any]]:
    """Discover components in a module by class name convention or base class.

    Parameters
    ----------
    module : Any
        Module to scan
    suffix : str | None
        Class name suffix to filter by (e.g., "Node", "Adapter")
    base_class : type | None
        Base class to filter by (alternative to suffix)

    Returns
    -------
    list[dict[str, Any]]
        List of component info dicts
    """
    result = []

    for name in dir(module):
        if name.startswith("_"):
            continue

        obj = getattr(module, name, None)
        if obj is None or not isinstance(obj, type):
            continue

        # Filter by suffix or base class
        matches = False
        if suffix and name.endswith(suffix):
            matches = True
        if base_class and issubclass(obj, base_class) and obj is not base_class:
            matches = True

        if matches:
            result.append({
                "name": name,
                "module": f"{module.__name__}.{name}",
                "description": (obj.__doc__ or "").split("\n")[0].strip(),
            })

    return result


# ============================================================================
# Component Discovery Tools (Dynamic from Registry)
# ============================================================================


@mcp.tool()  # type: ignore[misc]
def list_nodes() -> str:
    """List all available node types with auto-generated documentation.

    Returns detailed information about each node type including:
    - Node name, namespace, and module path
    - Description (from docstring)
    - Parameters summary (from _yaml_schema if available)
    - Required vs optional parameters

    Returns
    -------
        JSON string with available nodes grouped by namespace

    Examples
    --------
        >>> list_nodes()  # doctest: +SKIP
        {
          "core": [
            {
              "name": "ConditionalNode",
              "namespace": "core",
              "module_path": "hexdag.builtin.nodes.ConditionalNode",
              "description": "Multi-branch conditional router...",
              "parameters": {
                "required": ["branches"],
                "optional": ["else_action", "tie_break"]
              }
            }
          ]
        }
    """
    from hexdag.builtin import nodes as builtin_nodes

    nodes_by_namespace: dict[str, list[dict[str, Any]]] = {"core": []}

    # Scan builtin nodes module
    for name in dir(builtin_nodes):
        if name.startswith("_"):
            continue

        obj = getattr(builtin_nodes, name, None)
        if obj is None or not isinstance(obj, type):
            continue

        # Check if it's a node class (ends with Node or has BaseNodeFactory parent)
        if not name.endswith("Node"):
            continue

        # Extract description from _yaml_schema or docstring
        yaml_schema = getattr(obj, "_yaml_schema", None)
        if yaml_schema and isinstance(yaml_schema, dict):
            description = yaml_schema.get(
                "description", (obj.__doc__ or "No description").split("\n")[0].strip()
            )
            # Extract parameter info from schema
            properties = yaml_schema.get("properties", {})
            required = yaml_schema.get("required", [])
            optional = [k for k in properties if k not in required]
            params_info = {"required": required, "optional": optional}
        else:
            description = (obj.__doc__ or "No description available").split("\n")[0].strip()
            params_info = None

        node_info: dict[str, Any] = {
            "name": name,
            "namespace": "core",
            "module_path": f"hexdag.builtin.nodes.{name}",
            "description": description,
        }

        if params_info:
            node_info["parameters"] = params_info

        nodes_by_namespace["core"].append(node_info)

    return json.dumps(nodes_by_namespace, indent=2)


@mcp.tool()  # type: ignore[misc]
def list_adapters(port_type: str | None = None) -> str:
    """List all available adapters in the hexDAG registry.

    Args
    ----
        port_type: Optional filter by port type (e.g., "llm", "memory", "database", "secret")

    Returns
    -------
        JSON string with available adapters grouped by port type

    Examples
    --------
        >>> list_adapters(port_type="llm")  # doctest: +SKIP
        {
          "llm": [
            {
              "name": "openai",
              "namespace": "core",
              "port_type": "llm",
              "description": "OpenAI LLM adapter"
            }
          ]
        }
    """
    from hexdag.builtin import adapters as builtin_adapters

    adapters_by_port: dict[str, list[dict[str, Any]]] = {}

    # Scan builtin adapters module
    for name in dir(builtin_adapters):
        if name.startswith("_"):
            continue

        obj = getattr(builtin_adapters, name, None)
        if obj is None or not isinstance(obj, type):
            continue

        # Check if it's an adapter class
        if not name.endswith("Adapter"):
            continue

        # Guess port type from name
        guessed_port = _guess_port_type_from_name(name)

        # Filter by port type if specified
        if port_type and guessed_port != port_type:
            continue

        if guessed_port not in adapters_by_port:
            adapters_by_port[guessed_port] = []

        adapter_info = {
            "name": name,
            "namespace": "core",
            "module_path": f"hexdag.builtin.adapters.{name}",
            "port_type": guessed_port,
            "description": (obj.__doc__ or "No description available").split("\n")[0].strip(),
        }

        adapters_by_port[guessed_port].append(adapter_info)

    return json.dumps(adapters_by_port, indent=2)


def _guess_port_type_from_name(adapter_name: str) -> str:
    """Guess port type from adapter class name."""
    name_lower = adapter_name.lower()
    if "llm" in name_lower or "openai" in name_lower or "anthropic" in name_lower:
        return "llm"
    if "memory" in name_lower:
        return "memory"
    if "database" in name_lower or "sql" in name_lower:
        return "database"
    if "secret" in name_lower or "keyvault" in name_lower:
        return "secret"
    if "storage" in name_lower or "blob" in name_lower:
        return "storage"
    if "tool" in name_lower:
        return "tool_router"
    return "unknown"


@mcp.tool()  # type: ignore[misc]
def list_tools(namespace: str | None = None) -> str:
    """List all available tools in the hexDAG registry.

    Args
    ----
        namespace: Optional filter by namespace (e.g., "core", "user", "plugin")

    Returns
    -------
        JSON string with available tools and their schemas

    Examples
    --------
        >>> list_tools(namespace="core")  # doctest: +SKIP
        {
          "core": [
            {
              "name": "tool_end",
              "namespace": "core",
              "description": "End agent execution"
            }
          ]
        }
    """
    from hexdag.builtin.tools import builtin_tools

    tools_by_namespace: dict[str, list[dict[str, Any]]] = {"core": []}

    # Filter by namespace if specified (only core namespace for builtin)
    if namespace and namespace != "core":
        return json.dumps(tools_by_namespace, indent=2)

    # Scan builtin tools module
    for name in dir(builtin_tools):
        if name.startswith("_"):
            continue

        obj = getattr(builtin_tools, name, None)
        if obj is None:
            continue

        # Check if it's a callable (function or class)
        if not callable(obj):
            continue

        # Skip non-tool items
        if name in ("Any", "TypeVar"):
            continue

        tool_info = {
            "name": name,
            "namespace": "core",
            "module_path": f"hexdag.builtin.tools.{name}",
            "description": (obj.__doc__ or "No description available").split("\n")[0].strip(),
        }

        tools_by_namespace["core"].append(tool_info)

    return json.dumps(tools_by_namespace, indent=2)


@mcp.tool()  # type: ignore[misc]
def list_macros() -> str:
    """List all available macros in the hexDAG registry.

    Macros are reusable pipeline templates that expand into subgraphs.

    Returns
    -------
        JSON string with available macros and their descriptions

    Examples
    --------
        >>> list_macros()  # doctest: +SKIP
        [
          {
            "name": "reasoning_agent",
            "namespace": "core",
            "description": "ReAct reasoning agent pattern"
          }
        ]
    """
    from hexdag.builtin import macros as builtin_macros

    macros_list: list[dict[str, Any]] = []

    # Scan builtin macros module
    for name in dir(builtin_macros):
        if name.startswith("_"):
            continue

        obj = getattr(builtin_macros, name, None)
        if obj is None or not isinstance(obj, type):
            continue

        # Check if it's a macro class (ends with Macro or has ConfigurableMacro parent)
        if not name.endswith("Macro"):
            continue

        macro_info = {
            "name": name,
            "namespace": "core",
            "module_path": f"hexdag.builtin.macros.{name}",
            "description": (obj.__doc__ or "No description available").split("\n")[0].strip(),
        }
        macros_list.append(macro_info)

    return json.dumps(macros_list, indent=2)


@mcp.tool()  # type: ignore[misc]
def list_policies() -> str:
    """List all available policies in the hexDAG registry.

    Returns
    -------
        JSON string with available policies and their descriptions

    Examples
    --------
        >>> list_policies()  # doctest: +SKIP
        [
          {
            "name": "retry_policy",
            "namespace": "core",
            "description": "Retry failed operations"
          }
        ]
    """
    from hexdag.builtin import policies as builtin_policies

    policies_list: list[dict[str, Any]] = []

    # Scan builtin policies module
    for name in dir(builtin_policies):
        if name.startswith("_"):
            continue

        obj = getattr(builtin_policies, name, None)
        if obj is None or not isinstance(obj, type):
            continue

        # Check if it's a policy class (ends with Policy)
        if not name.endswith("Policy"):
            continue

        policy_info = {
            "name": name,
            "namespace": "core",
            "module_path": f"hexdag.builtin.policies.{name}",
            "description": (obj.__doc__ or "No description available").split("\n")[0].strip(),
        }
        policies_list.append(policy_info)

    return json.dumps(policies_list, indent=2)


@mcp.tool()  # type: ignore[misc]
def get_component_schema(
    component_type: str,
    name: str,
    namespace: str = "core",
) -> str:
    """Get detailed auto-generated schema for a specific component.

    Auto-extracts documentation from:
    - _yaml_schema class attribute (preferred, with full descriptions)
    - __call__ method signature (fallback)
    - Class/function docstrings

    Args
    ----
        component_type: Type of component (node, adapter, tool, macro, policy)
        name: Component name (class name or full module path)
        namespace: Component namespace (default: "core") - ignored if full path provided

    Returns
    -------
        JSON string with:
        - schema: Full JSON schema with property descriptions
        - parameters: Detailed parameter documentation
        - yaml_example: Ready-to-use YAML example
        - documentation: Full docstring

    Examples
    --------
        >>> get_component_schema("node", "ConditionalNode", "core")  # doctest: +SKIP
        {
          "name": "ConditionalNode",
          "type": "node",
          "schema": {...},
          "parameters": [
            {"name": "branches", "type": "array", "required": true, "description": "..."}
          ],
          "yaml_example": "..."
        }
    """
    try:
        # If name is a full module path, resolve directly
        if "." in name:
            component_obj = resolve(name)
        else:
            # Try to resolve from builtin modules based on component type
            module_map = {
                "node": "hexdag.builtin.nodes",
                "adapter": "hexdag.builtin.adapters",
                "tool": "hexdag.builtin.tools",
                "macro": "hexdag.builtin.macros",
                "policy": "hexdag.builtin.policies",
            }

            base_module = module_map.get(component_type)
            if not base_module:
                raise ResolveError(name, f"Unknown component type: {component_type}")

            # Try to resolve with full path
            full_path = f"{base_module}.{name}"
            component_obj = resolve(full_path)

        # Check for explicit _yaml_schema (preferred - has full descriptions)
        yaml_schema = getattr(component_obj, "_yaml_schema", None)

        if yaml_schema and isinstance(yaml_schema, dict):
            # Use explicit schema with full documentation
            schema = yaml_schema

            # Extract detailed parameter documentation
            parameters = _extract_parameters_from_schema(yaml_schema)

            # Generate rich YAML example from schema
            yaml_example = _generate_yaml_example_from_schema(name, yaml_schema)
        else:
            # Fall back to signature introspection
            if isinstance(component_obj, type):
                try:
                    component_instance = component_obj()
                    schema = SchemaGenerator.from_callable(component_instance)  # type: ignore[arg-type]
                except TypeError:
                    schema = SchemaGenerator.from_callable(component_obj)  # type: ignore[arg-type]
            else:
                schema = SchemaGenerator.from_callable(component_obj)  # type: ignore[arg-type]

            parameters = _extract_parameters_from_schema(schema) if isinstance(schema, dict) else []
            yaml_example = (
                SchemaGenerator.generate_example_yaml(name, schema)
                if isinstance(schema, dict) and schema
                else ""
            )

        # Extract documentation from docstring
        doc = ""
        if hasattr(component_obj, "__doc__") and component_obj.__doc__:
            doc = component_obj.__doc__.strip()

        result = {
            "name": name,
            "namespace": namespace,
            "type": component_type,
            "schema": schema,
            "parameters": parameters,
            "yaml_example": yaml_example,
            "documentation": doc,
        }

        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps(
            {
                "error": str(e),
                "component_type": component_type,
                "name": name,
                "namespace": namespace,
            },
            indent=2,
        )


def _extract_parameters_from_schema(schema: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract detailed parameter documentation from JSON schema.

    Args
    ----
        schema: JSON schema dict with properties

    Returns
    -------
        List of parameter dicts with name, type, required, default, description
    """
    parameters = []
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    for prop_name, prop_schema in properties.items():
        param: dict[str, Any] = {
            "name": prop_name,
            "type": prop_schema.get("type", "any"),
            "required": prop_name in required,
        }

        if "description" in prop_schema:
            param["description"] = prop_schema["description"]

        if "default" in prop_schema:
            param["default"] = prop_schema["default"]

        if "enum" in prop_schema:
            param["allowed_values"] = prop_schema["enum"]

        # Handle nested objects (like branch items)
        if prop_schema.get("type") == "array" and "items" in prop_schema:
            items_schema = prop_schema["items"]
            if items_schema.get("type") == "object" and "properties" in items_schema:
                param["item_properties"] = list(items_schema["properties"].keys())

        parameters.append(param)

    return parameters


def _generate_yaml_example_from_schema(node_name: str, schema: dict[str, Any]) -> str:
    """Generate a rich YAML example from schema with comments.

    Args
    ----
        node_name: Name of the node type
        schema: JSON schema dict

    Returns
    -------
        YAML string with example values and comments
    """
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    # Build example spec
    spec: dict[str, Any] = {}

    for prop_name, prop_schema in properties.items():
        prop_type = prop_schema.get("type", "string")

        # Use default if available
        if "default" in prop_schema:
            if prop_name in required:
                spec[prop_name] = prop_schema["default"]
            continue  # Skip optional with defaults

        # Generate example value based on type
        if prop_type == "string":
            if "enum" in prop_schema:
                spec[prop_name] = prop_schema["enum"][0]
            else:
                spec[prop_name] = f"<{prop_name}>"
        elif prop_type == "integer":
            spec[prop_name] = 0
        elif prop_type == "number":
            spec[prop_name] = 0.0
        elif prop_type == "boolean":
            spec[prop_name] = False
        elif prop_type == "array":
            items = prop_schema.get("items", {})
            if items.get("type") == "object":
                # Generate example array item
                item_props = items.get("properties", {})
                example_item = {}
                for item_key, item_schema in item_props.items():
                    if item_schema.get("type") == "string":
                        example_item[item_key] = f"<{item_key}>"
                    else:
                        example_item[item_key] = f"<{item_key}>"
                spec[prop_name] = [example_item]
            else:
                spec[prop_name] = []
        elif prop_type == "object":
            spec[prop_name] = {}
        else:
            spec[prop_name] = f"<{prop_name}>"

    # Build full YAML structure
    example = {
        "kind": node_name.lower().replace("node", "_node") if "Node" in node_name else node_name,
        "metadata": {"name": f"my_{node_name.lower().replace('node', '')}"},
        "spec": spec,
        "dependencies": [],
    }

    return yaml.dump(example, sort_keys=False, default_flow_style=False)


@mcp.tool()  # type: ignore[misc]
def get_syntax_reference() -> str:
    """Get reference for hexDAG YAML syntax including variable references.

    Returns comprehensive documentation on:
    - $input.field - Reference initial pipeline input
    - {{node.output}} - Jinja2 template for node outputs
    - ${ENV_VAR} - Environment variables
    - input_mapping syntax and usage

    Returns
    -------
        Detailed syntax reference documentation

    Examples
    --------
        >>> get_syntax_reference()  # doctest: +SKIP
        # hexDAG Variable Reference Syntax
        ...
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
```

**Key Points:**
- Double curly braces `{{}}` for Jinja2 templates
- `{{node_name.field}}` extracts field from named node's output
- Only available in `prompt_template` fields
- Resolved at runtime during LLM call

## 3. Environment Variables: ${VAR}

Environment variables are resolved in two phases:

### Non-Secrets (Build-time resolution)
```yaml
spec:
  ports:
    llm:
      config:
        model: ${MODEL}              # Resolved when YAML is parsed
        timeout: ${TIMEOUT:30}       # Default value if not set
```

### Secrets (Runtime resolution)
Secret-like variables are deferred to runtime for security:
```yaml
spec:
  ports:
    llm:
      config:
        api_key: ${OPENAI_API_KEY}   # Resolved when adapter is created
```

**Secret Patterns (deferred to runtime):**
- `*_API_KEY` (e.g., OPENAI_API_KEY)
- `*_SECRET` (e.g., DB_SECRET)
- `*_TOKEN` (e.g., AUTH_TOKEN)
- `*_PASSWORD` (e.g., DB_PASSWORD)
- `*_CREDENTIAL` (e.g., SERVICE_CREDENTIAL)
- `SECRET_*` (e.g., SECRET_KEY)

**Default Values:**
- `${VAR:default}` - Use "default" if VAR is not set
- `${VAR:}` - Use empty string if VAR is not set

## 4. Input Mapping

The `input_mapping` field transforms input data for a node:

```yaml
- kind: function_node
  metadata:
    name: merger
  spec:
    fn: myapp.merge_results
    input_mapping:
      # From initial pipeline input
      request_id: $input.id

      # From specific dependency outputs
      analysis: analyzer.result
      validation_status: validator.is_valid

      # Nested path extraction
      score: analyzer.metadata.confidence_score
  dependencies: [analyzer, validator]
```

**Mapping Sources:**
- `$input.path` - Extract from initial pipeline input
- `$input` - Entire initial input
- `node_name.path` - Extract from specific node's output
- `field_name` - Extract from base input (single dependency case)

## 5. Node Aliases

Define short aliases for node module paths:

```yaml
spec:
  aliases:
    fn: hexdag.builtin.nodes.FunctionNode
    my_processor: myapp.nodes.ProcessorNode
  nodes:
    - kind: fn                    # Uses alias!
      metadata:
        name: parser
      spec:
        fn: json.loads
```

## Quick Reference Table

| Syntax | Location | Purpose |
|--------|----------|---------|
| `$input.field` | input_mapping | Access initial pipeline input |
| `$input` | input_mapping | Entire initial input |
| `{{node.field}}` | prompt_template | Jinja2 template reference |
| `${VAR}` | Any string value | Environment variable |
| `${VAR:default}` | Any string value | Env var with default |
| `node.path` | input_mapping | Dependency output extraction |
"""


@mcp.tool()  # type: ignore[misc]
def validate_yaml_pipeline_lenient(yaml_content: str) -> str:
    """Validate YAML pipeline structure without requiring environment variables.

    Use this for CI/CD validation where secrets aren't available.
    This validates structure only, without instantiating adapters.

    Validates:
    - YAML syntax
    - Node structure and dependencies
    - Port configuration format
    - Manifest format (apiVersion, kind, metadata, spec)

    Does NOT validate:
    - Environment variable values
    - Adapter instantiation
    - Module path resolution

    Args
    ----
        yaml_content: YAML pipeline configuration as a string

    Returns
    -------
        JSON string with validation results (success/errors/warnings)

    Examples
    --------
        >>> validate_yaml_pipeline_lenient(pipeline_yaml)  # doctest: +SKIP
        {
          "valid": true,
          "message": "Pipeline structure is valid",
          "node_count": 3,
          "nodes": ["step1", "step2", "step3"],
          "warnings": []
        }
    """
    try:
        # Parse YAML
        parsed = yaml.safe_load(yaml_content)

        if not isinstance(parsed, dict):
            return json.dumps(
                {
                    "valid": False,
                    "error": "YAML must be a dictionary",
                    "error_type": "ParseError",
                },
                indent=2,
            )

        warnings: list[str] = []
        nodes: list[str] = []

        # Check manifest format
        if "kind" not in parsed:
            return json.dumps(
                {
                    "valid": False,
                    "error": "Missing 'kind' field. Use declarative manifest format.",
                    "error_type": "ManifestError",
                },
                indent=2,
            )

        if "metadata" not in parsed:
            warnings.append("Missing 'metadata' field")

        if "spec" not in parsed:
            return json.dumps(
                {
                    "valid": False,
                    "error": "Missing 'spec' field",
                    "error_type": "ManifestError",
                },
                indent=2,
            )

        spec = parsed.get("spec", {})

        # Check nodes
        nodes_list = spec.get("nodes", [])
        if not nodes_list:
            warnings.append("No nodes defined in pipeline")

        node_ids = set()
        for i, node in enumerate(nodes_list):
            if not isinstance(node, dict):
                return json.dumps(
                    {
                        "valid": False,
                        "error": f"Node {i} is not a dictionary",
                        "error_type": "NodeError",
                    },
                    indent=2,
                )

            metadata = node.get("metadata", {})
            node_id = metadata.get("name")
            if not node_id:
                warnings.append(f"Node {i} missing 'metadata.name'")
                node_id = f"unnamed_{i}"

            if node_id in node_ids:
                return json.dumps(
                    {
                        "valid": False,
                        "error": f"Duplicate node name: {node_id}",
                        "error_type": "NodeError",
                    },
                    indent=2,
                )

            node_ids.add(node_id)
            nodes.append(node_id)

            # Check dependencies reference valid nodes
            deps = node.get("dependencies", [])
            all_node_names = node_ids | {n.get("metadata", {}).get("name") for n in nodes_list}
            warnings.extend(
                f"Node '{node_id}' depends on '{dep}' which may not exist"
                for dep in deps
                if dep not in all_node_names
            )

        # Check ports structure
        ports = spec.get("ports", {})
        for port_name, port_config in ports.items():
            if not isinstance(port_config, dict):
                warnings.append(f"Port '{port_name}' config is not a dictionary")
            elif "adapter" not in port_config and "name" not in port_config:
                warnings.append(f"Port '{port_name}' missing 'adapter' field")

        return json.dumps(
            {
                "valid": True,
                "message": "Pipeline structure is valid",
                "node_count": len(nodes),
                "nodes": nodes,
                "ports": list(ports.keys()),
                "warnings": warnings,
            },
            indent=2,
        )

    except yaml.YAMLError as e:
        return json.dumps(
            {
                "valid": False,
                "error": f"YAML parse error: {e}",
                "error_type": "ParseError",
            },
            indent=2,
        )
    except Exception as e:
        return json.dumps(
            {
                "valid": False,
                "error": str(e),
                "error_type": type(e).__name__,
            },
            indent=2,
        )


# ============================================================================
# Helper Functions
# ============================================================================


def _normalize_for_yaml(obj: Any) -> Any:
    """Recursively convert enum values to strings for YAML serialization.

    This ensures that enums are serialized using their .value attribute
    instead of their name, preventing validation errors when the YAML
    is loaded back.

    Args
    ----
        obj: Object to normalize (can be dict, list, enum, or primitive)

    Returns
    -------
        Normalized object with enums converted to their string values

    Examples
    --------
        >>> from enum import Enum  # doctest: +SKIP
        >>> class Format(str, Enum):  # doctest: +SKIP
        ...     MIXED = "mixed"
        >>> _normalize_for_yaml({"format": Format.MIXED})  # doctest: +SKIP
        {'format': 'mixed'}
    """
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {k: _normalize_for_yaml(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_normalize_for_yaml(item) for item in obj]
    return obj


# ============================================================================
# YAML Pipeline Building Tools
# ============================================================================


@mcp.tool()  # type: ignore[misc]
def validate_yaml_pipeline(yaml_content: str) -> str:
    """Validate a YAML pipeline configuration.

    Args
    ----
        yaml_content: YAML pipeline configuration as a string

    Returns
    -------
        JSON string with validation results (success/errors)

    Examples
    --------
        >>> validate_yaml_pipeline(pipeline_yaml)  # doctest: +SKIP
        {
          "valid": true,
          "message": "Pipeline is valid",
          "node_count": 3,
          "nodes": ["step1", "step2", "step3"]
        }
    """
    try:
        # Attempt to build the pipeline
        builder = YamlPipelineBuilder()
        graph, config = builder.build_from_yaml_string(yaml_content)

        return json.dumps(
            {
                "valid": True,
                "message": "Pipeline is valid",
                "node_count": len(graph.nodes),
                "nodes": [node.name for node in graph.nodes.values()],
                "ports": list(config.ports.keys()) if config.ports else [],
            },
            indent=2,
        )
    except Exception as e:
        return json.dumps(
            {
                "valid": False,
                "error": str(e),
                "error_type": type(e).__name__,
            },
            indent=2,
        )


@mcp.tool()  # type: ignore[misc]
def generate_pipeline_template(
    pipeline_name: str,
    description: str,
    node_types: list[str],
) -> str:
    """Generate a YAML pipeline template with specified node types.

    Args
    ----
        pipeline_name: Name for the pipeline
        description: Pipeline description
        node_types: List of node types to include (e.g., ["llm_node", "agent_node"])

    Returns
    -------
        YAML pipeline template as a string

    Examples
    --------
        >>> generate_pipeline_template(  # doctest: +SKIP
        ...     "my-workflow",
        ...     "Example workflow",
        ...     ["llm_node", "function_node"]
        ... )
        apiVersion: hexdag/v1
        kind: Pipeline
        metadata:
          name: my-workflow
          description: Example workflow
        spec:
          nodes:
            - kind: llm_node
              metadata:
                name: llm_1
              spec:
                prompt_template: "Your prompt here: {{input}}"
                output_key: result
              dependencies: []
    """
    # Create basic pipeline structure using helper
    pipeline = _create_pipeline_base(pipeline_name, description)

    # Add node templates
    for i, node_type in enumerate(node_types, 1):
        # Remove '_node' suffix if present for cleaner names
        node_name_base = node_type.replace("_node", "")

        node_template = {
            "kind": node_type,
            "metadata": {
                "name": f"{node_name_base}_{i}",
            },
            "spec": _get_node_spec_template(node_type),
            "dependencies": [] if i == 1 else [f"{node_types[i - 2].replace('_node', '')}_{i - 1}"],
        }
        pipeline["spec"]["nodes"].append(node_template)  # type: ignore[index]

    # Normalize enums before serialization
    pipeline = _normalize_for_yaml(pipeline)
    return yaml.dump(pipeline, sort_keys=False, default_flow_style=False)


def _get_node_spec_template(node_type: str) -> dict[str, Any]:
    """Get a spec template for a given node type.

    Args
    ----
        node_type: Type of node (e.g., "llm_node", "agent_node")

    Returns
    -------
        Dict with common spec fields for the node type
    """
    templates = {
        "llm_node": {
            "prompt_template": "Your prompt here: {{input}}",
            "output_key": "result",
        },
        "agent_node": {
            "initial_prompt_template": "Task: {{task}}",
            "max_steps": 5,
            "output_key": "agent_result",
            "tools": [],
        },
        "function_node": {
            "fn": "your_module.your_function",
            "input_schema": {"param": "str"},
            "output_schema": {"result": "str"},
        },
        "conditional_node": {
            "condition": "{{input}} > 0",
            "true_path": [],
            "false_path": [],
        },
        "loop_node": {
            "loop_variable": "item",
            "items": "{{input_list}}",
            "body": [],
        },
    }

    return templates.get(node_type, {})  # type: ignore[return-value]


@mcp.tool()  # type: ignore[misc]
def build_yaml_pipeline_interactive(
    pipeline_name: str,
    description: str,
    nodes: list[dict[str, Any]],
    ports: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Build a complete YAML pipeline with full specifications.

    This is the recommended tool for building complete pipelines with LLM assistance.

    Args
    ----
        pipeline_name: Name for the pipeline
        description: Pipeline description
        nodes: List of node specifications with full config
        ports: Optional port configurations (llm, memory, database, etc.)

    Returns
    -------
        Complete YAML pipeline configuration

    Examples
    --------
        >>> build_yaml_pipeline_interactive(  # doctest: +SKIP
        ...     "analysis-pipeline",
        ...     "Analyze documents",
        ...     nodes=[
        ...         {
        ...             "kind": "llm_node",
        ...             "name": "analyzer",
        ...             "spec": {"prompt_template": "Analyze: {{input}}"},
        ...             "dependencies": []
        ...         }
        ...     ],
        ...     ports={
        ...         "llm": {
        ...             "adapter": "openai",
        ...             "config": {"api_key": "${OPENAI_API_KEY}", "model": "gpt-4"}
        ...         }
        ...     }
        ... )
        apiVersion: hexdag/v1
        kind: Pipeline
        metadata:
          name: analysis-pipeline
          description: Analyze documents
        spec:
          ports:
            llm:
              adapter: openai
              config:
                api_key: ${OPENAI_API_KEY}
                model: gpt-4
          nodes:
            - kind: llm_node
              metadata:
                name: analyzer
              spec:
                prompt_template: "Analyze: {{input}}"
              dependencies: []
    """
    # Create pipeline using helper
    pipeline = _create_pipeline_base(pipeline_name, description, ports)

    # Add nodes
    for node_def in nodes:
        node = {
            "kind": node_def["kind"],
            "metadata": {
                "name": node_def["name"],
            },
            "spec": node_def["spec"],
            "dependencies": node_def.get("dependencies", []),
        }
        pipeline["spec"]["nodes"].append(node)  # type: ignore[index]

    # Normalize enums before serialization
    pipeline = _normalize_for_yaml(pipeline)
    return yaml.dump(pipeline, sort_keys=False, default_flow_style=False)


@mcp.tool()  # type: ignore[misc]
def create_environment_pipelines(
    pipeline_name: str,
    description: str,
    nodes: list[dict[str, Any]],
    dev_ports: dict[str, dict[str, Any]] | None = None,
    staging_ports: dict[str, dict[str, Any]] | None = None,
    prod_ports: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Create dev, staging, and production YAML pipelines in one call.

    This generates up to 3 YAML files with environment-specific port configurations:
    - dev: Uses mock adapters (no API keys)
    - staging: Uses real APIs with staging credentials
    - prod: Uses real APIs with production credentials

    Args
    ----
        pipeline_name: Base name for pipelines (will be suffixed with -dev, -staging, -prod)
        description: Pipeline description
        nodes: Node specifications (shared across all environments)
        dev_ports: Port config for dev (defaults to mock adapters if not provided)
        staging_ports: Port config for staging (optional)
        prod_ports: Port config for production (optional)

    Returns
    -------
        JSON string with separate YAML content for each environment

    Examples
    --------
        >>> create_environment_pipelines(  # doctest: +SKIP
        ...     "research-agent",
        ...     "Research agent",
        ...     nodes=[{"kind": "macro_invocation", ...}],
        ...     prod_ports={"llm": {"adapter": "core:openai", ...}}
        ... )
        {
          "dev": "apiVersion: hexdag/v1\\nkind: Pipeline...",
          "staging": "apiVersion: hexdag/v1\\nkind: Pipeline...",
          "prod": "apiVersion: hexdag/v1\\nkind: Pipeline..."
        }
    """
    result = {}

    # Default dev ports to mock adapters
    if dev_ports is None:
        dev_ports = {
            "llm": {
                "adapter": "hexdag.builtin.adapters.mock.MockLLMAdapter",
                "config": {
                    "responses": [
                        "I'll search for information. INVOKE_TOOL: search(query='...')",
                        "Let me gather more details. INVOKE_TOOL: search(query='...')",
                        "Based on research, here are my findings: [Mock comprehensive answer]",
                    ],
                    "delay_seconds": 0.1,
                },
            },
            "tool_router": {
                "adapter": "hexdag.builtin.adapters.mock.MockToolRouterAdapter",
                "config": {"available_tools": ["search", "calculate"]},
            },
        }

    # Helper to build pipeline for an environment
    def build_env_pipeline(env_name: str, env_suffix: str, env_ports: dict[str, Any]) -> str:
        pipeline = _create_pipeline_base(
            f"{pipeline_name}-{env_name}",
            f"{description} ({env_suffix})",
            env_ports,
        )
        for node_def in nodes:
            node = {
                "kind": node_def["kind"],
                "metadata": {"name": node_def["name"]},
                "spec": node_def["spec"],
                "dependencies": node_def.get("dependencies", []),
            }
            pipeline["spec"]["nodes"].append(node)
        pipeline = _normalize_for_yaml(pipeline)
        return yaml.dump(pipeline, sort_keys=False, default_flow_style=False)

    # Build dev environment
    result["dev"] = build_env_pipeline("dev", "DEV - Mock Adapters", dev_ports)

    # Build staging environment (if provided)
    if staging_ports:
        result["staging"] = build_env_pipeline("staging", "STAGING", staging_ports)

    # Build production environment (if provided)
    if prod_ports:
        result["prod"] = build_env_pipeline("prod", "PRODUCTION", prod_ports)

    return json.dumps(result, indent=2)


@mcp.tool()  # type: ignore[misc]
def create_environment_pipelines_with_includes(
    pipeline_name: str,
    description: str,
    nodes: list[dict[str, Any]],
    dev_ports: dict[str, dict[str, Any]] | None = None,
    staging_ports: dict[str, dict[str, Any]] | None = None,
    prod_ports: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Create base + environment-specific YAML files using the include pattern.

    This generates a base YAML with shared nodes and separate environment configs
    that include the base file. This approach reduces duplication and makes it
    easier to maintain consistent logic across environments.

    Generated files:
    - base.yaml: Shared node definitions
    - dev.yaml: Includes base + dev ports (mock adapters)
    - staging.yaml: Includes base + staging ports (optional)
    - prod.yaml: Includes base + prod ports (optional)

    Args
    ----
        pipeline_name: Base name for pipelines
        description: Pipeline description
        nodes: Node specifications (shared across all environments)
        dev_ports: Port config for dev (defaults to mock adapters if not provided)
        staging_ports: Port config for staging (optional)
        prod_ports: Port config for production (optional)

    Returns
    -------
        JSON string with base YAML and environment-specific includes

    Examples
    --------
        >>> create_environment_pipelines_with_includes(  # doctest: +SKIP
        ...     "research-agent",
        ...     "Research agent",
        ...     nodes=[{"kind": "macro_invocation", ...}],
        ...     prod_ports={"llm": {"adapter": "core:openai", ...}}
        ... )
        {
          "base": "apiVersion: hexdag/v1\\n...",
          "dev": "include: ./research_agent_base.yaml\\nports:\\n  llm:\\n    adapter: "
                "hexdag.builtin.adapters.mock.MockLLMAdapter",
          "prod": "include: ./research_agent_base.yaml\\nports:\\n  llm:\\n    adapter: ..."
        }
    """
    result = {}

    # Default dev ports to mock adapters
    if dev_ports is None:
        dev_ports = {
            "llm": {
                "adapter": "hexdag.builtin.adapters.mock.MockLLMAdapter",
                "config": {
                    "responses": [
                        "I'll search for information. INVOKE_TOOL: search(query='...')",
                        "Let me gather more details. INVOKE_TOOL: search(query='...')",
                        "Based on research, here are my findings: [Mock comprehensive answer]",
                    ],
                    "delay_seconds": 0.1,
                },
            },
            "tool_router": {
                "adapter": "hexdag.builtin.adapters.mock.MockToolRouterAdapter",
                "config": {"available_tools": ["search", "calculate"]},
            },
        }

    # Build base YAML (nodes only, no ports)
    base_pipeline = _create_pipeline_base(
        f"{pipeline_name}-base",
        f"{description} (Base Configuration)",
    )
    for node_def in nodes:
        node = {
            "kind": node_def["kind"],
            "metadata": {"name": node_def["name"]},
            "spec": node_def["spec"],
            "dependencies": node_def.get("dependencies", []),
        }
        base_pipeline["spec"]["nodes"].append(node)
    base_pipeline = _normalize_for_yaml(base_pipeline)
    result["base"] = yaml.dump(base_pipeline, sort_keys=False, default_flow_style=False)

    # Helper to build environment include file
    def build_env_include(env_name: str, env_suffix: str, env_ports: dict[str, Any]) -> str:
        env_config = {
            "include": f"./{pipeline_name}_base.yaml",
            "metadata": {
                "name": f"{pipeline_name}-{env_name}",
                "description": f"{description} ({env_suffix})",
            },
            "ports": env_ports,
        }
        env_config = _normalize_for_yaml(env_config)
        return yaml.dump(env_config, sort_keys=False, default_flow_style=False)

    # Build environment include files
    result["dev"] = build_env_include("dev", "DEV - Mock Adapters", dev_ports)

    if staging_ports:
        result["staging"] = build_env_include("staging", "STAGING", staging_ports)

    if prod_ports:
        result["prod"] = build_env_include("prod", "PRODUCTION", prod_ports)

    return json.dumps(result, indent=2)


@mcp.tool()  # type: ignore[misc]
def explain_yaml_structure() -> str:
    """Explain the structure of hexDAG YAML pipelines.

    Returns
    -------
        Detailed explanation of YAML pipeline structure with examples
    """
    return """# hexDAG YAML Pipeline Structure

## Basic Structure

```yaml
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: pipeline-name
  description: What this pipeline does
spec:
  ports:       # Optional: Configure adapters
    llm:
      adapter: openai
      config:
        api_key: ${OPENAI_API_KEY}
        model: gpt-4
  nodes:       # Required: Pipeline nodes
    - kind: llm_node
      metadata:
        name: node_name
      spec:
        prompt_template: "Your prompt: {{input}}"
      dependencies: []
```

## Key Concepts

1. **apiVersion**: Always "hexdag/v1"
2. **kind**: Always "Pipeline" (or "Macro" for macro definitions)
3. **metadata**: Pipeline name and description
4. **spec**: Pipeline specification
   - **ports**: Adapter configurations (llm, memory, database, secret)
   - **nodes**: List of processing nodes

## Node Structure

Each node has:
- **kind**: Node type (llm_node, agent_node, function_node, etc.)
- **metadata.name**: Unique identifier for the node
- **spec**: Node-specific configuration
- **dependencies**: List of node names this node depends on

## Available Node Types

- **llm_node**: LLM interactions with prompt templates
- **agent_node**: ReAct agents with tool access
- **function_node**: Execute Python functions
- **conditional_node**: Conditional execution paths
- **loop_node**: Iterative processing

## Templating

Use Jinja2 syntax for dynamic values:
- `{{variable}}` - Reference node outputs or inputs
- `{{node_name.output_key}}` - Reference specific node outputs
- `${ENV_VAR}` - Environment variables (resolved at build or runtime)

## Port Configuration

```yaml
ports:
  llm:
    adapter: openai
    config:
      api_key: ${OPENAI_API_KEY}
      model: gpt-4
  memory:
    adapter: in_memory
  database:
    adapter: sqlite
    config:
      database_path: ./data.db
```

## Dependencies

Dependencies define execution order:
```yaml
nodes:
  - metadata:
      name: step1
    dependencies: []  # Runs first

  - metadata:
      name: step2
    dependencies: [step1]  # Runs after step1

  - metadata:
      name: step3
    dependencies: [step1, step2]  # Runs after both
```

## Secret Handling

Secret-like environment variables are deferred to runtime:
- `*_API_KEY`, `*_SECRET`, `*_TOKEN`, `*_PASSWORD`
- Allows building pipelines without secrets present
- Runtime injection via SecretPort → Memory

## Best Practices

1. Use descriptive node names
2. Add comprehensive descriptions
3. Leverage environment variables for secrets
4. Keep pipelines modular and reusable
5. Validate before execution using validate_yaml_pipeline()
6. Use macro_invocation for reusable patterns
"""


@mcp.tool()  # type: ignore[misc]
def get_custom_adapter_guide() -> str:
    """Get a comprehensive guide for creating custom adapters.

    Returns documentation on:
    - Creating adapters with the @adapter decorator
    - Secret handling with the secrets parameter
    - Using custom adapters in YAML pipelines
    - Testing patterns for adapters

    Returns
    -------
        Detailed guide for creating custom adapters

    Examples
    --------
        >>> get_custom_adapter_guide()  # doctest: +SKIP
        # Creating Custom Adapters in hexDAG
        ...
    """
    # Try to load auto-generated documentation first
    generated = _load_generated_doc("adapter_guide.md")
    if generated:
        return generated

    # Fallback to static documentation
    return '''# Creating Custom Adapters in hexDAG

## Overview

hexDAG uses a decorator-based pattern for creating adapters. Adapters implement
"ports" (interfaces) that connect your pipelines to external services like LLMs,
databases, and APIs.

## Quick Start

### Simple Adapter (No Secrets)

```python
from hexdag.core.registry import adapter

@adapter("cache", name="memory_cache")
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

### Adapter with Secrets

```python
from hexdag.core.registry import adapter

@adapter("llm", name="openai", secrets={"api_key": "OPENAI_API_KEY"})
class OpenAIAdapter:
    """OpenAI LLM adapter with automatic secret resolution."""

    def __init__(
        self,
        api_key: str,           # Auto-resolved from OPENAI_API_KEY env var
        model: str = "gpt-4",
        temperature: float = 0.7
    ):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature

    async def aresponse(self, messages: list) -> str:
        # Your OpenAI API implementation
        ...
```

### Adapter with Multiple Secrets

```python
@adapter(
    "database",
    name="postgres",
    secrets={
        "username": "DB_USERNAME",
        "password": "DB_PASSWORD"
    }
)
class PostgresAdapter:
    def __init__(
        self,
        username: str,          # From DB_USERNAME
        password: str,          # From DB_PASSWORD
        host: str = "localhost",
        port: int = 5432,
        database: str = "mydb"
    ):
        self.connection_string = (
            f"postgresql://{username}:{password}@{host}:{port}/{database}"
        )
```

## The @adapter Decorator

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `port_type` | str | Port this adapter implements ("llm", "memory", "database", etc.) |
| `name` | str | Unique adapter name for registration |
| `namespace` | str | Namespace (default: "plugin") |
| `secrets` | dict | Map of param names to env var names |

### Secret Resolution Order

Secrets are resolved in this order:
1. **Explicit kwargs** - Values passed directly to `__init__`
2. **Environment variables** - From the `secrets` mapping
3. **Memory port** - From orchestrator memory (with `secret:` prefix)
4. **Error** - If required and no default

## Using Custom Adapters in YAML

### Register Your Adapter

Your adapter module must be importable. Either:
- Install as a package
- Add to `PYTHONPATH`
- Place in `hexdag_plugins/` directory

### Reference in YAML Pipeline

```yaml
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: my-pipeline
spec:
  ports:
    llm:
      adapter: mycompany.adapters.CustomLLMAdapter
      config:
        api_key: ${MY_API_KEY}
        model: gpt-4-turbo
        temperature: 0.5

    database:
      adapter: mycompany.adapters.PostgresAdapter
      config:
        host: ${DB_HOST}
        port: 5432
        database: production

  nodes:
    - kind: llm_node
      metadata:
        name: analyzer
      spec:
        prompt_template: "Analyze: {{input}}"
      dependencies: []
```

### Using Aliases for Cleaner YAML

```yaml
spec:
  aliases:
    my_llm: mycompany.adapters.CustomLLMAdapter
    my_db: mycompany.adapters.PostgresAdapter

  ports:
    llm:
      adapter: my_llm  # Uses alias!
      config:
        model: gpt-4
```

## Plugin Directory Structure

For organized plugin development:

```
hexdag_plugins/
└── my_adapter/
    ├── __init__.py
    ├── my_adapter.py      # Adapter implementation
    ├── pyproject.toml     # Dependencies
    └── tests/
        └── test_my_adapter.py
```

### pyproject.toml for Plugin

```toml
[project]
name = "hexdag-my-adapter"
version = "0.1.0"
dependencies = [
    "hexdag>=0.2.0",
    "httpx>=0.25.0",  # Your adapter dependencies
]

[tool.hexdag]
plugins = ["hexdag_plugins.my_adapter"]
```

## Testing Your Adapter

### Unit Test Pattern

```python
import pytest
from mycompany.adapters import CustomLLMAdapter

@pytest.fixture
def adapter():
    return CustomLLMAdapter(
        api_key="test-key",
        model="gpt-4",
        temperature=0.5
    )

@pytest.mark.asyncio
async def test_adapter_response(adapter, mocker):
    # Mock external API call
    mock_response = mocker.patch.object(
        adapter, "_call_api",
        return_value="Test response"
    )

    result = await adapter.aresponse([{"role": "user", "content": "Hello"}])

    assert result == "Test response"
    mock_response.assert_called_once()
```

### Integration Test with Mock

```python
from hexdag.builtin.adapters.mock import MockLLM

def test_pipeline_with_mock():
    """Test pipeline logic without real API calls."""
    mock_llm = MockLLM(responses=["Analysis complete", "Summary done"])

    # Use mock_llm in your pipeline tests
```

## Common Port Types

| Port | Purpose | Key Methods |
|------|---------|-------------|
| `llm` | Language models | `aresponse(messages) -> str` |
| `memory` | Key-value storage | `aget(key)`, `aset(key, value)` |
| `database` | SQL/NoSQL databases | `aexecute_query(sql, params)` |
| `secret` | Secret management | `aget_secret(name)` |
| `tool_router` | Tool execution | `acall_tool(name, args)` |
| `observer_manager` | Event observation | `notify(event)` |
| `policy_manager` | Policy evaluation | `evaluate(context)` |

## Best Practices

1. **Async First**: Use `async def` for I/O operations
2. **Type Hints**: Add type annotations for better tooling
3. **Docstrings**: Document your adapter's purpose and config
4. **Error Handling**: Wrap external calls in try/except
5. **Logging**: Use `hexdag.core.logging.get_logger(__name__)`
6. **Secrets**: Never hardcode secrets; use the `secrets` parameter

## CLI Commands for Plugin Development

```bash
# Create a new plugin
hexdag plugin new my_adapter --port llm

# Lint and test
hexdag plugin lint my_adapter
hexdag plugin test my_adapter

# Install dependencies
hexdag plugin install my_adapter
```
'''


@mcp.tool()  # type: ignore[misc]
def get_custom_node_guide() -> str:
    """Get a comprehensive guide for creating custom nodes.

    Returns documentation on:
    - Creating nodes with the @node decorator
    - Node factory pattern
    - Input/output schemas
    - Using custom nodes in YAML pipelines

    Returns
    -------
        Detailed guide for creating custom nodes

    Examples
    --------
        >>> get_custom_node_guide()  # doctest: +SKIP
        # Creating Custom Nodes in hexDAG
        ...
    """
    # Try to load auto-generated documentation first
    generated = _load_generated_doc("node_guide.md")
    if generated:
        return generated

    # Fallback to static documentation
    return '''# Creating Custom Nodes in hexDAG

## Overview

Nodes are the building blocks of hexDAG pipelines. Each node performs a specific
task and can be connected to other nodes via dependencies.

## Quick Start

### Simple Function Node

The easiest way to create a custom node is using FunctionNode with your own function:

```yaml
# In YAML - reference any Python function by module path
- kind: function_node
  metadata:
    name: my_processor
  spec:
    fn: mycompany.processors.process_data
  dependencies: []
```

```python
# mycompany/processors.py
def process_data(input_data: dict) -> dict:
    """Your processing logic."""
    return {"result": input_data["value"] * 2}
```

### Custom Node Class

For more complex logic, create a node class:

```python
from hexdag.core.registry import node
from hexdag.builtin.nodes import BaseNodeFactory
from hexdag.core.domain.dag import NodeSpec

@node(name="custom_processor", namespace="plugin")
class CustomProcessorNode(BaseNodeFactory):
    """Custom node for specialized processing."""

    def __init__(self):
        super().__init__()

    def __call__(
        self,
        name: str,
        threshold: float = 0.5,
        mode: str = "standard",
        **kwargs
    ) -> NodeSpec:
        async def process_fn(input_data: dict) -> dict:
            # Your async processing logic
            if input_data.get("score", 0) > threshold:
                return {"status": "pass", "mode": mode}
            return {"status": "fail", "mode": mode}

        return NodeSpec(
            name=name,
            fn=process_fn,
            deps=frozenset(kwargs.get("deps", [])),
            params={"threshold": threshold, "mode": mode},
        )
```

## Using Custom Nodes in YAML

### With Full Module Path

```yaml
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: custom-pipeline
spec:
  nodes:
    - kind: mycompany.nodes.CustomProcessorNode
      metadata:
        name: processor
      spec:
        threshold: 0.7
        mode: strict
      dependencies: []
```

### With Aliases

```yaml
spec:
  aliases:
    processor: mycompany.nodes.CustomProcessorNode

  nodes:
    - kind: processor  # Uses alias!
      metadata:
        name: my_processor
      spec:
        threshold: 0.7
      dependencies: []
```

## Node Factory Pattern

hexDAG nodes use the factory pattern:

```python
class MyNode(BaseNodeFactory):
    def __call__(self, name: str, **params) -> NodeSpec:
        # Factory method creates NodeSpec when called
        return NodeSpec(
            name=name,
            fn=self._create_function(**params),
            deps=frozenset(params.get("deps", [])),
            params=params,
        )

    def _create_function(self, **params):
        async def node_function(input_data):
            # Actual processing logic
            return {"result": "processed"}
        return node_function
```

## Input/Output Schemas

Define schemas for type validation:

```python
from pydantic import BaseModel

class ProcessorInput(BaseModel):
    text: str
    options: dict = {}

class ProcessorOutput(BaseModel):
    result: str
    confidence: float

@node(name="typed_processor", namespace="plugin")
class TypedProcessorNode(BaseNodeFactory):
    def __call__(
        self,
        name: str,
        **kwargs
    ) -> NodeSpec:
        async def process_fn(input_data: ProcessorInput) -> ProcessorOutput:
            return ProcessorOutput(
                result=input_data.text.upper(),
                confidence=0.95
            )

        return NodeSpec(
            name=name,
            fn=process_fn,
            in_model=ProcessorInput,
            out_model=ProcessorOutput,
            deps=frozenset(kwargs.get("deps", [])),
        )
```

### In YAML

```yaml
- kind: mycompany.nodes.TypedProcessorNode
  metadata:
    name: processor
  spec:
    input_schema:
      text: str
      options: dict
    output_schema:
      result: str
      confidence: float
  dependencies: []
```

## Builder Pattern Nodes

For complex configuration, use the builder pattern:

```python
@node(name="configurable_node", namespace="plugin")
class ConfigurableNode(BaseNodeFactory):
    def __init__(self):
        super().__init__()
        self._name = None
        self._config = {}
        self._validators = []

    def name(self, n: str) -> "ConfigurableNode":
        self._name = n
        return self

    def config(self, **kwargs) -> "ConfigurableNode":
        self._config.update(kwargs)
        return self

    def validate_with(self, validator) -> "ConfigurableNode":
        self._validators.append(validator)
        return self

    def build(self) -> NodeSpec:
        async def process_fn(input_data):
            for validator in self._validators:
                input_data = validator(input_data)
            return {"processed": True, **self._config}

        return NodeSpec(
            name=self._name,
            fn=process_fn,
            params=self._config,
        )
```

Usage:
```python
node = (ConfigurableNode()
    .name("my_node")
    .config(threshold=0.5, mode="strict")
    .validate_with(my_validator)
    .build())
```

## Providing YAML Schema

For MCP tools to show proper schemas, add `_yaml_schema`:

```python
@node(name="documented_node", namespace="plugin")
class DocumentedNode(BaseNodeFactory):
    # Schema for MCP tools and documentation
    _yaml_schema = {
        "type": "object",
        "properties": {
            "threshold": {
                "type": "number",
                "description": "Processing threshold (0-1)",
                "default": 0.5
            },
            "mode": {
                "type": "string",
                "enum": ["standard", "strict", "lenient"],
                "description": "Processing mode"
            }
        },
        "required": ["mode"]
    }

    def __call__(self, name: str, threshold: float = 0.5, mode: str = "standard"):
        ...
```

## Best Practices

1. **Async Functions**: Use `async def` for the node function
2. **Immutable**: Don't modify input_data; return new dict
3. **Type Hints**: Add types for better IDE support
4. **Docstrings**: Document purpose and parameters
5. **Small & Focused**: Each node should do one thing well
6. **Testable**: Design for easy unit testing

## Testing Custom Nodes

```python
import pytest
from mycompany.nodes import CustomProcessorNode

@pytest.mark.asyncio
async def test_custom_processor():
    # Create node spec
    node_factory = CustomProcessorNode()
    node_spec = node_factory(name="test", threshold=0.5)

    # Test the function
    result = await node_spec.fn({"score": 0.8})

    assert result["status"] == "pass"
```
'''


@mcp.tool()  # type: ignore[misc]
def get_custom_tool_guide() -> str:
    """Get a guide for creating custom tools for agents.

    Returns documentation on creating tools that agents can use
    during execution.

    Returns
    -------
        Guide for creating custom tools
    """
    # Try to load auto-generated documentation first
    generated = _load_generated_doc("tool_guide.md")
    if generated:
        return generated

    # Fallback to static documentation
    return '''# Creating Custom Tools for hexDAG Agents

## Overview

Tools are functions that agents can invoke during execution. They enable
agents to interact with external systems, perform calculations, or access data.

## Quick Start

### Simple Tool Function

```python
from hexdag.core.registry import tool

@tool(name="calculate", namespace="custom", description="Perform calculations")
def calculate(expression: str) -> str:
    """Safely evaluate a mathematical expression.

    Args:
        expression: Math expression like "2 + 2" or "sqrt(16)"

    Returns:
        Result as a string
    """
    import ast
    import operator

    # Safe evaluation (simplified example)
    result = eval(expression, {"__builtins__": {}}, {"sqrt": math.sqrt})
    return str(result)
```

### Async Tool

```python
@tool(name="fetch_data", namespace="custom", description="Fetch data from API")
async def fetch_data(url: str, timeout: int = 30) -> dict:
    """Fetch JSON data from a URL.

    Args:
        url: API endpoint URL
        timeout: Request timeout in seconds

    Returns:
        JSON response as dict
    """
    import httpx

    async with httpx.AsyncClient() as client:
        response = await client.get(url, timeout=timeout)
        return response.json()
```

## Tool Schema Generation

Tool schemas are auto-generated from:
- Function signature (parameter types)
- Docstring (descriptions)
- Type hints (for validation)

```python
@tool(name="search", namespace="custom", description="Search documents")
def search(
    query: str,
    limit: int = 10,
    filters: dict | None = None
) -> list[dict]:
    """Search for documents matching query.

    Args:
        query: Search query string
        limit: Maximum results to return (default: 10)
        filters: Optional filters like {"category": "tech"}

    Returns:
        List of matching documents
    """
    # Implementation
    ...
```

This generates schema:
```json
{
  "name": "search",
  "description": "Search for documents matching query.",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {"type": "string", "description": "Search query string"},
      "limit": {"type": "integer", "default": 10},
      "filters": {"type": "object", "nullable": true}
    },
    "required": ["query"]
  }
}
```

## Using Tools with Agents

### In YAML Pipeline

```yaml
- kind: agent_node
  metadata:
    name: research_agent
  spec:
    initial_prompt_template: "Research: {{topic}}"
    max_steps: 5
    tools:
      - mycompany.tools.search
      - mycompany.tools.fetch_data
      - mycompany.tools.calculate
  dependencies: []
```

### Tool Invocation Format

Agents invoke tools using this format in their output:
```
INVOKE_TOOL: tool_name(param1="value", param2=123)
```

## Built-in Tools

hexDAG provides these built-in tools:

| Tool | Description |
|------|-------------|
| `tool_end` | Signal agent completion |
| `tool_noop` | No operation (thinking step) |

## Best Practices

1. **Type Hints**: Always add parameter and return types
2. **Docstrings**: Write clear descriptions for LLM understanding
3. **Error Handling**: Return error messages, don't raise exceptions
4. **Idempotent**: Tools should be safe to retry
5. **Async**: Use async for I/O operations
6. **Validation**: Validate inputs before processing
'''


@mcp.tool()  # type: ignore[misc]
def get_extension_guide(component_type: str | None = None) -> str:
    """Get a guide for extending hexDAG with custom components.

    Args
    ----
        component_type: Optional specific type (adapter, node, tool, macro, policy)
                       If not specified, returns overview of all extension points.

    Returns
    -------
        Guide for the requested extension type or overview

    Examples
    --------
        >>> get_extension_guide()  # doctest: +SKIP
        # Extending hexDAG - Overview
        ...
        >>> get_extension_guide("adapter")  # doctest: +SKIP
        # See get_custom_adapter_guide() for details
    """
    if component_type == "adapter":
        return "Use get_custom_adapter_guide() for detailed adapter documentation."
    if component_type == "node":
        return "Use get_custom_node_guide() for detailed node documentation."
    if component_type == "tool":
        return "Use get_custom_tool_guide() for detailed tool documentation."

    # Try to load auto-generated documentation first
    generated = _load_generated_doc("extension_guide.md")
    if generated:
        return generated

    # Fallback to static documentation
    return """# Extending hexDAG - Overview

## Extension Points

hexDAG can be extended at multiple levels:

| Component | Purpose | Decorator |
|-----------|---------|-----------|
| **Adapter** | Connect to external services | `@adapter()` |
| **Node** | Custom processing logic | `@node()` |
| **Tool** | Agent-callable functions | `@tool()` |
| **Macro** | Reusable pipeline patterns | `@macro()` |
| **Policy** | Execution control rules | `@policy()` |

## Quick Reference

### Adapters
```python
@adapter("llm", name="my_llm", secrets={"api_key": "MY_API_KEY"})
class MyLLMAdapter:
    def __init__(self, api_key: str, model: str = "default"):
        ...
```
→ Use `get_custom_adapter_guide()` for full documentation

### Nodes
```python
@node(name="my_node", namespace="plugin")
class MyNode(BaseNodeFactory):
    def __call__(self, name: str, **params) -> NodeSpec:
        ...
```
→ Use `get_custom_node_guide()` for full documentation

### Tools
```python
@tool(name="my_tool", namespace="plugin", description="Does something")
def my_tool(param: str) -> str:
    ...
```
→ Use `get_custom_tool_guide()` for full documentation

### Macros
```python
@macro(name="my_pattern", namespace="plugin")
class MyMacro(ConfigurableMacro):
    def expand(self, **params) -> list[NodeSpec]:
        # Return list of nodes that implement the pattern
        ...
```

### Policies
```python
@policy(name="my_policy", description="Custom retry logic")
class MyPolicy:
    def __init__(self, max_retries: int = 3):
        ...

    async def evaluate(self, context: PolicyContext) -> PolicyResponse:
        ...
```

## Plugin Structure

Organize extensions in `hexdag_plugins/`:

```
hexdag_plugins/
├── my_adapter/
│   ├── __init__.py
│   ├── adapter.py
│   ├── pyproject.toml
│   └── tests/
├── my_nodes/
│   ├── __init__.py
│   ├── processor.py
│   └── analyzer.py
└── my_tools/
    ├── __init__.py
    └── search.py
```

## Using Extensions in YAML

```yaml
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: extended-pipeline
spec:
  # Aliases for cleaner references
  aliases:
    my_processor: mycompany.nodes.ProcessorNode
    my_analyzer: mycompany.nodes.AnalyzerNode

  # Custom adapters
  ports:
    llm:
      adapter: mycompany.adapters.CustomLLMAdapter
      config:
        api_key: ${MY_API_KEY}

  # Custom nodes
  nodes:
    - kind: my_processor
      metadata:
        name: step1
      spec:
        mode: fast
      dependencies: []

    - kind: agent_node
      metadata:
        name: agent
      spec:
        tools:
          - mycompany.tools.search  # Custom tool
          - mycompany.tools.analyze
      dependencies: [step1]
```

## MCP Tools for Development

Use these MCP tools when building pipelines:

| Tool | Purpose |
|------|---------|
| `list_nodes()` | See available nodes |
| `list_adapters()` | See available adapters |
| `get_component_schema()` | Get config schema |
| `get_syntax_reference()` | Variable syntax help |
| `validate_yaml_pipeline()` | Validate your YAML |
| `get_custom_adapter_guide()` | Adapter creation guide |
| `get_custom_node_guide()` | Node creation guide |
| `get_custom_tool_guide()` | Tool creation guide |
| `init_pipeline()` | Create new empty pipeline |
| `add_node_to_pipeline()` | Add node to pipeline |
| `remove_node_from_pipeline()` | Remove node from pipeline |
| `update_node_config()` | Update node configuration |
| `list_pipeline_nodes()` | List nodes with dependencies |
"""


# ============================================================================
# Pipeline Manipulation Tools
# ============================================================================


def _parse_pipeline_yaml(yaml_content: str) -> tuple[dict[str, Any], str | None]:
    """Parse pipeline YAML with error handling.

    Parameters
    ----------
    yaml_content : str
        YAML content to parse

    Returns
    -------
    tuple[dict[str, Any], str | None]
        Tuple of (parsed_config, error_message)
        If error_message is not None, parsed_config will be empty dict
    """
    try:
        config = yaml.safe_load(yaml_content)
        if not isinstance(config, dict):
            return {}, "YAML must be a dictionary/object"
        return config, None
    except yaml.YAMLError as e:
        return {}, f"YAML parse error: {e}"


def _find_node_by_name(nodes: list[dict[str, Any]], name: str) -> int | None:
    """Find node index by metadata.name.

    Parameters
    ----------
    nodes : list[dict[str, Any]]
        List of node configurations
    name : str
        Node name to find

    Returns
    -------
    int | None
        Index of node if found, None otherwise
    """
    for i, node in enumerate(nodes):
        node_name = node.get("metadata", {}).get("name")
        if node_name == name:
            return i
    return None


def _get_node_names(nodes: list[dict[str, Any]]) -> set[str]:
    """Get set of all node names.

    Parameters
    ----------
    nodes : list[dict[str, Any]]
        List of node configurations

    Returns
    -------
    set[str]
        Set of node names
    """
    names = set()
    for node in nodes:
        name = node.get("metadata", {}).get("name")
        if name:
            names.add(name)
    return names


def _compute_execution_order(nodes: list[dict[str, Any]]) -> list[str]:
    """Compute topological execution order using Kahn's algorithm.

    Parameters
    ----------
    nodes : list[dict[str, Any]]
        List of node configurations

    Returns
    -------
    list[str]
        Node names in topological order
    """
    # Build adjacency list and in-degree count
    in_degree: dict[str, int] = {}
    graph: dict[str, list[str]] = {}
    all_nodes: set[str] = set()

    for node in nodes:
        name = node.get("metadata", {}).get("name")
        if not name:
            continue
        all_nodes.add(name)
        in_degree.setdefault(name, 0)
        graph.setdefault(name, [])

        deps = node.get("dependencies", [])
        for dep in deps:
            if dep in all_nodes or dep in in_degree:
                graph.setdefault(dep, []).append(name)
                in_degree[name] = in_degree.get(name, 0) + 1

    # Kahn's algorithm
    queue = [n for n in all_nodes if in_degree.get(n, 0) == 0]
    result: list[str] = []

    while queue:
        node = queue.pop(0)
        result.append(node)
        for neighbor in graph.get(node, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return result


@mcp.tool()  # type: ignore[misc]
def init_pipeline(name: str, description: str = "") -> str:
    """Create a new minimal pipeline YAML configuration.

    Creates an empty pipeline structure ready for adding nodes.

    Parameters
    ----------
    name : str
        Pipeline name (used in metadata.name)
    description : str, optional
        Pipeline description

    Returns
    -------
    str
        JSON with {success: bool, yaml_content: str}
    """
    # Use centralized helper for pipeline creation
    config = _create_pipeline_base(name, description)
    yaml_content = yaml.dump(config, sort_keys=False, default_flow_style=False)

    return json.dumps(
        {
            "success": True,
            "yaml_content": yaml_content,
            "message": f"Created empty pipeline '{name}'",
        },
        indent=2,
    )


@mcp.tool()  # type: ignore[misc]
def add_node_to_pipeline(yaml_content: str, node_config: dict[str, Any]) -> str:
    """Add a node to an existing pipeline YAML.

    Parameters
    ----------
    yaml_content : str
        Existing pipeline YAML as string
    node_config : dict[str, Any]
        Node configuration with keys:
        - kind: Node type (e.g., "llm_node", "function_node")
        - name: Unique node identifier
        - spec: Node-specific configuration dict
        - dependencies: Optional list of dependency node names

    Returns
    -------
    str
        JSON with {success: bool, yaml_content: str, warnings: list, node_count: int}
    """
    config, error = _parse_pipeline_yaml(yaml_content)
    if error:
        return json.dumps({"success": False, "error": error}, indent=2)

    # Validate required fields
    if "kind" not in node_config:
        return json.dumps(
            {"success": False, "error": "node_config must have 'kind' field"}, indent=2
        )
    if "name" not in node_config:
        return json.dumps(
            {"success": False, "error": "node_config must have 'name' field"}, indent=2
        )

    # Get or create nodes list
    spec = config.setdefault("spec", {})
    nodes = spec.setdefault("nodes", [])

    # Check for duplicate name
    existing_names = _get_node_names(nodes)
    node_name = node_config["name"]
    if node_name in existing_names:
        return json.dumps(
            {"success": False, "error": f"Node '{node_name}' already exists"}, indent=2
        )

    # Build node structure
    new_node: dict[str, Any] = {
        "kind": node_config["kind"],
        "metadata": {"name": node_name},
        "spec": node_config.get("spec", {}),
        "dependencies": node_config.get("dependencies", []),
    }

    # Check for missing dependencies (warn, don't fail)
    deps = node_config.get("dependencies", [])
    warnings: list[str] = [
        f"Dependency '{dep}' not found in pipeline" for dep in deps if dep not in existing_names
    ]

    nodes.append(new_node)

    yaml_output = yaml.dump(config, sort_keys=False, default_flow_style=False)

    return json.dumps(
        {
            "success": True,
            "yaml_content": yaml_output,
            "node_count": len(nodes),
            "warnings": warnings if warnings else None,
        },
        indent=2,
    )


@mcp.tool()  # type: ignore[misc]
def remove_node_from_pipeline(yaml_content: str, node_name: str) -> str:
    """Remove a node from a pipeline YAML.

    Parameters
    ----------
    yaml_content : str
        Existing pipeline YAML as string
    node_name : str
        Name of the node to remove

    Returns
    -------
    str
        JSON with {success: bool, yaml_content: str, warnings: list}
        Warns if other nodes depend on the removed node.
    """
    config, error = _parse_pipeline_yaml(yaml_content)
    if error:
        return json.dumps({"success": False, "error": error}, indent=2)

    nodes = config.get("spec", {}).get("nodes", [])

    # Find node index
    node_idx = _find_node_by_name(nodes, node_name)
    if node_idx is None:
        return json.dumps({"success": False, "error": f"Node '{node_name}' not found"}, indent=2)

    # Check for dependents
    warnings: list[str] = []
    for node in nodes:
        deps = node.get("dependencies", [])
        if node_name in deps:
            dependent_name = node.get("metadata", {}).get("name", "unknown")
            warnings.append(f"Node '{dependent_name}' depends on '{node_name}'")

    # Remove the node
    nodes.pop(node_idx)

    yaml_output = yaml.dump(config, sort_keys=False, default_flow_style=False)

    return json.dumps(
        {
            "success": True,
            "yaml_content": yaml_output,
            "node_count": len(nodes),
            "removed": True,
            "warnings": warnings if warnings else None,
        },
        indent=2,
    )


@mcp.tool()  # type: ignore[misc]
def update_node_config(yaml_content: str, node_name: str, config_updates: dict[str, Any]) -> str:
    """Update a node's configuration in pipeline YAML.

    Parameters
    ----------
    yaml_content : str
        Existing pipeline YAML as string
    node_name : str
        Name of the node to update
    config_updates : dict[str, Any]
        Updates to apply:
        - spec: Dict of spec fields to merge/update
        - dependencies: New dependencies list (replaces existing)
        - kind: New node type (use with caution)

    Returns
    -------
    str
        JSON with {success: bool, yaml_content: str}
    """
    config, error = _parse_pipeline_yaml(yaml_content)
    if error:
        return json.dumps({"success": False, "error": error}, indent=2)

    nodes = config.get("spec", {}).get("nodes", [])

    # Find node index
    node_idx = _find_node_by_name(nodes, node_name)
    if node_idx is None:
        return json.dumps({"success": False, "error": f"Node '{node_name}' not found"}, indent=2)

    node = nodes[node_idx]
    warnings: list[str] = []

    # Apply updates
    if "spec" in config_updates:
        # Deep merge spec updates
        node_spec = node.setdefault("spec", {})
        for key, value in config_updates["spec"].items():
            node_spec[key] = value

    if "dependencies" in config_updates:
        # Replace dependencies
        node["dependencies"] = config_updates["dependencies"]

    if "kind" in config_updates:
        # Change node type (warn user)
        old_kind = node.get("kind")
        new_kind = config_updates["kind"]
        if old_kind != new_kind:
            warnings.append(f"Changed node type from '{old_kind}' to '{new_kind}'")
        node["kind"] = new_kind

    yaml_output = yaml.dump(config, sort_keys=False, default_flow_style=False)

    return json.dumps(
        {
            "success": True,
            "yaml_content": yaml_output,
            "warnings": warnings if warnings else None,
        },
        indent=2,
    )


@mcp.tool()  # type: ignore[misc]
def list_pipeline_nodes(yaml_content: str) -> str:
    """List all nodes in a pipeline with their dependencies.

    Parameters
    ----------
    yaml_content : str
        Pipeline YAML as string

    Returns
    -------
    str
        JSON with:
        {
            success: bool,
            pipeline_name: str,
            node_count: int,
            nodes: [{name, kind, dependencies, dependents}],
            execution_order: [str]
        }
    """
    config, error = _parse_pipeline_yaml(yaml_content)
    if error:
        return json.dumps({"success": False, "error": error}, indent=2)

    pipeline_name = config.get("metadata", {}).get("name", "unknown")
    nodes = config.get("spec", {}).get("nodes", [])

    # Build node info with reverse dependencies
    node_infos: list[dict[str, Any]] = []
    all_names = _get_node_names(nodes)

    # Build reverse dependency map
    dependents_map: dict[str, list[str]] = {name: [] for name in all_names}
    for node in nodes:
        node_name = node.get("metadata", {}).get("name")
        deps = node.get("dependencies", [])
        for dep in deps:
            if dep in dependents_map:
                dependents_map[dep].append(node_name)

    for node in nodes:
        node_name = node.get("metadata", {}).get("name", "unknown")
        node_infos.append({
            "name": node_name,
            "kind": node.get("kind", "unknown"),
            "dependencies": node.get("dependencies", []),
            "dependents": dependents_map.get(node_name, []),
        })

    # Compute execution order
    execution_order = _compute_execution_order(nodes)

    return json.dumps(
        {
            "success": True,
            "pipeline_name": pipeline_name,
            "node_count": len(nodes),
            "nodes": node_infos,
            "execution_order": execution_order,
        },
        indent=2,
    )


# Run the server
if __name__ == "__main__":
    mcp.run()
