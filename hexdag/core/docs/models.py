"""Data models for documentation generation.

These Pydantic models represent extracted documentation from code artifacts,
providing a structured format for template rendering.
"""

from typing import Any

from pydantic import BaseModel, Field


class ParameterDoc(BaseModel):
    """Documentation for a single parameter.

    Attributes
    ----------
    name : str
        Parameter name
    type_hint : str
        Type annotation as string (e.g., "str", "int | None")
    description : str
        Parameter description from docstring
    required : bool
        Whether the parameter is required (no default value)
    default : str | None
        String representation of default value, if any
    enum_values : list[str] | None
        Allowed values for Literal types
    """

    name: str
    type_hint: str = "Any"
    description: str = ""
    required: bool = True
    default: str | None = None
    enum_values: list[str] | None = None


class ComponentDoc(BaseModel):
    """Base documentation for a component (adapter/node/tool).

    Attributes
    ----------
    name : str
        Component name (class or function name)
    module_path : str
        Full module path (e.g., "hexdag.builtin.adapters.OpenAIAdapter")
    description : str
        First line of docstring or explicit description
    full_docstring : str
        Complete docstring for detailed documentation
    parameters : list[ParameterDoc]
        Documented parameters
    examples : list[str]
        Code examples extracted from docstring
    yaml_example : str
        Generated YAML usage example
    """

    name: str
    module_path: str
    description: str = ""
    full_docstring: str = ""
    parameters: list[ParameterDoc] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)
    yaml_example: str = ""


class AdapterDoc(ComponentDoc):
    """Documentation for an adapter.

    Attributes
    ----------
    port_type : str
        Port type this adapter implements (e.g., "llm", "memory", "database")
    secrets : dict[str, str]
        Mapping of parameter names to environment variable names
    decorator_example : str
        Example @adapter decorator usage
    """

    port_type: str = "unknown"
    secrets: dict[str, str] = Field(default_factory=dict)
    decorator_example: str = ""


class NodeDoc(ComponentDoc):
    """Documentation for a node factory.

    Attributes
    ----------
    namespace : str
        Node namespace (e.g., "core", "plugin")
    yaml_schema : dict | None
        Explicit _yaml_schema if defined on the class
    kind : str
        Node kind for YAML (e.g., "llm_node", "function_node")
    """

    namespace: str = "core"
    yaml_schema: dict[str, Any] | None = None
    kind: str = ""


class ToolDoc(ComponentDoc):
    """Documentation for a tool function.

    Attributes
    ----------
    namespace : str
        Tool namespace (e.g., "core", "plugin")
    return_type : str
        Return type annotation as string
    is_async : bool
        Whether the tool is async
    """

    namespace: str = "core"
    return_type: str = "Any"
    is_async: bool = False
