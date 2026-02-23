"""YAML-defined macros for declarative pipeline composition.

This module enables defining macros entirely in YAML without Python code.
YAML macros provide the same capabilities as Python macros but with a
declarative syntax suitable for version control and team collaboration.

Architecture:
    YamlMacro - ConfigurableMacro subclass that expands from YAML definition
    Parameter validation - Same as nodes (Pydantic-based)
    Template expansion - Jinja2 for node generation
    Output mapping - Same as Python macros (DirectedGraph)

Examples
--------
YAML macro definition::

    apiVersion: hexdag/v1
    kind: Macro
    metadata:
      name: retry_workflow
      description: Retry a node with exponential backoff
    parameters:
      - name: max_retries
        type: int
        default: 3
      - name: base_delay
        type: float
        default: 1.0
    nodes:
      - kind: function_node
        metadata:
          name: "{{name}}_attempt"
        spec:
          fn: "{{fn}}"
          max_retries: "{{max_retries}}"

YAML macro usage::

    nodes:
      - kind: macro_invocation
        metadata:
          name: api_call
        spec:
          macro: user:retry_workflow
          config:
            max_retries: 5
          inputs:
            fn: "myapp.api.fetch_data"
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from jinja2 import ChainableUndefined, Environment, TemplateSyntaxError, UndefinedError
from pydantic import BaseModel, Field, field_validator

from hexdag.kernel.configurable import ConfigurableMacro, MacroConfig
from hexdag.kernel.domain.dag import DirectedGraph
from hexdag.kernel.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

logger = get_logger(__name__)


class PreserveUndefined(ChainableUndefined):
    """Custom Jinja2 Undefined that preserves template syntax for undefined variables.

    This allows partial template rendering where:
    - Known variables (from macro context) are replaced
    - Unknown variables (runtime references) are preserved as {{var}}
    """

    def __str__(self) -> str:
        """Return the original template syntax for undefined variables."""
        return f"{{{{{self._undefined_name}}}}}"

    def __getattr__(self, name: str) -> Any:
        """Handle attribute access on undefined variables.

        This preserves the full path for dotted references like {{node.output}}.
        """
        if name.startswith("_"):
            # Internal Jinja2 attributes - use parent implementation
            return super().__getattr__(name)

        # Create a new PreserveUndefined with the full path
        return self.__class__(
            name=f"{self._undefined_name}.{name}",
            exc=self._undefined_exception,
        )


class YamlMacroParameterSpec(BaseModel):
    """Schema for a single YAML macro parameter.

    Attributes
    ----------
    name : str
        Parameter name
    type : str
        Python type name (e.g., "str", "int", "list", "dict")
    description : str | None
        Parameter description for documentation
    required : bool
        Whether parameter is required (default: False)
    default : Any
        Default value if not provided
    enum : list[Any] | None
        Valid values for enumeration types
    """

    name: str
    type: str = "str"
    description: str | None = None
    required: bool = False
    default: Any = None
    enum: list[Any] | None = None

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        """Validate that type is a recognized Python type."""
        # Support common types and union syntax
        valid_base_types = {"str", "int", "float", "bool", "list", "dict", "Any"}
        # Split on | for union types
        types = [t.strip() for t in v.split("|")]
        for t in types:
            if t not in valid_base_types:
                raise ValueError(
                    f"Invalid type '{t}'. Must be one of: {', '.join(valid_base_types)}"
                )
        return v


class YamlMacroConfig(MacroConfig):
    """Configuration for YAML-defined macros.

    This config is dynamically generated from the YAML macro definition.
    It stores the macro structure (parameters, nodes, outputs) that will
    be expanded when the macro is invoked.

    Attributes
    ----------
    macro_name : str
        Name of the macro
    macro_description : str | None
        Description from metadata
    parameters : list[YamlMacroParameterSpec]
        Parameter definitions
    nodes : list[dict[str, Any]]
        Node templates (will be rendered with Jinja2)
    outputs : dict[str, str] | None
        Output mappings (optional, like Python macros)
    """

    macro_name: str
    macro_description: str | None = None
    parameters: list[YamlMacroParameterSpec] = Field(default_factory=list)
    nodes: list[dict[str, Any]] = Field(default_factory=list)
    outputs: dict[str, str] | None = None


class YamlMacro(ConfigurableMacro):
    """Macro defined entirely in YAML.

    This class provides the runtime implementation for YAML macro expansion.
    It's instantiated by MacroDefinitionPlugin when processing `kind: Macro`
    declarations in YAML files.

    The expansion process:
    1. Validate provided config against parameter definitions
    2. Build Jinja2 context from parameters + inputs
    3. Render node templates
    4. Build DirectedGraph from rendered nodes
    5. Return expanded graph

    Examples
    --------
    YAML macro definition::

        apiVersion: hexdag/v1
        kind: Macro
        metadata:
          name: hitl_decision
          description: Human-in-the-loop decision point
        parameters:
          - name: mode
            type: str
            default: human
            enum: [human, auto]
          - name: timeout
            type: int
            default: 60
        nodes:
          - kind: conditional
            metadata:
              name: "{{name}}_route"
            spec:
              condition: "{{mode == 'human'}}"
              true_branch: "{{name}}_human"
              false_branch: "{{name}}_auto"

    Python invocation::

        from hexdag.kernel.resolver import resolve

        MacroClass = resolve("myapp.macros.HitlDecisionMacro")
        macro = MacroClass()
        graph = macro.expand(
            instance_name="approval",
            inputs={"mode": "auto"},
            dependencies=["validator"]
        )

    YAML invocation::

        nodes:
          - kind: macro_invocation
            metadata:
              name: approval
            spec:
              macro: user:hitl_decision
              config:
                mode: auto
              dependencies: [validator]
    """

    Config = YamlMacroConfig

    def __init__(self, **kwargs: Any) -> None:
        """Initialize YAML macro from config.

        Parameters
        ----------
        **kwargs : Any
            Configuration matching YamlMacroConfig schema
        """
        super().__init__(**kwargs)

        # Create Jinja2 environment for template rendering
        # Use PreserveUndefined to allow partial rendering:
        # - Macro variables ({{name}}, {{param}}) are replaced
        # - Runtime variables ({{node.output}}) are preserved
        # Note: autoescape=False is intentional - we're processing YAML, not HTML
        self.jinja_env = Environment(
            autoescape=False,  # nosec B701 - YAML processing, not HTML
            undefined=PreserveUndefined,
            keep_trailing_newline=True,
        )

        # Build parameter schema for validation
        self._param_schema = self._build_param_schema()

    def _build_param_schema(self) -> dict[str, YamlMacroParameterSpec]:
        """Build parameter schema from config for fast lookup."""
        config: YamlMacroConfig = self.config  # type: ignore[assignment]
        return {param.name: param for param in config.parameters}

    def expand(
        self,
        instance_name: str,
        inputs: dict[str, Any],
        dependencies: list[str],
        node_builder: Callable[[list[dict[str, Any]]], DirectedGraph] | None = None,
    ) -> DirectedGraph:
        """Expand YAML macro into a DirectedGraph.

        Parameters
        ----------
        instance_name : str
            Unique name for this macro instance (used in templates as {{name}})
        inputs : dict[str, Any]
            Input values for macro parameters (merged with defaults)
        dependencies : list[str]
            External node names this macro depends on
        node_builder : Callable | None
            Optional callback to build a DirectedGraph from rendered node configs.
            When provided, avoids importing the compiler (breaking circular dep).
            Signature: ``(rendered_nodes: list[dict]) -> DirectedGraph``

        Returns
        -------
        DirectedGraph
            Expanded graph with rendered nodes
        """
        config: YamlMacroConfig = self.config  # type: ignore[assignment]

        # Step 1: Validate and normalize inputs
        validated_inputs = self._validate_and_normalize_inputs(inputs)

        # Step 2: Build Jinja2 context
        context = self._build_template_context(instance_name, validated_inputs, dependencies)

        # Step 3: Render node templates
        rendered_nodes = self._render_node_templates(config.nodes, context)

        # Step 4: Build DirectedGraph from rendered nodes
        graph = self._build_graph_from_nodes(rendered_nodes, node_builder)

        logger.info(
            f"✅ Expanded YAML macro '{config.macro_name}' as '{instance_name}' "
            f"({len(graph)} nodes)"
        )

        return graph

    def _validate_and_normalize_inputs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Validate inputs against parameter schema and apply defaults.

        Parameters
        ----------
        inputs : dict[str, Any]
            Provided input values

        Returns
        -------
        dict[str, Any]
            Validated inputs with defaults applied

        Raises
        ------
        ValueError
            If required parameters are missing or enum validation fails
        """
        result = {}
        config: YamlMacroConfig = self.config  # type: ignore[assignment]

        for param in config.parameters:
            param_name = param.name

            # Check if value provided
            if param_name in inputs:
                value = inputs[param_name]

                # Enum validation
                if param.enum is not None and value not in param.enum:
                    raise ValueError(
                        f"Parameter '{param_name}' must be one of {param.enum}, got '{value}'"
                    )

                result[param_name] = value

            elif param.required:
                # Required but not provided
                raise ValueError(
                    f"Required parameter '{param_name}' not provided for macro "
                    f"'{config.macro_name}'"
                )

            elif param.default is not None:
                # Apply default
                result[param_name] = param.default

        return result

    def _build_template_context(
        self,
        instance_name: str,
        validated_inputs: dict[str, Any],
        dependencies: list[str],
    ) -> dict[str, Any]:
        """Build Jinja2 context for template rendering.

        Parameters
        ----------
        instance_name : str
            Macro instance name
        validated_inputs : dict[str, Any]
            Validated parameter values
        dependencies : list[str]
            External dependencies

        Returns
        -------
        dict[str, Any]
            Template context with special variables + parameters
        """
        return {
            # Special variables
            "name": instance_name,
            "dependencies": dependencies,
            # All validated parameters
            **validated_inputs,
        }

    def _render_node_templates(
        self,
        node_templates: list[dict[str, Any]],
        context: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Render node templates with Jinja2.

        Parameters
        ----------
        node_templates : list[dict[str, Any]]
            Node template definitions
        context : dict[str, Any]
            Template rendering context

        Returns
        -------
        list[dict[str, Any]]
            Rendered node configurations

        Raises
        ------
        ValueError
            If template rendering fails
        """
        rendered = []

        for i, node_template in enumerate(node_templates):
            try:
                rendered_node = self._render_dict_recursive(node_template, context)
                rendered.append(rendered_node)
            except (TemplateSyntaxError, UndefinedError) as e:
                config: YamlMacroConfig = self.config  # type: ignore[assignment]
                raise ValueError(
                    f"Failed to render node template {i} in macro '{config.macro_name}': {e}"
                ) from e

        return rendered

    def _render_dict_recursive(self, obj: Any, context: dict[str, Any]) -> Any:
        """Recursively render Jinja2 templates in nested structures.

        Uses PreserveUndefined to enable partial rendering:
        - Macro variables ({{name}}, {{param}}) are replaced with actual values
        - Runtime variables ({{node.output}}) are preserved as-is for execution time

        Example:
            Template: "Macro: {{name}}, Runtime: {{other_node.result}}"
            Context: {"name": "my_instance"}
            Result: "Macro: my_instance, Runtime: {{other_node.result}}"

        Parameters
        ----------
        obj : Any
            Object to render (dict, list, str, or primitive)
        context : dict[str, Any]
            Template context

        Returns
        -------
        Any
            Rendered object with same structure
        """
        if isinstance(obj, str):
            # Render string template (undefined vars are preserved)
            if "{{" in obj or "{%" in obj:
                template = self.jinja_env.from_string(obj)
                return template.render(context)
            return obj

        if isinstance(obj, dict):
            # Render dict values recursively
            return {k: self._render_dict_recursive(v, context) for k, v in obj.items()}

        if isinstance(obj, list):
            # Render list items recursively
            return [self._render_dict_recursive(item, context) for item in obj]

        # Primitives (int, float, bool, None)
        return obj

    def _build_graph_from_nodes(
        self,
        rendered_nodes: list[dict[str, Any]],
        node_builder: Callable[[list[dict[str, Any]]], DirectedGraph] | None = None,
    ) -> DirectedGraph:
        """Build DirectedGraph from rendered node configurations.

        This uses the same YamlPipelineBuilder logic to build nodes,
        ensuring consistency with regular YAML pipelines.

        Parameters
        ----------
        rendered_nodes : list[dict[str, Any]]
            Rendered node configurations
        node_builder : Callable | None
            Optional callback to build a DirectedGraph from node configs.
            When provided, avoids importing the compiler.

        Returns
        -------
        DirectedGraph
            Graph with all nodes added

        Raises
        ------
        ValueError
            If node building fails
        """
        # Validate no nested macro_invocations
        for node_config in rendered_nodes:
            if node_config.get("kind") == "macro_invocation":
                raise ValueError(
                    "YAML macros cannot contain nested macro_invocations. "
                    "Use Python macros for composition."
                )

        if node_builder is not None:
            return node_builder(rendered_nodes)

        # Fallback: lazy import for standalone usage
        from hexdag.compiler.yaml_builder import (  # lazy: mutual cycle with macro_definition
            NodeEntityPlugin,
            YamlPipelineBuilder,
        )

        builder = YamlPipelineBuilder()
        graph = DirectedGraph()
        node_plugin = NodeEntityPlugin(builder)

        for node_config in rendered_nodes:
            if not node_plugin.can_handle(node_config):
                kind = node_config.get("kind", "unknown")
                raise ValueError(f"Invalid node kind in YAML macro: {kind}")

            node_spec = node_plugin.build(node_config, builder, graph)
            graph += node_spec

        return graph

    def __repr__(self) -> str:
        """String representation for debugging."""
        config: YamlMacroConfig = self.config  # type: ignore[assignment]
        return (
            f"YamlMacro(name='{config.macro_name}', "
            f"parameters={len(config.parameters)}, "
            f"nodes={len(config.nodes)})"
        )
