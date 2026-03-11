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

import warnings
from typing import TYPE_CHECKING, Any

import yaml
from jinja2 import ChainableUndefined, Environment, TemplateSyntaxError, UndefinedError
from pydantic import BaseModel, Field, field_validator, model_validator

from hexdag.kernel.configurable import ConfigurableMacro, MacroConfig
from hexdag.kernel.domain.dag import DirectedGraph
from hexdag.kernel.exceptions import YamlPipelineBuilderError
from hexdag.kernel.logging import get_logger
from hexdag.kernel.validation.sanitized_types import _BASE_TYPE_MAP

if TYPE_CHECKING:
    from collections.abc import Callable

logger = get_logger(__name__)

# Extend kernel's _BASE_TYPE_MAP with collection types and Any for macro
# parameter validation.  int is accepted for float (widening).
_PARAM_TYPE_MAP: dict[str, tuple[type, ...]] = {
    **{k: (v,) for k, v in _BASE_TYPE_MAP.items()},
    "float": (int, float),  # widen: int → float
    "list": (list,),
    "dict": (dict,),
    "Any": (object,),
}


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

    @model_validator(mode="after")
    def _track_explicit_default(self) -> YamlMacroParameterSpec:
        """Track whether 'default' was explicitly provided (including None)."""
        object.__setattr__(self, "_has_explicit_default", "default" in self.model_fields_set)
        return self

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
                raise YamlPipelineBuilderError(
                    f"Invalid type '{t}'. Must be one of: {', '.join(valid_base_types)}"
                )
        return v


class PortRequirement(BaseModel):
    """Declares a port dependency for a YAML macro.

    When a macro uses LLM calls, database access, or other port-backed
    operations, it should declare those dependencies so the builder can
    validate at build time that the parent pipeline provides them.

    Attributes
    ----------
    name : str
        Port name as declared in the pipeline's ``spec.ports`` (e.g., ``"llm"``).
    protocol : str
        Expected protocol name for documentation (e.g., ``"LLM"``,
        ``"SupportsKeyValue"``).  Not enforced at build time — serves
        as a hint for pipeline authors.
    optional : bool
        If ``True``, the macro works without this port (graceful degradation).
        If ``False`` (default), a missing port raises a build-time error.
    """

    name: str
    protocol: str
    optional: bool = False


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
    nodes_raw : str | None
        Raw YAML string for pre-parse Jinja2 rendering.
        When present and contains ``{%`` block tags, the expansion path
        renders this string with Jinja2 first, then parses the result
        as YAML to produce the node list.  This enables dynamic node
        generation with ``{% for %}`` and ``{% if %}`` constructs.
    requires_ports : list[PortRequirement]
        Port dependencies that the parent pipeline must provide.
    outputs : dict[str, str] | None
        Output mappings (optional, like Python macros)
    """

    macro_name: str
    macro_description: str | None = None
    parameters: list[YamlMacroParameterSpec] = Field(default_factory=list)
    nodes: list[dict[str, Any]] = Field(default_factory=list)
    nodes_raw: str | None = None
    requires_ports: list[PortRequirement] = Field(default_factory=list)
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

        # Step 3: Render node templates — choose path based on nodes_raw
        if config.nodes_raw and "{%" in config.nodes_raw:
            # Pre-parse path: render raw YAML string with Jinja2, then parse
            rendered_nodes = self._render_nodes_from_raw(config.nodes_raw, context)
        else:
            # Post-parse path: render individual string fields in parsed dicts
            rendered_nodes = self._render_node_templates(config.nodes, context)

        # Step 4: Build DirectedGraph from rendered nodes
        graph = self._build_graph_from_nodes(rendered_nodes, node_builder)

        logger.info(
            "Expanded YAML macro '{}' as '{}' ({} nodes)",
            config.macro_name,
            instance_name,
            len(graph),
        )

        return graph

    def _render_nodes_from_raw(
        self,
        nodes_raw: str,
        context: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Pre-parse Jinja2 rendering for dynamic node generation.

        Renders the raw YAML nodes template string with Jinja2 first,
        then parses the result as YAML to produce the node list.  This
        enables ``{% for %}`` and ``{% if %}`` constructs at the node
        list level.

        Parameters
        ----------
        nodes_raw : str
            Raw YAML template string with Jinja2 block tags
        context : dict[str, Any]
            Template rendering context

        Returns
        -------
        list[dict[str, Any]]
            Parsed node configurations after template rendering

        Raises
        ------
        YamlPipelineBuilderError
            If template rendering or YAML parsing fails
        """
        config: YamlMacroConfig = self.config  # type: ignore[assignment]

        try:
            template = self.jinja_env.from_string(nodes_raw)
            rendered_yaml = template.render(context)
        except (TemplateSyntaxError, UndefinedError) as e:
            raise YamlPipelineBuilderError(
                f"Failed to render nodes_raw template in macro '{config.macro_name}': {e}"
            ) from e

        try:
            parsed = yaml.safe_load(rendered_yaml)
        except yaml.YAMLError as e:
            raise YamlPipelineBuilderError(
                f"Failed to parse rendered YAML in macro '{config.macro_name}': {e}"
            ) from e

        if not isinstance(parsed, list):
            raise YamlPipelineBuilderError(
                f"Rendered nodes_raw in macro '{config.macro_name}' must be "
                f"a YAML list, got {type(parsed).__name__}"
            )

        return parsed

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
        YamlPipelineBuilderError
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
                    raise YamlPipelineBuilderError(
                        f"Parameter '{param_name}' must be one of {param.enum}, got '{value}'"
                    )

                # Type validation
                if param.type != "Any":
                    expected: list[type] = []
                    for t in param.type.split("|"):
                        t = t.strip()
                        if t in _PARAM_TYPE_MAP:
                            expected.extend(_PARAM_TYPE_MAP[t])
                    if expected and not isinstance(value, tuple(expected)):
                        raise YamlPipelineBuilderError(
                            f"Parameter '{param_name}' expected type '{param.type}', "
                            f"got {type(value).__name__}: {value!r}"
                        )

                result[param_name] = value

            elif param.required:
                # Required but not provided
                raise YamlPipelineBuilderError(
                    f"Required parameter '{param_name}' not provided for macro "
                    f"'{config.macro_name}'"
                )

            elif getattr(param, "_has_explicit_default", param.default is not None):
                # Apply default (including explicit None defaults)
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
        YamlPipelineBuilderError
            If template rendering fails
        """
        rendered = []

        for i, node_template in enumerate(node_templates):
            try:
                rendered_node = self._render_dict_recursive(node_template, context)
                rendered.append(rendered_node)
            except (TemplateSyntaxError, UndefinedError) as e:
                config: YamlMacroConfig = self.config  # type: ignore[assignment]
                raise YamlPipelineBuilderError(
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
        YamlPipelineBuilderError
            If node building fails
        """
        if node_builder is not None:
            return node_builder(rendered_nodes)

        # Fallback: lazy import for standalone usage.
        # User-registered aliases and custom types are NOT available here.
        warnings.warn(
            "YamlMacro._build_graph_from_nodes called without node_builder. "
            "User-registered aliases and custom types will not be available.",
            stacklevel=2,
        )
        from hexdag.compiler.plugins.macro_entity import (  # lazy: mutual cycle
            MacroEntityPlugin,
        )
        from hexdag.compiler.yaml_builder import (  # lazy: mutual cycle with macro_definition
            NodeEntityPlugin,
            YamlPipelineBuilder,
        )

        builder = YamlPipelineBuilder()
        graph = DirectedGraph()
        node_plugin = NodeEntityPlugin(builder)
        macro_plugin = MacroEntityPlugin()

        for node_config in rendered_nodes:
            handled = False
            for plugin in (macro_plugin, node_plugin):
                if plugin.can_handle(node_config):
                    result = plugin.build(node_config, builder, graph)
                    if result is not None:
                        graph += result
                    handled = True
                    break
            if not handled:
                kind = node_config.get("kind", "unknown")
                raise YamlPipelineBuilderError(f"Invalid node kind in YAML macro: {kind}")

        return graph

    def __repr__(self) -> str:
        """String representation for debugging."""
        config: YamlMacroConfig = self.config  # type: ignore[assignment]
        return (
            f"YamlMacro(name='{config.macro_name}', "
            f"parameters={len(config.parameters)}, "
            f"nodes={len(config.nodes)})"
        )
