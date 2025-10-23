"""Simplified YAML Pipeline Builder with Plugin Architecture.

Philosophy:
- Core builder: Minimal YAML → DirectedGraph parser
- Preprocessing plugins: Environment variables, templating, validation
- Entity plugins: Macros, nodes (each entity type handles its own YAML)

Plugins provide clear value:
- EnvironmentVariablePlugin: Resolve ${VAR} and ${VAR:default}
- TemplatePlugin: Jinja2 templating in YAML values
- SchemaValidationPlugin: Validate schemas before execution
"""

import os
import re
from contextlib import suppress
from functools import lru_cache, singledispatch
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

import yaml
from jinja2 import Environment, TemplateSyntaxError, UndefinedError

from hexdag.core.bootstrap import ensure_bootstrapped
from hexdag.core.configurable import ConfigurableMacro
from hexdag.core.domain.dag import DirectedGraph, NodeSpec
from hexdag.core.logging import get_logger
from hexdag.core.pipeline_builder.pipeline_config import PipelineConfig
from hexdag.core.pipeline_builder.yaml_validator import YamlValidator
from hexdag.core.registry import registry
from hexdag.core.registry.exceptions import ComponentNotFoundError
from hexdag.core.registry.models import NAMESPACE_SEPARATOR

if TYPE_CHECKING:
    from collections.abc import Callable

logger = get_logger(__name__)


class YamlPipelineBuilderError(Exception):
    """YAML pipeline building errors."""

    pass


# ============================================================================
# Plugin Protocol
# ============================================================================


class PreprocessPlugin(Protocol):
    """Plugin for preprocessing YAML before building (env vars, templating, etc.)."""

    def process(self, config: dict[str, Any]) -> dict[str, Any]:
        """Process entire config, returning modified version."""
        ...


class EntityPlugin(Protocol):
    """Plugin for building specific entity types (macros, nodes, etc.)."""

    def can_handle(self, node_config: dict[str, Any]) -> bool:
        """Return True if this plugin can handle the node config."""
        ...

    def build(
        self, node_config: dict[str, Any], builder: "YamlPipelineBuilder", graph: DirectedGraph
    ) -> NodeSpec | None:
        """Build entity from config.

        Args
        -----
            node_config: The node configuration dictionary.
            builder: The YamlPipelineBuilder instance.
            graph: The DirectedGraph instance.

        Return NodeSpec or None if handled (e.g., macro merged into graph)."""
        ...


# ============================================================================
# Core Builder
# ============================================================================


class YamlPipelineBuilder:
    """YAML → DirectedGraph builder with plugin support.

    Workflow:
    1. Parse YAML
    2. Select environment
    3. Validate structure
    4. Run preprocessing plugins (env vars, templates)
    5. Build graph using entity plugins
    6. Extract pipeline config
    """

    def __init__(self) -> None:
        """Initialize builder."""
        ensure_bootstrapped()

        self.validator = YamlValidator()

        # Plugins
        self.preprocess_plugins: list[PreprocessPlugin] = []
        self.entity_plugins: list[EntityPlugin] = []

        self._register_default_plugins()

    def _register_default_plugins(self) -> None:
        """Register default plugins for common use cases."""
        # Preprocessing plugins (run before building)
        self.preprocess_plugins.append(EnvironmentVariablePlugin())
        self.preprocess_plugins.append(TemplatePlugin())

        # Entity plugins (build specific entity types)
        self.entity_plugins.append(MacroEntityPlugin())
        self.entity_plugins.append(NodeEntityPlugin(self))

    # --- Public API ---

    def build_from_yaml_file(
        self, yaml_path: str, use_cache: bool = True
    ) -> tuple[DirectedGraph, PipelineConfig]:
        """Build from YAML file."""
        yaml_content = Path(yaml_path).read_text(encoding="utf-8")
        return self.build_from_yaml_string(yaml_content, use_cache=use_cache)

    def build_from_yaml_string(
        self, yaml_content: str, use_cache: bool = True, environment: str | None = None
    ) -> tuple[DirectedGraph, PipelineConfig]:
        """Build DirectedGraph + PipelineConfig from YAML string."""
        # Step 1: Parse YAML
        documents = self._parse_yaml(yaml_content, use_cache=use_cache)

        # Step 2: Select environment
        config = self._select_environment(documents, environment)

        # Step 3: Validate structure
        config = self._validate_config(config)

        # Step 4: Preprocess (env vars, templating, etc.)
        for plugin in self.preprocess_plugins:
            config = plugin.process(config)

        # Step 5: Build graph
        graph = self._build_graph(config)

        # Step 6: Extract pipeline config
        pipeline_config = self._extract_pipeline_config(config)

        logger.info(
            "✅ Built pipeline '{name}' with {nodes} nodes, {ports} ports, {policies} policies",
            name=pipeline_config.metadata.get("name", "unknown"),
            nodes=len(graph.nodes),
            ports=len(pipeline_config.ports),
            policies=len(pipeline_config.policies),
        )

        return graph, pipeline_config

    # --- Core Logic ---

    def _parse_yaml(self, yaml_content: str, use_cache: bool) -> list[dict[str, Any]]:
        """Parse YAML into list of documents."""
        if "---" in yaml_content:
            return list(yaml.safe_load_all(yaml_content))
        parsed = _parse_yaml_cached(yaml_content) if use_cache else yaml.safe_load(yaml_content)
        return [parsed]

    def _select_environment(
        self, documents: list[dict[str, Any]], environment: str | None
    ) -> dict[str, Any]:
        """Select document by environment name."""
        if environment:
            for doc in documents:
                if doc.get("metadata", {}).get("namespace") == environment:
                    logger.info(f"Selected environment '{environment}' from multi-document YAML")
                    return doc

            available_envs = [
                doc.get("metadata", {}).get("namespace", "default") for doc in documents
            ]
            raise YamlPipelineBuilderError(
                f"Environment '{environment}' not found in YAML. "
                f"Available environments: {', '.join(available_envs)}"
            )

        if len(documents) > 1:
            logger.warning(
                f"Multi-document YAML detected ({len(documents)} documents) "
                "but no environment specified. "
                "Using first document. Specify environment parameter to select specific config."
            )

        return documents[0]

    def _validate_config(self, config: Any) -> dict[str, Any]:
        """Validate YAML structure."""
        if not isinstance(config, dict):
            raise YamlPipelineBuilderError(
                f"YAML document must be a dictionary, got {type(config).__name__}"
            )

        self._validate_manifest_format(config)

        result = self.validator.validate(config)
        if not result.is_valid:
            errors = "\n".join(f"  ERROR: {error}" for error in result.errors)
            raise YamlPipelineBuilderError(f"YAML validation failed:\n{errors}")

        for warning in result.warnings:
            logger.warning(f"YAML validation warning: {warning}")

        return config

    @staticmethod
    def _validate_manifest_format(config: dict[str, Any]) -> None:
        """Validate declarative manifest format."""
        if "kind" not in config:
            raise YamlPipelineBuilderError(
                "YAML must use declarative manifest format with 'kind' field. "
                "Example:\n"
                "apiVersion: v1\n"
                "kind: Pipeline\n"
                "metadata:\n"
                "  name: my-pipeline\n"
                "spec:\n"
                "  nodes: [...]"
            )

        if "spec" not in config:
            raise YamlPipelineBuilderError("Manifest YAML must have 'spec' field")

        if "metadata" not in config:
            raise YamlPipelineBuilderError("Manifest YAML must have 'metadata' field")

    def _build_graph(self, config: dict[str, Any]) -> DirectedGraph:
        """Build DirectedGraph using entity plugins."""
        graph = DirectedGraph()
        nodes_list = config.get("spec", {}).get("nodes", [])

        for node_config in nodes_list:
            # Find plugin that can handle this entity
            for plugin in self.entity_plugins:
                if plugin.can_handle(node_config):
                    result = plugin.build(node_config, self, graph)
                    if result is not None:
                        graph += result
                    break
            else:
                # No plugin handled it - error
                kind = node_config.get("kind", "unknown")
                raise YamlPipelineBuilderError(f"No plugin can handle kind: {kind}")

        return graph

    @staticmethod
    def _extract_pipeline_config(config: dict[str, Any]) -> PipelineConfig:
        """Extract PipelineConfig from YAML."""
        spec = config.get("spec", {})
        metadata = config.get("metadata", {})

        return PipelineConfig(
            ports=spec.get("ports", {}),
            type_ports=spec.get("type_ports", {}),
            policies=spec.get("policies", {}),
            metadata=metadata,
            nodes=spec.get("nodes", []),
        )


# ============================================================================
# Entity Plugins - Each handles one entity type
# ============================================================================


class MacroEntityPlugin:
    """Plugin for handling macro_invocation entities."""

    def can_handle(self, node_config: dict[str, Any]) -> bool:
        """Handle macro_invocation kind."""
        return node_config.get("kind") == "macro_invocation"

    def build(
        self, node_config: dict[str, Any], builder: YamlPipelineBuilder, graph: DirectedGraph
    ) -> NodeSpec | None:
        """Expand macro into subgraph and merge into main graph."""
        instance_name = node_config["metadata"]["name"]
        spec = node_config.get("spec", {})
        macro_ref = spec.get("macro")
        if not macro_ref:
            raise YamlPipelineBuilderError(f"Macro '{instance_name}' missing spec.macro field")

        # Parse macro reference
        macro_name, namespace = self._parse_macro_ref(macro_ref)

        # Get config params for macro initialization
        config_params = spec.get("config", {}).copy()
        inputs = spec.get("inputs", {})
        dependencies = spec.get("dependencies", [])

        # Get macro from registry with init_params (macro handles schema conversion)
        try:
            macro_instance_obj: object = registry.get(
                macro_name, namespace=namespace, init_params=config_params
            )
        except ComponentNotFoundError as e:
            raise YamlPipelineBuilderError(
                f"Macro '{namespace}:{macro_name}' not found: {e}"
            ) from e
        except Exception as e:
            raise YamlPipelineBuilderError(
                f"Failed to instantiate macro '{macro_name}': {e}"
            ) from e

        # Validate it's actually a macro
        if not isinstance(macro_instance_obj, ConfigurableMacro):
            type_name = type(macro_instance_obj).__name__
            raise YamlPipelineBuilderError(
                f"Component '{macro_name}' is not a ConfigurableMacro (got {type_name})"
            )

        macro_instance: ConfigurableMacro = macro_instance_obj

        # Expand macro
        try:
            subgraph = macro_instance.expand(
                instance_name=instance_name, inputs=inputs, dependencies=dependencies
            )
        except Exception as e:
            raise YamlPipelineBuilderError(
                f"Failed to expand macro '{macro_name}' (instance '{instance_name}'): {e}"
            ) from e

        # Merge subgraph into main graph
        self._merge_subgraph(graph, subgraph, dependencies)

        logger.info(
            "✅ Expanded macro '{macro}' as '{instance}' ({nodes} nodes)",
            macro=macro_ref,
            instance=instance_name,
            nodes=len(subgraph.nodes),
        )

        # Return None - subgraph already merged into graph
        return None

    @staticmethod
    def _merge_subgraph(
        graph: DirectedGraph, subgraph: DirectedGraph, external_deps: list[str]
    ) -> None:
        """Merge subgraph into main graph with external dependencies."""
        if not external_deps:
            graph |= subgraph
        else:
            # Add external deps to entry nodes
            modified = DirectedGraph()
            for node in subgraph.nodes.values():
                if not subgraph.get_dependencies(node.name):
                    # Entry node - add external dependencies
                    modified += node.after(*external_deps)
                else:
                    # Internal node - keep original
                    modified += node
            graph |= modified

    @staticmethod
    def _parse_macro_ref(ref: str) -> tuple[str, str]:
        """Parse 'namespace:name' or 'name' into (name, namespace)."""
        return _parse_namespaced_ref(ref)


class NodeEntityPlugin:
    """Plugin for handling all node types (llm, function, agent, etc.)."""

    def __init__(self, builder: YamlPipelineBuilder):
        """Initialize with reference to builder for shared state."""
        self.builder = builder

    def can_handle(self, node_config: dict[str, Any]) -> bool:
        """Handle everything except macro_invocation."""
        return node_config.get("kind") != "macro_invocation"

    def build(
        self, node_config: dict[str, Any], builder: YamlPipelineBuilder, graph: DirectedGraph
    ) -> NodeSpec:
        """Build node from config."""
        # Validate structure
        if "kind" not in node_config:
            raise YamlPipelineBuilderError("Node missing 'kind' field")
        if "metadata" not in node_config or "name" not in node_config["metadata"]:
            raise YamlPipelineBuilderError(
                f"Node '{node_config.get('kind')}' missing metadata.name"
            )

        kind = node_config["kind"]
        node_id = node_config["metadata"]["name"]
        spec = node_config.get("spec", {}).copy()
        deps = spec.pop("dependencies", [])

        # Parse kind into (type, namespace)
        node_type, namespace = self._parse_kind(kind)

        # Get factory from registry
        factory_obj: object = registry.get(f"{node_type}_node", namespace=namespace)

        # Validate it's callable
        if not callable(factory_obj):
            raise YamlPipelineBuilderError(
                f"Node factory for '{node_type}_node' "
                f"is not callable (got {type(factory_obj).__name__})"
            )

        factory = cast("Callable[..., NodeSpec]", factory_obj)

        # Create node
        node: NodeSpec = factory(node_id, **spec)

        # Add dependencies
        if deps:
            node = node.after(*deps) if isinstance(deps, list) else node.after(deps)

        return node

    @staticmethod
    def _parse_kind(kind: str) -> tuple[str, str]:
        """Parse 'namespace:type' or 'type' into (type, namespace)."""
        node_kind, namespace = _parse_namespaced_ref(kind)
        node_type = node_kind.removesuffix("_node")
        return node_type, namespace


# ============================================================================
# Preprocessing Plugins
# ============================================================================


class EnvironmentVariablePlugin:
    """Resolve ${VAR} and ${VAR:default} in YAML."""

    ENV_VAR_PATTERN = re.compile(r"\$\{([A-Z_][A-Z0-9_]*?)(?::([^}]*))?\}")

    def process(self, config: dict[str, Any]) -> dict[str, Any]:
        """Recursively resolve environment variables."""
        result = _resolve_env_vars(config, self.ENV_VAR_PATTERN)
        assert isinstance(result, dict)  # nosec B101 - type narrowing
        return result


@singledispatch
def _resolve_env_vars(obj: Any, pattern: re.Pattern[str]) -> Any:
    """Recursively resolve ${VAR} in any structure."""
    return obj


@_resolve_env_vars.register
def _(obj: str, pattern: re.Pattern[str]) -> str | int | float | bool:
    """Resolve ${VAR} in strings with type coercion."""

    def replacer(match: re.Match[str]) -> str:
        var_name, default = match.group(1), match.group(2)
        env_value = os.environ.get(var_name)
        if env_value is None:
            if default is not None:
                return default
            raise YamlPipelineBuilderError(f"Environment variable '${{{var_name}}}' not set")
        return env_value

    resolved = pattern.sub(replacer, obj)
    if resolved != obj:  # Type coercion if changed
        if resolved.lower() in ("true", "yes", "1"):
            return True
        if resolved.lower() in ("false", "no", "0"):
            return False
        with suppress(ValueError):
            return int(resolved)
        with suppress(ValueError):
            return float(resolved)
    return resolved


@_resolve_env_vars.register(dict)
def _(obj: dict, pattern: re.Pattern[str]) -> dict[str, Any]:
    """Resolve ${VAR} in dict values."""
    return {k: _resolve_env_vars(v, pattern) for k, v in obj.items()}


@_resolve_env_vars.register(list)
def _(obj: list, pattern: re.Pattern[str]) -> list[Any]:
    """Resolve ${VAR} in list items."""
    return [_resolve_env_vars(item, pattern) for item in obj]


class TemplatePlugin:
    """Render Jinja2 templates in YAML (e.g., {{ variables.model }})."""

    def __init__(self) -> None:
        # YAML config templates don't contain HTML, so autoescape=False is safe
        # This is for configuration files, not web content
        self.env = Environment(autoescape=False, keep_trailing_newline=True)  # nosec B701

    def process(self, config: dict[str, Any]) -> dict[str, Any]:
        """Render Jinja2 templates with config as context."""
        result = _render_templates(config, config, self.env)
        assert isinstance(result, dict)  # nosec B101 - type narrowing
        return result


@singledispatch
def _render_templates(obj: Any, context: dict[str, Any], env: Any) -> Any:
    """Recursively render Jinja2 templates."""
    return obj


@_render_templates.register
def _(obj: str, context: dict[str, Any], env: Any) -> str | int | float | bool:
    """Render Jinja2 template in string."""
    if "{{" not in obj and "{%" not in obj:
        return obj

    try:
        rendered: str = env.from_string(obj).render(context)
        # Type coercion
        with suppress(ValueError):
            return int(rendered)
        with suppress(ValueError):
            return float(rendered)
        if rendered.lower() in ("true", "yes"):
            return True
        if rendered.lower() in ("false", "no"):
            return False
        return rendered
    except TemplateSyntaxError as e:
        raise YamlPipelineBuilderError(
            f"Invalid Jinja2 template syntax: {e}\nTemplate: {obj}"
        ) from e
    except UndefinedError as e:
        raise YamlPipelineBuilderError(
            f"Undefined variable in template: {e}\nTemplate: {obj}"
        ) from e


@_render_templates.register(dict)
def _(obj: dict, context: dict[str, Any], env: Any) -> dict[str, Any]:
    """Render templates in dict values."""
    return {k: _render_templates(v, context, env) for k, v in obj.items()}


@_render_templates.register(list)
def _(obj: list, context: dict[str, Any], env: Any) -> list[Any]:
    """Render templates in list items."""
    return [_render_templates(item, context, env) for item in obj]


# ============================================================================
# Utilities
# ============================================================================


@lru_cache(maxsize=32)
def _parse_yaml_cached(yaml_content: str) -> Any:
    """Cached YAML parsing.

    Returns dict[str, Any] in practice, but yaml.safe_load returns Any.
    """
    return yaml.safe_load(yaml_content)


def _parse_namespaced_ref(ref: str, default_namespace: str = "core") -> tuple[str, str]:
    """Parse 'namespace:name' or 'name' into (name, namespace).

    Args:
        ref: Reference string (e.g., "core:llm" or "llm")
        default_namespace: Namespace to use if not specified

    Returns:
        Tuple of (name, namespace)
    """
    if NAMESPACE_SEPARATOR in ref:
        namespace, name = ref.split(NAMESPACE_SEPARATOR, 1)
        return name, namespace
    return ref, default_namespace
