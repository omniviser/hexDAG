"""YAML Pipeline Builder — slim coordinator with plugin architecture.

The builder orchestrates the pipeline from YAML to DirectedGraph:
1. Parse YAML → 2. Select environment → 3. Validate → 4. Preprocess → 5. Build graph

Plugins are extracted into focused modules:
- ``preprocessing/``: IncludePreprocessPlugin, EnvironmentVariablePlugin, TemplatePlugin
- ``plugins/``: MacroDefinitionPlugin, MacroEntityPlugin, NodeEntityPlugin
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

import yaml

# Re-export extracted classes for backward compatibility.
# Existing code that imports from ``yaml_builder`` continues to work.
from hexdag.compiler.plugins.config_definition import ConfigDefinitionPlugin  # noqa: F401
from hexdag.compiler.plugins.macro_definition import MacroDefinitionPlugin  # noqa: F401
from hexdag.compiler.plugins.macro_entity import MacroEntityPlugin  # noqa: F401
from hexdag.compiler.plugins.node_entity import NodeEntityPlugin  # noqa: F401
from hexdag.compiler.preprocessing.env_vars import EnvironmentVariablePlugin  # noqa: F401
from hexdag.compiler.preprocessing.include import IncludePreprocessPlugin  # noqa: F401
from hexdag.compiler.preprocessing.template import TemplatePlugin  # noqa: F401
from hexdag.compiler.reference_resolver import (
    extract_refs_from_expressions,
    extract_refs_from_mapping,
    extract_refs_from_template,
)
from hexdag.compiler.yaml_validator import YamlValidator
from hexdag.kernel.domain.dag import DirectedGraph
from hexdag.kernel.domain.pipeline_config import BaseNodeConfig, PipelineConfig
from hexdag.kernel.exceptions import YamlPipelineBuilderError  # noqa: F401
from hexdag.kernel.logging import get_logger
from hexdag.kernel.resolver import register_alias
from hexdag.kernel.validation.sanitized_types import register_type_from_config

logger = get_logger(__name__)


# ============================================================================
# Plugin Protocols
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
        self, node_config: dict[str, Any], builder: YamlPipelineBuilder, graph: DirectedGraph
    ) -> Any:
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
    """YAML -> DirectedGraph builder with plugin support.

    Template Rendering Phases
    -------------------------
    hexDAG uses a multi-phase rendering strategy:

    **Phase 1** (build-time): ``${VAR}`` env var substitution via
    ``EnvironmentVariablePlugin``. Secret patterns (``*_API_KEY``, etc.)
    are deferred to Phase 3b.

    **Phase 2** (build-time): ``{{ var }}`` Jinja2 rendering via
    ``TemplatePlugin``. Node ``spec`` fields and pipeline ``outputs``
    are **skipped** — preserved for Phase 3.

    **Phase 3** (runtime): ``{{var}}`` rendering via ``PromptTemplate``
    when nodes execute with actual dependency outputs.

    **Phase 3b** (runtime): Deferred ``${VAR}`` resolution at adapter
    instantiation via ``_resolve_deferred_env_vars``.

    Workflow:
    1. Parse YAML
    2. Select environment
    3. Validate structure
    4. Run preprocessing plugins (Phase 1 → Phase 2)
    5. Build graph using entity plugins
    6. Extract pipeline config
    """

    def __init__(self, base_path: Path | None = None) -> None:
        """Initialize builder.

        Args:
            base_path: Base directory for resolving includes (default: cwd)
        """
        self.base_path = base_path or Path.cwd()
        self.validator = YamlValidator()

        # Plugins
        self.preprocess_plugins: list[PreprocessPlugin] = []
        self.entity_plugins: list[EntityPlugin] = []

        # Inline config from kind: Config documents in multi-doc YAML
        self._inline_config: Any = None  # HexDAGConfig | None (avoid circular import)

        self._register_default_plugins()

    def _register_default_plugins(self) -> None:
        """Register default plugins for common use cases."""
        # Preprocessing plugins (run before building)
        self.preprocess_plugins.append(IncludePreprocessPlugin(base_path=self.base_path))
        self.preprocess_plugins.append(EnvironmentVariablePlugin())
        self.preprocess_plugins.append(TemplatePlugin())

        # Entity plugins (build specific entity types)
        self.entity_plugins.append(ConfigDefinitionPlugin())  # Process Config definitions
        self.entity_plugins.append(MacroDefinitionPlugin())  # Process Macro definitions
        self.entity_plugins.append(MacroEntityPlugin())  # Then macro invocations
        self.entity_plugins.append(NodeEntityPlugin(self))  # Finally regular nodes

    @contextmanager
    def _temporary_base_path(self, new_base: Path) -> Any:
        """Context manager for temporarily changing base_path.

        Args:
            new_base: Temporary base path to use

        Yields:
            None
        """
        original_base = self.base_path
        self.base_path = new_base

        # Update include plugin base paths
        for plugin in self.preprocess_plugins:
            if isinstance(plugin, IncludePreprocessPlugin):
                plugin.base_path = new_base

        try:
            yield
        finally:
            # Always restore original state
            self.base_path = original_base
            for plugin in self.preprocess_plugins:
                if isinstance(plugin, IncludePreprocessPlugin):
                    plugin.base_path = original_base

    # --- Public API ---

    def build_from_yaml_file(
        self, yaml_path: str, use_cache: bool = True
    ) -> tuple[DirectedGraph, PipelineConfig]:
        """Build from YAML file.

        Args:
            yaml_path: Path to YAML file
            use_cache: Whether to use cached YAML parsing

        Returns:
            Tuple of (DirectedGraph, PipelineConfig)
        """
        yaml_file = Path(yaml_path)
        yaml_content = yaml_file.read_text(encoding="utf-8")

        # Use context manager to temporarily change base_path for relative includes
        with self._temporary_base_path(yaml_file.parent):
            return self.build_from_yaml_string(yaml_content, use_cache=use_cache)

    def build_from_yaml_string(
        self, yaml_content: str, use_cache: bool = True, environment: str | None = None
    ) -> tuple[DirectedGraph, PipelineConfig]:
        """Build DirectedGraph + PipelineConfig from YAML string."""
        # Step 1: Parse YAML
        documents = self._parse_yaml(yaml_content, use_cache=use_cache)

        # Step 2: Process ALL documents for definitions first
        #         This allows Config and Macros to be defined in multi-document YAML
        #         before the pipeline that uses them
        for doc in documents:
            kind = doc.get("kind")
            if kind == "Config":
                # Process Config definition (skip template rendering)
                processed_doc = doc
                for plugin in self.preprocess_plugins:
                    if not isinstance(plugin, TemplatePlugin):
                        processed_doc = plugin.process(processed_doc)
                self._process_config_definition(processed_doc)
            elif kind == "Macro":
                # Process includes for macro definitions (but skip template rendering)
                # Templates in macros should be preserved for expansion-time rendering
                processed_doc = doc
                for plugin in self.preprocess_plugins:
                    # Skip TemplatePlugin for Macro definitions - templates should be preserved
                    if not isinstance(plugin, TemplatePlugin):
                        processed_doc = plugin.process(processed_doc)
                # Validate and process macro definition
                processed_doc = self._validate_config(processed_doc)
                # Register macro (MacroDefinitionPlugin handles this)
                self._process_macro_definitions([processed_doc])

        # Step 3: Select pipeline environment
        config = self._select_environment(documents, environment)

        # Skip if selected document is a Macro (already processed above)
        if config.get("kind") == "Macro":
            # Return empty graph for macro-only documents
            logger.info("Document contains only macro definitions, no pipeline to build")
            return DirectedGraph(), PipelineConfig()

        # Step 4: Preprocess FIRST (includes must resolve before validation)
        for plugin in self.preprocess_plugins:
            config = plugin.process(config)

        # Step 4.5: Register user-defined aliases BEFORE validation
        # so that custom node kinds can pass validation
        self._register_aliases(config)

        # Step 4.6: Register custom sanitized types BEFORE validation
        # so that output_schema type names can pass validation
        self._register_custom_types(config)

        # Step 5: Validate structure (after includes are resolved)
        config = self._validate_config(config)

        # Step 6: Build graph
        graph = self._build_graph(config)

        # Step 7: Extract pipeline config
        pipeline_config = self._extract_pipeline_config(config)

        logger.info(
            "Built pipeline '{name}' with {nodes} nodes, {ports} ports, {policies} policies",
            name=pipeline_config.metadata.get("name", "unknown"),
            nodes=len(graph),
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
        """Select document by environment name.

        Skips Macro definitions when selecting the Pipeline document.
        """
        # Filter out Config and Macro definitions - they're processed separately
        pipeline_docs = [doc for doc in documents if doc.get("kind") not in ("Macro", "Config")]

        if not pipeline_docs:
            # No pipeline documents, return first macro (for macro-only files)
            return documents[0]

        if environment:
            for doc in pipeline_docs:
                if doc.get("metadata", {}).get("namespace") == environment:
                    logger.info("Selected environment '{}' from multi-document YAML", environment)
                    return doc

            available_envs = [
                doc.get("metadata", {}).get("namespace", "default") for doc in pipeline_docs
            ]
            raise YamlPipelineBuilderError(
                f"Environment '{environment}' not found in YAML. "
                f"Available environments: {', '.join(available_envs)}"
            )

        if len(pipeline_docs) > 1:
            logger.warning(
                "Multi-document YAML detected ({} pipeline documents) "
                "but no environment specified. "
                "Using first pipeline. Specify environment parameter to select specific config.",
                len(pipeline_docs),
            )

        return pipeline_docs[0]

    def _validate_config(self, config: Any) -> dict[str, Any]:
        """Validate YAML structure."""
        if not isinstance(config, dict):
            raise YamlPipelineBuilderError(
                f"YAML document must be a dictionary, got {type(config).__name__}"
            )

        self._validate_manifest_format(config)

        result = self.validator.validate(config)
        if not result:
            errors = "\n".join(f"  ERROR: {error}" for error in result.errors)
            raise YamlPipelineBuilderError(f"YAML validation failed:\n{errors}")

        for warning in result.warnings:
            logger.warning("YAML validation warning: {}", warning)

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

        # Macro and Config definitions have different structure
        kind = config.get("kind")
        if kind == "Macro":
            # Macro has: metadata, parameters, nodes (no spec)
            if "metadata" not in config:
                raise YamlPipelineBuilderError("Macro definition must have 'metadata' field")
            if "nodes" not in config:
                raise YamlPipelineBuilderError("Macro definition must have 'nodes' field")
        elif kind == "Config":
            # Config has: metadata, spec (no nodes)
            if "spec" not in config:
                raise YamlPipelineBuilderError("Config manifest must have 'spec' field")
        else:
            # Pipeline and other kinds require spec
            if "spec" not in config:
                raise YamlPipelineBuilderError("Manifest YAML must have 'spec' field")

            if "metadata" not in config:
                raise YamlPipelineBuilderError("Manifest YAML must have 'metadata' field")

    def _process_macro_definitions(self, macro_configs: list[dict[str, Any]]) -> None:
        """Process macro definitions and register them.

        Parameters
        ----------
        macro_configs : list[dict[str, Any]]
            List of validated macro configuration dictionaries
        """
        # Create temporary graph (not used for macros, but required by plugin interface)
        temp_graph = DirectedGraph()

        # Find MacroDefinitionPlugin
        macro_plugin = next(
            (p for p in self.entity_plugins if isinstance(p, MacroDefinitionPlugin)), None
        )

        if macro_plugin is None:
            raise YamlPipelineBuilderError(
                "MacroDefinitionPlugin not found. Cannot process macro definitions."
            )

        for macro_config in macro_configs:
            # Wrap in a fake node config format expected by entity plugin
            node_config = macro_config
            macro_plugin.build(node_config, self, temp_graph)

    def _process_config_definition(self, config_doc: dict[str, Any]) -> None:
        """Process a kind: Config document and store its settings.

        Parameters
        ----------
        config_doc : dict[str, Any]
            Validated Config document
        """
        temp_graph = DirectedGraph()

        config_plugin = next(
            (p for p in self.entity_plugins if isinstance(p, ConfigDefinitionPlugin)), None
        )

        if config_plugin is None:
            raise YamlPipelineBuilderError(
                "ConfigDefinitionPlugin not found. Cannot process Config definitions."
            )

        config_plugin.build(config_doc, self, temp_graph)

    def _register_aliases(self, config: dict[str, Any]) -> None:
        """Register user-defined aliases from spec.aliases before validation.

        This allows custom node kinds to be used in YAML without requiring
        full module paths.

        Parameters
        ----------
        config : dict[str, Any]
            The YAML configuration dict
        """
        spec = config.get("spec", {})
        aliases = spec.get("aliases", {})

        for alias, full_path in aliases.items():
            register_alias(alias, full_path)
            logger.debug("Registered alias: {} -> {}", alias, full_path)

    def _register_custom_types(self, config: dict[str, Any]) -> None:
        """Register custom sanitized types from ``spec.custom_types`` before validation.

        This allows output_schema fields to use custom type names defined
        declaratively in the pipeline YAML.

        Parameters
        ----------
        config : dict[str, Any]
            The YAML configuration dict
        """
        spec = config.get("spec", {})
        custom_types = spec.get("custom_types", {})

        for type_name, type_config in custom_types.items():
            register_type_from_config(type_name, type_config)
            logger.debug("Registered custom type: {}", type_name)

    def _build_graph(self, config: dict[str, Any]) -> DirectedGraph:
        """Build DirectedGraph using entity plugins.

        Uses a two-pass approach:

        **Pass 1** — Collect all node names so the reference resolver can
        distinguish ``node_name.field`` from arbitrary identifiers.

        **Pass 2** — For each node, infer dependencies from ``input_mapping``,
        ``expressions``, and ``prompt_template``/``template`` fields.  Also
        supports the ``wait_for`` keyword for ordering-only relationships.

        Implicit sequential dependencies are preserved: if a node has no
        ``dependencies``, no ``wait_for``, no ``input_mapping``, and no
        inline references, it chains to the previous node (unless it is the
        first node, in which case it is a root).

        Explicit ``dependencies: []`` still marks a node as root.
        """
        graph = DirectedGraph()
        spec = config.get("spec", {})
        nodes_list = spec.get("nodes", [])

        # --- Pass 1: collect all node names ---
        known_nodes = frozenset(
            nc.get("metadata", {}).get("name")
            for nc in nodes_list
            if nc.get("metadata", {}).get("name")
        )

        # --- Pass 2: infer deps & build ---
        previous_node_id: str | None = None

        for node_config in nodes_list:
            node_id = node_config.get("metadata", {}).get("name")
            node_spec = node_config.get("spec", {})

            # Parse base fields via the model — single source of truth
            base = BaseNodeConfig.from_node_config(node_config)
            has_deps_key = "dependencies" in node_config or "dependencies" in node_spec

            # Infer deps from input_mapping / expressions / templates
            inferred_deps = self._infer_deps(node_config, known_nodes)

            # Merge wait_for into deps (ordering only, no data flow)
            wait_for = list(base.wait_for) if base.wait_for else []

            if has_deps_key:
                # Explicit dependencies provided — use them, but warn if redundant
                explicit_deps = set(base.dependencies or [])
                if inferred_deps and explicit_deps and explicit_deps <= inferred_deps:
                    warnings.warn(
                        f"Node '{node_id}': explicit 'dependencies' {sorted(explicit_deps)} "
                        f"can be inferred from input_mapping/expressions/templates. "
                        f"Consider removing the redundant 'dependencies' key.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                # Merge wait_for into explicit deps
                if wait_for:
                    merged = list(explicit_deps | set(wait_for))
                    node_config = {**node_config, "dependencies": merged}
            elif inferred_deps or wait_for:
                # No explicit deps — use inferred + wait_for
                merged_deps = inferred_deps | set(wait_for)
                node_config = {**node_config, "dependencies": sorted(merged_deps)}
            elif previous_node_id is not None:
                # Implicit sequential dependency (no refs, no wait_for, not first)
                node_config = {**node_config, "dependencies": [previous_node_id]}
            # else: first node, no deps → root

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

            # Track for next iteration
            if node_id:
                previous_node_id = node_id

        return graph

    @staticmethod
    def _infer_deps(node_config: dict[str, Any], known_nodes: frozenset[str]) -> set[str]:
        """Infer dependencies from input_mapping, expressions, and templates.

        Parameters
        ----------
        node_config : dict[str, Any]
            Node configuration dictionary.
        known_nodes : frozenset[str]
            All node names in the pipeline.

        Returns
        -------
        set[str]
            Inferred dependency node names.
        """
        refs: set[str] = set()
        spec = node_config.get("spec", {})
        node_id = node_config.get("metadata", {}).get("name")

        # Exclude self-references
        other_nodes = known_nodes - {node_id} if node_id else known_nodes

        # 1. input_mapping
        input_mapping = spec.get("input_mapping")
        if isinstance(input_mapping, dict):
            refs |= extract_refs_from_mapping(input_mapping, other_nodes)

        # 2. expressions
        expressions = spec.get("expressions")
        if isinstance(expressions, dict):
            refs |= extract_refs_from_expressions(expressions, other_nodes)

        # 3. prompt_template / template
        for key in ("prompt_template", "template", "initial_prompt_template", "main_prompt"):
            template = spec.get(key)
            if isinstance(template, str):
                refs |= extract_refs_from_template(template, other_nodes)

        return refs

    @staticmethod
    def _extract_pipeline_config(config: dict[str, Any]) -> PipelineConfig:
        """Extract PipelineConfig from YAML.

        Feeds the raw ``spec`` dict straight into ``model_validate`` so
        field names are defined once — on the Pydantic model.
        ``metadata`` lives at the pipeline level, not inside ``spec``,
        so it is injected separately.
        """
        spec = config.get("spec", {})
        return PipelineConfig.model_validate({**spec, "metadata": config.get("metadata", {})})


# ============================================================================
# Utilities
# ============================================================================


@lru_cache(maxsize=32)
def _parse_yaml_cached(yaml_content: str) -> Any:
    """Cached YAML parsing.

    Returns dict[str, Any] in practice, but yaml.safe_load returns Any.
    """
    return yaml.safe_load(yaml_content)
