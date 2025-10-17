"""Pipeline Builder - Converts YAML configurations to DirectedGraphs.

Simple pipeline builder that focuses on basic YAML processing with simple data mapping.
"""

from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from hexdag.core.ports.observer_manager import ObserverManagerPort

# ObserverManager is now a port - passed as dependency
from hexdag.builtin.nodes.mapped_input import FieldMappingRegistry
from hexdag.core.bootstrap import ensure_bootstrapped
from hexdag.core.configurable import ConfigurableMacro
from hexdag.core.domain.dag import DirectedGraph, NodeSpec
from hexdag.core.logging import get_logger
from hexdag.core.orchestration.prompt.template import ChatPromptTemplate
from hexdag.core.pipeline_builder.component_instantiator import ComponentInstantiator
from hexdag.core.pipeline_builder.pipeline_config import PipelineConfig
from hexdag.core.pipeline_builder.yaml_validator import YamlValidator
from hexdag.core.registry import registry
from hexdag.core.registry.exceptions import ComponentNotFoundError
from hexdag.core.registry.models import NAMESPACE_SEPARATOR

logger = get_logger(__name__)


@lru_cache(maxsize=32)
def _parse_yaml_cached(yaml_content: str) -> Any:
    """Parse and validate YAML content with caching.

    This is a module-level function so lru_cache doesn't cause memory leaks.
    Cache key is the YAML content string itself.

    Parameters
    ----------
    yaml_content : str
        Raw YAML content to parse

    Returns
    -------
    Any
        Parsed YAML content (typically dict, but could be list, str, etc.)
    """
    return yaml.safe_load(yaml_content)


class YamlPipelineBuilderError(Exception):
    """Custom exception for YAML pipeline building errors."""

    pass


class YamlPipelineBuilder:
    """Simple pipeline builder with basic data mapping support and intelligent auto-conversion."""

    def __init__(self, event_manager: "ObserverManagerPort | None" = None) -> None:
        """Initialize the pipeline builder.

        Args
        ----
            event_manager: Optional event manager for observer pattern
        """
        # Ensure registry is bootstrapped before creating validator
        # This allows the validator to discover all registered node types including plugins
        ensure_bootstrapped()

        self.registered_functions: dict[str, Any] = {}

        self.event_manager = event_manager  # Now expects a port implementation
        self.field_mapping_registry = FieldMappingRegistry()
        self.validator = YamlValidator()
        self.component_instantiator = ComponentInstantiator()

    def register_function(self, name: str, func: Any) -> None:
        """Register a function for use in YAML pipelines."""
        self.registered_functions[name] = func

    def build_from_yaml_file(
        self, yaml_file_path: str, use_cache: bool = True
    ) -> tuple[DirectedGraph, PipelineConfig]:
        """Build DirectedGraph and PipelineConfig from YAML file with caching.

        Parameters
        ----------
        yaml_file_path : str
            Path to the YAML file to build from
        use_cache : bool
            Whether to use LRU caching (default: True)

        Returns
        -------
        tuple[DirectedGraph, PipelineConfig]
            Tuple of (DirectedGraph, PipelineConfig with ports and policies)

        Raises
        ------
        YamlPipelineBuilderError
            If the YAML file cannot be loaded or parsed

        Notes
        -----
        Caching is based on file content hash, so changes to the YAML file
        will automatically invalidate the cache. The cache is in-memory (LRU)
        with a maximum of 32 entries.
        """
        try:
            yaml_path = Path(yaml_file_path)
            yaml_content = yaml_path.read_text(encoding="utf-8")

            # Use cached or uncached version based on flag
            return self.build_from_yaml_string(yaml_content, use_cache=use_cache)

        except OSError as e:
            raise YamlPipelineBuilderError(
                f"Failed to load YAML file '{yaml_file_path}': {e}"
            ) from e

    def build_from_yaml_string(
        self, yaml_content: str, use_cache: bool = True
    ) -> tuple[DirectedGraph, PipelineConfig]:
        """Convert declarative YAML manifest to DirectedGraph and PipelineConfig.

        Expects declarative manifest format with K8s-style structure:
        ```yaml
        apiVersion: hexdag.omniviser.io/v1alpha1
        kind: Pipeline
        metadata:
          name: my-pipeline
        spec:
          ports:
            llm: core:openai(model=gpt-4)
          policies:
            retry: core:retry(max_retries=3)
          nodes:
            - kind: llm_node
              metadata:
                name: processor
              spec:
                prompt_template: "Process {{input}}"
                dependencies: []
        ```

        Parameters
        ----------
        yaml_content : str
            YAML content string to parse
        use_cache : bool
            Whether to use cached YAML parsing (default: True)

        Returns
        -------
        tuple[DirectedGraph, PipelineConfig]
            Tuple of (DirectedGraph, PipelineConfig with ports and policies)
        """
        config = self._parse_and_validate_yaml(yaml_content, use_cache=use_cache)

        # Extract pipeline configuration including ports and policies
        pipeline_config = self._extract_pipeline_config(config)

        # Register common mappings for backward compatibility
        self._register_common_mappings(config)

        # Build the graph
        graph = self._build_graph_from_config(config)

        logger.info(
            "✅ Built pipeline '{name}' with {nodes} nodes, {ports} ports, {policies} policies",
            name=pipeline_config.metadata.get("name", "unknown"),
            nodes=len(graph.nodes),
            ports=len(pipeline_config.ports),
            policies=len(pipeline_config.policies),
        )

        return graph, pipeline_config

    def _parse_and_validate_yaml(self, yaml_content: str, use_cache: bool = True) -> dict[str, Any]:
        """Parse YAML content and validate the configuration.

        Parameters
        ----------
        yaml_content : str
            YAML content string to parse
        use_cache : bool
            Whether to use cached parsing (default: True)

        Returns
        -------
        dict[str, Any]
            Validated configuration dictionary

        Raises
        ------
        YamlPipelineBuilderError
            If YAML parsing or validation fails
        """
        try:
            # Use cached function (module-level) or direct parsing
            config = _parse_yaml_cached(yaml_content) if use_cache else yaml.safe_load(yaml_content)
        except yaml.YAMLError as e:
            raise YamlPipelineBuilderError(f"Invalid YAML content: {e}") from e

        # Ensure config is a dictionary
        if not isinstance(config, dict):
            raise YamlPipelineBuilderError(
                f"YAML root must be a dictionary, got {type(config).__name__}"
            )

        # Validate manifest format
        self._validate_manifest_format(config)

        # Validate configuration
        validation_result = self.validator.validate(config)

        if not validation_result.is_valid:
            errors = "\n".join(f"  ERROR: {error}" for error in validation_result.errors)
            raise YamlPipelineBuilderError(f"YAML validation failed:\n{errors}")

        # Log warnings and suggestions
        for warning in validation_result.warnings:
            logger.warning("YAML validation warning: {msg}", msg=warning)
        for suggestion in validation_result.suggestions:
            logger.info("YAML validation suggestion: {msg}", msg=suggestion)

        return config

    def _build_graph_from_config(self, config: dict[str, Any]) -> DirectedGraph:
        """Build DirectedGraph from validated configuration.

        Parameters
        ----------
        config : dict[str, Any]
            Validated YAML configuration

        Returns
        -------
        DirectedGraph
            Constructed graph with all nodes
        """
        graph = DirectedGraph()
        nodes_list = config.get("spec", {}).get("nodes", [])

        for node_config in nodes_list:
            kind = node_config.get("kind", "")

            if kind == "macro_invocation":
                # Handle macro expansion
                self._expand_macro_into_graph(graph, node_config)
            else:
                # Regular node creation using += operator
                graph += self._build_node_from_config(node_config)

        return graph

    def _expand_macro_into_graph(self, graph: DirectedGraph, macro_config: dict[str, Any]) -> None:
        """Expand a macro invocation and merge the resulting subgraph into the main graph.

        Parameters
        ----------
        graph : DirectedGraph
            Main graph to expand the macro into
        macro_config : dict[str, Any]
            Macro invocation configuration from YAML

        Raises
        ------
        YamlPipelineBuilderError
            If macro cannot be found or expanded
        """
        # Extract macro configuration
        instance_name = macro_config.get("metadata", {}).get("name")
        if not instance_name:
            raise YamlPipelineBuilderError("macro_invocation must have metadata.name field")

        spec = macro_config.get("spec", {})
        macro_ref = spec.get("macro")
        if not macro_ref:
            raise YamlPipelineBuilderError(
                f"macro_invocation '{instance_name}' must specify spec.macro field"
            )

        # Parse macro reference (e.g., "core:research" -> ("research", "core"))
        macro_name, namespace = self._parse_macro_reference(macro_ref)

        # Extract configuration and inputs
        config_params = spec.get("config", {})
        inputs = spec.get("inputs", {})
        dependencies = spec.get("dependencies", [])

        # Get and instantiate macro from registry, then expand
        try:
            # Registry automatically instantiates the macro with init_params
            macro_instance = registry.get(
                macro_name, namespace=namespace, init_params=config_params
            )
        except ComponentNotFoundError as e:
            raise YamlPipelineBuilderError(
                f"Macro '{namespace}:{macro_name}' not found in registry: {e}"
            ) from e
        except Exception as e:
            raise YamlPipelineBuilderError(
                f"Failed to instantiate macro '{macro_name}': {e}"
            ) from e

        # Validate it's actually a macro
        if not isinstance(macro_instance, ConfigurableMacro):
            type_name = type(macro_instance).__name__
            raise YamlPipelineBuilderError(
                f"Component '{macro_name}' is not a ConfigurableMacro (got {type_name})"
            )

        # Expand the macro into a subgraph
        try:
            subgraph = macro_instance.expand(
                instance_name=instance_name, inputs=inputs, dependencies=dependencies
            )
        except Exception as e:
            raise YamlPipelineBuilderError(
                f"Failed to expand macro '{macro_name}' (instance '{instance_name}'): {e}"
            ) from e

        # Merge subgraph into main graph
        self._merge_subgraph_into_graph(graph, subgraph, dependencies)

        logger.info(
            "✅ Expanded macro '{macro}' as '{instance}' ({nodes} nodes)",
            macro=macro_ref,
            instance=instance_name,
            nodes=len(subgraph.nodes),
        )

    @staticmethod
    def _parse_macro_reference(macro_ref: str) -> tuple[str, str]:
        """Parse macro reference into (name, namespace).

        Parameters
        ----------
        macro_ref : str
            Macro reference like "core:research" or "research"

        Returns
        -------
        tuple[str, str]
            (macro_name, namespace)
        """
        if NAMESPACE_SEPARATOR in macro_ref:
            namespace, macro_name = macro_ref.split(NAMESPACE_SEPARATOR, 1)
        else:
            namespace = "core"
            macro_name = macro_ref

        return macro_name, namespace

    @staticmethod
    def _merge_subgraph_into_graph(
        graph: DirectedGraph, subgraph: DirectedGraph, external_dependencies: list[str]
    ) -> None:
        """Merge a subgraph into the main graph.

        Parameters
        ----------
        graph : DirectedGraph
            Main graph to merge into
        subgraph : DirectedGraph
            Subgraph to merge (from macro expansion)
        external_dependencies : list[str]
            List of node names in the main graph that the subgraph depends on
        """
        # Merge subgraph using |= operator for cleaner code
        if not external_dependencies:
            # Simple case: no external dependencies, direct merge
            graph |= subgraph
        else:
            # Need to add external dependencies to entry nodes before merging
            modified_subgraph = DirectedGraph()
            for node in subgraph:  # Using iterator instead of .nodes.values()
                internal_deps = subgraph.get_dependencies(node.name)
                if not internal_deps:
                    # Entry node - add external dependencies
                    modified_subgraph += node.after(*external_dependencies)
                else:
                    # Internal node - keep original dependencies
                    modified_subgraph += node

            # Merge the modified subgraph using |= operator
            graph |= modified_subgraph

    def _build_node_from_config(self, node_config: dict[str, Any]) -> NodeSpec:
        """Build a single NodeSpec from node configuration.

        Parameters
        ----------
        node_config : dict[str, Any]
            Node configuration from YAML

        Returns
        -------
        NodeSpec
            Constructed node specification
        """
        # Extract node configuration
        node_id = node_config.get("metadata", {}).get("name")
        node_type, namespace = self._parse_kind(node_config["kind"])
        params = node_config.get("spec", {}).copy()  # Copy to avoid modifying original
        deps = params.pop("dependencies", [])

        # Process parameters
        params = self._process_node_params(node_id, node_type, params)

        # Create node using factory
        node = self._create_node_from_factory(node_id, node_type, namespace, params)

        # Add dependencies
        if deps:
            node = node.after(*deps) if isinstance(deps, list) else node.after(deps)

        return node

    def _process_node_params(
        self, node_id: str, node_type: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Process and transform node parameters.

        Parameters
        ----------
        node_id : str
            Node identifier
        node_type : str
            Type of node (e.g., "llm", "function")
        params : dict[str, Any]
            Raw node parameters

        Returns
        -------
        dict[str, Any]
            Processed parameters
        """
        # Handle field_mapping parameter
        field_mapping = params.get("field_mapping")
        if field_mapping:
            resolved_mapping = self.field_mapping_registry.get(field_mapping)
            params["field_mapping"] = resolved_mapping
            logger.debug(
                "Node '{node_id}' using field mapping: {mapping}",
                node_id=node_id,
                mapping=resolved_mapping,
            )

        # Auto-convert LLM nodes with incompatible template + schema combinations
        if node_type == "llm":
            logger.debug(
                "📋 LLM node '{node_id}' original params: {params}",
                node_id=node_id,
                params=list(params.keys()),
            )
            params = self._auto_convert_llm_node(node_id, params)
            logger.debug(
                "📋 LLM node '{node_id}' final params: {params}",
                node_id=node_id,
                params=list(params.keys()),
            )

        # Resolve function references
        if node_type == "function" and "fn" in params:
            func_ref = params["fn"]
            if isinstance(func_ref, str) and func_ref in self.registered_functions:
                params["fn"] = self.registered_functions[func_ref]

        return params

    @staticmethod
    def _create_node_from_factory(
        node_id: str, node_type: str, namespace: str, params: dict[str, Any]
    ) -> NodeSpec:
        """Create NodeSpec using registry factory.

        Parameters
        ----------
        node_id : str
            Node identifier
        node_type : str
            Type of node
        namespace : str
            Component namespace
        params : dict[str, Any]
            Processed node parameters

        Returns
        -------
        NodeSpec
            Created node specification

        Raises
        ------
        TypeError
            If factory is not callable or doesn't return NodeSpec
        """
        ensure_bootstrapped()

        factory_name = f"{node_type}_node"
        factory = registry.get(factory_name, namespace=namespace)

        if not callable(factory):
            raise TypeError(f"Expected callable factory for {factory_name}, got {type(factory)}")

        node = factory(node_id, **params)

        if not isinstance(node, NodeSpec):
            raise TypeError(f"Factory {factory_name} did not return a NodeSpec")

        return node

    @staticmethod
    def _validate_manifest_format(config: dict[str, Any]) -> None:
        """Validate that config uses declarative manifest format.

        Args
        ----
            config: Raw YAML configuration

        Raises
        ------
            YamlPipelineBuilderError: If config is not in manifest format
        """
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

    @staticmethod
    def _parse_kind(kind: str) -> tuple[str, str]:
        """Parse kind into (node_type, namespace).

        Supports both simple kinds and namespace-qualified kinds:
        - "llm_node" -> ("llm", "core")
        - "my-plugin:dalle_node" -> ("dalle", "my-plugin")

        Args
        ----
            kind: Kind string from YAML

        Returns
        -------
            Tuple of (node_type, namespace)
        """
        if NAMESPACE_SEPARATOR in kind:
            # Namespace-qualified kind
            namespace, node_kind = kind.split(NAMESPACE_SEPARATOR, 1)
        else:
            # Simple kind - assume core namespace
            namespace = "core"
            node_kind = kind

        # Remove '_node' suffix if present (e.g., "llm_node" -> "llm")
        node_type = node_kind.removesuffix("_node")

        return node_type, namespace

    def _extract_pipeline_config(self, config: dict[str, Any]) -> PipelineConfig:
        """Extract complete pipeline configuration including ports and policies.

        Parameters
        ----------
        config : dict[str, Any]
            Raw YAML configuration

        Returns
        -------
        PipelineConfig
            Complete pipeline configuration with ports, policies, and metadata
        """
        metadata_section = config.get("metadata", {})
        spec = config.get("spec", {})

        # Extract ports configuration
        ports = spec.get("ports", {})
        type_ports = spec.get("type_ports", {})
        policies = spec.get("policies", {})

        # Build metadata dict using comprehension with merge
        base_fields = ["name", "description"]
        optional_fields = ["version", "author", "tags", "environment"]

        metadata = {k: metadata_section.get(k) for k in base_fields} | {
            k: metadata_section[k] for k in optional_fields if k in metadata_section
        }

        # Create PipelineConfig
        pipeline_config = PipelineConfig(
            ports=ports,
            type_ports=type_ports,
            policies=policies,
            metadata=metadata,
            nodes=spec.get("nodes", []),
        )

        logger.debug(
            "Extracted pipeline config: {ports} ports, {policies} policies",
            ports=len(ports),
            policies=len(policies),
        )

        return pipeline_config

    def _register_common_mappings(self, config: dict[str, Any]) -> None:
        """Register common field mappings from manifest config.

        Args
        ----
            config: Pipeline configuration
        """
        common_mappings = config.get("spec", {}).get("common_field_mappings", {})

        for name, mapping in common_mappings.items():
            self.field_mapping_registry.register(name, mapping)
            logger.debug(
                "Registered common field mapping '{name}': {mapping}", name=name, mapping=mapping
            )

    @staticmethod
    def _auto_convert_llm_node(node_id: str, params: dict[str, Any]) -> dict[str, Any]:
        """Auto-convert LLM node parameters to handle common configuration incompatibilities.

        Specifically handles the case where prompt_template (string) is used with output_schema,
        which requires structured templates. Automatically converts to ChatPromptTemplate.

        Args
        ----
            node_id: Node identifier for logging
            params: Original node parameters

        Returns
        -------
            Updated parameters with auto-conversions applied
        """
        # Create a copy to avoid modifying the original
        params = params.copy()

        # Check for incompatible prompt_template + output_schema combination
        has_string_template = "prompt_template" in params and isinstance(
            params["prompt_template"], str
        )
        has_output_schema = "output_schema" in params
        has_parse_as_json = params.get("parse_as_json", False)

        if has_string_template and (has_output_schema or has_parse_as_json):
            # Auto-convert string template to structured template
            original_template = params["prompt_template"]

            # Remove parse_as_json (incompatible with structured templates)
            if "parse_as_json" in params:
                del params["parse_as_json"]

            # Convert the string template to ChatPromptTemplate object
            # Keep the prompt_template parameter name but change the value type
            chat_template = ChatPromptTemplate(human_message=original_template)
            params["prompt_template"] = chat_template

            logger.info(
                f"🔄 Auto-converted LLM node '{node_id}': "
                f"prompt_template (string) + output_schema → prompt_template (ChatPromptTemplate)"
            )

        return params
