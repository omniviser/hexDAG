"""YAML Pipeline Validator - Validates pipeline configurations."""

from collections.abc import Iterator
from typing import Any

from hexdag.kernel.domain.dag import DirectedGraph
from hexdag.kernel.domain.pipeline_config import BaseNodeConfig

# Separator for namespace:name format
NAMESPACE_SEPARATOR = ":"

# Lazy-loaded known node types (derived from resolver's builtin aliases)
_known_node_types: frozenset[str] | None = None


def _get_known_node_types() -> frozenset[str]:
    """Lazily load known node types from resolver's builtin aliases.

    This derives valid node types from the aliases registered by hexdag.stdlib,
    maintaining hexagonal architecture (core doesn't import from builtin).
    Types are cached after first load.
    """
    global _known_node_types
    if _known_node_types is None:
        from hexdag.kernel.resolver import (
            get_builtin_aliases,  # lazy: avoid circular import with node discovery
        )

        _known_node_types = frozenset(get_builtin_aliases().keys())
    return _known_node_types


# Keep KNOWN_NODE_TYPES as a module-level reference for backwards compatibility
# in tests that may import it directly.
KNOWN_NODE_TYPES = _get_known_node_types()


class ValidationReport:
    """Container for validation results with optimized memory usage."""

    __slots__ = ("_errors", "_warnings", "_suggestions")

    def __init__(self) -> None:
        """Initialize validation result."""
        self._errors: list[str] = []
        self._warnings: list[str] = []
        self._suggestions: list[str] = []

    @property
    def is_valid(self) -> bool:
        """Check if validation passed (no errors).

        Returns
        -------
        bool
            True if no errors are present, False otherwise
        """
        return len(self._errors) == 0

    def add_error(self, message: str) -> None:
        """Add an error message."""
        self._errors.append(message)

    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self._warnings.append(message)

    def add_suggestion(self, message: str) -> None:
        """Add a suggestion message."""
        self._suggestions.append(message)

    @property
    def errors(self) -> list[str]:
        """Get all error messages.

        Returns
        -------
        list[str]
            List of error messages
        """
        return self._errors

    @property
    def warnings(self) -> list[str]:
        """Get all warning messages.

        Returns
        -------
        list[str]
            List of warning messages
        """
        return self._warnings

    @property
    def suggestions(self) -> list[str]:
        """Get all suggestion messages.

        Returns
        -------
        list[str]
            List of suggestion messages
        """
        return self._suggestions

    def __bool__(self) -> bool:
        """True if validation passed (no errors)."""
        return not self._errors

    def __len__(self) -> int:
        """Total number of issues (errors + warnings + suggestions)."""
        return len(self._errors) + len(self._warnings) + len(self._suggestions)

    def __iter__(self) -> Iterator[str]:
        """Iterate over all issues: errors first, then warnings, then suggestions."""
        yield from self._errors
        yield from self._warnings
        yield from self._suggestions


class _SchemaValidator:
    """Validates YAML node specs against known schemas.

    Since we no longer have a registry, schema validation is simplified
    to basic structural validation.

    Note: This is an internal class. Use YamlValidator for public validation API.
    """

    def validate_node_spec(
        self,
        node_type: str,
        spec: dict[str, Any],
        namespace: str = "core",
    ) -> list[str]:
        """Validate a node's spec with basic structural checks.

        Args
        ----
            node_type: Type of node (e.g., "llm", "agent", "function")
            spec: Node specification from YAML manifest
            namespace: Component namespace (default: "core")

        Returns
        -------
            List of validation error messages (empty if valid)
        """
        # Basic validation - without registry, we can only do structural checks
        errors: list[str] = []

        # LLM nodes require template or prompt_template
        if (
            node_type in ("llm", "llm_node")
            and "template" not in spec
            and "prompt_template" not in spec
        ):
            errors.append("Missing required field 'template' (or 'prompt_template')")

        # Prompt nodes require template
        if (
            node_type in ("prompt", "prompt_node")
            and "template" not in spec
            and "prompt_ref" not in spec
        ):
            errors.append("Missing required field 'template' or 'prompt_ref'")

        # Agent nodes require initial_prompt_template or main_prompt
        if (
            node_type in ("agent", "agent_node")
            and "initial_prompt_template" not in spec
            and "main_prompt" not in spec
        ):
            errors.append("Missing required field 'initial_prompt_template' (or 'main_prompt')")

        # Function nodes require fn
        if node_type in ("function", "function_node") and "fn" not in spec:
            errors.append("Missing required field 'fn'")

        # Composite nodes require mode
        if node_type in ("composite", "composite_node"):
            errors.extend(self._validate_composite_spec(spec))

        return errors

    def _validate_composite_spec(self, spec: dict[str, Any]) -> list[str]:
        """Validate composite_node specific requirements.

        Parameters
        ----------
        spec : dict[str, Any]
            Node specification from YAML manifest

        Returns
        -------
        list[str]
            List of validation error messages
        """
        errors: list[str] = []

        if "mode" not in spec:
            errors.append("Missing required field 'mode'")
            return errors

        mode = spec.get("mode")
        valid_modes = ("while", "for-each", "times", "if-else", "switch")
        if mode not in valid_modes:
            errors.append(f"Invalid mode '{mode}'. Valid modes: {', '.join(valid_modes)}")
            return errors

        # Mode-specific validation
        match mode:
            case "while":
                if "condition" not in spec:
                    errors.append("Mode 'while' requires 'condition' field")
            case "for-each":
                if "items" not in spec:
                    errors.append("Mode 'for-each' requires 'items' field")
            case "times":
                if "count" not in spec:
                    errors.append("Mode 'times' requires 'count' field")
                elif not isinstance(spec.get("count"), int):
                    errors.append("Field 'count' must be an integer")
            case "if-else":
                if "condition" not in spec:
                    errors.append("Mode 'if-else' requires 'condition' field")
            case "switch":
                if "branches" not in spec:
                    errors.append("Mode 'switch' requires 'branches' field")
                elif not isinstance(spec.get("branches"), list):
                    errors.append("Field 'branches' must be a list")

        # Validate body field if present (can be string, list, or callable from !py)
        body = spec.get("body")
        body_pipeline = spec.get("body_pipeline")

        if body is not None and body_pipeline is not None:
            errors.append("Cannot specify both 'body' and 'body_pipeline'")

        # If body is a callable (from !py tag), it's already validated at parse time
        # If body is a list, it should be inline nodes
        if isinstance(body, list):
            for i, node_config in enumerate(body):
                if not isinstance(node_config, dict):
                    errors.append(f"body[{i}] must be a node configuration dict")
                elif "kind" not in node_config:
                    errors.append(f"body[{i}] missing required 'kind' field")

        return errors


class YamlValidator:
    """Validates YAML pipeline configurations with optimized performance."""

    def __init__(
        self,
        valid_node_types: set[str] | frozenset[str] | None = None,
    ) -> None:
        """Initialize validator with configurable node types.

        Args
        ----
            valid_node_types: Set of valid node type names. If None, uses defaults.
        """
        self._provided_node_types = (
            frozenset(valid_node_types) if valid_node_types is not None else None
        )
        self._cached_node_types: frozenset[str] | None = None

        # Schema validator for spec validation
        self.schema_validator = _SchemaValidator()

    @property
    def valid_node_types(self) -> frozenset[str]:
        """Get valid node types.

        Returns
        -------
        frozenset[str]
            Set of valid node type names
        """
        # If user provided explicit node types, use those
        if self._provided_node_types is not None:
            return self._provided_node_types

        # Otherwise, use auto-discovered node types
        if self._cached_node_types is None:
            self._cached_node_types = _get_known_node_types()

        return self._cached_node_types

    def validate(self, config: Any) -> ValidationReport:
        """Validate complete YAML configuration with optimized caching.

        Expects declarative manifest format: {kind: Pipeline,
        spec: {nodes: [{kind, metadata, spec}]}}

        Args
        ----
            config: Parsed YAML configuration

        Returns
        -------
        ValidationReport
            ValidationReport with errors, warnings, and suggestions
        """
        result = ValidationReport()

        # Validate manifest structure
        self._validate_manifest_structure(config, result)

        if not result:
            return result

        spec = config.get("spec", {})
        nodes = spec.get("nodes", [])

        # Validate nodes and cache the IDs and macro instances for reuse
        node_ids, macro_instances = self._validate_nodes(nodes, result)

        # Reuse cached node_ids and macro_instances for dependency validation
        self._validate_dependencies_with_cache(nodes, result, node_ids, macro_instances)

        return result

    def _validate_manifest_structure(self, config: Any, result: ValidationReport) -> None:
        """Validate declarative manifest YAML structure.

        Args
        ----
            config: Parsed YAML configuration
            result: ValidationReport to add errors to
        """
        if not isinstance(config, dict):
            result.add_error("Configuration must be a dictionary")
            return

        if "kind" not in config:
            result.add_error(
                "Configuration must contain 'kind' field (declarative manifest format required)"
            )
            return

        if "metadata" not in config:
            result.add_error("Configuration must contain 'metadata' field")
            return

        # Macro and Config definitions have different structure
        kind = config.get("kind")
        if kind == "Macro":
            # Macro has: metadata, parameters, nodes (no spec)
            if "nodes" not in config:
                result.add_error("Macro definition must contain 'nodes' field")
                return
            # Skip rest of validation for Macro kind
            return

        if kind == "Config":
            # Config has: metadata, spec (no nodes)
            if "spec" not in config:
                result.add_error("Config definition must contain 'spec' field")
            # Skip pipeline-specific validation for Config kind
            return

        # For Pipeline and other kinds, validate spec
        if "spec" not in config:
            result.add_error("Configuration must contain 'spec' field")
            return

        spec = config.get("spec", {})
        if not isinstance(spec, dict):
            result.add_error("'spec' field must be a dictionary")
            return

        if "nodes" not in spec:
            result.add_error("'spec' must contain 'nodes' field")
            return

        if not isinstance(spec["nodes"], list):
            result.add_error("'spec.nodes' field must be a list")
            return

        if len(spec["nodes"]) == 0:
            result.add_warning("Pipeline has no nodes defined")

        # Validate common_field_mappings structure if present
        common_mappings = spec.get("common_field_mappings")
        if common_mappings is not None and not isinstance(common_mappings, dict):
            result.add_error("'spec.common_field_mappings' must be a dictionary")

    def _validate_nodes(
        self, nodes: list[dict[str, Any]], result: ValidationReport
    ) -> tuple[set[str], set[str]]:
        """Validate nodes and return node IDs and macro instance names.

        Expects declarative node format: {kind, metadata: {name}, spec: {dependencies}}

        Returns
        -------
        tuple[set[str], set[str]]
            Tuple of (node_ids, macro_instance_names) for caching and reuse in dependency validation
        """
        node_ids = set()
        macro_instances = set()

        for i, node in enumerate(nodes):
            # Validate node has required fields
            if "kind" not in node:
                result.add_error(f"Node {i}: Missing 'kind' field")
                continue

            if "metadata" not in node:
                result.add_error(f"Node {i}: Missing 'metadata' field")
                continue

            node_id = node.get("metadata", {}).get("name")
            if not node_id:
                result.add_error(f"Node {i}: Missing 'metadata.name'")
                continue

            kind = node.get("kind", "")

            # Check node ID uniqueness
            if node_id in node_ids:
                result.add_error(f"Duplicate node ID: '{node_id}'")
            node_ids.add(node_id)

            # Special case: macro_invocation is not a node type, skip node type validation
            if kind == "macro_invocation":
                # Validate macro invocation spec (macro reference required)
                spec = node.get("spec", {})
                if "macro" not in spec:
                    result.add_error(
                        f"Node '{node_id}': macro_invocation must specify 'spec.macro' field"
                    )
                macro_instances.add(node_id)
                continue

            # Merge settings + spec for parameter validation
            # settings: contains literal config, spec: contains dynamic wiring
            _settings = node.get("settings", {})
            _spec = node.get("spec", {})
            merged_params = {**_settings, **_spec}

            # Handle module paths (e.g., hexdag.stdlib.nodes.LLMNode)
            if "." in kind and ":" not in kind:
                # This is a full module path, skip node type validation
                # (resolution will happen at build time)
                continue

            # Handle user-registered aliases (e.g., "fn" -> "hexdag.stdlib.nodes.FunctionNode")
            from hexdag.kernel.resolver import get_registered_aliases

            if kind in get_registered_aliases():
                # This is a registered alias, skip node type validation
                # (resolution will happen at build time via resolver)
                continue

            if NAMESPACE_SEPARATOR in kind:
                namespace, node_kind = kind.split(NAMESPACE_SEPARATOR, 1)
            else:
                namespace = "core"
                node_kind = kind

            # Remove '_node' suffix if present
            node_type = node_kind[:-5] if node_kind.endswith("_node") else node_kind

            qualified_node_type = f"{namespace}:{node_type}"

            params = merged_params

            # Validate node type
            # Support both qualified (namespace:type) and simple (type) formats
            if (
                qualified_node_type not in self.valid_node_types
                and node_type not in self.valid_node_types
                and node_kind not in self.valid_node_types
            ):
                # Show available types grouped by namespace
                by_namespace: dict[str, list[str]] = {}
                simple_types: list[str] = []
                has_namespaced = False

                for valid_type in sorted(self.valid_node_types):
                    if ":" in valid_type:
                        has_namespaced = True
                        ns, nt = valid_type.split(":", 1)
                        by_namespace.setdefault(ns, []).append(nt)
                    else:
                        # Legacy format without namespace
                        simple_types.append(valid_type)

                parts = []
                if by_namespace:
                    parts.append(
                        ", ".join(
                            f"{ns}:[{', '.join(types)}]"
                            for ns, types in sorted(by_namespace.items())
                        )
                    )
                if simple_types:
                    parts.append(", ".join(sorted(simple_types)))

                valid_types_str = ", ".join(parts) if parts else "none"

                # Use simple node_type in error if no valid types have namespaces (legacy mode)
                invalid_type_str = node_type if not has_namespaced else qualified_node_type

                result.add_error(
                    f"Node '{node_id}': Invalid type '{invalid_type_str}'. "
                    f"Valid types: {valid_types_str}"
                )

            # Validate node-specific requirements and schema
            self._validate_node_params(node_id, node_type, params, namespace, result)

        return node_ids, macro_instances

    def _validate_node_params(
        self,
        node_id: str | None,
        node_type: str,
        params: dict[str, Any],
        namespace: str,
        result: ValidationReport,
    ) -> None:
        """Validate node-specific parameters using basic structural validation.

        Args
        ----
            node_id: Node identifier
            node_type: Type of node (e.g., "llm", "function")
            params: Node spec parameters
            namespace: Component namespace
            result: ValidationReport to add errors to
        """
        # Schema-based validation
        schema_errors = self.schema_validator.validate_node_spec(
            node_type, params, namespace=namespace
        )
        for error in schema_errors:
            result.add_error(f"Node '{node_id}': {error}")

    def _validate_dependencies_with_cache(
        self,
        nodes: list[dict[str, Any]],
        result: ValidationReport,
        node_ids: set[str],
        macro_instances: set[str],
    ) -> None:
        """Validate node dependencies using cached node IDs and check for cycles.

        Dependencies are in spec.dependencies field.

        Parameters
        ----------
        nodes : list[dict[str, Any]]
            List of node configurations
        result : ValidationReport
            Report to add errors to
        node_ids : set[str]
            Cached set of valid node IDs from _validate_nodes
        macro_instances : set[str]
            Set of macro instance names (nodes will be generated at runtime)
        """
        dependency_graph = {}

        for node in nodes:
            node_id = node.get("metadata", {}).get("name")
            if not node_id:
                continue

            # Parse base fields via the model — single source of truth
            base = BaseNodeConfig.from_node_config(node)
            all_deps = list(base.dependencies or []) + list(base.wait_for or [])

            # Check all dependencies exist using cached node_ids
            valid_deps = set()
            for dep in all_deps:
                if dep in node_ids:
                    valid_deps.add(dep)
                    continue

                is_macro_generated = False
                for macro_instance in macro_instances:
                    if dep.startswith(f"{macro_instance}_"):
                        is_macro_generated = True
                        valid_deps.add(dep)
                        break

                # If not a known node and not macro-generated, it's an error
                if not is_macro_generated:
                    result.add_error(f"Node '{node_id}': Dependency '{dep}' does not exist")

            dependency_graph[node_id] = valid_deps

        # Check for cycles using DirectedGraph's public static method
        if cycle_message := DirectedGraph.detect_cycle(dependency_graph):
            result.add_error(cycle_message)
