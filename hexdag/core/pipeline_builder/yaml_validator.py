"""YAML Pipeline Validator - Validates pipeline configurations."""

from typing import Any

from hexdag.core.domain.dag import DirectedGraph
from hexdag.core.registry.exceptions import ComponentNotFoundError
from hexdag.core.registry.models import NAMESPACE_SEPARATOR, ComponentType
from hexdag.core.registry.registry import registry


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


class _SchemaValidator:
    """Validates YAML node specs against auto-generated schemas from registry.

    This ensures YAML manifests and Python DSL have exactly the same options,
    using the registry as the single source of truth.

    Note: This is an internal class. Use YamlValidator for public validation API.
    """

    def validate_node_spec(
        self,
        node_type: str,
        spec: dict[str, Any],
        namespace: str = "core",
    ) -> list[str]:
        """Validate a node's spec against its auto-generated schema.

        Args
        ----
            node_type: Type of node (e.g., "llm", "agent", "function")
            spec: Node specification from YAML manifest
            namespace: Component namespace (default: "core")

        Returns
        -------
            List of validation error messages (empty if valid)
        """

        # Get schema from registry
        factory_name = f"{node_type}_node"
        try:
            schema_dict = registry.get_schema(factory_name, namespace=namespace, format="dict")
        except (KeyError, ValueError, ComponentNotFoundError):
            # If schema doesn't exist, skip validation
            # This allows for custom nodes that don't have schemas yet
            return []

        if not isinstance(schema_dict, dict):
            return []

        # Get properties from schema
        properties = schema_dict.get("properties", {})
        required = schema_dict.get("required", [])

        # Check required fields
        errors: list[str] = [
            f"Missing required field '{field}'" for field in required if field not in spec
        ]

        # Validate provided fields
        for field_name, field_value in spec.items():
            # Skip special fields (dependencies, etc.)
            if field_name in ("dependencies",):
                continue

            # Check if field exists in schema
            if field_name not in properties:
                errors.append(
                    f"Unknown field '{field_name}'. "
                    f"Valid fields: {', '.join(sorted(properties.keys()))}"
                )
                continue

            # Get field schema
            field_schema = properties[field_name]

            # Validate field type
            validation_error = self._validate_field(field_name, field_value, field_schema)
            if validation_error:
                errors.append(validation_error)

        return errors

    def _validate_field(
        self, field_name: str, value: Any, field_schema: dict[str, Any]
    ) -> str | None:
        """Validate a single field against its schema.

        Args
        ----
            field_name: Name of the field being validated
            value: Value from the YAML manifest
            field_schema: Schema definition for this field

        Returns
        -------
            Error message if validation fails, None if valid
        """
        # Get expected type(s)
        field_type = field_schema.get("type")
        if not field_type:
            # No type specified, skip validation
            return None

        # Handle anyOf (union types)
        if "anyOf" in field_schema:
            # Try validating against each option
            errors = []
            for option in field_schema["anyOf"]:
                error = self._validate_field(field_name, value, option)
                if error is None:
                    return None  # Valid for at least one option
                errors.append(error)
            # Invalid for all options
            types = [opt.get("type", "unknown") for opt in field_schema["anyOf"]]
            return f"Field '{field_name}' must be one of types: {', '.join(set(types))}"

        # Validate basic type
        if not self._check_type(value, field_type):
            return (
                f"Field '{field_name}' must be of type '{field_type}', got '{type(value).__name__}'"
            )

        # Validate enum constraints
        if "enum" in field_schema and value not in field_schema["enum"]:
            return f"Field '{field_name}' must be one of {field_schema['enum']}, got '{value}'"

        # Validate numeric constraints
        if field_type in ("integer", "number"):
            # Check minimum
            if "minimum" in field_schema and value < field_schema["minimum"]:
                return f"Field '{field_name}' must be >= {field_schema['minimum']}, got {value}"

            # Check maximum
            if "maximum" in field_schema and value > field_schema["maximum"]:
                return f"Field '{field_name}' must be <= {field_schema['maximum']}, got {value}"

        # Validate string constraints
        if field_type == "string":
            # Check min length
            if "minLength" in field_schema and len(value) < field_schema["minLength"]:
                return f"Field '{field_name}' must have at least {field_schema['minLength']} \
                        characters"

            # Check max length
            if "maxLength" in field_schema and len(value) > field_schema["maxLength"]:
                return (
                    f"Field '{field_name}' must have at most {field_schema['maxLength']} characters"
                )

        return None

    def _check_type(self, value: Any, expected_type: str | list[str]) -> bool:
        """Check if value matches expected JSON Schema type.

        Args
        ----
            value: Value to check
            expected_type: JSON Schema type name or list of type names

        Returns
        -------
            True if type matches, False otherwise
        """
        type_mapping = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
            "null": type(None),
        }

        # Handle array of types (e.g., ["object", "null"])
        if isinstance(expected_type, list):
            # Check if value matches any of the types
            return any(self._check_type(value, t) for t in expected_type)

        expected_python_type = type_mapping.get(expected_type)
        if not expected_python_type:
            # Unknown type, skip validation
            return True

        # Type ignore for mypy - expected_python_type is guaranteed to be a type or tuple of types
        return isinstance(value, expected_python_type)  # type: ignore[arg-type]


class YamlValidator:
    """Validates YAML pipeline configurations with optimized performance."""

    def __init__(
        self,
        valid_node_types: set[str] | frozenset[str] | None = None,
    ) -> None:
        """Initialize validator with configurable node types.

        Args
        ----
            valid_node_types: Set of valid node type names. If None, uses registry.
        """
        # Store the provided node types, or None to indicate we should use registry
        self._provided_node_types = (
            frozenset(valid_node_types) if valid_node_types is not None else None
        )
        self._cached_node_types: frozenset[str] | None = None

        # Schema validator for spec validation (always enabled - no fallback)
        self.schema_validator = _SchemaValidator()

    @property
    def valid_node_types(self) -> frozenset[str]:
        """Get valid node types from registry or cache.

        Returns
        -------
        frozenset[str]
            Set of valid node type names
        """
        # If user provided explicit node types, use those
        if self._provided_node_types is not None:
            return self._provided_node_types

        # Otherwise, lazily get from registry and cache
        if self._cached_node_types is None:
            # Get all node factories from registry (e.g., "function_node", "llm_node")
            # and extract the node type (e.g., "function", "llm")
            node_components = registry.list_components(component_type=ComponentType.NODE)
            node_types = {
                comp.name.removesuffix("_node")
                for comp in node_components
                if comp.name.endswith("_node")
            }

            # If registry is empty (not bootstrapped yet), use core node types as fallback
            if not node_types:
                node_types = {"function", "llm", "agent", "loop", "conditional", "passthrough"}

            self._cached_node_types = frozenset(node_types)

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

        if not result.is_valid:
            return result

        # Extract nodes from spec
        spec = config.get("spec", {})
        nodes = spec.get("nodes", [])

        # Validate nodes and cache the IDs for reuse
        node_ids = self._validate_nodes(nodes, result)

        # Reuse cached node_ids for dependency validation
        self._validate_dependencies_with_cache(nodes, result, node_ids)

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

    def _validate_nodes(self, nodes: list[dict[str, Any]], result: ValidationReport) -> set[str]:
        """Validate individual nodes and return the set of node IDs for reuse.

        Expects declarative node format: {kind, metadata: {name}, spec: {dependencies}}

        Returns
        -------
        set[str]
            Set of valid node IDs for caching and reuse in dependency validation
        """
        node_ids = set()

        for i, node in enumerate(nodes):
            # Validate node has required fields
            if "kind" not in node:
                result.add_error(f"Node {i}: Missing 'kind' field")
                continue

            if "metadata" not in node:
                result.add_error(f"Node {i}: Missing 'metadata' field")
                continue

            # Extract node ID
            node_id = node.get("metadata", {}).get("name")
            if not node_id:
                result.add_error(f"Node {i}: Missing 'metadata.name'")
                continue

            # Extract node type and namespace from kind (e.g., "llm_node" -> "llm")
            kind = node.get("kind", "")
            if NAMESPACE_SEPARATOR in kind:
                namespace, node_kind = kind.split(NAMESPACE_SEPARATOR, 1)
            else:
                namespace = "core"
                node_kind = kind

            # Remove '_node' suffix if present
            node_type = node_kind[:-5] if node_kind.endswith("_node") else node_kind

            # Get params from spec
            params = node.get("spec", {})

            # Check node ID uniqueness
            if node_id in node_ids:
                result.add_error(f"Duplicate node ID: '{node_id}'")
            node_ids.add(node_id)

            # Validate node type
            if node_type not in self.valid_node_types:
                result.add_error(
                    f"Node '{node_id}': Invalid type '{node_type}'. "
                    f"Valid types: {', '.join(sorted(self.valid_node_types))}"
                )

            # Validate node-specific requirements and schema
            self._validate_node_params(node_id, node_type, params, namespace, result)

        return node_ids

    def _validate_node_params(
        self,
        node_id: str | None,
        node_type: str,
        params: dict[str, Any],
        namespace: str,
        result: ValidationReport,
    ) -> None:
        """Validate node-specific parameters using registry schema validation.

        Uses auto-generated schemas from the registry as the single source of truth.

        Args
        ----
            node_id: Node identifier
            node_type: Type of node (e.g., "llm", "function")
            params: Node spec parameters
            namespace: Component namespace
            result: ValidationReport to add errors to
        """
        # Schema-based validation using registry
        schema_errors = self.schema_validator.validate_node_spec(
            node_type, params, namespace=namespace
        )
        for error in schema_errors:
            result.add_error(f"Node '{node_id}': {error}")

    def _validate_dependencies_with_cache(
        self, nodes: list[dict[str, Any]], result: ValidationReport, node_ids: set[str]
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
        """
        dependency_graph = {}

        for node in nodes:
            node_id = node.get("metadata", {}).get("name")
            if not node_id:
                continue

            deps = node.get("spec", {}).get("dependencies", [])

            if not isinstance(deps, list):
                deps = [deps]

            # Check all dependencies exist using cached node_ids
            valid_deps = set()
            for dep in deps:
                if dep not in node_ids:
                    result.add_error(f"Node '{node_id}': Dependency '{dep}' does not exist")
                else:
                    valid_deps.add(dep)

            dependency_graph[node_id] = valid_deps

        # Check for cycles using DirectedGraph's public static method
        cycle_message = DirectedGraph.detect_cycle(dependency_graph)
        if cycle_message:
            result.add_error(cycle_message)
