"""YAML Pipeline Validator - Validates pipeline configurations."""

from typing import Any


class ValidationResult:
    """Container for validation results."""

    def __init__(self) -> None:
        """Initialize validation result."""
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.suggestions: list[str] = []

    @property
    def is_valid(self) -> bool:
        """Check if validation passed (no errors)."""
        return len(self.errors) == 0

    def add_error(self, message: str) -> None:
        """Add an error message."""
        self.errors.append(message)

    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)

    def add_suggestion(self, message: str) -> None:
        """Add a suggestion message."""
        self.suggestions.append(message)


class YamlValidator:
    """Validates YAML pipeline configurations."""

    VALID_NODE_TYPES = {"function", "llm", "agent", "loop"}
    REQUIRED_NODE_FIELDS = {"id"}

    def validate(self, config: Any) -> ValidationResult:
        """Validate complete YAML configuration.

        Args
        ----
            config: Parsed YAML configuration

        Returns
        -------
            ValidationResult with errors, warnings, and suggestions

        """
        result = ValidationResult()

        # Validate structure
        self._validate_structure(config, result)

        if result.is_valid:
            # Only validate nodes if structure is valid
            self._validate_nodes(config.get("nodes", []), result)
            self._validate_dependencies(config.get("nodes", []), result)
            self._validate_field_mappings(config, result)

        return result

    def _validate_structure(self, config: Any, result: ValidationResult) -> None:
        """Validate overall YAML structure."""
        if not isinstance(config, dict):
            result.add_error("Configuration must be a dictionary")
            return

        if "nodes" not in config:
            result.add_error("Configuration must contain 'nodes' field")
            return

        if not isinstance(config["nodes"], list):
            result.add_error("'nodes' field must be a list")
            return

        if len(config["nodes"]) == 0:
            result.add_warning("Pipeline has no nodes defined")

    def _validate_nodes(self, nodes: list[dict[str, Any]], result: ValidationResult) -> None:
        """Validate individual nodes."""
        node_ids = set()

        for i, node in enumerate(nodes):
            # Check required fields
            for field in self.REQUIRED_NODE_FIELDS:
                if field not in node:
                    result.add_error(f"Node {i}: Missing required field '{field}'")

            # Check node ID uniqueness
            node_id = node.get("id")
            if node_id:
                if node_id in node_ids:
                    result.add_error(f"Duplicate node ID: '{node_id}'")
                node_ids.add(node_id)

            # Validate node type
            node_type = node.get("type", "function")
            if node_type not in self.VALID_NODE_TYPES:
                result.add_error(
                    f"Node '{node_id}': Invalid type '{node_type}'. "
                    f"Valid types: {', '.join(self.VALID_NODE_TYPES)}"
                )

            # Validate node-specific requirements
            self._validate_node_params(node_id, node_type, node.get("params", {}), result)

    def _validate_node_params(
        self, node_id: str | None, node_type: str, params: dict[str, Any], result: ValidationResult
    ) -> None:
        """Validate node-specific parameters."""
        if node_type == "function":
            if "fn" not in params:
                result.add_error(f"Node '{node_id}': Function nodes require 'fn' parameter")

        elif node_type == "llm":
            if "prompt_template" not in params:
                result.add_error(f"Node '{node_id}': LLM nodes require 'prompt_template' parameter")

        elif node_type == "agent":
            if "initial_prompt_template" not in params:
                result.add_warning(
                    f"Node '{node_id}': Agent nodes should have 'initial_prompt_template'"
                )

    def _validate_dependencies(self, nodes: list[dict[str, Any]], result: ValidationResult) -> None:
        """Validate node dependencies and check for cycles."""
        node_ids = {node.get("id") for node in nodes if node.get("id")}
        dependency_graph = {}

        for node in nodes:
            node_id = node.get("id")
            if not node_id:
                continue

            deps = node.get("depends_on", [])
            if not isinstance(deps, list):
                deps = [deps]

            # Check all dependencies exist
            for dep in deps:
                if dep not in node_ids:
                    result.add_error(f"Node '{node_id}': Dependency '{dep}' does not exist")

            dependency_graph[node_id] = set(deps)

        # Check for cycles
        if self._has_cycle(dependency_graph):
            result.add_error("Dependency cycle detected in pipeline")

    def _has_cycle(self, graph: dict[str, set[str]]) -> bool:
        """Check if dependency graph has cycles using DFS."""
        visited = set()
        rec_stack = set()

        def visit(node: str) -> bool:
            if node in rec_stack:
                return True  # Cycle detected
            if node in visited:
                return False

            visited.add(node)
            rec_stack.add(node)

            for neighbor in graph.get(node, []):
                if visit(neighbor):
                    return True

            rec_stack.remove(node)
            return False

        for node in graph:
            if node not in visited:
                if visit(node):
                    return True

        return False

    def _validate_field_mappings(self, config: dict[str, Any], result: ValidationResult) -> None:
        """Validate field mapping configurations."""
        common_mappings = config.get("common_field_mappings", {})

        if common_mappings and not isinstance(common_mappings, dict):
            result.add_error("common_field_mappings must be a dictionary")
            return

        nodes = config.get("nodes", [])
        for node in nodes:
            node_id = node.get("id")
            params = node.get("params", {})
            field_mapping = params.get("field_mapping")
            input_mapping = params.get("input_mapping")

            mapping = field_mapping or input_mapping
            if not mapping:
                continue

            # Validate mapping format
            if isinstance(mapping, str):
                # String reference to common mapping
                if mapping not in common_mappings:
                    result.add_warning(
                        f"Node '{node_id}': References unknown field mapping '{mapping}'"
                    )
            elif isinstance(mapping, dict):
                # Inline mapping
                deps = set(node.get("depends_on", []))
                for target, source in mapping.items():
                    if not isinstance(source, str):
                        result.add_error(f"Node '{node_id}': Invalid mapping source for '{target}'")
                    else:
                        # Check if source node is in dependencies
                        source_node = source.split(".")[0] if "." in source else None
                        if source_node and source_node not in deps and source_node != node_id:
                            result.add_suggestion(
                                f"Node '{node_id}': Consider adding '{source_node}' to dependencies"
                            )
            else:
                result.add_error(f"Node '{node_id}': field_mapping must be a string or dictionary")
