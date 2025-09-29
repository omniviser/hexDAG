"""YAML Pipeline Validator - Validates pipeline configurations."""

from typing import Any


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


class YamlValidator:
    """Validates YAML pipeline configurations with optimized performance."""

    # Default valid node types - can be overridden in __init__
    DEFAULT_NODE_TYPES = frozenset({"function", "llm", "agent", "loop"})
    REQUIRED_NODE_FIELDS = frozenset({"id"})

    def __init__(self, valid_node_types: set[str] | frozenset[str] | None = None) -> None:
        """Initialize validator with configurable node types.

        Args
        ----
            valid_node_types: Set of valid node type names. If None, uses defaults.
        """
        # Convert to frozenset for O(1) membership testing and immutability
        if valid_node_types is None:
            self.valid_node_types = self.DEFAULT_NODE_TYPES
        else:
            self.valid_node_types = frozenset(valid_node_types)

    def validate(self, config: Any) -> ValidationReport:
        """Validate complete YAML configuration with optimized caching.

        Args
        ----
            config: Parsed YAML configuration

        Returns
        -------
        ValidationReport
            ValidationReport with errors, warnings, and suggestions
        """
        result = ValidationReport()

        # Validate structure
        self._validate_structure(config, result)

        if result.is_valid:
            # Extract nodes once to avoid repeated lookups
            nodes = config.get("nodes", [])

            # Validate nodes and cache the IDs for reuse
            node_ids = self._validate_nodes(nodes, result)

            # Reuse cached node_ids for dependency validation
            self._validate_dependencies_with_cache(nodes, result, node_ids)

            # Validate field mappings
            self._validate_field_mappings(config, result)

        return result

    def _validate_structure(self, config: Any, result: ValidationReport) -> None:
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

    def _validate_nodes(self, nodes: list[dict[str, Any]], result: ValidationReport) -> set[str]:
        """Validate individual nodes and return the set of node IDs for reuse.

        Returns
        -------
        set[str]
            Set of valid node IDs for caching and reuse in dependency validation
        """
        node_ids = set()

        for i, node in enumerate(nodes):
            # Check required fields efficiently using set difference
            missing_fields = self.REQUIRED_NODE_FIELDS - node.keys()
            for field in missing_fields:
                result.add_error(f"Node {i}: Missing required field '{field}'")

            # Check node ID uniqueness
            node_id = node.get("id")
            if node_id:
                if node_id in node_ids:
                    result.add_error(f"Duplicate node ID: '{node_id}'")
                node_ids.add(node_id)

            # Validate node type
            node_type = node.get("type", "function")
            if node_type not in self.valid_node_types:
                result.add_error(
                    f"Node '{node_id}': Invalid type '{node_type}'. "
                    f"Valid types: {', '.join(sorted(self.valid_node_types))}"
                )

            # Validate node-specific requirements
            self._validate_node_params(node_id, node_type, node.get("params", {}), result)

        return node_ids

    def _validate_node_params(
        self, node_id: str | None, node_type: str, params: dict[str, Any], result: ValidationReport
    ) -> None:
        """Validate node-specific parameters."""
        if node_type == "function":
            if "fn" not in params:
                result.add_error(f"Node '{node_id}': Function nodes require 'fn' parameter")

        elif node_type == "llm":
            if "prompt_template" not in params:
                result.add_error(f"Node '{node_id}': LLM nodes require 'prompt_template' parameter")

        elif node_type == "agent" and "initial_prompt_template" not in params:
            result.add_warning(
                f"Node '{node_id}': Agent nodes should have 'initial_prompt_template'"
            )

    def _validate_dependencies_with_cache(
        self, nodes: list[dict[str, Any]], result: ValidationReport, node_ids: set[str]
    ) -> None:
        """Validate node dependencies using cached node IDs and check for cycles.

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
            node_id = node.get("id")
            if not node_id:
                continue

            deps = node.get("depends_on", [])
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

        # Check for cycles using optimized algorithm
        if self._has_cycle(dependency_graph):
            result.add_error("Dependency cycle detected in pipeline")

    def _has_cycle(self, graph: dict[str, set[str]]) -> bool:
        """Check if dependency graph has cycles using optimized DFS with colors.

        Uses integer colors for better performance than set operations:
        - 0 (WHITE): Not visited
        - 1 (GRAY): Currently being processed (in recursion stack)
        - 2 (BLACK): Completely processed

        Returns
        -------
        bool
            True if a cycle is detected, False otherwise
        """
        # Use integers for colors (more efficient than set operations)
        white, gray, black = 0, 1, 2
        colors = dict.fromkeys(graph, white)

        def visit(node: str) -> bool:
            if colors[node] == gray:
                return True  # Back edge found - cycle detected
            if colors[node] == black:
                return False  # Already processed

            colors[node] = gray
            for neighbor in graph.get(node, []):
                if visit(neighbor):
                    return True

            colors[node] = black
            return False

        return any(colors[node] == white and visit(node) for node in graph)

    def _validate_field_mappings(self, config: dict[str, Any], result: ValidationReport) -> None:
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
