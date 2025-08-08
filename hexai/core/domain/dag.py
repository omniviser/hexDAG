"""DAG primitives: NodeSpec and DirectedGraph.

This module provides the core building blocks for defining and executing
directed acyclic graphs of agents in the Hex-DAG framework.
"""

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Callable

from hexai.core.validation import can_convert_schema


class Color(Enum):
    """Colors for DFS cycle detection algorithm."""

    WHITE = 0  # Unvisited
    GRAY = 1  # Currently being processed (in recursion stack)
    BLACK = 2  # Completely processed


@dataclass(frozen=True)
class NodeSpec:
    """Immutable representation of a node in a DAG.

    A NodeSpec defines:
    - A unique name within the DAG
    - The function to execute (agent)
    - Input/output types for validation
    - Dependencies (explicit and computed)
    - Arbitrary metadata parameters

    Supports fluent chaining via .after() method.
    """

    name: str
    fn: Callable[..., Any]
    in_type: type | None = None
    out_type: type | None = None
    deps: set[str] = field(default_factory=set)
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Ensure deps and params are immutable."""
        object.__setattr__(self, "deps", frozenset(self.deps))
        object.__setattr__(self, "params", MappingProxyType(self.params))

    def after(self, *node_names: str) -> "NodeSpec":
        """Create a new NodeSpec that depends on the specified nodes.

        Args
        ----
            *node_names: Names of nodes this node should run after

        Returns
        -------
            New NodeSpec with updated dependencies

        Example
        -------
            node_b = NodeSpec("b", my_fn).after("a")
            node_c = NodeSpec("c", my_fn).after("a", "b")
        """
        new_deps = self.deps | set(node_names)

        return NodeSpec(
            name=self.name,
            fn=self.fn,
            in_type=self.in_type,
            out_type=self.out_type,
            deps=new_deps,
            params=dict(self.params),
        )

    def __repr__(self) -> str:
        """Readable representation for debugging."""
        deps_str = f", deps={sorted(self.deps)}" if self.deps else ""
        types_str = ""
        if self.in_type or self.out_type:
            in_name = (
                getattr(self.in_type, "__name__", str(self.in_type)) if self.in_type else "Any"
            )
            out_name = (
                getattr(self.out_type, "__name__", str(self.out_type)) if self.out_type else "Any"
            )
            types_str = f", {in_name} -> {out_name}"

        params_str = f", params={dict(self.params)}" if self.params else ""

        return f"NodeSpec('{self.name}'{types_str}{deps_str}{params_str})"


class DirectedGraphError(Exception):
    """Base exception for DirectedGraph errors."""

    pass


class CycleDetectedError(DirectedGraphError):
    """Raised when a cycle is detected in the DAG."""

    pass


class MissingDependencyError(DirectedGraphError):
    """Raised when a node depends on a non-existent node."""

    pass


class DuplicateNodeError(DirectedGraphError):
    """Raised when attempting to add a node with an existing name."""

    pass


class SchemaCompatibilityError(DirectedGraphError):
    """Raised when connected nodes have incompatible schemas."""

    pass


class DirectedGraph:
    """A directed acyclic graph (DAG) for orchestrating NodeSpec instances.

    Provides:
    - Node management with cycle detection
    - Dependency validation
    - Topological sorting into execution waves
    """

    def __init__(self, nodes: list[NodeSpec] | None = None) -> None:
        """Initialize DirectedGraph, optionally with a list of nodes.

        Args
        ----
            nodes: Optional list of NodeSpec instances to add to the graph
        """
        self.nodes: dict[str, NodeSpec] = {}
        self._forward_edges: dict[str, set[str]] = {}  # Private: node -> set of dependents
        self._reverse_edges: dict[str, set[str]] = {}  # Private: node -> set of dependencies

        # Add nodes if provided
        if nodes:
            self.add_many(*nodes)

    def add(self, node_spec: NodeSpec) -> "DirectedGraph":
        """Add a NodeSpec to the graph.

        Args
        ----
            node_spec: NodeSpec instance to add to the graph

        Returns
        -------
            Self for method chaining

        Raises
        ------
            DuplicateNodeError: If a node with the same name already exists
        """
        if node_spec.name in self.nodes:
            raise DuplicateNodeError(f"Node '{node_spec.name}' already exists in the graph")

        self.nodes[node_spec.name] = node_spec
        self._forward_edges[node_spec.name] = set()
        self._reverse_edges[node_spec.name] = set(node_spec.deps)

        # Update forward edges from dependencies to this node
        for dep in node_spec.deps:
            if dep in self._forward_edges:
                self._forward_edges[dep].add(node_spec.name)
            else:
                # Dependency doesn't exist yet, create placeholder
                self._forward_edges[dep] = {node_spec.name}

        return self

    def add_many(self, *node_specs: NodeSpec) -> "DirectedGraph":
        """Add multiple nodes to the graph.

        Args
        ----
            *node_specs: Variable number of NodeSpec instances to add

        Returns
        -------
            Self for method chaining

        Raises
        ------
            DuplicateNodeError: If any node with the same name already exists

        Example
        -------
            graph.add_many(
                NodeSpec("fetch", fetch_fn),
                NodeSpec("process", process_fn).after("fetch"),
                NodeSpec("analyze", analyze_fn).after("process")
            )
        """
        # First, validate all nodes can be added (check for duplicates)
        for node_spec in node_specs:
            if node_spec.name in self.nodes:
                raise DuplicateNodeError(f"Node '{node_spec.name}' already exists in the graph")

        # If validation passes, add all nodes
        for node_spec in node_specs:
            self.add(node_spec)
        return self

    def get_dependencies(self, node_name: str) -> set[str]:
        """Get the dependencies (parents) of a node.

        Args
        ----
            node_name: Name of the node

        Returns
        -------
            Set of node names that this node depends on

        Raises
        ------
            KeyError: If the node doesn't exist
        """
        if node_name not in self.nodes:
            raise KeyError(f"Node '{node_name}' not found in graph")
        return set(self.nodes[node_name].deps)

    def get_dependents(self, node_name: str) -> set[str]:
        """Get the dependents (children) of a node.

        Args
        ----
            node_name: Name of the node

        Returns
        -------
            Set of node names that depend on this node

        Raises
        ------
            KeyError: If the node doesn't exist
        """
        if node_name not in self.nodes:
            raise KeyError(f"Node '{node_name}' not found in graph")
        return self._forward_edges.get(node_name, set()).copy()

    def validate(self) -> None:
        """Validate the DAG structure.

        Checks for:
        - Missing dependencies
        - Cycles in the graph
        - Schema compatibility between connected nodes

        Raises
        ------
            MissingDependencyError: If any node depends on a non-existent node
            CycleDetectedError: If a cycle is detected in the graph
            SchemaCompatibilityError: If connected nodes have incompatible schemas
        """
        # Check for missing dependencies
        missing_deps = []
        for node_name, node_spec in self.nodes.items():
            for dep in node_spec.deps:
                if dep not in self.nodes:
                    missing_deps.append(f"Node '{node_name}' depends on missing node '{dep}'")

        if missing_deps:
            raise MissingDependencyError("; ".join(missing_deps))

        # Check for cycles using DFS
        self._detect_cycles()

        # Check schema compatibility between connected nodes
        self._validate_schema_compatibility()

    def _detect_cycles(self) -> None:
        """Detect cycles using depth-first search with three states.

        States:
        - WHITE (0): Unvisited
        - GRAY (1): Currently being processed (in recursion stack)
        - BLACK (2): Completely processed

        Raises
        ------
            CycleDetectedError: If a cycle is detected
        """
        colors = {node: Color.WHITE for node in self.nodes}

        def dfs(node: str, path: list[str]) -> None:
            if colors[node] == Color.GRAY:
                # Found a back edge - cycle detected
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                raise CycleDetectedError(f"Cycle detected: {' -> '.join(cycle)}")

            if colors[node] == Color.BLACK:
                return

            colors[node] = Color.GRAY
            path.append(node)

            # Visit all dependencies
            for dep in self.nodes[node].deps:
                if dep in self.nodes:
                    dfs(dep, path)

            path.pop()
            colors[node] = Color.BLACK

        for node in self.nodes:
            if colors[node] == Color.WHITE:
                dfs(node, [])

    def _validate_schema_compatibility(self) -> None:
        """Validate that connected nodes have compatible input/output schemas."""
        for node_name, node_spec in self.nodes.items():
            # Skip nodes with input mapping - validation framework handles conversion
            if node_spec.params.get("input_mapping"):
                continue

            # Skip nodes with multiple dependencies - orchestrator handles aggregation
            if len(node_spec.deps) > 1 and node_spec.in_type == dict:
                continue

            # Check each dependency connection
            for dep_name in node_spec.deps:
                dep_node = self.nodes[dep_name]

                # Skip if either node doesn't specify types
                if node_spec.in_type is None or dep_node.out_type is None:
                    continue

                # Check compatibility using validation framework
                if not can_convert_schema(dep_node.out_type, node_spec.in_type):
                    raise SchemaCompatibilityError(
                        f"Schema mismatch: Node '{dep_name}' outputs {dep_node.out_type} "
                        f"but node '{node_name}' expects {node_spec.in_type}"
                    )

    def waves(self) -> list[list[str]]:
        """Compute execution waves using topological sorting.

        Returns
        -------
            List of waves, where each wave is a list of node names that can
            be executed in parallel.

        Example
        -------
            # For DAG: A -> B -> D, A -> C -> D
            # Returns: [["A"], ["B", "C"], ["D"]]
        """
        if not self.nodes:
            return []

        # Calculate in-degrees (number of dependencies)
        in_degrees = {node: len(self.nodes[node].deps) for node in self.nodes}
        waves = []

        while in_degrees:
            # Find nodes with no remaining dependencies
            current_wave = [node for node, degree in in_degrees.items() if degree == 0]

            if not current_wave:
                # This shouldn't happen if the graph is acyclic and validate() passed
                remaining_nodes = list(in_degrees.keys())
                raise CycleDetectedError(
                    f"No nodes with zero in-degree found. Remaining nodes: {remaining_nodes}"
                )

            waves.append(sorted(current_wave))  # Sort for deterministic output

            # Remove current wave nodes and update in-degrees
            for node in current_wave:
                del in_degrees[node]

                # Reduce in-degree for all dependents
                for dependent in self._forward_edges.get(node, set()):
                    if dependent in in_degrees:
                        in_degrees[dependent] -= 1

        return waves

    def __repr__(self) -> str:
        """Readable representation for debugging."""
        node_names = sorted(self.nodes.keys())
        return f"DirectedGraph(nodes={node_names})"
