"""DAG primitives: NodeSpec and DirectedGraph.

This module provides the core building blocks for defining and executing
directed acyclic graphs of agents in the Hex-DAG framework.
"""

import sys
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from enum import Enum, auto
from types import MappingProxyType
from typing import Any, Literal, TypeVar

from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from hexai.core.protocols import is_dict_convertible

T = TypeVar("T", bound=BaseModel)

# Performance optimization: reuse empty frozenset instead of creating new set() objects
_EMPTY_SET: frozenset[str] = frozenset()


class ValidationError(Exception):
    """Domain-specific validation error for DAG validation.

    Note: This is separate from hexai.core.exceptions.ValidationError
    which is used for general field validation. This exception is specifically
    for DAG node input/output validation failures.
    """

    __slots__ = ()


class Color(Enum):
    """Colors for DFS cycle detection algorithm."""

    WHITE = auto()  # Unvisited
    GRAY = auto()  # Currently being processed (in recursion stack)
    BLACK = auto()  # Completely processed


@dataclass(frozen=True, slots=True)
class NodeSpec:
    """Immutable representation of a node in a DAG.

    A NodeSpec defines:
    - A unique name within the DAG
    - The function to execute (agent)
    - Input/output types for validation (Pydantic models or legacy types)
    - Dependencies (explicit and computed)
    - Arbitrary metadata parameters

    Supports fluent chaining via .after() method.
    """

    name: str
    fn: Callable[..., Any]
    in_model: type[BaseModel] | None = None  # Pydantic model for input validation
    out_model: type[BaseModel] | None = None  # Pydantic model for output validation
    deps: frozenset[str] = field(default_factory=frozenset)
    params: dict[str, Any] = field(default_factory=dict)
    timeout: float | None = None  # Optional timeout in seconds for this node

    def __post_init__(self) -> None:
        """Ensure deps and params are immutable, and intern strings for performance."""
        # Intern node name for memory efficiency and faster comparisons
        object.__setattr__(self, "name", sys.intern(self.name))
        # Intern dependency names as well
        object.__setattr__(self, "deps", frozenset(sys.intern(d) for d in self.deps))
        object.__setattr__(self, "params", MappingProxyType(self.params))

    def _validate_with_model(
        self, data: Any, model: type[T] | None, validation_type: Literal["input", "output"]
    ) -> T | Any:
        """Validate data using the provided Pydantic model.

        Parameters
        ----------
        data : Any
            Data to validate
        model : Type[T] | None
            Pydantic model to validate against
        validation_type : Literal["input", "output"]
            Type of validation for error messages

        Returns
        -------
        T | Any
            Validated model instance or original data if no model

        Raises
        ------
        ValidationError
            If validation fails
        """
        if model is None:
            # No validation needed
            return data

        # If already the correct type, return as-is
        if isinstance(data, model):
            return data

        try:
            # Check if data is dict-convertible using protocol
            if is_dict_convertible(data):
                # Pydantic model or dict-like - convert to dict first
                return model.model_validate(
                    data.model_dump() if not isinstance(data, dict) else data
                )
            # Data is other type
            return model.model_validate(data)
        except PydanticValidationError as e:
            # Format Pydantic validation errors nicely
            error_msg = (
                f"{validation_type.capitalize()} validation failed for node '{self.name}': {e}"
            )
            raise ValidationError(error_msg) from e
        except Exception as e:
            # Other unexpected errors
            error_msg = (
                f"{validation_type.capitalize()} validation error for node '{self.name}': {e}"
            )
            raise ValidationError(error_msg) from e

    def validate_input(self, data: Any) -> BaseModel | Any:
        """Validate and convert input data using Pydantic model if available.

        Parameters
        ----------
        data : Any
            Input data to validate

        Returns
        -------
        BaseModel | Any
            Validated/converted data
        """
        return self._validate_with_model(data, self.in_model, "input")

    def validate_output(self, data: Any) -> BaseModel | Any:
        """Validate and convert output data using Pydantic model if available.

        Parameters
        ----------
        data : Any
            Output data to validate

        Returns
        -------
        BaseModel | Any
            Validated/converted data
        """
        return self._validate_with_model(data, self.out_model, "output")

    def after(self, *node_names: str) -> "NodeSpec":
        """Create a new NodeSpec that depends on the specified nodes.

        Args
        ----
            *node_names: Names of nodes this node should run after

        Returns
        -------
            New NodeSpec with updated dependencies

        Examples
        --------
            node_b = NodeSpec("b", my_fn).after("a")
            node_c = NodeSpec("c", my_fn).after("a", "b")
        """
        new_deps = self.deps | frozenset(node_names)
        return replace(self, deps=new_deps)

    def __repr__(self) -> str:
        """Readable representation for debugging.

        Returns
        -------
            String representation of the NodeSpec.
        """
        deps_str = f", deps={sorted(self.deps)}" if self.deps else ""
        types_str = ""

        # Show Pydantic models if available
        if self.in_model or self.out_model:
            in_name = self.in_model.__name__ if self.in_model else "Any"
            out_name = self.out_model.__name__ if self.out_model else "Any"
            types_str = f", {in_name} -> {out_name}"

        params_str = f", params={dict(self.params)}" if self.params else ""

        return f"NodeSpec('{self.name}'{types_str}{deps_str}{params_str})"


class DirectedGraphError(Exception):
    """Base exception for DirectedGraph errors."""

    __slots__ = ()


class CycleDetectedError(DirectedGraphError):
    """Raised when a cycle is detected in the DAG."""

    __slots__ = ()


class MissingDependencyError(DirectedGraphError):
    """Raised when a node depends on a non-existent node."""

    __slots__ = ()


class DuplicateNodeError(DirectedGraphError):
    """Raised when attempting to add a node with an existing name."""

    __slots__ = ()


class SchemaCompatibilityError(DirectedGraphError):
    """Raised when connected nodes have incompatible schemas."""

    __slots__ = ()


class DirectedGraph:
    """A directed acyclic graph (DAG) for orchestrating NodeSpec instances.

    Provides:
    - Node management with cycle detection
    - Dependency validation
    - Topological sorting into execution waves
    - Optional Pydantic model compatibility checking
    """

    def __init__(
        self,
        nodes: list[NodeSpec] | None = None,
    ) -> None:
        """Initialize DirectedGraph, optionally with a list of nodes.

        Args
        ----
            nodes: Optional list of NodeSpec instances to add to the graph
        """
        self.nodes: dict[str, NodeSpec] = {}
        self._forward_edges: defaultdict[str, set[str]] = defaultdict(
            set
        )  # node -> set of dependents
        self._reverse_edges: defaultdict[str, set[str]] = defaultdict(
            set
        )  # node -> set of dependencies

        # Cache for expensive operations
        self._waves_cache: list[list[str]] | None = None
        self._validation_cache: bool = False

        # Add nodes if provided
        if nodes:
            self.add_many(*nodes)

    @staticmethod
    def detect_cycle(graph: Mapping[str, set[str] | frozenset[str]]) -> str | None:
        """Detect cycles in a dependency graph using DFS with three-state coloring.

        This is a public static method that can be used to detect cycles in simple
        dependency graphs before constructing a full DirectedGraph.

        Parameters
        ----------
        graph : Mapping[str, set[str] | frozenset[str]]
            Dependency graph where keys are node names and values are sets of dependencies

        Returns
        -------
        str | None
            Cycle description if found, None otherwise

        Examples
        --------
        >>> graph = {"a": {"b"}, "b": {"c"}, "c": {"a"}}  # a->b->c->a
        >>> DirectedGraph.detect_cycle(graph)
        'Cycle detected: a -> b -> c -> a'

        >>> graph = {"a": {"b"}, "b": {"c"}, "c": set()}  # No cycle
        >>> result = DirectedGraph.detect_cycle(graph)
        >>> result is None
        True
        """
        colors = dict.fromkeys(graph, Color.WHITE)

        def dfs(node: str, path: list[str]) -> str | None:
            if colors[node] == Color.GRAY:
                # Found a back edge - cycle detected
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                return f"Cycle detected: {' -> '.join(cycle)}"

            if colors[node] == Color.BLACK:
                return None

            colors[node] = Color.GRAY
            path.append(node)

            # Visit all dependencies
            for dep in graph.get(node, set()):
                if dep in colors:  # Only visit nodes that exist in graph
                    result = dfs(dep, path)
                    if result:  # If cycle found, propagate it up
                        return result

            path.pop()
            colors[node] = Color.BLACK
            return None

        for node in graph:
            if colors[node] == Color.WHITE:
                result = dfs(node, [])
                if result:  # If cycle found, return it
                    return result

        return None  # No cycles found

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
        DuplicateNodeError
            If a node with the same name already exists.
        """
        if node_spec.name in self.nodes:
            raise DuplicateNodeError(f"Node '{node_spec.name}' already exists in the graph")

        self.nodes[node_spec.name] = node_spec
        self._forward_edges[node_spec.name]  # Ensure key exists (defaultdict creates empty set)
        self._reverse_edges[node_spec.name] = set(node_spec.deps)

        # Update forward edges from dependencies to this node
        for dep in node_spec.deps:
            self._forward_edges[dep].add(node_spec.name)

        # Invalidate caches when graph structure changes
        self._invalidate_caches()

        return self

    def _invalidate_caches(self) -> None:
        """Invalidate cached results when graph structure changes."""
        self._waves_cache = None
        self._validation_cache = False

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
        DuplicateNodeError
            If any node with the same name already exists.

        Examples
        --------
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

    def get_dependencies(self, node_name: str) -> frozenset[str]:
        """Get the dependencies (parents) of a node.

        Args
        ----
            node_name: Name of the node

        Returns
        -------
            Immutable set of node names that this node depends on

        Raises
        ------
        KeyError
            If the node doesn't exist.
        """
        if node_name not in self.nodes:
            raise KeyError(f"Node '{node_name}' not found in graph")
        return self.nodes[node_name].deps

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
        KeyError
            If the node doesn't exist.
        """
        if node_name not in self.nodes:
            raise KeyError(f"Node '{node_name}' not found in graph")
        return set(self._forward_edges.get(node_name, _EMPTY_SET))

    def validate(self, check_type_compatibility: bool = True) -> None:
        """Validate the DAG structure and optionally type compatibility with caching.

        Caching behavior:
        - Structural validation (dependencies, cycles) is cached after first success
        - Type compatibility validation is NOT cached (expensive but changes with node specs)
        - Cache invalidated when graph structure changes (add/remove nodes)

        Checks for:
        - Missing dependencies
        - Cycles in the graph
        - Type compatibility between connected nodes (optional)

        Parameters
        ----------
        check_type_compatibility : bool
            If True, validates that connected nodes have compatible types

        Raises
        ------
        MissingDependencyError
            If any node depends on a non-existent node.
        CycleDetectedError
            If a cycle is detected in the graph.
        SchemaCompatibilityError
            If connected nodes have incompatible types.
        """
        # Skip structural validation if already validated and graph hasn't changed
        if not self._validation_cache:
            # Check for missing dependencies
            missing_deps: list[str] = []
            for node_name, node_spec in self.nodes.items():
                missing_deps.extend(
                    f"Node '{node_name}' depends on missing node '{dep}'"
                    for dep in node_spec.deps
                    if dep not in self.nodes
                )

            if missing_deps:
                raise MissingDependencyError("; ".join(missing_deps))

            # Check for cycles using DFS
            cycle_message = self._detect_cycles()
            if cycle_message:
                raise CycleDetectedError(cycle_message)

            # Cache structural validation result
            self._validation_cache = True

        # Check type compatibility between connected nodes (not cached - less frequent)
        if check_type_compatibility:
            incompatibilities = self._validate_type_compatibility()
            if incompatibilities:
                raise SchemaCompatibilityError("; ".join(incompatibilities))

    def _detect_cycles(self) -> str | None:
        """Detect cycles using depth-first search with three states.

        Returns
        -------
        str | None
            Cycle detected message or None if no cycle is detected
        """
        # Build a simple dependency graph from NodeSpecs and use static method
        graph = {name: node_spec.deps for name, node_spec in self.nodes.items()}
        return DirectedGraph.detect_cycle(graph)

    def _validate_type_compatibility(self) -> list[str]:
        """Validate type compatibility between connected nodes.

        For nodes with multiple dependencies, we check if the dependent node
        can handle the aggregated output types from its dependencies.

        Returns
        -------
            List of incompatibility messages or empty list if no incompatibilities are found
        """
        incompatibilities = []

        for node_name, node_spec in self.nodes.items():
            if not node_spec.deps:
                continue

            # Check if this node has input validation defined
            if not node_spec.in_model:
                # No input model means it accepts Any, so skip validation
                continue

            # For each dependency, check if output is compatible
            for dep_name in node_spec.deps:
                dep_node = self.nodes[dep_name]

                # If dependency has no output model, we can't validate
                if not dep_node.out_model:
                    continue

                # Check basic type compatibility
                # For single dependencies only - multiple deps are handled by aggregation
                if len(node_spec.deps) == 1 and dep_node.out_model != node_spec.in_model:
                    # Types don't match exactly - report incompatibility
                    incompatibilities.append(
                        f"Node '{node_name}' expects {node_spec.in_model.__name__} "
                        f"but dependency '{dep_name}' outputs {dep_node.out_model.__name__}"
                    )

        return incompatibilities

    def waves(self) -> list[list[str]]:
        """Compute execution waves using topological sorting with caching.

        Caches the result since waves() is called multiple times during orchestration:
        1. During DAG validation
        2. At pipeline start for event emission
        3. For each wave execution

        Returns
        -------
            List of waves, where each wave is a list of node names that can
            be executed in parallel.

        Raises
        ------
        CycleDetectedError
            If a cycle is detected (no nodes with zero in-degree found).

        Examples
        --------
            # For DAG: A -> B -> D, A -> C -> D
            # Returns: [["A"], ["B", "C"], ["D"]]
        """
        # Return cached result if available
        if self._waves_cache is not None:
            return self._waves_cache

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
                for dependent in self._forward_edges.get(node, _EMPTY_SET):
                    if dependent in in_degrees:
                        in_degrees[dependent] -= 1

        # Cache the result
        self._waves_cache = waves
        return waves

    def __repr__(self) -> str:
        """Readable representation for debugging.

        Returns
        -------
            String representation of the DirectedGraph.
        """
        node_names = sorted(self.nodes.keys())
        return f"DirectedGraph(nodes={node_names})"

    def __len__(self) -> int:
        """Return the number of nodes in the graph."""
        return len(self.nodes)

    def __bool__(self) -> bool:
        """Return True if the graph has nodes."""
        return bool(self.nodes)

    def __contains__(self, node_name: str) -> bool:
        """Check if a node exists in the graph."""
        return node_name in self.nodes
