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
from typing import TYPE_CHECKING, Any, Literal, TypeVar

if TYPE_CHECKING:
    from collections.abc import ItemsView, Iterator, KeysView, ValuesView  # noqa: F401

from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from hexdag.core.protocols import is_dict_convertible

T = TypeVar("T", bound=BaseModel)

# Performance optimization: reuse empty frozenset instead of creating new set() objects
_EMPTY_SET: frozenset[str] = frozenset()


class ValidationError(Exception):
    """Domain-specific validation error for DAG validation.

    Note: This is separate from hexdag.core.exceptions.ValidationError
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

    def __rshift__(self, other: "NodeSpec") -> "NodeSpec":
        """Create dependency using >> operator: node_a >> node_b means "b depends on a".

        This operator provides a visual, left-to-right data flow representation.
        The node on the right depends on the node on the left.

        Parameters
        ----------
        other : NodeSpec
            The downstream node that will depend on this node

        Returns
        -------
        NodeSpec
            A new NodeSpec with the dependency added to 'other'

        Examples
        --------
        >>> node_a = NodeSpec("a", lambda: "data")
        >>> node_b = NodeSpec("b", lambda x: x.upper())
        >>> node_b_with_dep = node_a >> node_b  # b depends on a
        >>> "a" in node_b_with_dep.deps
        True

        Chain multiple dependencies:
        >>> graph = DirectedGraph()
        >>> dummy = lambda: None
        >>> a = NodeSpec("a", dummy)
        >>> b = NodeSpec("b", dummy)
        >>> c = NodeSpec("c", dummy)
        >>> graph += a
        >>> b_with_dep = a >> b  # b depends on a
        >>> graph += b_with_dep
        >>> c_with_dep = b >> c  # c depends on b
        >>> graph += c_with_dep
        >>> len(graph)
        3
        >>> "a" in graph.nodes["b"].deps
        True
        >>> "b" in graph.nodes["c"].deps
        True

        Notes
        -----
        The >> operator reads naturally as "flows into" or "feeds into".
        For multiple dependencies, use .after() method instead.
        """
        return replace(other, deps=other.deps | frozenset([self.name]))

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

        self._waves_cache: list[list[str]] | None = None
        self._validation_cache: bool = False

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
                if dep in colors and (result := dfs(dep, path)):
                    return result

            path.pop()
            colors[node] = Color.BLACK
            return None

        for node in graph:
            if colors[node] == Color.WHITE and (result := dfs(node, [])):
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
        if not self._validation_cache:
            missing_deps: list[str] = []
            for node_name, node_spec in self.nodes.items():
                for dep in node_spec.deps:
                    if dep not in self.nodes:
                        msg = f"Node '{node_name}' depends on missing node '{dep}'"
                        missing_deps.append(msg)

            if missing_deps:
                raise MissingDependencyError("; ".join(missing_deps))

            if cycle_message := self._detect_cycles():
                raise CycleDetectedError(cycle_message)

            self._validation_cache = True

        if check_type_compatibility and (incompatibilities := self._validate_type_compatibility()):
            raise SchemaCompatibilityError("; ".join(incompatibilities))

    def _detect_cycles(self) -> str | None:
        """Detect cycles using depth-first search with three states.

        Returns
        -------
        str | None
            Cycle detected message or None if no cycle is detected
        """
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
        # Use list comprehension for more declarative validation
        return [
            (
                f"Node '{node_name}' expects {node_spec.in_model.__name__} "
                f"but dependency '{dep_name}' outputs {dep_node.out_model.__name__}"
            )
            for node_name, node_spec in self.nodes.items()
            if node_spec.deps and node_spec.in_model  # Has dependencies and input validation
            for dep_name in node_spec.deps
            if (dep_node := self.nodes[dep_name]).out_model  # Dependency has output model
            # Single dep type mismatch:
            if len(node_spec.deps) == 1 and dep_node.out_model != node_spec.in_model
        ]

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
        if self._waves_cache is not None:
            return self._waves_cache

        if not self.nodes:
            return []

        in_degrees = {node: len(self.nodes[node].deps) for node in self.nodes}
        waves = []

        while in_degrees:
            current_wave = []
            for node, degree in in_degrees.items():
                if degree == 0:
                    current_wave.append(node)

            if not current_wave:
                remaining_nodes = list(in_degrees.keys())
                raise CycleDetectedError(
                    f"No nodes with zero in-degree found. Remaining nodes: {remaining_nodes}"
                )

            waves.append(sorted(current_wave))

            for node in current_wave:
                del in_degrees[node]
                for dependent in self._forward_edges.get(node, _EMPTY_SET):
                    if dependent in in_degrees:
                        in_degrees[dependent] -= 1

        self._waves_cache = waves
        return waves

    def __repr__(self) -> str:
        """Developer-friendly representation for debugging.

        Shows all node names for inspection in REPL and debugging.

        Returns
        -------
        str
            Debug representation like 'DirectedGraph(nodes={'a', 'b', 'c'})'

        Examples
        --------
        >>> graph = DirectedGraph()
        >>> graph += NodeSpec("a", lambda: None)
        >>> graph += NodeSpec("b", lambda: None)
        >>> 'a' in repr(graph) and 'b' in repr(graph)
        True
        """
        if not self.nodes:
            return "DirectedGraph(nodes=set())"
        node_names = sorted(self.nodes.keys())
        return f"DirectedGraph(nodes={set(node_names)!r})"

    def __str__(self) -> str:
        """User-friendly string representation.

        Returns
        -------
        str
            Readable string showing node names

        Examples
        --------
        >>> graph = DirectedGraph()
        >>> graph += NodeSpec("a", lambda: None)
        >>> graph += NodeSpec("b", lambda: None)
        >>> str(graph)
        'DirectedGraph(2 nodes: a, b)'
        """
        if not self.nodes:
            return "DirectedGraph(empty)"

        node_names = sorted(self.nodes.keys())
        if len(node_names) <= 5:
            names_str = ", ".join(node_names)
            return f"DirectedGraph({len(node_names)} nodes: {names_str})"
        # Show first 5 nodes if more than 5
        names_str = ", ".join(node_names[:5])
        return f"DirectedGraph({len(node_names)} nodes: {names_str}, ...)"

    def __len__(self) -> int:
        """Return the number of nodes in the graph."""
        return len(self.nodes)

    def __bool__(self) -> bool:
        """Return True if the graph has nodes."""
        return bool(self.nodes)

    def __contains__(self, node_name: str) -> bool:
        """Check if a node exists in the graph.

        Examples
        --------
        >>> graph = DirectedGraph()
        >>> _ = graph.add(NodeSpec("a", lambda: None))
        >>> "a" in graph
        True
        >>> "b" in graph
        False
        """
        return node_name in self.nodes

    def __iadd__(self, other: NodeSpec | list[NodeSpec]) -> "DirectedGraph":
        """Add node(s) to graph in-place using += operator.

        This is a convenience operator that delegates to add() or add_many().
        Provides a more Pythonic way to build graphs.

        Parameters
        ----------
        other : NodeSpec | list[NodeSpec]
            Single node or list of nodes to add

        Returns
        -------
        DirectedGraph
            Self for method chaining

        Examples
        --------
        >>> graph = DirectedGraph()
        >>> node = NodeSpec("a", lambda: "result")
        >>> graph += node  # Add single node
        >>> len(graph)
        1
        >>> graph += [NodeSpec("b", lambda: None), NodeSpec("c", lambda: None)]
        >>> len(graph)
        3
        """
        if isinstance(other, list):
            return self.add_many(*other)
        return self.add(other)

    def __iter__(self) -> "Iterator[NodeSpec]":
        """Iterate over NodeSpec instances in the graph.

        Returns
        -------
        Iterator[NodeSpec]
            Iterator over node specifications in the graph

        Examples
        --------
        >>> graph = DirectedGraph()
        >>> graph += NodeSpec("a", lambda: None)
        >>> graph += NodeSpec("b", lambda: None)
        >>> for node in graph:
        ...     print(node.name)
        a
        b
        """
        return iter(self.nodes.values())

    def keys(self) -> "KeysView[str]":
        """Get an iterator over node names (dict-like interface).

        Returns
        -------
        KeysView
            View of node names in the graph

        Examples
        --------
        >>> graph = DirectedGraph()
        >>> graph += NodeSpec("a", lambda: None)
        >>> list(graph.keys())
        ['a']
        """
        return self.nodes.keys()

    def values(self) -> "ValuesView[NodeSpec]":
        """Get an iterator over NodeSpec instances (dict-like interface).

        Returns
        -------
        ValuesView
            View of NodeSpec instances in the graph

        Examples
        --------
        >>> graph = DirectedGraph()
        >>> graph += NodeSpec("a", lambda: None)
        >>> nodes = list(graph.values())
        >>> len(nodes)
        1
        """
        return self.nodes.values()

    def items(self) -> "ItemsView[str, NodeSpec]":
        """Get an iterator over (name, NodeSpec) pairs (dict-like interface).

        Returns
        -------
        ItemsView
            View of (name, NodeSpec) tuples

        Examples
        --------
        >>> graph = DirectedGraph()
        >>> graph += NodeSpec("a", lambda: None)
        >>> for name, spec in graph.items():
        ...     print(f"{name}: {spec.fn}")
        a: <function...>
        """
        return self.nodes.items()

    def __ior__(self, other: "DirectedGraph") -> "DirectedGraph":
        """Merge another graph into this one using |= operator.

        This operator provides in-place merging of graphs, useful for composing
        subgraphs (especially from macro expansions) into a main graph.

        Parameters
        ----------
        other : DirectedGraph
            The graph to merge into this one

        Returns
        -------
        DirectedGraph
            Self, for method chaining

        Examples
        --------
        Merge graphs:

        .. code-block:: python

            main_graph = DirectedGraph()
            main_graph += NodeSpec("a", lambda: None)
            subgraph = DirectedGraph()
            subgraph += NodeSpec("b", lambda: None)
            subgraph += NodeSpec("c", lambda: None)
            main_graph |= subgraph  # Merge subgraph into main
            assert len(main_graph) == 3
        """
        for node in other:  # Using iterator instead of .nodes.values()
            self.add(node)
        return self

    def __lshift__(self, other: NodeSpec | tuple) -> "DirectedGraph":
        """Fluent chaining with << operator: graph << node or graph << (a >> b).

        This operator provides a fluent interface for building graphs with
        a visual left-to-right flow.

        Parameters
        ----------
        other : NodeSpec | tuple
            Single node or tuple of nodes to add

        Returns
        -------
        DirectedGraph
            Self, for method chaining

        Examples
        --------
        Fluent chaining:

        .. code-block:: python

            graph = DirectedGraph()
            a = NodeSpec("a", lambda: None)
            b = NodeSpec("b", lambda: None)
            graph << a << b  # Fluent chaining
            assert len(graph) == 2

            # With pipeline operator:
            graph2 = DirectedGraph()
            c = NodeSpec("c", lambda: None)
            graph2 << (a >> b >> c)  # Add pipeline
        """
        if isinstance(other, tuple):
            for node in other:
                self.add(node)
        else:
            self.add(other)
        return self
