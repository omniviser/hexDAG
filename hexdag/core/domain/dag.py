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
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from collections.abc import ItemsView, Iterator, KeysView, ValuesView  # noqa: F401

from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

T = TypeVar("T", bound=BaseModel)

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


class ValidationCacheState(Enum):
    """State of the validation cache."""

    INVALID = auto()  # Cache invalidated or never validated
    VALID = auto()  # Structural validation passed


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
        self, data: Any, model: type[T] | None, validation_type: str
    ) -> T | Any:
        """Validate data using the provided Pydantic model.

        Parameters
        ----------
        data : Any
            Data to validate
        model : Type[T] | None
            Pydantic model to validate against
        validation_type : str
            Type of validation for error messages ('input' or 'output')

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
            return data

        # Fast path: if already the correct type, return as-is
        if isinstance(data, model):
            return data

        try:
            # If data is a different Pydantic model, convert to dict first
            # This allows schema transformation between incompatible models
            if isinstance(data, BaseModel):
                return model.model_validate(data.model_dump())

            # For dict, primitives, and other types, validate directly
            return model.model_validate(data)
        except PydanticValidationError as e:
            error_msg = (
                f"{validation_type.capitalize()} validation failed for node '{self.name}': {e}"
            )
            raise ValidationError(error_msg) from e
        except Exception as e:
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
        strict_add: bool = False,
    ) -> None:
        """Initialize DirectedGraph, optionally with a list of nodes.

        Args
        ----
            nodes: Optional list of NodeSpec instances to add to the graph
            strict_add: If True, validate dependencies and cycles immediately on add().
                If False (default), allow adding nodes with missing dependencies
                and defer validation to validate() call. Set to True for dynamic
                graphs to catch errors early.
        """
        self.nodes: dict[str, NodeSpec] = {}
        self._forward_edges: defaultdict[str, set[str]] = defaultdict(
            set
        )  # node -> set of dependents
        self._reverse_edges: defaultdict[str, set[str]] = defaultdict(
            set
        )  # node -> set of dependencies

        self._waves_cache: list[list[str]] | None = None
        self._validation_cache: ValidationCacheState = ValidationCacheState.INVALID
        self._strict_add = strict_add

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
        MissingDependencyError
            If strict_add=True and the node depends on non-existent nodes.
        CycleDetectedError
            If strict_add=True and adding the node would create a cycle.

        Notes
        -----
        When strict_add=False (default), nodes can be added with missing dependencies.
        Validation happens later when validate() is called.

        When strict_add=True (optimized for dynamic graphs), validation happens
        immediately with O(deps) complexity instead of O(n²) for full graph validation.
        """
        if node_spec.name in self.nodes:
            raise DuplicateNodeError(f"Node '{node_spec.name}' already exists in the graph")

        # Incremental validation only if strict_add=True (for dynamic graphs)
        if self._strict_add:
            missing_deps = [dep for dep in node_spec.deps if dep not in self.nodes]
            if missing_deps:
                raise MissingDependencyError(
                    f"Node '{node_spec.name}' depends on missing node(s): {missing_deps}"
                )

            # Incremental cycle detection: only check if new node creates cycle
            if self._would_create_cycle(node_spec):
                raise CycleDetectedError(
                    f"Adding node '{node_spec.name}' would create a cycle in the graph"
                )

        self.nodes[node_spec.name] = node_spec
        self._forward_edges[node_spec.name]  # Ensure key exists (defaultdict creates empty set)
        self._reverse_edges[node_spec.name] = set(node_spec.deps)

        for dep in node_spec.deps:
            self._forward_edges[dep].add(node_spec.name)

        # Invalidate caches when graph structure changes
        self._invalidate_caches()

        return self

    def _would_create_cycle(self, new_node: NodeSpec) -> bool:
        """Fast incremental cycle detection for a new node.

        Only checks if adding this specific node would create a cycle,
        without validating the entire graph. This is much faster than
        full graph validation for dynamic graphs with 100+ nodes.

        Strategy: Check if new_node is reachable from any of its dependencies.
        If we can reach new_node by following forward edges from any of its deps,
        adding new_node would create a cycle.

        Parameters
        ----------
        new_node : NodeSpec
            The node being added

        Returns
        -------
        bool
            True if adding the node would create a cycle

        Examples
        --------
        >>> graph = DirectedGraph()
        >>> a = NodeSpec("a", lambda: None)
        >>> b = NodeSpec("b", lambda: None).after("a")
        >>> c = NodeSpec("c", lambda: None).after("a")
        >>> graph += [a, b, c]
        >>> bad_node = NodeSpec("a_cycle", lambda: None, deps=frozenset(["c"]))
        >>> # Adding a_cycle with dep on c would create: a -> c -> a_cycle
        >>> # No cycle since a_cycle is new
        >>> graph._would_create_cycle(bad_node)
        False
        >>> # But if a_cycle tries to connect back to a's dependents:
        >>> graph += NodeSpec("d", lambda: None).after("b")
        >>> cycle_node = NodeSpec("b", lambda: None).after("d")  # b->d->b
        >>> # This would be caught as duplicate, but demonstrates the concept
        """
        if not new_node.deps:
            return False  # No dependencies = no cycle possible

        visited: set[str] = set()

        def can_reach_new_node(current: str) -> bool:
            """DFS to check if we can reach new_node from current."""
            if current == new_node.name:
                return True  # Found cycle!

            if current in visited:
                return False

            visited.add(current)

            # Check all nodes that depend on current (forward edges)
            for dependent in self._forward_edges.get(current, _EMPTY_SET):
                if can_reach_new_node(dependent):
                    return True

            return False

        # Check if new_node is reachable from any of its dependencies
        return any(can_reach_new_node(dep) for dep in new_node.deps)

    def _invalidate_caches(self) -> None:
        """Invalidate cached results when graph structure changes."""
        self._waves_cache = None
        self._validation_cache = ValidationCacheState.INVALID

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
        if self._validation_cache == ValidationCacheState.INVALID:
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

            self._validation_cache = ValidationCacheState.VALID

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

        Checks single-dependency nodes for type mismatches. Multi-dependency nodes
        are not validated automatically as they require custom aggregation logic.

        Returns
        -------
            List of incompatibility messages or empty list if no incompatibilities are found
        """
        incompatibilities = []

        for node_name, node_spec in self.nodes.items():
            # Skip nodes without input validation or dependencies
            if not node_spec.in_model or not node_spec.deps:
                continue

            # Only validate single-dependency nodes (multi-dep requires custom logic)
            if len(node_spec.deps) == 1:
                dep_name = next(iter(node_spec.deps))
                dep_node = self.nodes[dep_name]

                # Check if dependency has output model and it mismatches
                if dep_node.out_model and dep_node.out_model != node_spec.in_model:
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

    def waves_remaining(self, completed: frozenset[str] | set[str]) -> list[list[str]]:
        """Compute execution waves for remaining nodes after some have completed.

        This is MUCH faster than recomputing waves() for the entire graph when
        nodes are added dynamically during execution. Only computes in-degrees
        for nodes that haven't completed yet.

        Optimized for dynamic graphs where the graph is modified during execution:
        - Macro expansions add new nodes
        - Some nodes have already completed
        - Need to compute remaining execution order

        Performance: O(remaining nodes) instead of O(total nodes)

        Parameters
        ----------
        completed : frozenset[str] | set[str]
            Set of node names that have already been executed

        Returns
        -------
        list[list[str]]
            List of remaining waves for parallel execution

        Raises
        ------
        CycleDetectedError
            If a cycle is detected in remaining nodes

        Examples
        --------
        Dynamic execution with graph expansion:

        >>> graph = DirectedGraph()
        >>> graph += NodeSpec("a", lambda: None)
        >>> graph += NodeSpec("b", lambda: None).after("a")
        >>> graph += NodeSpec("c", lambda: None).after("a")
        >>> # After executing 'a', compute remaining waves
        >>> remaining = graph.waves_remaining(frozenset(["a"]))
        >>> remaining  # [["b", "c"]]
        [['b', 'c']]
        >>> # Now dynamically add more nodes
        >>> graph += NodeSpec("d", lambda: None).after("b", "c")
        >>> # Compute waves for b, c, d (a already done)
        >>> remaining = graph.waves_remaining(frozenset(["a"]))
        >>> len(remaining)  # Two waves: [b,c] then [d]
        2

        Notes
        -----
        For static graphs (no dynamic expansion), use waves() instead as it caches results.
        This method is optimized for the dynamic case where caching isn't beneficial.
        """
        if not completed:
            # No nodes completed yet - use full waves() with caching
            return self.waves()

        if not self.nodes:
            return []

        # Only compute in-degrees for nodes that haven't completed
        remaining_nodes = self.nodes.keys() - completed
        in_degrees: dict[str, int] = {}

        for node in remaining_nodes:
            # Count only dependencies that haven't been completed
            deps_remaining = self.nodes[node].deps - completed
            in_degrees[node] = len(deps_remaining)

        # Kahn's algorithm on remaining nodes only
        waves: list[list[str]] = []

        while in_degrees:
            current_wave = [node for node, degree in in_degrees.items() if degree == 0]

            if not current_wave:
                remaining = list(in_degrees.keys())
                raise CycleDetectedError(
                    f"No nodes with zero in-degree found. Remaining nodes: {remaining}"
                )

            waves.append(sorted(current_wave))

            for node in current_wave:
                del in_degrees[node]
                # Update in-degrees for nodes that depend on completed node
                for dependent in self._forward_edges.get(node, _EMPTY_SET):
                    if dependent in in_degrees:
                        in_degrees[dependent] -= 1

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
        if isinstance(other, DirectedGraph):
            return self.merge(other)
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

    def merge(self, other: "DirectedGraph") -> "DirectedGraph":
        """Merge another graph into this one with optimized batching.

        This method provides explicit graph merging, useful for dynamic
        graph expansion during execution (e.g., from macro expansions).

        Optimized for performance with large subgraphs (10+ nodes):
        - Single validation pass instead of per-node validation
        - Batched cycle detection
        - Faster than calling add() for each node individually

        Performance: O(n) instead of O(n²) for n nodes being merged

        Parameters
        ----------
        other : DirectedGraph
            The graph to merge into this one

        Returns
        -------
        DirectedGraph
            Self, for method chaining

        Raises
        ------
        DuplicateNodeError
            If any nodes from the other graph already exist in this graph
        MissingDependencyError
            If merged nodes have dependencies not in either graph
        CycleDetectedError
            If merging would create a cycle

        Examples
        --------
        Dynamic graph expansion:

        .. code-block:: python

            main_graph = DirectedGraph()
            main_graph += NodeSpec("llm", llm_fn)

            # At runtime, expand tool calls into subgraph
            tool_graph = create_tool_subgraph(tool_calls)
            main_graph.merge(tool_graph)  # Add tool nodes dynamically

        Performance comparison for 50 node merge:
        - Old: 50 × O(n) validation = O(n²) ≈ 2500 operations
        - New: 1 × O(n) validation = O(n) ≈ 50 operations
        """
        if not other.nodes:
            return self  # Nothing to merge

        # Fast path for small merges (< 5 nodes) - use existing add()
        if len(other.nodes) < 5:
            for node in other:
                self.add(node)
            return self

        # Optimized batch merge for large subgraphs
        # Step 1: Check for duplicate nodes (O(n) set intersection)
        overlap = self.nodes.keys() & other.nodes.keys()
        if overlap:
            raise DuplicateNodeError(f"Cannot merge: duplicate node(s) found: {sorted(overlap)}")

        # Step 2: Check dependencies exist (O(n) validation)
        # Dependencies can be in either graph (self or other)
        combined_nodes = self.nodes.keys() | other.nodes.keys()
        missing_deps = [
            f"Node '{node_name}' depends on missing '{dep}'"
            for node_name, node_spec in other.nodes.items()
            for dep in node_spec.deps
            if dep not in combined_nodes
        ]

        if missing_deps:
            raise MissingDependencyError("; ".join(missing_deps))

        # Step 3: Batch add all nodes (skip per-node validation)
        for node_name, node_spec in other.nodes.items():
            # Direct insertion - validation already done
            self.nodes[node_name] = node_spec
            self._forward_edges[node_name]  # Ensure key exists
            self._reverse_edges[node_name] = set(node_spec.deps)

            for dep in node_spec.deps:
                self._forward_edges[dep].add(node_name)

        # Step 4: Single cycle check for entire merged graph
        # Only check if merge creates cycles (incremental check)
        if self._detect_cycles():
            # Rollback the merge
            for node_name in other.nodes:
                del self.nodes[node_name]
                if node_name in self._forward_edges:
                    del self._forward_edges[node_name]
                if node_name in self._reverse_edges:
                    del self._reverse_edges[node_name]

            raise CycleDetectedError("Merging graphs would create a cycle")

        # Step 5: Invalidate caches once
        self._invalidate_caches()

        return self

    def get_exit_nodes(self) -> list[str]:
        """Get nodes with no dependents (exit/leaf nodes).

        Exit nodes are nodes that have no other nodes depending on them.
        These are typically the final outputs of a subgraph.

        Returns
        -------
        list[str]
            List of node names with no dependents

        Examples
        --------
        Find exit nodes:

        .. code-block:: python

            graph = DirectedGraph()
            graph += NodeSpec("a", lambda: None)
            graph += NodeSpec("b", lambda: None).after("a")
            graph += NodeSpec("c", lambda: None).after("a")

            exit_nodes = graph.get_exit_nodes()
            assert set(exit_nodes) == {"b", "c"}  # Both are leaves
        """
        return [node_name for node_name in self.nodes if not self._forward_edges.get(node_name)]

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
        return self.merge(other)

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
