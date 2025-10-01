"""Input mapper for node data preparation.

This module provides the InputMapper class that handles preparing input data
for nodes based on their dependencies.
"""

from typing import Any

from hexai.core.domain.dag import NodeSpec


class InputMapper:
    """Handles input data preparation and mapping for nodes.

    This component is responsible for preparing the input data for each node
    based on its dependencies. It implements a smart mapping strategy:

    - **No dependencies**: Pass through the initial input
    - **Single dependency**: Pass through that dependency's result directly
    - **Multiple dependencies**: Create a dict with dependency names as keys

    This strategy simplifies node function signatures while maintaining clear
    data flow through the DAG.

    Single Responsibility: Map dependencies to node inputs.

    Examples
    --------
    >>> mapper = InputMapper()
    >>>
    >>> # No dependencies - gets initial input
    >>> input_data = mapper.prepare_node_input(
    ...     node_spec=NodeSpec("start", fn),
    ...     node_results={},
    ...     initial_input="Hello"
    ... )
    >>> assert input_data == "Hello"
    >>>
    >>> # Single dependency - gets that result directly
    >>> input_data = mapper.prepare_node_input(
    ...     node_spec=NodeSpec("process", fn, deps={"start"}),
    ...     node_results={"start": "processed"},
    ...     initial_input="Hello"
    ... )
    >>> assert input_data == "processed"
    >>>
    >>> # Multiple dependencies - gets dict of results
    >>> input_data = mapper.prepare_node_input(
    ...     node_spec=NodeSpec("combine", fn, deps={"start", "process"}),
    ...     node_results={"start": "A", "process": "B"},
    ...     initial_input="Hello"
    ... )
    >>> assert input_data == {"start": "A", "process": "B"}
    """

    def prepare_node_input(
        self, node_spec: NodeSpec, node_results: dict[str, Any], initial_input: Any
    ) -> Any:
        """Prepare input data for node execution with simplified data mapping.

        The mapping strategy is:
        1. **No dependencies** → initial_input (entry point)
        2. **Single dependency** → results[dependency_name] (pass-through)
        3. **Multiple dependencies** → {dep1: result1, dep2: result2, ...} (namespace)

        This approach balances simplicity (pass-through for single deps) with
        clarity (named dict for multiple deps).

        Parameters
        ----------
        node_spec : NodeSpec
            Node specification containing dependencies
        node_results : dict[str, Any]
            Results from previously executed nodes
        initial_input : Any
            Initial input data for the pipeline

        Returns
        -------
        Any
            Prepared input data for the node:
            - initial_input if no dependencies
            - dependency result if single dependency
            - dict of dependency results if multiple dependencies

        Examples
        --------
        >>> # Pipeline: start → process → combine
        >>> mapper = InputMapper()
        >>>
        >>> # start node (no deps)
        >>> start_input = mapper.prepare_node_input(
        ...     NodeSpec("start", lambda x: x.upper()),
        ...     node_results={},
        ...     initial_input="hello"
        ... )
        >>> # start_input = "hello"
        >>>
        >>> # After start executes: node_results = {"start": "HELLO"}
        >>> process_input = mapper.prepare_node_input(
        ...     NodeSpec("process", lambda x: x + "!", deps={"start"}),
        ...     node_results={"start": "HELLO"},
        ...     initial_input="hello"
        ... )
        >>> # process_input = "HELLO" (direct pass-through)
        >>>
        >>> # After process executes: node_results = {"start": "HELLO", "process": "HELLO!"}
        >>> combine_input = mapper.prepare_node_input(
        ...     NodeSpec("combine", lambda data: ..., deps={"start", "process"}),
        ...     node_results={"start": "HELLO", "process": "HELLO!"},
        ...     initial_input="hello"
        ... )
        >>> # combine_input = {"start": "HELLO", "process": "HELLO!"}

        Notes
        -----
        The multi-dependency dict preserves node names as keys, making it clear
        where each piece of data came from. This is especially useful for
        debugging and for nodes that need to treat different dependencies
        differently.
        """
        if not node_spec.deps:
            # No dependencies - use initial input (entry point)
            return initial_input

        if len(node_spec.deps) == 1:
            # Single dependency - pass through directly
            # This simplifies node functions: they receive the data directly
            # rather than wrapped in a dict
            dep_name = next(iter(node_spec.deps))
            return node_results.get(dep_name, initial_input)

        # Multiple dependencies - preserve namespace structure
        # Each dependency is keyed by its node name for clarity
        aggregated_data = {}

        for dep_name in node_spec.deps:
            if dep_name in node_results:
                aggregated_data[dep_name] = node_results[dep_name]

        return aggregated_data
