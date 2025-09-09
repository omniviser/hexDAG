"""Simple compile-time type safety checker for pipelines."""

from hexai.core.domain.dag import DirectedGraph
from hexai.core.validation import can_convert_schema


def check_pipeline_type_safety(
    graph: DirectedGraph, pipeline_name: str = "unnamed"
) -> tuple[bool, list[str]]:
    """Check if pipeline is type safe at compile time.

    Args
    ----
    graph: The DirectedGraph to check
    pipeline_name: Name of the pipeline for error reporting

    Returns
    -------
    Tuple of (is_type_safe, list_of_error_messages)
    """
    errors = []

    # Check each connection between nodes
    for node_name, node_spec in graph.nodes.items():
        # Skip nodes without dependencies
        if not node_spec.deps:
            continue

        # Skip nodes with input mapping (they handle their own conversion)
        if "input_mapping" in node_spec.params:
            continue

        # Check each dependency
        for dep_name in node_spec.deps:
            dep_spec = graph.nodes.get(dep_name)
            if not dep_spec:
                errors.append(f"Node '{node_name}' depends on missing node '{dep_name}'")
                continue

            # Check type compatibility if both have types specified
            if (
                node_spec.in_type
                and dep_spec.out_type
                and not can_convert_schema(dep_spec.out_type, node_spec.in_type)
            ):
                errors.append(
                    f"Type mismatch: '{dep_name}' outputs {dep_spec.out_type.__name__} "
                    f"but '{node_name}' expects {node_spec.in_type.__name__}"
                )

    return len(errors) == 0, errors


def validate_pipeline_compilation(graph: DirectedGraph, pipeline_name: str) -> None:
    """Validate pipeline for compilation - raises error if not type safe.

    Args
    ----
    graph: The DirectedGraph to validate
    pipeline_name: Name of the pipeline

    Raises
    ------
    ValueError: If pipeline is not type safe for compilation
    """
    is_type_safe, errors = check_pipeline_type_safety(graph, pipeline_name)

    if not is_type_safe:
        error_msg = f"❌ Pipeline '{pipeline_name}' has type safety issues:\n"
        for i, error in enumerate(errors[:5], 1):  # Show first 5 errors
            error_msg += f"  {i}. {error}\n"
        if len(errors) > 5:
            error_msg += f"  ... and {len(errors) - 5} more errors\n"
        error_msg += "💡 Fix type compatibility issues before compilation"
        raise ValueError(error_msg)
