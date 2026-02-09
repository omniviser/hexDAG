"""Pipeline YAML manipulation API.

Provides unified functions for creating and modifying hexDAG pipeline YAML.
"""

from __future__ import annotations

from typing import Any

import yaml


def init(name: str, description: str = "") -> dict[str, Any]:
    """Create a new minimal pipeline YAML configuration.

    Creates an empty pipeline structure ready for adding nodes.

    Parameters
    ----------
    name : str
        Pipeline name (used in metadata.name)
    description : str, optional
        Pipeline description

    Returns
    -------
    dict
        Result with keys:
        - success: bool
        - yaml_content: str - The generated YAML
        - message: str

    Examples
    --------
    >>> result = init("my_pipeline", "A sample pipeline")
    >>> result["success"]
    True
    >>> "apiVersion" in result["yaml_content"]
    True
    """
    config = _create_pipeline_base(name, description)
    yaml_content = yaml.dump(config, sort_keys=False, default_flow_style=False)

    return {
        "success": True,
        "yaml_content": yaml_content,
        "message": f"Created empty pipeline '{name}'",
    }


def add_node(
    yaml_content: str,
    kind: str,
    name: str,
    spec: dict[str, Any] | None = None,
    dependencies: list[str] | None = None,
) -> dict[str, Any]:
    """Add a node to an existing pipeline YAML.

    Parameters
    ----------
    yaml_content : str
        Existing pipeline YAML as string
    kind : str
        Node type (e.g., "llm_node", "function_node")
    name : str
        Unique node identifier
    spec : dict | None
        Node-specific configuration dict
    dependencies : list[str] | None
        List of dependency node names

    Returns
    -------
    dict
        Result with keys:
        - success: bool
        - yaml_content: str - Updated YAML
        - node_count: int
        - warnings: list | None

    Examples
    --------
    >>> from hexdag.api.pipeline import init
    >>> yaml_content = init("test")["yaml_content"]
    >>> result = add_node(
    ...     yaml_content,
    ...     kind="llm_node",
    ...     name="analyzer",
    ...     spec={"prompt_template": "Analyze: {{input}}"},
    ...     dependencies=[]
    ... )
    >>> result["success"]
    True
    """
    config, error = _parse_pipeline_yaml(yaml_content)
    if error:
        return {"success": False, "error": error}

    if spec is None:
        spec = {}
    if dependencies is None:
        dependencies = []

    # Get or create nodes list
    spec_section = config.setdefault("spec", {})
    nodes = spec_section.setdefault("nodes", [])

    # Check for duplicate name
    existing_names = _get_node_names(nodes)
    if name in existing_names:
        return {"success": False, "error": f"Node '{name}' already exists"}

    # Build node structure
    new_node: dict[str, Any] = {
        "kind": kind,
        "metadata": {"name": name},
        "spec": spec,
        "dependencies": dependencies,
    }

    # Check for missing dependencies (warn, don't fail)
    warnings: list[str] = [
        f"Dependency '{dep}' not found in pipeline"
        for dep in dependencies
        if dep not in existing_names
    ]

    nodes.append(new_node)

    yaml_output = yaml.dump(config, sort_keys=False, default_flow_style=False)

    return {
        "success": True,
        "yaml_content": yaml_output,
        "node_count": len(nodes),
        "warnings": warnings if warnings else None,
    }


def remove_node(yaml_content: str, node_name: str) -> dict[str, Any]:
    """Remove a node from a pipeline YAML.

    Parameters
    ----------
    yaml_content : str
        Existing pipeline YAML as string
    node_name : str
        Name of the node to remove

    Returns
    -------
    dict
        Result with keys:
        - success: bool
        - yaml_content: str - Updated YAML
        - node_count: int
        - removed: bool
        - warnings: list | None (warns if other nodes depend on removed node)

    Examples
    --------
    >>> from hexdag.api.pipeline import init, add_node
    >>> yaml_content = init("test")["yaml_content"]
    >>> res = add_node(yaml_content, "data_node", "old_node", {"output": {"v": 1}}, [])
    >>> yaml_content = res["yaml_content"]
    >>> result = remove_node(yaml_content, "old_node")
    >>> result["success"]
    True
    """
    config, error = _parse_pipeline_yaml(yaml_content)
    if error:
        return {"success": False, "error": error}

    nodes = config.get("spec", {}).get("nodes", [])

    # Find node index
    node_idx = _find_node_by_name(nodes, node_name)
    if node_idx is None:
        return {"success": False, "error": f"Node '{node_name}' not found"}

    # Check for dependents
    warnings: list[str] = []
    for node in nodes:
        deps = node.get("dependencies", [])
        if node_name in deps:
            dependent_name = node.get("metadata", {}).get("name", "unknown")
            warnings.append(f"Node '{dependent_name}' depends on '{node_name}'")

    # Remove the node
    nodes.pop(node_idx)

    yaml_output = yaml.dump(config, sort_keys=False, default_flow_style=False)

    return {
        "success": True,
        "yaml_content": yaml_output,
        "node_count": len(nodes),
        "removed": True,
        "warnings": warnings if warnings else None,
    }


def update_node(
    yaml_content: str,
    node_name: str,
    spec: dict[str, Any] | None = None,
    dependencies: list[str] | None = None,
    kind: str | None = None,
) -> dict[str, Any]:
    """Update a node's configuration in pipeline YAML.

    Parameters
    ----------
    yaml_content : str
        Existing pipeline YAML as string
    node_name : str
        Name of the node to update
    spec : dict | None
        Spec fields to merge/update
    dependencies : list[str] | None
        New dependencies list (replaces existing)
    kind : str | None
        New node type (use with caution)

    Returns
    -------
    dict
        Result with keys:
        - success: bool
        - yaml_content: str - Updated YAML
        - warnings: list | None

    Examples
    --------
    >>> from hexdag.api.pipeline import init, add_node
    >>> yaml_content = init("test")["yaml_content"]
    >>> res = add_node(yaml_content, "llm_node", "analyzer", {"prompt_template": "Old"}, [])
    >>> yaml_content = res["yaml_content"]
    >>> result = update_node(
    ...     yaml_content,
    ...     "analyzer",
    ...     spec={"prompt_template": "New prompt: {{input}}"}
    ... )
    >>> result["success"]
    True
    """
    config, error = _parse_pipeline_yaml(yaml_content)
    if error:
        return {"success": False, "error": error}

    nodes = config.get("spec", {}).get("nodes", [])

    # Find node index
    node_idx = _find_node_by_name(nodes, node_name)
    if node_idx is None:
        return {"success": False, "error": f"Node '{node_name}' not found"}

    node = nodes[node_idx]
    warnings: list[str] = []

    # Apply updates
    if spec is not None:
        # Deep merge spec updates
        node_spec = node.setdefault("spec", {})
        for key, value in spec.items():
            node_spec[key] = value

    if dependencies is not None:
        # Replace dependencies
        node["dependencies"] = dependencies

    if kind is not None:
        # Change node type (warn user)
        old_kind = node.get("kind")
        if old_kind != kind:
            warnings.append(f"Changed node type from '{old_kind}' to '{kind}'")
        node["kind"] = kind

    yaml_output = yaml.dump(config, sort_keys=False, default_flow_style=False)

    return {
        "success": True,
        "yaml_content": yaml_output,
        "warnings": warnings if warnings else None,
    }


def list_nodes(yaml_content: str) -> dict[str, Any]:
    """List all nodes in a pipeline with their dependencies.

    Parameters
    ----------
    yaml_content : str
        Pipeline YAML as string

    Returns
    -------
    dict
        Result with keys:
        - success: bool
        - pipeline_name: str
        - node_count: int
        - nodes: list of {name, kind, dependencies, dependents}
        - execution_order: list[str]

    Examples
    --------
    >>> from hexdag.api.pipeline import init, add_node
    >>> yaml_content = init("test")["yaml_content"]
    >>> res = add_node(yaml_content, "data_node", "a", {"output": {"v": 1}}, [])
    >>> yaml_content = res["yaml_content"]
    >>> result = list_nodes(yaml_content)
    >>> result["success"]
    True
    >>> len(result["nodes"]) == result["node_count"]
    True
    """
    config, error = _parse_pipeline_yaml(yaml_content)
    if error:
        return {"success": False, "error": error}

    pipeline_name = config.get("metadata", {}).get("name", "unknown")
    nodes = config.get("spec", {}).get("nodes", [])

    # Build node info with reverse dependencies
    node_infos: list[dict[str, Any]] = []
    all_names = _get_node_names(nodes)

    # Build reverse dependency map
    dependents_map: dict[str, list[str]] = {name: [] for name in all_names}
    for node in nodes:
        node_name = node.get("metadata", {}).get("name")
        deps = node.get("dependencies", [])
        for dep in deps:
            if dep in dependents_map:
                dependents_map[dep].append(node_name)

    for node in nodes:
        node_name = node.get("metadata", {}).get("name", "unknown")
        node_infos.append({
            "name": node_name,
            "kind": node.get("kind", "unknown"),
            "dependencies": node.get("dependencies", []),
            "dependents": dependents_map.get(node_name, []),
        })

    # Compute execution order
    execution_order = _compute_execution_order(nodes)

    return {
        "success": True,
        "pipeline_name": pipeline_name,
        "node_count": len(nodes),
        "nodes": node_infos,
        "execution_order": execution_order,
    }


# =============================================================================
# Helper Functions
# =============================================================================


def _create_pipeline_base(name: str, description: str = "") -> dict[str, Any]:
    """Create base pipeline configuration structure."""
    config: dict[str, Any] = {
        "apiVersion": "hexdag/v1",
        "kind": "Pipeline",
        "metadata": {
            "name": name,
        },
        "spec": {
            "nodes": [],
        },
    }

    if description:
        config["metadata"]["description"] = description

    return config


def _parse_pipeline_yaml(yaml_content: str) -> tuple[dict[str, Any], str | None]:
    """Parse pipeline YAML with error handling.

    Returns
    -------
    tuple[dict[str, Any], str | None]
        Tuple of (parsed_config, error_message)
    """
    try:
        config = yaml.safe_load(yaml_content)
        if not isinstance(config, dict):
            return {}, "YAML must be a dictionary/object"
        return config, None
    except yaml.YAMLError as e:
        return {}, f"YAML parse error: {e}"


def _find_node_by_name(nodes: list[dict[str, Any]], name: str) -> int | None:
    """Find node index by metadata.name."""
    for i, node in enumerate(nodes):
        node_name = node.get("metadata", {}).get("name")
        if node_name == name:
            return i
    return None


def _get_node_names(nodes: list[dict[str, Any]]) -> set[str]:
    """Get set of all node names."""
    names = set()
    for node in nodes:
        name = node.get("metadata", {}).get("name")
        if name:
            names.add(name)
    return names


def _compute_execution_order(nodes: list[dict[str, Any]]) -> list[str]:
    """Compute topological execution order using Kahn's algorithm."""
    # Build adjacency list and in-degree count
    in_degree: dict[str, int] = {}
    graph: dict[str, list[str]] = {}
    all_nodes: set[str] = set()

    for node in nodes:
        name = node.get("metadata", {}).get("name")
        if not name:
            continue
        all_nodes.add(name)
        in_degree.setdefault(name, 0)
        graph.setdefault(name, [])

        deps = node.get("dependencies", [])
        for dep in deps:
            if dep in all_nodes or dep in in_degree:
                graph.setdefault(dep, []).append(name)
                in_degree[name] = in_degree.get(name, 0) + 1

    # Kahn's algorithm
    queue = [n for n in all_nodes if in_degree.get(n, 0) == 0]
    result: list[str] = []

    while queue:
        current_node = queue.pop(0)
        result.append(current_node)
        for neighbor in graph.get(current_node, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return result
