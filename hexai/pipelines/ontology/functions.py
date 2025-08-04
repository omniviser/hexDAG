"""Functions for the ontology analysis pipeline."""

from typing import Any

from pydantic import BaseModel, Field

# Constants
DEFAULT_ONTOLOGY_NAME = "Default Ontology"
UNKNOWN_NODE_NAME = "Unknown"
DEFAULT_MAX_DEPTH = 5
DEFAULT_INCLUDE_METADATA = False


class OntologyPipelineInput(BaseModel):
    """Input model for the ontology pipeline."""

    user_query: str = Field(default="", description="User query for ontology analysis")


async def load_ontology_context(input_data: dict[str, Any], **ports: Any) -> dict[str, Any]:
    """Load ontology context and nodes from the ontology port.

    Args
    ----
        input_data: Input data containing user query
        **ports: Injected ports including 'ontology'

    Returns
    -------
        Dictionary containing ontology nodes and user query
    """
    event_manager = ports.get("event_manager")
    if event_manager:
        event_manager.add_trace("load_ontology_context", "Loading ontology context...")

    ontology_port = ports.get("ontology")
    if not ontology_port:
        raise ValueError("No ontology port provided")

    # Handle both dict and Pydantic model input
    if isinstance(input_data, dict):
        user_query = input_data.get("user_query", "")
    else:
        # Assume it's a Pydantic model
        data = input_data.model_dump()
        user_query = data.get("user_query", "")

    # Use new schema structure
    ontology_nodes = ontology_port.get_ontology_nodes()
    ontology_relations = ontology_port.get_ontology_relations()

    # Format for the OntologyParserAgent (grouped by ontology)
    database_nodes: dict[str, list[str]] = {}
    for node in ontology_nodes:
        # Get the ontology name - try to find it from metadata or use default
        ontology_name = DEFAULT_ONTOLOGY_NAME
        ontologies = ontology_port.get_ontologies()
        for ont in ontologies:
            if ont.get("id") == node.get("ontology_id"):
                ontology_name = ont.get("name", DEFAULT_ONTOLOGY_NAME)
                break

        node_name = node.get("name", UNKNOWN_NODE_NAME)

        if ontology_name not in database_nodes:
            database_nodes[ontology_name] = []
        database_nodes[ontology_name].append(node_name)

    if event_manager:
        event_manager.add_trace(
            "ontology_context",
            f"Loaded {len(ontology_nodes)} nodes and {len(ontology_relations)} relations",
        )

    return {
        "user_query": user_query,
        "database_nodes": database_nodes,
        "ontology_nodes": ontology_nodes,
        "ontology_relations": ontology_relations,
        "metadata": {"node_name": "load_ontology_context"},
    }


async def load_ontology_data(input_data: dict[str, Any], **ports: Any) -> dict[str, Any]:
    """Load complete ontology data for analysis.

    Args
    ----
        input_data: Input data
        **ports: Injected ports including 'ontology'

    Returns
    -------
        Dictionary containing complete ontology data
    """
    event_manager = ports.get("event_manager")
    if event_manager:
        event_manager.add_trace("load_ontology_data", "Loading complete ontology data...")

    ontology_port = ports.get("ontology")
    if not ontology_port:
        raise ValueError("No ontology port provided")

    # Use new schema structure
    ontologies = ontology_port.get_ontologies()
    ontology_nodes = ontology_port.get_ontology_nodes()
    ontology_relations = ontology_port.get_ontology_relations()

    # Extract user query from input
    user_query = input_data.get("user_query", "")

    if event_manager:
        event_manager.add_trace(
            "ontology_data",
            f"Loaded {len(ontologies)} ontologies, {len(ontology_nodes)} nodes, "
            f"{len(ontology_relations)} relations",
        )

    return {
        "ontologies": ontologies,
        "ontology_nodes": ontology_nodes,
        "ontology_relations": ontology_relations,
        "user_query": user_query,
        "metadata": {"node_name": "load_ontology_data"},
    }


async def metadata_resolver(input_data: dict[str, Any], **ports: Any) -> dict[str, Any]:
    """Resolve metadata for ontology nodes and relations.

    Args
    ----
        input_data: Input data containing ontology information
        **ports: Injected ports

    Returns
    -------
        Dictionary with resolved metadata
    """
    event_manager = ports.get("event_manager")
    if event_manager:
        event_manager.add_trace("metadata_resolver", "Resolving ontology metadata...")

    ontology_nodes = input_data.get("ontology_nodes", [])
    ontology_relations = input_data.get("ontology_relations", [])
    user_query = input_data.get("user_query", "")

    # Resolve metadata for nodes
    node_metadata = {}
    for node in ontology_nodes:
        node_id = node.get("id")
        if node_id:
            node_metadata[node_id] = {
                "name": node.get("name", ""),
                "description": node.get("description", ""),
                "type": node.get("type", ""),
                "ontology_id": node.get("ontology_id", ""),
            }

    # Resolve metadata for relations
    relation_metadata = {}
    for relation in ontology_relations:
        relation_id = relation.get("id")
        if relation_id:
            relation_metadata[relation_id] = {
                "name": relation.get("name", ""),
                "source_node_id": relation.get("source_node_id", ""),
                "target_node_id": relation.get("target_node_id", ""),
                "type": relation.get("type", ""),
            }

    if event_manager:
        event_manager.add_trace(
            "metadata_resolver",
            f"Resolved metadata for {len(node_metadata)} nodes and "
            f"{len(relation_metadata)} relations",
        )

    return {
        "node_metadata": node_metadata,
        "relation_metadata": relation_metadata,
        "user_query": user_query,
        "ontology_nodes": ontology_nodes,
        "ontology_relations": ontology_relations,
        "metadata": {"node_name": "metadata_resolver"},
    }


async def ontology_analyzer(input_data: dict[str, Any], **ports: Any) -> dict[str, Any]:
    """Analyze ontology structure and relationships.

    Args
    ----
        input_data: Input data containing ontology information
        **ports: Injected ports

    Returns
    -------
        Dictionary with ontology analysis results
    """
    event_manager = ports.get("event_manager")
    if event_manager:
        event_manager.add_trace("ontology_analyzer", "Analyzing ontology structure...")

    ontology_nodes = input_data.get("ontology_nodes", [])
    ontology_relations = input_data.get("ontology_relations", [])
    user_query = input_data.get("user_query", "")

    # Analyze node types
    node_types = {}
    for node in ontology_nodes:
        node_type = node.get("type", "unknown")
        if node_type not in node_types:
            node_types[node_type] = 0
        node_types[node_type] += 1

    # Analyze relation types
    relation_types = {}
    for relation in ontology_relations:
        relation_type = relation.get("type", "unknown")
        if relation_type not in relation_types:
            relation_types[relation_type] = 0
        relation_types[relation_type] += 1

    # Calculate basic statistics
    total_nodes = len(ontology_nodes)
    total_relations = len(ontology_relations)
    avg_relations_per_node = total_relations / total_nodes if total_nodes > 0 else 0

    analysis_result = {
        "total_nodes": total_nodes,
        "total_relations": total_relations,
        "node_types": node_types,
        "relation_types": relation_types,
        "avg_relations_per_node": avg_relations_per_node,
        "user_query": user_query,
    }

    if event_manager:
        event_manager.add_trace(
            "ontology_analyzer",
            f"Analyzed ontology: {total_nodes} nodes, {total_relations} relations",
        )

    return {
        "user_query": user_query,
        "analysis_result": analysis_result,
        "ontology_nodes": ontology_nodes,
        "ontology_relations": ontology_relations,
        "metadata": {"node_name": "ontology_analyzer"},
    }


async def query_matcher(input_data: dict[str, Any], **ports: Any) -> dict[str, Any]:
    """Match user query against ontology nodes and relations.

    Args
    ----
        input_data: Input data containing ontology and query information
        **ports: Injected ports

    Returns
    -------
        Dictionary with query matching results
    """
    event_manager = ports.get("event_manager")
    if event_manager:
        event_manager.add_trace("query_matcher", "Matching user query against ontology...")

    ontology_nodes = input_data.get("ontology_nodes", [])
    ontology_relations = input_data.get("ontology_relations", [])
    user_query = input_data.get("user_query", "")
    user_query_lower = user_query.lower()  # Use lowercase for matching only

    # Simple keyword matching
    matched_nodes = []
    matched_relations = []

    # Match nodes
    for node in ontology_nodes:
        node_name = node.get("name", "").lower()
        node_description = node.get("description", "").lower()

        if user_query_lower in node_name or user_query_lower in node_description:
            matched_nodes.append(node)

    # Match relations
    for relation in ontology_relations:
        relation_name = relation.get("name", "").lower()

        if user_query_lower in relation_name:
            matched_relations.append(relation)

    if event_manager:
        event_manager.add_trace(
            "query_matcher",
            f"Query matched {len(matched_nodes)} nodes and {len(matched_relations)} relations",
        )

    return {
        "matched_entities": matched_nodes,  # Alias for backwards compatibility
        "matched_nodes": matched_nodes,
        "matched_relations": matched_relations,
        "ontology_nodes": ontology_nodes,
        "ontology_relations": ontology_relations,
        "user_query": user_query,
        "total_matches": len(matched_nodes) + len(matched_relations),
        "metadata": {"node_name": "query_matcher"},
    }


async def result_formatter(input_data: dict[str, Any], **ports: Any) -> dict[str, Any]:
    """Format the final ontology analysis results.

    Args
    ----
        input_data: Input data containing analysis results
        **ports: Injected ports

    Returns
    -------
        Dictionary with formatted results
    """
    event_manager = ports.get("event_manager")
    if event_manager:
        event_manager.add_trace("result_formatter", "Formatting ontology analysis results...")

    # Extract results from previous nodes
    analysis_result = input_data.get("analysis_result", {})
    matched_nodes = input_data.get("matched_nodes", [])
    matched_relations = input_data.get("matched_relations", [])
    ontology_nodes = input_data.get("ontology_nodes", [])
    ontology_relations = input_data.get("ontology_relations", [])
    user_query = input_data.get("user_query", "")

    # Format the final output
    formatted_result = {
        "query": user_query,
        "analysis": analysis_result,
        "matches": {
            "nodes": matched_nodes,
            "relations": matched_relations,
            "total": len(matched_nodes) + len(matched_relations),
        },
        "summary": {
            "total_nodes_analyzed": analysis_result.get("total_nodes", 0),
            "total_relations_analyzed": analysis_result.get("total_relations", 0),
            "matching_nodes": len(matched_nodes),
            "matching_relations": len(matched_relations),
        },
    }

    if event_manager:
        event_manager.add_trace(
            "result_formatter", "Ontology analysis results formatted successfully"
        )

    return {
        "user_query": user_query,
        "ontology_nodes": ontology_nodes,
        "ontology_relations": ontology_relations,
        "formatted_result": formatted_result,
        "raw_data": input_data,
        "metadata": {"node_name": "result_formatter"},
    }
