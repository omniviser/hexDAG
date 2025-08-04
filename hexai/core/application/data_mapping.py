"""Data mapping system for explicit data flow between nodes.

This module provides utilities for mapping data between nodes with different schemas, enabling clear
and explicit data relationships in pipelines.
"""

from typing import Any, Literal

from pydantic import BaseModel

# Field mapping mode types
FieldMappingMode = Literal["none", "default", "vector", "custom"]

# Default field name mappings - general purpose
DEFAULT_FIELD_MAPPINGS = {
    "text": ["content", "data", "input", "message", "body"],
    "content": ["text", "data", "input", "message", "body"],
    "data": ["content", "text", "input", "payload", "body"],
    "metadata": ["config", "options", "params", "headers", "meta"],
    "config": ["metadata", "options", "params", "settings"],
    "result": ["output", "response", "data", "value"],
    "output": ["result", "response", "data", "value"],
    "response": ["result", "output", "data", "value"],
    "error": ["err", "exception", "failure"],
    "status": ["state", "code", "result_code"],
    "id": ["identifier", "key", "name", "_id", "user_id"],
    "name": ["title", "label", "identifier"],
    "value": ["data", "content", "result"],
    "timestamp": ["time", "created_at", "updated_at", "datetime"],
    "created_at": ["timestamp", "time", "datetime", "date"],
}


class DataMapper:
    """Maps data between nodes using explicit field mappings."""

    def __init__(self, mapping: dict[str, str] | None = None) -> None:
        """Initialize data mapper with optional field mappings.

        Args:
        ----
            mapping: Dict of target_field -> source_path mappings
                    e.g., {"content": "processor.text", "config": "validator.metadata"}
        """
        self.mapping = mapping or {}

    def map_data(
        self, node_results: dict[str, Any], input_mapping: dict[str, str]
    ) -> dict[str, Any]:
        """Map data from node results using explicit mappings.

        Args
        ----
            node_results: Results from executed nodes {node_name: result}
            input_mapping: Field mappings {target_field: source_path}

        Returns
        -------
            Mapped data ready for node input
        """
        mapped_data = {}

        for target_field, source_path in input_mapping.items():
            value = self._resolve_source_path(source_path, node_results)
            if value is not None:
                mapped_data[target_field] = value

        return mapped_data

    def _resolve_source_path(self, source_path: str, node_results: dict[str, Any]) -> Any:
        """Resolve a source path to actual data.

        Args
        ----
            source_path: Path like "node_name.field" or "node_name"
            node_results: Available node results

        Returns
        -------
            Resolved value or None if not found
        """
        if "." in source_path:
            # Nested path: "processor.text"
            node_name, field_path = source_path.split(".", 1)
            if node_name in node_results:
                node_output = node_results[node_name]
                return self._extract_field(node_output, field_path)
        else:
            # Direct node reference: "processor"
            return node_results.get(source_path)

        return None

    def _extract_field(self, data: Any, field_path: str) -> Any:
        """Extract field from data using dot notation.

        Args
        ----
            data: Source data (dict, BaseModel, or primitive)
            field_path: Field path like "text" or "metadata.config"

        Returns
        -------
            Extracted value or None if not found
        """
        if data is None:
            return None

        # Convert Pydantic model to dict
        if isinstance(data, BaseModel):
            data = data.model_dump()
        elif not isinstance(data, dict):
            # Primitive value - only direct access
            return data if field_path == "" else None

        # Navigate nested fields
        current = data
        for field in field_path.split("."):
            if isinstance(current, dict) and field in current:
                current = current[field]
            else:
                return None

        return current


class SchemaAligner:
    """Schema alignment between connected nodes with configurable field mappings."""

    def __init__(
        self,
        mode: FieldMappingMode = "default",
        custom_mappings: dict[str, list[str]] | None = None,
    ) -> None:
        """Initialize with field mapping mode.

        Args
        ----
            mode: Field mapping mode - "none", "default", or "custom"
            custom_mappings: Required when mode="custom", ignored otherwise
        """
        if mode == "none":
            self.field_mappings: dict[str, list[str]] = {}
        elif mode == "default":
            self.field_mappings = DEFAULT_FIELD_MAPPINGS
        elif mode == "custom":
            if custom_mappings is None:
                raise ValueError("custom_mappings required when mode='custom'")
            self.field_mappings = custom_mappings
        else:
            raise ValueError(f"Field mapping mode: {mode} not supported.")

    def align_schemas(
        self, source_schema: dict[str, Any], target_schema: dict[str, Any]
    ) -> dict[str, str]:
        """Create automatic field mapping between schemas.

        Args:
        ----
            source_schema: Schema of source node output
            target_schema: Schema of target node input

        Returns
        -------
            Field mapping dict {target_field: source_field}
        """
        mapping = {}

        # Direct field name matches
        for target_field in target_schema:
            if target_field in source_schema:
                mapping[target_field] = target_field
                continue

            # Look for alternative field names
            alternatives = self.field_mappings.get(target_field, [])
            for alt_field in alternatives:
                if alt_field in source_schema:
                    mapping[target_field] = alt_field
                    break

        return mapping

    def suggest_mappings(
        self, source_schema: dict[str, Any], target_schema: dict[str, Any]
    ) -> dict[str, list[str]]:
        """Suggest possible field mappings for manual review.

        Args
        ----
            source_schema: Schema of source node output
            target_schema: Schema of target node input

        Returns
        -------
            Suggestions dict {target_field: [possible_source_fields]}
        """
        suggestions = {}

        for target_field in target_schema:
            possible_sources = []

            # Exact match
            if target_field in source_schema:
                possible_sources.append(target_field)

            # Alternative names only (no fuzzy matching)
            alternatives = self.field_mappings.get(target_field, [])
            for alt_field in alternatives:
                if alt_field in source_schema and alt_field not in possible_sources:
                    possible_sources.append(alt_field)

            if possible_sources:
                suggestions[target_field] = possible_sources

        return suggestions


class DataAggregator:
    """Structured data aggregation preserving node namespaces."""

    def aggregate_structured(
        self, node_results: dict[str, Any], dependencies: list[str]
    ) -> dict[str, Any]:
        """Aggregate data from multiple dependencies with namespace preservation.

        Args
        ----
            node_results: Results from executed nodes
            dependencies: List of dependency node names

        Returns
        -------
            Aggregated data with preserved structure
        """
        if not dependencies:
            return {}
        elif len(dependencies) == 1:
            # Single dependency - return directly
            dep_name = dependencies[0]
            return node_results.get(dep_name, {})  # type: ignore[no-any-return]
        else:
            # Multiple dependencies - preserve namespaces
            aggregated = {}
            for dep_name in dependencies:
                if dep_name in node_results:
                    dep_output = node_results[dep_name]

                    # Preserve structure with namespace
                    if isinstance(dep_output, BaseModel):
                        aggregated[dep_name] = dep_output.model_dump()
                    elif isinstance(dep_output, dict):
                        aggregated[dep_name] = dep_output
                    else:
                        aggregated[dep_name] = dep_output

            return aggregated


# Utility functions for common data mapping patterns
def create_passthrough_mapping(fields: list[str]) -> dict[str, str]:
    """Create a passthrough mapping for field names.

    Args
    ----
        fields: List of field names to pass through

    Returns
    -------
        Mapping dict with identical source and target names
    """
    return {field: field for field in fields}


def create_rename_mapping(renames: dict[str, str]) -> dict[str, str]:
    """Create a field rename mapping.

    Args
    ----
        renames: Dict of {new_name: old_name}

    Returns
    -------
        Mapping dict for renaming fields
    """
    return renames


def create_prefixed_mapping(
    fields: list[str], source_node: str, prefix: str = ""
) -> dict[str, str]:
    """Create mapping with node prefixes.

    Args
    ----
        fields: List of field names
        source_node: Source node name
        prefix: Optional prefix for target fields

    Returns
    -------
        Mapping dict with node prefixes
    """
    mapping = {}
    for field in fields:
        target_field = f"{prefix}{field}" if prefix else field
        mapping[target_field] = f"{source_node}.{field}"
    return mapping
