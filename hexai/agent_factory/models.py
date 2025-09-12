"""Domain models for pipeline applications.

This module contains data models specific to pipeline applications, particularly for ontology
analytics and text-to-SQL use cases.

Optimized with Pydantic v2 best practices:
- Annotated types for Rust-level validation
- Optimized model configurations
- TypeAdapter caching for bulk operations
"""

from datetime import datetime
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

# --- Optimized type definitions with Annotated constraints ---
# These push validation into pydantic-core (Rust) for better performance

PositiveInt = Annotated[int, Field(gt=0)]
NonEmptyStr = Annotated[str, Field(min_length=1, max_length=255)]
Identifier = Annotated[int, Field(gt=0)]
TenantId = Annotated[int, Field(gt=0)]
Confidence = Annotated[float, Field(ge=0.0, le=1.0)]
QueryId = Annotated[str, Field(pattern=r"^[a-zA-Z0-9_-]+$", min_length=1, max_length=100)]

# --- Optimized base configuration ---
OPTIMIZED_CONFIG = ConfigDict(
    frozen=True,  # Immutable models (hashable, thread-safe)
    validate_default=False,  # Skip validation of default values
    validate_assignment=False,  # No re-validation on assignment (frozen anyway)
    use_enum_values=True,  # Use enum values directly
    cache_strings="keys",  # Cache string operations on keys
    extra="forbid",  # Fail fast on extra fields
    str_strip_whitespace=True,  # Auto-strip whitespace from strings
)

# Ontology Analytics Domain Models - Aligned with Database Schema


class RelationshipType(str, Enum):
    """Basic relationship types."""

    ONE_TO_ONE = "one_to_one"
    ONE_TO_MANY = "one_to_many"
    MANY_TO_ONE = "many_to_one"
    MANY_TO_MANY = "many_to_many"


class Ontology(BaseModel):
    """Ontology definition matching database schema."""

    model_config = OPTIMIZED_CONFIG

    id: Identifier = Field(description="Unique identifier for the ontology")
    name: NonEmptyStr = Field(description="Name of the ontology")
    is_active: bool = Field(default=True, description="Boolean flag if this ontology is in use")
    created_at: datetime = Field(description="When the ontology was created")
    tenant_id: TenantId = Field(description="Foreign key to the tenant or organization")
    last_synced_at: datetime | None = Field(default=None, description="Last sync timestamp")


class OntologyNode(BaseModel):
    """Ontology node (business concept) matching database schema."""

    model_config = OPTIMIZED_CONFIG

    id: Identifier = Field(description="Unique ID for the ontology node")
    ontology_id: Identifier = Field(description="FK to Ontology.id")
    name: NonEmptyStr = Field(description="Business name of the node")
    alias: NonEmptyStr | None = Field(default=None, description="Alias for the node")
    permissed_roles: list[str] = Field(
        default_factory=list, description="Roles allowed to access this node"
    )
    last_synced_at: datetime | None = Field(default=None, description="Last sync timestamp")


class OntologyRelation(BaseModel):
    """Ontology relation matching database schema."""

    model_config = OPTIMIZED_CONFIG

    id: Identifier = Field(description="Unique ID for this relationship")
    ontology_id: Identifier = Field(description="FK to Ontology.id")
    parent_id: Identifier = Field(description="FK to source OntologyNode.id")
    child_id: Identifier = Field(description="FK to target OntologyNode.id")
    description: str = Field(default="", description="Human-readable explanation")
    last_synced_at: datetime | None = Field(default=None, description="Last sync timestamp")


# Query Models


class QueryIntent(BaseModel):
    """Parsed query intent."""

    model_config = OPTIMIZED_CONFIG

    original_query: NonEmptyStr = Field(description="Original query")
    intent_type: NonEmptyStr = Field(description="Query type")
    target_nodes: list[str] = Field(default_factory=list, description="Target ontology nodes")
    confidence: Confidence = Field(default=0.0, description="Parsing confidence")


class SQLQuery(BaseModel):
    """Generated SQL query."""

    model_config = OPTIMIZED_CONFIG

    query_id: QueryId = Field(description="Unique identifier")
    sql_text: NonEmptyStr = Field(description="SQL text")
    query_type: NonEmptyStr = Field(description="Query type")


# --- Bulk validation helper using TypeAdapter ---

# Pre-compiled adapters for bulk operations
_ontology_adapter = TypeAdapter(Ontology)
_node_adapter = TypeAdapter(OntologyNode)
_relation_adapter = TypeAdapter(OntologyRelation)
_query_intent_adapter = TypeAdapter(QueryIntent)
_sql_query_adapter = TypeAdapter(SQLQuery)


class BulkValidator:
    """Optimized bulk validation using cached TypeAdapters."""

    @staticmethod
    def validate_ontologies(data_list: list[dict]) -> list[Ontology]:
        """Validate multiple ontologies efficiently."""
        return [_ontology_adapter.validate_python(d) for d in data_list]

    @staticmethod
    def validate_nodes(data_list: list[dict]) -> list[OntologyNode]:
        """Validate multiple nodes efficiently."""
        return [_node_adapter.validate_python(d) for d in data_list]

    @staticmethod
    def validate_relations(data_list: list[dict]) -> list[OntologyRelation]:
        """Validate multiple relations efficiently."""
        return [_relation_adapter.validate_python(d) for d in data_list]


# --- Fast construction for trusted internal data ---


def construct_ontology_unsafe(data: dict) -> Ontology:
    """Construct Ontology without validation for trusted internal data.

    WARNING: Only use with pre-validated or trusted internal data!
    """
    return Ontology.model_construct(**data)


def construct_node_unsafe(data: dict) -> OntologyNode:
    """Construct OntologyNode without validation for trusted internal data.

    WARNING: Only use with pre-validated or trusted internal data!
    """
    return OntologyNode.model_construct(**data)
