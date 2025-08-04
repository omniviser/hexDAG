"""Domain models for pipeline applications.

This module contains data models specific to pipeline applications, particularly for ontology
analytics and text-to-SQL use cases.
"""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

# Ontology Analytics Domain Models - Aligned with Database Schema


class RelationshipType(str, Enum):
    """Basic relationship types."""

    ONE_TO_ONE = "one_to_one"
    ONE_TO_MANY = "one_to_many"
    MANY_TO_ONE = "many_to_one"
    MANY_TO_MANY = "many_to_many"


class Ontology(BaseModel):
    """Ontology definition matching database schema."""

    id: int = Field(..., description="Unique identifier for the ontology")
    name: str = Field(..., description="Name of the ontology")
    is_active: bool = Field(True, description="Boolean flag if this ontology is in use")
    created_at: datetime = Field(..., description="When the ontology was created")
    tenant_id: int = Field(..., description="Foreign key to the tenant or organization")
    last_synced_at: datetime | None = Field(None, description="Last sync timestamp")


class OntologyNode(BaseModel):
    """Ontology node (business concept) matching database schema."""

    id: int = Field(..., description="Unique ID for the ontology node")
    ontology_id: int = Field(..., description="FK to Ontology.id")
    name: str = Field(..., description="Business name of the node")
    alias: str | None = Field(None, description="Alias for the node")
    permissed_roles: list[str] = Field(
        default_factory=list, description="Roles allowed to access this node"
    )
    last_synced_at: datetime | None = Field(None, description="Last sync timestamp")


class OntologyRelation(BaseModel):
    """Ontology relation matching database schema."""

    id: int = Field(..., description="Unique ID for this relationship")
    ontology_id: int = Field(..., description="FK to Ontology.id")
    parent_id: int = Field(..., description="FK to source OntologyNode.id")
    child_id: int = Field(..., description="FK to target OntologyNode.id")
    description: str = Field("", description="Human-readable explanation")
    last_synced_at: datetime | None = Field(None, description="Last sync timestamp")


# Query Models


class QueryIntent(BaseModel):
    """Parsed query intent."""

    original_query: str = Field(..., description="Original query")
    intent_type: str = Field(..., description="Query type")
    target_nodes: list[str] = Field(default_factory=list, description="Target ontology nodes")
    confidence: float = Field(0.0, description="Parsing confidence")


class SQLQuery(BaseModel):
    """Generated SQL query."""

    query_id: str = Field(..., description="Unique identifier")
    sql_text: str = Field(..., description="SQL text")
    query_type: str = Field(..., description="Query type")
