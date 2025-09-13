"""Tests for optimized domain models in agent_factory/models.py."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from hexai.agent_factory.models import (
    BulkValidator,
    Ontology,
    OntologyNode,
    OntologyRelation,
    QueryIntent,
    RelationshipType,
    SQLQuery,
    construct_node_unsafe,
    construct_ontology_unsafe,
)


class TestOntologyModels:
    """Test suite for Ontology-related models."""

    def test_ontology_with_annotated_constraints(self):
        """Test Ontology model with Annotated type constraints."""
        # Valid ontology
        ontology = Ontology(id=1, name="TestOntology", created_at=datetime.now(), tenant_id=100)
        assert ontology.id == 1
        assert ontology.name == "TestOntology"
        assert ontology.is_active is True  # Default value

        # Test validation errors with Annotated types
        with pytest.raises(ValidationError) as exc_info:
            Ontology(
                id=-1,  # Should fail: PositiveInt constraint
                name="Test",
                created_at=datetime.now(),
                tenant_id=100,
            )
        assert "greater than 0" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            Ontology(
                id=1,
                name="",  # Should fail: NonEmptyStr constraint
                tenant_id=100,
                created_at=datetime.now(),
            )
        assert "at least 1 character" in str(exc_info.value)

    def test_ontology_immutability(self):
        """Test that models are frozen (immutable)."""
        ontology = Ontology(
            id=1, name="ImmutableOntology", created_at=datetime.now(), tenant_id=100
        )

        # Pydantic v2 raises ValidationError for frozen models
        with pytest.raises(ValidationError) as exc_info:
            ontology.name = "NewName"
        assert "frozen" in str(exc_info.value)

    def test_ontology_node_with_roles(self):
        """Test OntologyNode with role validation."""
        node = OntologyNode(
            id=1, ontology_id=1, name="Customer", permissed_roles=["admin", "viewer"]
        )
        assert len(node.permissed_roles) == 2
        assert "admin" in node.permissed_roles

    def test_ontology_relation(self):
        """Test OntologyRelation model."""
        relation = OntologyRelation(
            id=1, ontology_id=1, parent_id=1, child_id=2, description="Parent-child relationship"
        )
        assert relation.parent_id == 1
        assert relation.child_id == 2
        assert relation.description == "Parent-child relationship"

    def test_relationship_type_enum(self):
        """Test RelationshipType enum."""
        assert RelationshipType.ONE_TO_MANY == "one_to_many"
        assert RelationshipType.MANY_TO_MANY == "many_to_many"


class TestQueryModels:
    """Test suite for Query-related models."""

    def test_query_intent_confidence_validation(self):
        """Test QueryIntent with Confidence constraint."""
        # Valid confidence
        query = QueryIntent(original_query="Find customers", intent_type="SELECT", confidence=0.95)
        assert query.confidence == 0.95

        # Invalid confidence (> 1.0)
        with pytest.raises(ValidationError):
            QueryIntent(
                original_query="Find customers",
                intent_type="SELECT",
                confidence=1.5,  # Should fail: > 1.0
            )

        # Invalid confidence (< 0.0)
        with pytest.raises(ValidationError):
            QueryIntent(
                original_query="Find customers",
                intent_type="SELECT",
                confidence=-0.1,  # Should fail: < 0.0
            )

    def test_sql_query_id_pattern(self):
        """Test SQLQuery with QueryId pattern constraint."""
        # Valid query ID
        query = SQLQuery(
            query_id="query-123", sql_text="SELECT * FROM customers", query_type="SELECT"
        )
        assert query.query_id == "query-123"

        # Invalid query ID pattern
        with pytest.raises(ValidationError):
            SQLQuery(
                query_id="query@#$",  # Invalid characters
                sql_text="SELECT * FROM customers",
                query_type="SELECT",
            )

    def test_empty_string_validation(self):
        """Test that NonEmptyStr fields reject empty strings."""
        with pytest.raises(ValidationError):
            QueryIntent(
                original_query="",
                intent_type="SELECT",
                confidence=0.5,  # Empty string should fail
            )


class TestBulkOperations:
    """Test suite for bulk validation operations."""

    def test_bulk_ontology_validation(self):
        """Test BulkValidator for efficient batch processing of ontologies."""
        # Create test data
        ontology_data = [
            {
                "id": i,
                "name": f"Ontology{i}",
                "created_at": datetime.now().isoformat(),
                "tenant_id": 100,
                "is_active": True,
            }
            for i in range(1, 11)
        ]

        # Bulk validate
        ontologies = BulkValidator.validate_ontologies(ontology_data)
        assert len(ontologies) == 10
        assert all(isinstance(o, Ontology) for o in ontologies)
        assert ontologies[0].id == 1
        assert ontologies[9].id == 10

    def test_bulk_node_validation(self):
        """Test bulk validation of OntologyNode objects."""
        node_data = [
            {"id": i, "ontology_id": 1, "name": f"Node{i}", "permissed_roles": ["viewer"]}
            for i in range(1, 6)
        ]

        nodes = BulkValidator.validate_nodes(node_data)
        assert len(nodes) == 5
        assert all(isinstance(n, OntologyNode) for n in nodes)
        assert nodes[0].name == "Node1"

    def test_bulk_relation_validation(self):
        """Test bulk validation of OntologyRelation objects."""
        relation_data = [
            {
                "id": i,
                "ontology_id": 1,
                "parent_id": i,
                "child_id": i + 1,
                "description": f"Relation {i}",
            }
            for i in range(1, 4)
        ]

        relations = BulkValidator.validate_relations(relation_data)
        assert len(relations) == 3
        assert all(isinstance(r, OntologyRelation) for r in relations)


class TestUnsafeConstruction:
    """Test suite for unsafe construction methods."""

    def test_construct_ontology_unsafe(self):
        """Test model_construct for trusted data paths."""
        # This bypasses validation for performance
        data = {
            "id": 1,
            "name": "FastOntology",
            "is_active": True,
            "created_at": datetime.now(),
            "tenant_id": 100,
        }

        # Fast construction without validation
        ontology = construct_ontology_unsafe(data)
        assert ontology.id == 1
        assert ontology.name == "FastOntology"

        # Note: Invalid data would pass through unchecked
        invalid_data = {
            "id": -1,  # Invalid but not caught
            "name": "",  # Invalid but not caught
            "is_active": True,
            "created_at": datetime.now(),
            "tenant_id": 100,
        }
        # This won't raise an error with model_construct
        ontology = construct_ontology_unsafe(invalid_data)
        assert ontology.id == -1  # Invalid data passed through

    def test_construct_node_unsafe(self):
        """Test unsafe construction of OntologyNode."""
        data = {"id": 1, "ontology_id": 1, "name": "FastNode", "permissed_roles": []}

        node = construct_node_unsafe(data)
        assert node.id == 1
        assert node.name == "FastNode"


class TestModelConfiguration:
    """Test model configuration settings."""

    def test_string_stripping(self):
        """Test automatic string stripping from str_strip_whitespace config."""
        query = QueryIntent(
            original_query="  Find customers  ",  # Has whitespace
            intent_type="  SELECT  ",
            confidence=0.9,
        )
        # Strings should be automatically stripped
        assert query.original_query == "Find customers"
        assert query.intent_type == "SELECT"

    def test_frozen_model_hashability(self):
        """Test that frozen models are hashable."""
        ontology1 = Ontology(id=1, name="Test", created_at=datetime(2024, 1, 1), tenant_id=100)
        ontology2 = Ontology(id=1, name="Test", created_at=datetime(2024, 1, 1), tenant_id=100)

        # Frozen models should be hashable
        assert hash(ontology1) == hash(ontology2)

        # Can use in sets/dicts
        ontology_set = {ontology1, ontology2}
        assert len(ontology_set) == 1  # Same hash, so only one item

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden due to extra='forbid' config."""
        with pytest.raises(ValidationError) as exc_info:
            Ontology(
                id=1,
                name="Test",
                created_at=datetime.now(),
                tenant_id=100,
                unknown_field="should_fail",  # Extra field
            )
        assert "Extra inputs are not permitted" in str(exc_info.value)
