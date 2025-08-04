"""Tests for ontology models with simplified schema."""

from datetime import datetime

from hexai.pipelines.models import Ontology, OntologyNode, OntologyRelation, RelationshipType
import pytest


class TestOntology:
    """Test the Ontology model."""

    def test_ontology_creation(self):
        """Test creating an ontology with integer ID."""
        ontology = Ontology(
            id=1, name="Test Ontology", is_active=True, created_at=datetime.now(), tenant_id=1
        )

        assert ontology.id == 1
        assert ontology.name == "Test Ontology"
        assert ontology.is_active is True
        assert ontology.tenant_id == 1
        assert ontology.last_synced_at is None

    def test_ontology_with_sync_time(self):
        """Test ontology with last_synced_at."""
        sync_time = datetime.now()
        ontology = Ontology(
            id=2,
            name="Synced Ontology",
            is_active=False,
            created_at=datetime.now(),
            tenant_id=2,
            last_synced_at=sync_time,
        )

        assert ontology.last_synced_at == sync_time
        assert ontology.is_active is False


class TestOntologyNode:
    """Test the OntologyNode model."""

    def test_node_creation(self):
        """Test creating an ontology node."""
        node = OntologyNode(
            id=1,
            ontology_id=1,
            name="Customer",
            alias="Cust",
            permissed_roles=["analyst", "manager"],
        )

        assert node.id == 1
        assert node.ontology_id == 1
        assert node.name == "Customer"
        assert node.alias == "Cust"
        assert node.permissed_roles == ["analyst", "manager"]

    def test_node_without_alias(self):
        """Test node creation without alias."""
        node = OntologyNode(id=2, ontology_id=1, name="Order")

        assert node.alias is None
        assert node.permissed_roles == []

    def test_node_validation(self):
        """Test node field validation."""
        with pytest.raises(ValueError):
            # Missing required fields
            OntologyNode()


class TestOntologyRelation:
    """Test the OntologyRelation model."""

    def test_relation_creation(self):
        """Test creating an ontology relation."""
        relation = OntologyRelation(
            id=1, ontology_id=1, parent_id=1, child_id=2, description="Customer places orders"
        )

        assert relation.id == 1
        assert relation.ontology_id == 1
        assert relation.parent_id == 1
        assert relation.child_id == 2
        assert relation.description == "Customer places orders"

    def test_relation_without_description(self):
        """Test relation creation without description."""
        relation = OntologyRelation(id=2, ontology_id=1, parent_id=2, child_id=3)

        assert relation.description == ""


class TestRelationshipType:
    """Test the RelationshipType enum."""

    def test_relationship_types(self):
        """Test all relationship type values."""
        assert RelationshipType.ONE_TO_ONE == "one_to_one"
        assert RelationshipType.ONE_TO_MANY == "one_to_many"
        assert RelationshipType.MANY_TO_ONE == "many_to_one"
        assert RelationshipType.MANY_TO_MANY == "many_to_many"

    def test_relationship_type_validation(self):
        """Test that invalid relationship types are rejected."""
        valid_types = [
            RelationshipType.ONE_TO_ONE,
            RelationshipType.ONE_TO_MANY,
            RelationshipType.MANY_TO_ONE,
            RelationshipType.MANY_TO_MANY,
        ]

        for rel_type in valid_types:
            assert isinstance(rel_type, str)


class TestModelIntegration:
    """Test model integration and relationships."""

    def test_complete_ontology_setup(self):
        """Test creating a complete ontology with nodes and relations."""
        # Create ontology
        ontology = Ontology(
            id=1, name="Business Ontology", is_active=True, created_at=datetime.now(), tenant_id=1
        )

        # Create nodes
        customer_node = OntologyNode(id=1, ontology_id=ontology.id, name="Customer", alias="Cust")

        order_node = OntologyNode(id=2, ontology_id=ontology.id, name="Order", alias="Ord")

        # Create relation
        relation = OntologyRelation(
            id=1,
            ontology_id=ontology.id,
            parent_id=customer_node.id,
            child_id=order_node.id,
            description="Customer places orders",
        )

        # Verify relationships
        assert customer_node.ontology_id == ontology.id
        assert order_node.ontology_id == ontology.id
        assert relation.ontology_id == ontology.id
        assert relation.parent_id == customer_node.id
        assert relation.child_id == order_node.id

    def test_model_serialization(self):
        """Test that models can be serialized to dict."""
        node = OntologyNode(id=1, ontology_id=1, name="Product", permissed_roles=["analyst"])

        data = node.model_dump()

        assert data["id"] == 1
        assert data["ontology_id"] == 1
        assert data["name"] == "Product"
        assert data["permissed_roles"] == ["analyst"]
        assert "last_synced_at" in data
