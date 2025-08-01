"""Tests for MockOntologyPort with simplified schema."""

from hexai.adapters.mock import MockOntologyPort


class TestMockOntologyPort:
    """Test the MockOntologyPort implementation."""

    def setup_method(self):
        """Set up test instance."""
        self.mock_ontology = MockOntologyPort()

    def test_get_ontologies(self):
        """Test getting ontologies."""
        ontologies = self.mock_ontology.get_ontologies()

        assert len(ontologies) == 1
        ontology = ontologies[0]

        assert ontology["id"] == 1
        assert ontology["name"] == "Bart Ontology"
        assert ontology["is_active"] is True
        assert ontology["tenant_id"] == 1
        assert "created_at" in ontology
        assert "last_synced_at" in ontology

    def test_get_ontology_nodes(self):
        """Test getting ontology nodes."""
        nodes = self.mock_ontology.get_ontology_nodes()

        assert len(nodes) == 4

        # Check customer node
        customer_node = next(node for node in nodes if node["name"] == "Customer")
        assert customer_node["id"] == 1
        assert customer_node["ontology_id"] == 1
        assert customer_node["alias"] == "Cust"
        assert "analyst" in customer_node["permissed_roles"]

        # Check order node
        order_node = next(node for node in nodes if node["name"] == "Order")
        assert order_node["id"] == 2
        assert order_node["alias"] == "Ord"

    def test_get_ontology_nodes_filtered(self):
        """Test getting ontology nodes filtered by ontology ID."""
        nodes = self.mock_ontology.get_ontology_nodes(ontology_id=1)

        assert len(nodes) == 4
        for node in nodes:
            assert node["ontology_id"] == 1

        # Test with non-existent ontology ID
        nodes = self.mock_ontology.get_ontology_nodes(ontology_id=999)
        assert len(nodes) == 0

    def test_get_ontology_relations(self):
        """Test getting ontology relations."""
        relations = self.mock_ontology.get_ontology_relations()

        assert len(relations) == 2

        # Check customer-order relation
        customer_order_rel = next(
            rel for rel in relations if rel["parent_id"] == 1 and rel["child_id"] == 2
        )
        assert customer_order_rel["id"] == 1
        assert customer_order_rel["ontology_id"] == 1
        assert customer_order_rel["description"] == "Customer places orders"

        # Check order-product relation
        order_product_rel = next(
            rel for rel in relations if rel["parent_id"] == 2 and rel["child_id"] == 3
        )
        assert order_product_rel["id"] == 2
        assert order_product_rel["description"] == "Orders contain products"

    def test_get_ontology_relations_filtered(self):
        """Test getting ontology relations filtered by ontology ID."""
        relations = self.mock_ontology.get_ontology_relations(ontology_id=1)

        assert len(relations) == 2
        for relation in relations:
            assert relation["ontology_id"] == 1

        # Test with non-existent ontology ID
        relations = self.mock_ontology.get_ontology_relations(ontology_id=999)
        assert len(relations) == 0

    def test_get_context(self):
        """Test getting context for query understanding."""
        # Test with customer query
        context = self.mock_ontology.get_context("Show me customer data")

        assert len(context) == 1
        assert context[0]["type"] == "ontology_node"
        assert context[0]["name"] == "Customer"
        assert context[0]["relevance"] == 0.8

        # Test with order query
        context = self.mock_ontology.get_context("Order information")

        assert len(context) == 1
        assert context[0]["type"] == "ontology_node"
        assert context[0]["name"] == "Order"

        # Test with no matches
        context = self.mock_ontology.get_context("Unrelated query")
        assert len(context) == 0

    def test_node_id_consistency(self):
        """Test that node IDs are consistent across calls."""
        nodes1 = self.mock_ontology.get_ontology_nodes()
        nodes2 = self.mock_ontology.get_ontology_nodes()

        # Should get same data
        assert len(nodes1) == len(nodes2)

        for node1, node2 in zip(nodes1, nodes2):
            assert node1["id"] == node2["id"]
            assert node1["name"] == node2["name"]

    def test_relation_references_valid_nodes(self):
        """Test that relations reference valid node IDs."""
        nodes = self.mock_ontology.get_ontology_nodes()
        relations = self.mock_ontology.get_ontology_relations()

        node_ids = {node["id"] for node in nodes}

        for relation in relations:
            assert relation["parent_id"] in node_ids
            assert relation["child_id"] in node_ids

    def test_data_structure_integrity(self):
        """Test overall data structure integrity."""
        ontologies = self.mock_ontology.get_ontologies()
        nodes = self.mock_ontology.get_ontology_nodes()
        relations = self.mock_ontology.get_ontology_relations()

        # All nodes should reference the ontology
        ontology_id = ontologies[0]["id"]
        for node in nodes:
            assert node["ontology_id"] == ontology_id

        # All relations should reference the ontology
        for relation in relations:
            assert relation["ontology_id"] == ontology_id

        # Relations should form valid connections
        for relation in relations:
            parent_exists = any(node["id"] == relation["parent_id"] for node in nodes)
            child_exists = any(node["id"] == relation["child_id"] for node in nodes)
            assert parent_exists
            assert child_exists


class TestMockOntologyPortEdgeCases:
    """Test edge cases and error conditions."""

    def setup_method(self):
        """Set up test instance."""
        self.mock_ontology = MockOntologyPort()

    def test_empty_query_context(self):
        """Test context with empty query."""
        context = self.mock_ontology.get_context("")
        assert len(context) == 0

    def test_case_insensitive_context(self):
        """Test that context matching is case insensitive."""
        context_lower = self.mock_ontology.get_context("customer")
        context_upper = self.mock_ontology.get_context("CUSTOMER")
        context_mixed = self.mock_ontology.get_context("Customer")

        assert len(context_lower) == 1
        assert len(context_upper) == 1
        assert len(context_mixed) == 1

        for context in [context_lower, context_upper, context_mixed]:
            assert context[0]["name"] == "Customer"

    def test_data_immutability(self):
        """Test that returned data cannot modify internal state."""
        nodes = self.mock_ontology.get_ontology_nodes()
        original_count = len(nodes)

        # Modify returned data
        nodes.append({"id": 999, "name": "Test"})

        # Should not affect subsequent calls
        new_nodes = self.mock_ontology.get_ontology_nodes()
        assert len(new_nodes) == original_count
