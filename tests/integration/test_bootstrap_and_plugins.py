"""Integration tests for the complete HexDAG system.

These tests verify that all components work together correctly:
- Registry bootstrap with plugins
- Orchestrator with registered nodes
- Event system integration
- YAML pipeline execution
- End-to-end workflows

Setup Instructions:
-------------------
For SQLite tests:
  No special setup required - SQLite is a built-in adapter.

For MySQL tests (if applicable):
  uv pip install -e hexai_plugins/mysql_adapter/

For LLM adapter tests:
  - OpenAI: Set OPENAI_API_KEY environment variable
  - Anthropic: Set ANTHROPIC_API_KEY environment variable
"""

import asyncio
import os
from unittest.mock import patch

import pytest

from hexai.core.bootstrap import bootstrap_registry, ensure_bootstrapped
from hexai.core.config.models import HexDAGConfig
from hexai.core.registry import registry as global_registry


class TestSystemBootstrap:
    """Test system initialization and bootstrap."""

    @pytest.fixture(autouse=True)
    def cleanup_registry(self):
        """Ensure registry is clean before and after each test."""
        if global_registry.ready:
            global_registry._cleanup_state()
        yield
        if global_registry.ready:
            global_registry._cleanup_state()

    def test_bootstrap_loads_core_components(self):
        """Test that bootstrap loads all core components."""
        config = HexDAGConfig(
            modules=[
                "hexai.core.ports",
                "hexai.core.application.nodes",
            ],
            plugins=[],
        )

        with patch("hexai.core.bootstrap.load_config", return_value=config):
            bootstrap_registry()

        # Verify core components are loaded
        components = global_registry.list_components()

        # Check ports are registered
        port_names = [c.name for c in components if c.component_type.value == "port"]
        assert "llm" in port_names
        assert "memory" in port_names
        assert "database" in port_names

        # Check nodes are registered
        node_names = [c.name for c in components if c.component_type.value == "node"]
        assert "llm_node" in node_names
        assert "function_node" in node_names
        assert "agent_node" in node_names

    def test_bootstrap_with_plugins(self):
        """Test bootstrap with plugins that have met requirements."""
        config = HexDAGConfig(
            modules=["hexai.core.ports"],
            plugins=[
                "hexai.adapters.mock",
                "hexai.adapters.llm.openai_adapter",
            ],
        )

        # Mock OpenAI as available
        with (
            patch("hexai.core.bootstrap.load_config", return_value=config),
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
        ):
            bootstrap_registry()

        components = global_registry.list_components()
        adapter_names = [c.name for c in components if c.component_type.value == "adapter"]

        # Mock should always be available
        assert "mock_llm" in adapter_names
        # OpenAI should be available with API key
        assert "openai" in adapter_names

    def test_ensure_bootstrapped_idempotent(self):
        """Test that ensure_bootstrapped can be called multiple times safely."""
        config = HexDAGConfig(modules=["hexai.core.ports"], plugins=["hexai.adapters.mock"])

        with patch("hexai.core.bootstrap.load_config", return_value=config):
            # First call bootstraps
            ensure_bootstrapped()
            first_count = len(global_registry.list_components())

            # Second call should not re-bootstrap
            ensure_bootstrapped()
            second_count = len(global_registry.list_components())

            assert first_count == second_count
            assert global_registry.ready


class TestPluginSystemIntegration:
    """Test plugin system with real adapter usage."""

    @pytest.mark.asyncio
    async def test_plugin_adapter_selection(self):
        """Test that correct adapter is selected based on availability."""
        # Test with no API keys - should use mock
        if global_registry.ready:
            global_registry._cleanup_state()

        config = HexDAGConfig(
            modules=["hexai.core.ports"],
            plugins=[
                "hexai.adapters.llm.openai_adapter",
                "hexai.adapters.mock",
            ],
        )

        with (
            patch("hexai.core.bootstrap.load_config", return_value=config),
            patch.dict(os.environ, {}, clear=True),
        ):
            bootstrap_registry()

        # Only mock should be available
        adapters = global_registry.list_components()
        adapter_names = [a.name for a in adapters if a.component_type.value == "adapter"]
        assert "mock_llm" in adapter_names
        assert "openai" not in adapter_names

        # Get adapter and test it works
        mock_adapter = global_registry.get("mock_llm", namespace="plugin")
        from hexai.core.ports.llm import Message

        response = await mock_adapter.aresponse([Message(role="user", content="What is 2+2?")])
        assert "Mock response" in response

    @pytest.mark.asyncio
    async def test_multiple_adapters_concurrent(self):
        """Test multiple adapters working concurrently."""
        if global_registry.ready:
            global_registry._cleanup_state()

        config = HexDAGConfig(modules=["hexai.core.ports"], plugins=["hexai.adapters.mock"])

        with patch("hexai.core.bootstrap.load_config", return_value=config):
            bootstrap_registry()

        # Create multiple mock adapter instances
        adapter1 = global_registry.get("mock_llm", namespace="plugin")
        adapter2 = global_registry.get("mock_llm", namespace="plugin")

        from hexai.core.ports.llm import Message

        # Run concurrent requests
        async def make_request(adapter, query):
            return await adapter.aresponse([Message(role="user", content=query)])

        results = await asyncio.gather(
            make_request(adapter1, "Query 1"), make_request(adapter2, "Query 2")
        )

        assert len(results) == 2
        assert all("Mock response" in r for r in results)

    def test_database_adapters_registration(self):
        """Test that database adapters (SQLite and MySQL) are properly registered."""
        if global_registry.ready:
            global_registry._cleanup_state()
        # Import MySQL adapter if available
        config = HexDAGConfig(
            modules=[
                "hexai.core.ports",
            ],
            plugins=[
                "hexai.adapters.database.sqlite",  # SQLite database adapter
            ],
        )

        with patch("hexai.core.bootstrap.load_config", return_value=config):
            bootstrap_registry()

        components = global_registry.list_components()
        adapter_names = [c.name for c in components if c.component_type.value == "adapter"]

        # SQLite should be registered as a database adapter
        assert "sqlite" in adapter_names

        # Verify SQLite can be retrieved
        sqlite_info = global_registry.get_info("sqlite", namespace="plugin")
        assert sqlite_info.name == "sqlite"
        assert sqlite_info.namespace == "plugin"
        assert sqlite_info.component_type.value == "adapter"

    @pytest.mark.asyncio
    async def test_sqlite_adapter_functionality(self):
        """Test SQLite adapter basic functionality."""
        if global_registry.ready:
            global_registry._cleanup_state()

        config = HexDAGConfig(
            modules=[
                "hexai.core.ports",
            ],
            plugins=[
                "hexai.adapters.database.sqlite",
            ],
        )

        with patch("hexai.core.bootstrap.load_config", return_value=config):
            bootstrap_registry()

        # Get SQLite adapter
        sqlite = global_registry.get(
            "sqlite", namespace="plugin", init_params={"db_path": ":memory:"}
        )

        # Test basic operations
        collection = "test_collection"

        # Insert a document
        doc_id = await sqlite.ainsert(collection, {"name": "Test", "value": 42})
        assert doc_id is not None

        # Retrieve the document
        doc = await sqlite.aget(collection, doc_id)
        assert doc["name"] == "Test"
        assert doc["value"] == 42

        # Update the document
        success = await sqlite.aupdate(collection, doc_id, {"status": "updated"})
        assert success

        # Query documents
        docs = await sqlite.aquery(collection)
        assert len(docs) == 1
        assert docs[0]["status"] == "updated"

        # Delete the document
        success = await sqlite.adelete(collection, doc_id)
        assert success

        # Verify deletion
        doc = await sqlite.aget(collection, doc_id)
        assert doc is None
