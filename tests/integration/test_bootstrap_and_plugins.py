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
            global_registry._reset_for_testing()
        yield
        if global_registry.ready:
            global_registry._reset_for_testing()

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
            global_registry._reset_for_testing()

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

        # Both adapters should be registered (registration happens regardless of API keys)
        adapters = global_registry.list_components()
        adapter_names = [a.name for a in adapters if a.component_type.value == "adapter"]
        assert "mock_llm" in adapter_names
        assert "openai" in adapter_names

        # Get adapter and test it works
        mock_adapter = global_registry.get("mock_llm", namespace="plugin")
        from hexai.core.ports.llm import Message

        response = await mock_adapter.aresponse([Message(role="user", content="What is 2+2?")])
        assert "Mock response" in response

    @pytest.mark.asyncio
    async def test_multiple_adapters_concurrent(self):
        """Test multiple adapters working concurrently."""
        if global_registry.ready:
            global_registry._reset_for_testing()

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
            global_registry._reset_for_testing()
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
            global_registry._reset_for_testing()

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

        # Test DatabasePort interface

        # Create a test table
        await sqlite.aexecute_query("""
            CREATE TABLE test_table (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                value INTEGER
            )
        """)

        # Insert data using parameterized query
        await sqlite.aexecute_query(
            "INSERT INTO test_table (name, value) VALUES (:name, :value)",
            {"name": "Test", "value": 42},
        )

        # Query data
        results = await sqlite.aexecute_query(
            "SELECT * FROM test_table WHERE name = :name", {"name": "Test"}
        )
        assert len(results) == 1
        assert results[0]["name"] == "Test"
        assert results[0]["value"] == 42

        # Test schema introspection
        schemas = await sqlite.aget_table_schemas()
        assert "test_table" in schemas
        assert schemas["test_table"]["columns"]["name"] == "TEXT"
        assert schemas["test_table"]["columns"]["value"] == "INTEGER"

        # Test table statistics
        stats = await sqlite.aget_table_statistics()
        assert "test_table" in stats
        assert stats["test_table"]["row_count"] == 1

        # Cleanup: close the adapter connection
        await sqlite.close()
