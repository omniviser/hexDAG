"""Integration tests without patching - using real config files."""

import os
import tempfile
from pathlib import Path

import pytest

from hexai.core.bootstrap import bootstrap_registry
from hexai.core.registry import registry as global_registry


class TestBootstrapWithRealConfig:
    """Test bootstrap using actual config files instead of patches."""

    @pytest.fixture(autouse=True)
    def cleanup_registry(self):
        """Ensure registry is clean before and after each test."""
        if global_registry.ready:
            global_registry._cleanup_state()
        yield
        if global_registry.ready:
            global_registry._cleanup_state()

    def test_bootstrap_with_config_file(self):
        """Test bootstrap using a real TOML config file."""
        # Create a temporary config file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            config_content = """
modules = [
    "hexai.core.ports",
    "hexai.core.application.nodes"
]
plugins = [
    "hexai.adapters.mock"
]
dev_mode = true
"""
            f.write(config_content)
            config_path = f.name

        try:
            # Bootstrap with the config file
            bootstrap_registry(config_path)

            # Verify components are loaded
            components = global_registry.list_components()
            component_names = [c.name for c in components]

            # Check that expected components are present
            assert "llm" in component_names  # Port
            assert "llm_node" in component_names  # Node
            assert "mock_llm" in component_names  # Adapter from plugin

        finally:
            # Clean up the temp file
            Path(config_path).unlink()

    def test_bootstrap_with_env_specific_config(self):
        """Test bootstrap with environment-specific configuration."""
        # Create a temp directory for our project
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a pyproject.toml with hexdag config
            pyproject_path = Path(tmpdir) / "pyproject.toml"
            config_content = """
[tool.hexdag]
modules = ["hexai.core.ports"]
plugins = ["hexai.adapters.mock"]

[tool.hexdag.settings]
log_level = "DEBUG"
enable_metrics = false
"""

            with pyproject_path.open("w") as f:
                f.write(config_content)

            # Change to the temp directory and bootstrap
            original_cwd = Path.cwd()
            try:
                os.chdir(tmpdir)
                bootstrap_registry()  # Should auto-discover pyproject.toml

                # Verify it loaded correctly
                components = global_registry.list_components()
                adapter_names = [c.name for c in components if c.component_type.value == "adapter"]
                assert "mock_llm" in adapter_names

            finally:
                os.chdir(original_cwd)

    @pytest.mark.asyncio
    async def test_end_to_end_with_real_bootstrap(self):
        """Test end-to-end functionality with real bootstrap."""
        # Create config for a minimal but functional setup
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            config_content = """
modules = [
    "hexai.core.ports",
    "hexai.core.application.nodes"
]
plugins = [
    "hexai.adapters.mock",
    "hexai.adapters.database.sqlite"
]
"""
            f.write(config_content)
            config_path = f.name

        try:
            # Bootstrap the system
            bootstrap_registry(config_path)

            # Get components and test they work
            mock_llm = global_registry.get("mock_llm", namespace="plugin")
            sqlite = global_registry.get(
                "sqlite", namespace="plugin", init_params={"db_path": ":memory:"}
            )

            # Test LLM adapter
            from hexai.core.ports.llm import Message

            response = await mock_llm.aresponse([Message(role="user", content="test")])
            assert "Mock response" in response

            # Test database adapter
            await sqlite.aexecute_query("""
                CREATE TABLE test (id INTEGER PRIMARY KEY, data TEXT)
            """)
            await sqlite.aexecute_query(
                "INSERT INTO test (data) VALUES (:data)", {"data": "test_value"}
            )
            results = await sqlite.aexecute_query("SELECT * FROM test")
            assert len(results) == 1
            assert results[0]["data"] == "test_value"

        finally:
            Path(config_path).unlink()

    def test_plugin_availability_based_on_env(self):
        """Test that plugins are loaded based on environment variables."""
        # Create config with conditional plugins
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            config_content = """
modules = ["hexai.core.ports"]
plugins = [
    "hexai.adapters.mock",  # Always available
    "hexai.adapters.llm.openai_adapter",  # Requires OPENAI_API_KEY
    "hexai.adapters.llm.anthropic_adapter"  # Requires ANTHROPIC_API_KEY
]
"""
            f.write(config_content)
            config_path = f.name

        try:
            # Test 1: No API keys - only mock should be available
            env_backup = os.environ.copy()
            os.environ.pop("OPENAI_API_KEY", None)
            os.environ.pop("ANTHROPIC_API_KEY", None)

            if global_registry.ready:
                global_registry._cleanup_state()

            bootstrap_registry(config_path)
            adapters = [
                c.name
                for c in global_registry.list_components()
                if c.component_type.value == "adapter"
            ]
            assert "mock_llm" in adapters
            # Adapters are registered regardless of API keys
            # Validation happens at instantiation time
            assert "openai" in adapters
            assert "anthropic" in adapters

            # Test 2: With OpenAI key
            os.environ["OPENAI_API_KEY"] = "test-key"
            global_registry._cleanup_state()

            bootstrap_registry(config_path)
            adapters = [
                c.name
                for c in global_registry.list_components()
                if c.component_type.value == "adapter"
            ]
            assert "mock_llm" in adapters
            assert "openai" in adapters  # Always registered
            assert "anthropic" in adapters  # Always registered

            # Restore environment
            os.environ.clear()
            os.environ.update(env_backup)

        finally:
            Path(config_path).unlink()
