"""Tests for the new bootstrap-based registry architecture."""

import os
import sys
import tempfile
import textwrap

import pytest

from hexai.core.bootstrap import bootstrap_registry
from hexai.core.config import ManifestEntry
from hexai.core.registry.decorators import node
from hexai.core.registry.exceptions import RegistryAlreadyBootstrappedError, RegistryImmutableError
from hexai.core.registry.models import ComponentType
from hexai.core.registry.registry import ComponentRegistry


class TestBootstrapArchitecture:
    """Test the new bootstrap-based registry lifecycle."""

    def test_registry_starts_empty(self):
        """Registry should start with no components."""
        registry = ComponentRegistry()
        assert not registry.ready
        assert registry.manifest is None
        assert len(registry.list_components()) == 0

    def test_decorators_dont_register(self):
        """Decorators should only add metadata, not register."""
        registry = ComponentRegistry()

        @node(name="test_node")
        class TestNode:
            """A test node."""

            pass

        # Decorator should add metadata
        assert hasattr(TestNode, "__hexdag_metadata__")
        assert TestNode.__hexdag_metadata__.name == "test_node"
        assert TestNode.__hexdag_metadata__.type == "node"

        # But should NOT register in registry
        assert len(registry.list_components()) == 0
        from hexai.core.registry.registry import ComponentNotFoundError

        with pytest.raises(ComponentNotFoundError):
            registry.get("test_node")

    def test_bootstrap_from_manifest(self):
        """Bootstrap should populate registry from manifest."""
        registry = ComponentRegistry()

        # Create entries pointing to our sample module
        entries = [
            ManifestEntry(namespace="test", module="tests.hexai.core.registry.sample_components")
        ]

        # Bootstrap
        registry.bootstrap(entries, dev_mode=True)

        # Registry should now be ready
        assert registry.ready
        assert len(registry.manifest) == 1

        # Components should be registered (3 decorated + 1 port = 4 total)
        components = registry.list_components()
        # Filter out the port since list_components might include it
        non_port_components = [c for c in components if c.component_type != ComponentType.PORT]
        assert len(non_port_components) == 3  # sample_node, sample_tool, sample_adapter

        # Check specific components
        node_names = {c.name for c in non_port_components}
        assert "sample_node" in node_names
        assert "sample_tool" in node_names
        assert "sample_adapter" in node_names

        # All should be in test namespace
        assert all(c.namespace == "test" for c in components)

    def test_no_registration_after_bootstrap_prod(self):
        """In production mode, registration after bootstrap should fail."""
        registry = ComponentRegistry()

        # Bootstrap with empty entries
        entries: list[ManifestEntry] = []
        registry.bootstrap(entries, dev_mode=False)

        assert registry.ready

        # Try to register after bootstrap - should fail
        with pytest.raises(RegistryImmutableError) as exc_info:
            registry.register(
                name="late_component",
                component=lambda: None,
                component_type="tool",
                namespace="user",
            )

        assert "read-only in production mode" in str(exc_info.value)

    def test_registration_allowed_in_dev_mode(self):
        """In dev mode, registration after bootstrap should work."""
        registry = ComponentRegistry()

        # Bootstrap with dev_mode=True
        entries: list[ManifestEntry] = []
        registry.bootstrap(entries, dev_mode=True)

        assert registry.ready

        # Registration should work in dev mode
        registry.register(
            name="dev_component",
            component=lambda: None,
            component_type="tool",
            namespace="user",
        )

        components = registry.list_components()
        assert len(components) == 1
        assert components[0].name == "dev_component"

    def test_bootstrap_idempotent(self):
        """Bootstrap should be idempotent - can't bootstrap twice."""
        registry = ComponentRegistry()

        entries: list[ManifestEntry] = []
        registry.bootstrap(entries)

        # Second bootstrap should fail
        with pytest.raises(RegistryAlreadyBootstrappedError) as exc_info:
            registry.bootstrap(entries)

        assert "already been bootstrapped" in str(exc_info.value)

    def test_reset_allows_re_bootstrap(self):
        """Reset should allow re-bootstrapping."""
        # This test requires a reset() method which doesn't exist
        # Skipping for now
        pytest.skip("reset() method not implemented")

    def test_manifest_validation(self):
        """Manifest should validate for duplicates."""
        registry = ComponentRegistry()
        with pytest.raises(ValueError) as exc_info:
            entries = [
                ManifestEntry(namespace="test", module="module1"),
                ManifestEntry(namespace="test", module="module1"),  # Duplicate
            ]
            registry.bootstrap(entries)

        assert "Duplicate manifest entry" in str(exc_info.value)

    def test_adapter_validation_during_bootstrap(self):
        """Test that adapter validation happens during bootstrap."""
        # Create a test file with an invalid adapter (missing required method)
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a module with an invalid adapter
            module_path = os.path.join(tmpdir, "invalid_adapter.py")
            with open(module_path, "w") as f:
                f.write(
                    textwrap.dedent("""
                    from hexai.core.registry.decorators import adapter
                    from hexai.core.registry.models import (
                        ClassComponent, ComponentMetadata, ComponentType, PortMetadata
                    )

                    # Define a port
                    class TestPort:
                        def required_method(self): pass

                    # Invalid adapter - missing required_method
                    @adapter(implements_port="test_port", name="invalid_adapter")
                    class InvalidAdapter:
                        pass  # Missing required_method!

                    def register_components(registry, namespace):
                        # Register the port first
                        port_meta = ComponentMetadata(
                            name="test_port",
                            component_type=ComponentType.PORT,
                            component=ClassComponent(value=TestPort),
                            namespace=namespace,
                            port_metadata=PortMetadata(
                                protocol_class=TestPort,
                                required_methods=["required_method"],
                                optional_methods=[],
                            ),
                        )
                        if namespace not in registry._components:
                            registry._components[namespace] = {}
                        registry._components[namespace]["test_port"] = port_meta

                        # Try to register the invalid adapter
                        from hexai.core.registry.discovery import discover_components
                        import sys
                        module = sys.modules[__name__]
                        components = discover_components(module)

                        for _, component in components:
                            if hasattr(component, "__hexdag_metadata__"):
                                metadata = getattr(component, "__hexdag_metadata__")
                                registry.register(
                                    name=metadata.name,
                                    component=component,
                                    component_type=metadata.type,
                                    namespace=namespace,
                                    adapter_metadata=getattr(metadata, "adapter_metadata", None),
                                )
                        return 1
                """)
                )

            # Add tmpdir to sys.path
            sys.path.insert(0, tmpdir)
            try:
                registry = ComponentRegistry()
                entries = [ManifestEntry(namespace="test", module="invalid_adapter")]

                # Bootstrap should fail due to adapter validation
                with pytest.raises(Exception) as exc_info:
                    registry.bootstrap(entries, dev_mode=True)

                # Should fail with InvalidComponentError mentioning missing method
                assert "does not implement required methods" in str(exc_info.value)
                assert "required_method" in str(exc_info.value)
            finally:
                # Clean up
                sys.path.remove(tmpdir)
                if "invalid_adapter" in sys.modules:
                    del sys.modules["invalid_adapter"]

    def test_import_error_rollback(self):
        """Failed imports should rollback bootstrap."""
        registry = ComponentRegistry()

        # Create entries with non-existent module
        entries = [ManifestEntry(namespace="test", module="non_existent_module_xyz")]

        # Bootstrap should fail and rollback
        with pytest.raises(ImportError):
            registry.bootstrap(entries)

        # Registry should be rolled back to empty state
        assert not registry.ready
        assert registry.manifest is None
        assert len(registry.list_components()) == 0


class TestTOMLBootstrap:
    """Test bootstrap functionality with TOML configuration."""

    def setup_method(self):
        """Reset registry before each test."""
        from hexai.core.registry import registry

        # Clear registry if it's bootstrapped
        if registry.ready:
            registry._components.clear()
            registry._protected_components.clear()
            registry._ready = False
            registry._manifest = None
            registry._bootstrap_context = False

    def test_bootstrap_from_toml(self):
        """Test bootstrapping from TOML configuration."""
        from hexai.core.registry import registry

        config_content = """
modules = [
    "tests.hexai.core.registry.sample_components",
]

dev_mode = true

[bindings]
llm = "mock_llm"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config_content)
            config_path = f.name

        try:
            bootstrap_registry(config_path)
            assert registry.ready
            assert registry.dev_mode

            # Check that components were loaded
            components = registry.list_components()
            assert len(components) > 0
        finally:
            os.unlink(config_path)
            # Clean up registry
            registry._components.clear()
            registry._protected_components.clear()
            registry._ready = False

    def test_bootstrap_dev_mode_from_config(self):
        """Test that dev_mode is read from TOML config."""
        from hexai.core.registry import registry

        config_content = """
modules = ["tests.hexai.core.registry.sample_components"]
dev_mode = true
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config_content)
            config_path = f.name

        try:
            # Bootstrap without explicit dev_mode parameter
            bootstrap_registry(config_path)

            # Should pick up dev_mode from config
            assert registry.dev_mode is True
        finally:
            os.unlink(config_path)
            # Clean up registry
            registry._components.clear()
            registry._protected_components.clear()
            registry._ready = False

    def test_bootstrap_dev_mode_override(self):
        """Test that parameter dev_mode overrides config."""
        from hexai.core.registry import registry

        config_content = """
modules = ["tests.hexai.core.registry.sample_components"]
dev_mode = false
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config_content)
            config_path = f.name

        try:
            # Override config's dev_mode=false with parameter
            bootstrap_registry(config_path, dev_mode=True)

            # Parameter should override config
            assert registry.dev_mode is True
        finally:
            os.unlink(config_path)
            # Clean up registry
            registry._components.clear()
            registry._protected_components.clear()
            registry._ready = False
