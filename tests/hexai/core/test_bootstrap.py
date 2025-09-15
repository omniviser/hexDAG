"""Tests for the new bootstrap-based registry architecture."""

import pytest

from hexai.core.registry.decorators import node
from hexai.core.registry.exceptions import RegistryAlreadyBootstrappedError, RegistryImmutableError
from hexai.core.registry.manifest import ComponentManifest
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

        # Create manifest pointing to our sample module
        manifest = ComponentManifest(
            [{"namespace": "test", "module": "tests.hexai.core.registry.sample_components"}]
        )

        # Bootstrap
        registry.bootstrap(manifest, dev_mode=True)

        # Registry should now be ready
        assert registry.ready
        assert registry.manifest == manifest

        # Components should be registered
        components = registry.list_components()
        assert len(components) == 3  # sample_node, sample_tool, sample_adapter

        # Check specific components
        node_names = {c.name for c in components}
        assert "sample_node" in node_names
        assert "sample_tool" in node_names
        assert "sample_adapter" in node_names

        # All should be in test namespace
        assert all(c.namespace == "test" for c in components)

    def test_no_registration_after_bootstrap_prod(self):
        """In production mode, registration after bootstrap should fail."""
        registry = ComponentRegistry()

        # Bootstrap with empty manifest
        manifest = ComponentManifest([])
        registry.bootstrap(manifest, dev_mode=False)

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
        manifest = ComponentManifest([])
        registry.bootstrap(manifest, dev_mode=True)

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

        manifest = ComponentManifest([])
        registry.bootstrap(manifest)

        # Second bootstrap should fail
        with pytest.raises(RegistryAlreadyBootstrappedError) as exc_info:
            registry.bootstrap(manifest)

        assert "already been bootstrapped" in str(exc_info.value)

    def test_reset_allows_re_bootstrap(self):
        """Reset should allow re-bootstrapping."""
        # This test requires a reset() method which doesn't exist
        # Skipping for now
        pytest.skip("reset() method not implemented")

    def test_manifest_validation(self):
        """Manifest should validate for duplicates."""
        with pytest.raises(ValueError) as exc_info:
            ComponentManifest(
                [
                    {"namespace": "test", "module": "module1"},
                    {"namespace": "test", "module": "module1"},  # Duplicate
                ]
            ).validate()

        assert "Duplicate manifest entry" in str(exc_info.value)

    def test_import_error_rollback(self):
        """Failed imports should rollback bootstrap."""
        registry = ComponentRegistry()

        # Create manifest with non-existent module
        manifest = ComponentManifest([{"namespace": "test", "module": "non_existent_module_xyz"}])

        # Bootstrap should fail and rollback
        with pytest.raises(ImportError):
            registry.bootstrap(manifest)

        # Registry should be rolled back to empty state
        assert not registry.ready
        assert registry.manifest is None
        assert len(registry.list_components()) == 0
