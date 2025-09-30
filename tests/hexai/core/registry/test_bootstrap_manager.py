"""Tests for BootstrapManager."""

import pytest

from hexai.core.config.models import ManifestEntry
from hexai.core.registry.bootstrap_manager import BootstrapManager
from hexai.core.registry.component_store import ComponentStore
from hexai.core.registry.exceptions import RegistryAlreadyBootstrappedError


class TestBootstrapManager:
    """Test BootstrapManager functionality."""

    @pytest.fixture
    def store(self):
        """Create a component store."""
        return ComponentStore()

    @pytest.fixture
    def manager(self, store):
        """Create a bootstrap manager."""
        return BootstrapManager(store)

    def test_initial_state(self, manager):
        """Test initial state of manager."""
        assert not manager.ready
        assert manager.manifest is None
        assert not manager.dev_mode
        assert not manager.in_bootstrap_context

    def test_can_register_before_bootstrap(self, manager):
        """Test can_register returns True before bootstrap."""
        assert manager.can_register()

    def test_can_register_after_bootstrap_production(self, manager, store):
        """Test can_register returns False after bootstrap in production."""
        manifest = [ManifestEntry(module="hexai.core.ports", namespace="core")]

        def mock_register_fn(_, ns, mod):
            return 0

        manager.bootstrap(manifest, dev_mode=False, register_components_fn=mock_register_fn)

        assert not manager.can_register()

    def test_can_register_after_bootstrap_dev_mode(self, manager, store):
        """Test can_register returns True after bootstrap in dev mode."""
        manifest = [ManifestEntry(module="hexai.core.ports", namespace="core")]

        def mock_register_fn(_, ns, mod):
            return 0

        manager.bootstrap(manifest, dev_mode=True, register_components_fn=mock_register_fn)

        assert manager.can_register()

    def test_bootstrap_sets_ready(self, manager, store):
        """Test that bootstrap sets ready flag."""
        manifest = [ManifestEntry(module="hexai.core.ports", namespace="core")]

        def mock_register_fn(_, ns, mod):
            return 1

        manager.bootstrap(manifest, dev_mode=False, register_components_fn=mock_register_fn)

        assert manager.ready
        assert manager.manifest == manifest

    def test_bootstrap_twice_raises_error(self, manager, store):
        """Test that bootstrapping twice raises error."""
        manifest = [ManifestEntry(module="hexai.core.ports", namespace="core")]

        def mock_register_fn(_, ns, mod):
            return 0

        manager.bootstrap(manifest, dev_mode=False, register_components_fn=mock_register_fn)

        # Second bootstrap should fail
        with pytest.raises(RegistryAlreadyBootstrappedError):
            manager.bootstrap(manifest, dev_mode=False, register_components_fn=mock_register_fn)

    def test_bootstrap_duplicate_manifest_entries_raises_error(self, manager, store):
        """Test that duplicate manifest entries raise error."""
        manifest = [
            ManifestEntry(module="hexai.core.ports", namespace="core"),
            ManifestEntry(module="hexai.core.ports", namespace="core"),  # Duplicate
        ]

        def mock_register_fn(_, ns, mod):
            return 0

        with pytest.raises(ValueError) as exc_info:
            manager.bootstrap(manifest, dev_mode=False, register_components_fn=mock_register_fn)

        assert "Duplicate manifest entry" in str(exc_info.value)

    def test_is_core_module(self, manager):
        """Test identifying core modules."""
        # Explicit core namespace
        entry = ManifestEntry(module="some.module", namespace="core")
        assert manager._is_core_module(entry)

        # Framework module
        entry = ManifestEntry(module="hexai.core.ports", namespace="plugin")
        assert manager._is_core_module(entry)

        # Builtin tools
        entry = ManifestEntry(module="hexai.tools.builtin_tools", namespace="plugin")
        assert manager._is_core_module(entry)

        # Plugin module
        entry = ManifestEntry(module="hexai.adapters.mock.mock_llm", namespace="plugin")
        assert not manager._is_core_module(entry)

    def test_check_plugin_requirements_existing_module(self, manager):
        """Test checking requirements for existing module."""
        result = manager._check_plugin_requirements("hexai.core.ports")
        assert result is None  # Module exists

    def test_check_plugin_requirements_missing_module(self, manager):
        """Test checking requirements for non-existent module."""
        result = manager._check_plugin_requirements("nonexistent.module.path")
        assert result is not None  # Should return reason
        assert "not found" in result.lower()

    def test_reset(self, manager, store):
        """Test reset functionality."""
        manifest = [ManifestEntry(module="hexai.core.ports", namespace="core")]

        def mock_register_fn(_, ns, mod):
            return 1

        manager.bootstrap(manifest, dev_mode=True, register_components_fn=mock_register_fn)

        assert manager.ready
        assert manager.dev_mode

        manager.reset()

        assert not manager.ready
        assert manager.manifest is None
        assert not manager.dev_mode
        assert not manager.in_bootstrap_context

    def test_bootstrap_context_flag(self, manager, store):
        """Test that bootstrap context flag is set during bootstrap."""
        manifest = [ManifestEntry(module="hexai.core.ports", namespace="core")]

        context_during_bootstrap = []

        def mock_register_fn(_, ns, mod):
            context_during_bootstrap.append(manager.in_bootstrap_context)
            return 1

        manager.bootstrap(manifest, dev_mode=False, register_components_fn=mock_register_fn)

        # Should have been True during bootstrap
        assert any(context_during_bootstrap)
        # Should be False after bootstrap
        assert not manager.in_bootstrap_context
