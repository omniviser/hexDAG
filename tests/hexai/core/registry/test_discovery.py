"""Tests for the discovery module."""

from unittest.mock import patch

from hexai.core.registry.discovery import (
    create_plugin,
    discover_entry_points,
    discover_plugins,
    hookimpl,
    hookspec,
)


class TestPluggyMarkers:
    """Test that pluggy markers are properly exported."""

    def test_hookimpl_marker(self):
        """Test hookimpl marker is available."""
        assert hookimpl is not None
        assert hookimpl.project_name == "hexdag"

    def test_hookspec_marker(self):
        """Test hookspec marker is available."""
        assert hookspec is not None
        assert hookspec.project_name == "hexdag"

    def test_markers_can_decorate(self):
        """Test that markers can be used as decorators."""

        # Create a class with hook specs
        class TestSpec:
            @hookspec
            def test_method(self):
                pass

        # Create implementation
        class TestImpl:
            @hookimpl
            def test_method(self):
                pass

        # Check markers were applied
        assert hasattr(TestSpec.test_method, "hexdag_spec")
        assert hasattr(TestImpl.test_method, "hexdag_impl")


class TestCreatePlugin:
    """Test the create_plugin helper function."""

    def test_create_simple_plugin(self):
        """Test creating a simple plugin class."""
        PluginClass = create_plugin("test_plugin")

        # Check class properties
        assert PluginClass.__name__ == "Test_PluginPlugin"

        # Create instance
        plugin = PluginClass()
        assert plugin.namespace == "test_plugin"

        # Check it has the hook
        assert hasattr(plugin, "hexdag_initialize")

    def test_plugin_hook_implementation(self):
        """Test that created plugin has proper hook."""
        PluginClass = create_plugin("my_plugin")
        plugin = PluginClass()

        # Check hook exists
        assert hasattr(plugin, "hexdag_initialize")

        # Should run without error
        with patch("hexai.core.registry.discovery.logger") as mock_logger:
            plugin.hexdag_initialize()
            mock_logger.debug.assert_called_with("Plugin 'my_plugin' initialized")

    def test_different_namespaces(self):
        """Test creating plugins with different namespaces."""
        Plugin1 = create_plugin("plugin_one")
        Plugin2 = create_plugin("plugin_two")

        p1 = Plugin1()
        p2 = Plugin2()

        assert p1.namespace == "plugin_one"
        assert p2.namespace == "plugin_two"
        assert Plugin1.__name__ == "Plugin_OnePlugin"
        assert Plugin2.__name__ == "Plugin_TwoPlugin"

    def test_plugin_usage_example(self):
        """Test the documented usage pattern."""
        # Simulate plugin __init__.py
        Plugin = create_plugin("example_plugin")

        def register():
            """Entry point function."""
            # Would normally import modules here
            return Plugin()

        # Test the register function
        plugin_instance = register()
        assert isinstance(plugin_instance, Plugin)
        assert plugin_instance.namespace == "example_plugin"


class TestLegacyFunctions:
    """Test backward compatibility functions."""

    @patch("hexai.core.registry.discovery.logger")
    def test_discover_entry_points_deprecated(self, mock_logger):
        """Test that discover_entry_points logs deprecation warning."""
        result = discover_entry_points("test.group")

        # Should return empty dict
        assert result == {}

        # Should log warning
        mock_logger.warning.assert_called_once()
        warning_msg = mock_logger.warning.call_args[0][0]
        assert "deprecated" in warning_msg
        assert "test.group" in warning_msg

    @patch("hexai.core.registry.discovery.logger")
    def test_discover_plugins_deprecated(self, mock_logger):
        """Test that discover_plugins logs deprecation warning."""
        result = discover_plugins()

        # Should return 0
        assert result == 0

        # Should log warning
        mock_logger.warning.assert_called_once()
        warning_msg = mock_logger.warning.call_args[0][0]
        assert "deprecated" in warning_msg
        assert "automatically" in warning_msg


class TestPluginIntegration:
    """Test plugin integration patterns."""

    def test_plugin_with_hooks(self):
        """Test creating a plugin that implements hooks."""
        # Create plugin class
        PluginClass = create_plugin("advanced_plugin")

        # Extend it with custom hooks
        class AdvancedPlugin(PluginClass):
            def __init__(self):
                super().__init__()
                self.components_registered = False

            @hookimpl
            def hexdag_initialize(self):
                """Override to do actual initialization."""
                super().hexdag_initialize()
                # Simulate importing modules
                self.components_registered = True

            @hookimpl
            def hexdag_configure(self, config):
                """Additional hook implementation."""
                self.config = config

        # Test the plugin
        plugin = AdvancedPlugin()
        assert plugin.namespace == "advanced_plugin"

        # Test initialization
        plugin.hexdag_initialize()
        assert plugin.components_registered

        # Test configuration
        test_config = {"key": "value"}
        plugin.hexdag_configure(test_config)
        assert plugin.config == test_config

    def test_multiple_plugins_pattern(self):
        """Test pattern for multiple plugins."""
        plugins = {}

        # Create multiple plugins
        for name in ["plugin_a", "plugin_b", "plugin_c"]:
            PluginClass = create_plugin(name)
            plugins[name] = PluginClass()

        # Verify each has unique namespace
        assert plugins["plugin_a"].namespace == "plugin_a"
        assert plugins["plugin_b"].namespace == "plugin_b"
        assert plugins["plugin_c"].namespace == "plugin_c"

        # Each should have the hook
        for plugin in plugins.values():
            assert hasattr(plugin, "hexdag_initialize")


class TestDocumentation:
    """Test that documented examples work."""

    def test_entry_point_example(self):
        """Test the documented entry point pattern."""
        # This is what would be in a plugin's __init__.py
        from hexai.core.registry.discovery import create_plugin

        Plugin = create_plugin("documented_plugin")

        def register():
            """Entry point for plugin registration."""
            # from . import nodes  # Would import decorated modules
            return Plugin()

        # Test that this pattern works
        plugin = register()
        assert plugin.namespace == "documented_plugin"

        # Plugin should be ready to use with pluggy
        assert hasattr(plugin, "hexdag_initialize")

        # Can call the hook
        plugin.hexdag_initialize()  # Should not raise
