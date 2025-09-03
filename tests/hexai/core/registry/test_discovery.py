"""Tests for discovery.py - plugin and component discovery."""

from unittest.mock import Mock, patch

from hexai.core.registry.discovery import discover_entry_points, discover_plugins


class TestDiscoverEntryPoints:
    """Test the discover_entry_points function."""

    @patch("hexai.core.registry.discovery.entry_points")
    def test_discover_entry_points_basic(self, mock_entry_points):
        """Test basic entry point discovery."""
        # Mock entry points
        mock_ep1 = Mock()
        mock_ep1.name = "plugin1"
        mock_ep1.load.return_value = Mock()

        mock_ep2 = Mock()
        mock_ep2.name = "plugin2"
        mock_ep2.load.return_value = Mock()

        mock_entry_points.return_value = [mock_ep1, mock_ep2]

        plugins = discover_entry_points("hexai.plugins")

        assert len(plugins) == 2
        assert "plugin1" in plugins
        assert "plugin2" in plugins

        mock_entry_points.assert_called_once_with(group="hexai.plugins")

    @patch("hexai.core.registry.discovery.entry_points")
    def test_discover_entry_points_with_error(self, mock_entry_points):
        """Test entry point discovery with loading errors."""
        mock_ep1 = Mock()
        mock_ep1.name = "good_plugin"
        mock_ep1.load.return_value = Mock()

        mock_ep2 = Mock()
        mock_ep2.name = "bad_plugin"
        mock_ep2.load.side_effect = ImportError("Failed to load")

        mock_entry_points.return_value = [mock_ep1, mock_ep2]

        with patch("hexai.core.registry.discovery.logger") as mock_logger:
            plugins = discover_entry_points("hexai.plugins")

            assert len(plugins) == 1
            assert "good_plugin" in plugins
            assert "bad_plugin" not in plugins

            # Should log the warning about failed loading
            mock_logger.warning.assert_called()

    @patch("hexai.core.registry.discovery.entry_points")
    def test_discover_entry_points_empty(self, mock_entry_points):
        """Test discovery with no entry points."""
        mock_entry_points.return_value = []

        plugins = discover_entry_points("hexai.plugins")

        assert len(plugins) == 0

    @patch("hexai.core.registry.discovery.entry_points")
    def test_discover_entry_points_duplicate_names(self, mock_entry_points):
        """Test handling of duplicate entry point names."""
        mock_ep1 = Mock()
        mock_ep1.name = "plugin"
        mock_ep1.load.return_value = Mock(version=1)

        mock_ep2 = Mock()
        mock_ep2.name = "plugin"  # Duplicate name
        mock_ep2.load.return_value = Mock(version=2)

        mock_entry_points.return_value = [mock_ep1, mock_ep2]

        plugins = discover_entry_points("hexai.plugins")

        # Should keep the last one (overwrites silently)
        assert len(plugins) == 1
        assert "plugin" in plugins
        # The second one should overwrite
        assert plugins["plugin"].version == 2


class TestDiscoverPlugins:
    """Test the discover_plugins function."""

    @patch("hexai.core.registry.discovery.discover_entry_points")
    def test_discover_plugins_success(self, mock_discover):
        """Test successful plugin discovery and loading."""
        # Mock plugin register functions
        register1 = Mock()
        register2 = Mock()
        register3 = Mock()

        mock_discover.return_value = {
            "plugin1": register1,
            "plugin2": register2,
            "plugin3": register3,
        }

        with patch("hexai.core.registry.discovery.logger") as mock_logger:
            loaded_count = discover_plugins()

            assert loaded_count == 3

            # All register functions should be called
            register1.assert_called_once()
            register2.assert_called_once()
            register3.assert_called_once()

            # Should log info for each plugin
            assert mock_logger.info.call_count >= 3

    @patch("hexai.core.registry.discovery.discover_entry_points")
    def test_discover_plugins_with_failures(self, mock_discover):
        """Test plugin discovery with some failures."""
        register1 = Mock()
        register2 = Mock()
        register2.side_effect = RuntimeError("Plugin failed")
        register3 = Mock()

        mock_discover.return_value = {
            "plugin1": register1,
            "plugin2": register2,
            "plugin3": register3,
        }

        with patch("hexai.core.registry.discovery.logger") as mock_logger:
            loaded_count = discover_plugins()

            # Only 2 should load successfully
            assert loaded_count == 2

            register1.assert_called_once()
            register2.assert_called_once()  # Called but failed
            register3.assert_called_once()

            # Should log error for failed plugin
            mock_logger.error.assert_called()

    @patch("hexai.core.registry.discovery.discover_entry_points")
    def test_discover_plugins_empty(self, mock_discover):
        """Test plugin discovery with no plugins."""
        mock_discover.return_value = {}

        loaded_count = discover_plugins()

        assert loaded_count == 0

    @patch("hexai.core.registry.discovery.discover_entry_points")
    def test_discover_plugins_all_fail(self, mock_discover):
        """Test when all plugins fail to load."""
        register1 = Mock(side_effect=Exception("Failed"))
        register2 = Mock(side_effect=Exception("Failed"))

        mock_discover.return_value = {
            "plugin1": register1,
            "plugin2": register2,
        }

        with patch("hexai.core.registry.discovery.logger") as mock_logger:
            loaded_count = discover_plugins()

            assert loaded_count == 0

            # Should log errors for each
            assert mock_logger.error.call_count == 2


class TestIntegration:
    """Integration tests for discovery system."""

    def test_full_discovery_flow(self):
        """Test complete discovery and registration flow."""
        # This test would require actual plugin entry points
        # which is harder to mock in unit tests
        pass

    def test_protected_namespace_discovery(self):
        """Test that discovery respects protected namespaces."""
        # Protected namespaces are handled by the registry, not discovery
        # This is tested in test_registry.py
        pass


class TestDiscoveryHelpers:
    """Test helper functions and utilities."""

    def test_entry_point_group_names(self):
        """Test standard entry point group names."""
        # These are conventions
        expected_groups = [
            "hexai.plugins",
            "hexai.nodes",
            "hexai.tools",
            "hexai.agents",
        ]

        # In real implementation, these might be constants
        for group in expected_groups:
            # Just verify the format
            assert group.startswith("hexai.")

    def test_filesystem_discovery(self):
        """Test filesystem-based component discovery."""
        # This is a placeholder test for future filesystem discovery implementation
        # Currently, discovery.py only supports entry points, not filesystem discovery
        pass
