"""Tests for plugin loader functionality."""

import sys
from unittest.mock import MagicMock, patch

from hexai.core.registry.plugin_loader import PluginLoader


class TestPluginLoader:
    """Test plugin loading functionality."""

    def test_plugin_loader_init(self):
        """Test plugin loader initialization."""
        mock_registry = MagicMock()
        loader = PluginLoader(mock_registry)
        assert loader._loaded_plugins == set()

    @patch("importlib.metadata.entry_points")
    def test_discover_plugins_python_310_plus(self, mock_entry_points):
        """Test plugin discovery on Python 3.10+."""
        # Mock entry points with select method (Python 3.10+)
        mock_ep = MagicMock()
        mock_ep.name = "test_plugin"
        mock_ep.value = "test_module:setup"

        mock_entry_points.return_value.select = MagicMock(return_value=[mock_ep])

        mock_registry = MagicMock()
        loader = PluginLoader(mock_registry)

        # Mock the entry point loading
        with patch.object(loader, "_load_entry_point", return_value=True) as mock_load:
            loader.load_plugins()
            mock_load.assert_called_once_with(mock_ep)

    @patch("importlib.metadata.entry_points")
    def test_discover_plugins_python_39(self, mock_entry_points):
        """Test plugin discovery on Python 3.9."""
        # Mock entry points as callable returning dict (Python 3.9)
        mock_ep = MagicMock()
        mock_ep.name = "test_plugin"
        mock_ep.value = "test_module:setup"

        # For Python 3.9, entry_points() returns a callable that returns a dict
        mock_eps_callable = MagicMock(return_value={"hexdag.plugins": [mock_ep]})
        # Make sure it doesn't have a select attribute
        del mock_eps_callable.select
        mock_entry_points.return_value = mock_eps_callable

        mock_registry = MagicMock()
        loader = PluginLoader(mock_registry)

        with patch.object(loader, "_load_entry_point", return_value=True) as mock_load:
            loader.load_plugins()
            mock_load.assert_called_once_with(mock_ep)

    def test_load_entry_point_success(self):
        """Test successful entry point loading."""
        mock_registry = MagicMock()
        loader = PluginLoader(mock_registry)

        mock_ep = MagicMock()
        mock_ep.name = "test_plugin"
        mock_setup = MagicMock()
        mock_ep.load.return_value = mock_setup

        result = loader._load_entry_point(mock_ep)

        assert result is True
        assert "test_plugin" in loader._loaded_plugins
        mock_setup.assert_called_once_with(mock_registry)

    def test_load_entry_point_failure(self):
        """Test entry point loading failure handling."""
        mock_registry = MagicMock()
        loader = PluginLoader(mock_registry)

        mock_ep = MagicMock()
        mock_ep.name = "bad_plugin"
        mock_ep.load.side_effect = ImportError("Module not found")

        result = loader._load_entry_point(mock_ep)

        assert result is False
        assert "bad_plugin" not in loader._loaded_plugins

    def test_load_entry_point_duplicate(self):
        """Test loading duplicate plugin is skipped."""
        mock_registry = MagicMock()
        loader = PluginLoader(mock_registry)
        loader._loaded_plugins.add("test_plugin")

        mock_ep = MagicMock()
        mock_ep.name = "test_plugin"

        result = loader._load_entry_point(mock_ep)

        assert result is False
        mock_ep.load.assert_not_called()

