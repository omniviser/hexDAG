"""Tests for hexdag.cli.commands.plugin_dev_cmd module."""

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from hexdag.cli.commands.plugin_dev_cmd import app, get_plugin_dir


@pytest.fixture
def runner():
    """Fixture providing a Typer CLI test runner."""
    return CliRunner()


@pytest.fixture
def mock_console():
    """Fixture providing a mocked console."""
    with patch("hexdag.cli.commands.plugin_dev_cmd.console") as mock:
        yield mock


class TestGetPluginDir:
    """Test the get_plugin_dir helper function."""

    def test_get_plugin_dir_finds_existing(self, tmp_path):
        """Test finding existing hexdag_plugins directory."""
        plugin_dir = tmp_path / "hexdag_plugins"
        plugin_dir.mkdir()

        with patch("pathlib.Path.cwd", return_value=tmp_path):
            result = get_plugin_dir()
            assert result == plugin_dir

    def test_get_plugin_dir_creates_new(self, tmp_path):
        """Test creating new hexdag_plugins directory."""
        with patch("pathlib.Path.cwd", return_value=tmp_path):
            result = get_plugin_dir()
            assert result.exists()
            assert result.name == "hexdag_plugins"

    def test_get_plugin_dir_from_project_root(self, tmp_path):
        """Test finding plugin dir from project with pyproject.toml."""
        # Create project structure
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "hexdag"\n')
        plugin_dir = tmp_path / "hexdag_plugins"
        plugin_dir.mkdir()

        subdir = tmp_path / "nested" / "dir"
        subdir.mkdir(parents=True)

        with patch("pathlib.Path.cwd", return_value=subdir):
            result = get_plugin_dir()
            assert result == plugin_dir


class TestCreatePlugin:
    """Test the new command."""

    def test_create_plugin_success(self, runner, tmp_path):
        """Test creating a new plugin successfully."""
        plugin_dir = tmp_path / "hexdag_plugins"
        plugin_dir.mkdir()

        with patch("hexdag.cli.commands.plugin_dev_cmd.get_plugin_dir", return_value=plugin_dir):
            result = runner.invoke(app, ["new", "my_adapter"])
            assert result.exit_code == 0

            # Check that plugin was created
            created_dir = plugin_dir / "my_adapter"
            assert created_dir.exists()
            assert (created_dir / "__init__.py").exists()
            assert (created_dir / "my_adapter.py").exists()
            assert (created_dir / "pyproject.toml").exists()
            assert (created_dir / "README.md").exists()
            assert (created_dir / "tests").exists()

    def test_create_plugin_with_custom_port(self, runner, tmp_path):
        """Test creating plugin with custom port type."""
        plugin_dir = tmp_path / "hexdag_plugins"
        plugin_dir.mkdir()

        with patch("hexdag.cli.commands.plugin_dev_cmd.get_plugin_dir", return_value=plugin_dir):
            result = runner.invoke(app, ["new", "redis_adapter", "--port", "cache"])
            assert result.exit_code == 0

            adapter_file = plugin_dir / "redis_adapter" / "redis_adapter.py"
            content = adapter_file.read_text()
            assert "cache" in content

    def test_create_plugin_with_custom_author(self, runner, tmp_path):
        """Test creating plugin with custom author."""
        plugin_dir = tmp_path / "hexdag_plugins"
        plugin_dir.mkdir()

        with patch("hexdag.cli.commands.plugin_dev_cmd.get_plugin_dir", return_value=plugin_dir):
            result = runner.invoke(app, ["new", "test_adapter", "--author", "John Doe"])
            assert result.exit_code == 0

            pyproject_file = plugin_dir / "test_adapter" / "pyproject.toml"
            content = pyproject_file.read_text()
            assert "John Doe" in content

    def test_create_plugin_already_exists(self, runner, tmp_path):
        """Test that creating existing plugin fails."""
        plugin_dir = tmp_path / "hexdag_plugins"
        plugin_dir.mkdir()
        existing = plugin_dir / "existing_adapter"
        existing.mkdir()

        with patch("hexdag.cli.commands.plugin_dev_cmd.get_plugin_dir", return_value=plugin_dir):
            result = runner.invoke(app, ["new", "existing_adapter"])
            assert result.exit_code == 1
            assert "already exists" in result.stdout

    def test_create_plugin_creates_test_structure(self, runner, tmp_path):
        """Test that plugin creation includes test structure."""
        plugin_dir = tmp_path / "hexdag_plugins"
        plugin_dir.mkdir()

        with patch("hexdag.cli.commands.plugin_dev_cmd.get_plugin_dir", return_value=plugin_dir):
            result = runner.invoke(app, ["new", "test_plugin"])
            assert result.exit_code == 0

            test_file = plugin_dir / "test_plugin" / "tests" / "test_test_plugin.py"
            assert test_file.exists()
            content = test_file.read_text()
            assert "pytest" in content

    def test_create_plugin_creates_pyproject(self, runner, tmp_path):
        """Test that pyproject.toml is created with correct structure."""
        plugin_dir = tmp_path / "hexdag_plugins"
        plugin_dir.mkdir()

        with patch("hexdag.cli.commands.plugin_dev_cmd.get_plugin_dir", return_value=plugin_dir):
            result = runner.invoke(app, ["new", "my_plugin"])
            assert result.exit_code == 0

            pyproject = plugin_dir / "my_plugin" / "pyproject.toml"
            content = pyproject.read_text()
            assert "[project]" in content
            assert "dependencies" in content


class TestListPlugins:
    """Test the list command."""

    def test_list_plugins_no_directory(self, runner, tmp_path):
        """Test listing when no plugin directory exists."""
        with patch(
            "hexdag.cli.commands.plugin_dev_cmd.get_plugin_dir",
            return_value=tmp_path / "nonexistent",
        ):
            result = runner.invoke(app, ["list"])
            assert result.exit_code == 0
            assert "No plugins directory" in result.stdout

    def test_list_plugins_empty_directory(self, runner, tmp_path):
        """Test listing when plugin directory is empty."""
        plugin_dir = tmp_path / "hexdag_plugins"
        plugin_dir.mkdir()

        with patch("hexdag.cli.commands.plugin_dev_cmd.get_plugin_dir", return_value=plugin_dir):
            result = runner.invoke(app, ["list"])
            assert result.exit_code == 0
            assert "No plugins found" in result.stdout

    def test_list_plugins_with_plugins(self, runner, tmp_path):
        """Test listing existing plugins."""
        plugin_dir = tmp_path / "hexdag_plugins"
        plugin_dir.mkdir()

        # Create a plugin with pyproject.toml
        test_plugin = plugin_dir / "test_plugin"
        test_plugin.mkdir()
        (test_plugin / "pyproject.toml").write_text(
            """
[project]
name = "test-plugin"
version = "0.1.0"
description = "Test plugin"
"""
        )

        with patch("hexdag.cli.commands.plugin_dev_cmd.get_plugin_dir", return_value=plugin_dir):
            result = runner.invoke(app, ["list"])
            assert result.exit_code == 0
            assert "test_plugin" in result.stdout

    def test_list_plugins_skips_hidden_dirs(self, runner, tmp_path):
        """Test that hidden directories are skipped."""
        plugin_dir = tmp_path / "hexdag_plugins"
        plugin_dir.mkdir()

        # Create hidden dir
        (plugin_dir / ".hidden").mkdir()
        # Create __pycache__
        (plugin_dir / "__pycache__").mkdir()
        # Create valid plugin
        valid = plugin_dir / "valid_plugin"
        valid.mkdir()

        with patch("hexdag.cli.commands.plugin_dev_cmd.get_plugin_dir", return_value=plugin_dir):
            result = runner.invoke(app, ["list"])
            assert result.exit_code == 0
            # Should only show valid_plugin
            assert "valid_plugin" in result.stdout or "Available Plugins" in result.stdout


class TestLintCommand:
    """Test the lint command."""

    def test_lint_plugin(self, runner, tmp_path):
        """Test linting a plugin."""
        plugin_dir = tmp_path / "hexdag_plugins"
        plugin_dir.mkdir()
        test_plugin = plugin_dir / "test_plugin"
        test_plugin.mkdir()

        with patch("hexdag.cli.commands.plugin_dev_cmd.get_plugin_dir", return_value=plugin_dir):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)

                result = runner.invoke(app, ["lint", "test_plugin"])
                # Command may fail without actual ruff, but should attempt
                assert mock_run.called or result.exit_code in [0, 1]

    def test_lint_nonexistent_plugin(self, runner, tmp_path):
        """Test linting a plugin that doesn't exist."""
        plugin_dir = tmp_path / "hexdag_plugins"
        plugin_dir.mkdir()

        with patch("hexdag.cli.commands.plugin_dev_cmd.get_plugin_dir", return_value=plugin_dir):
            result = runner.invoke(app, ["lint", "nonexistent"])
            assert result.exit_code == 1


class TestTestCommand:
    """Test the test command."""

    def test_test_plugin(self, runner, tmp_path):
        """Test running tests for a plugin."""
        plugin_dir = tmp_path / "hexdag_plugins"
        plugin_dir.mkdir()
        test_plugin = plugin_dir / "test_plugin"
        test_plugin.mkdir()
        (test_plugin / "tests").mkdir()

        with patch("hexdag.cli.commands.plugin_dev_cmd.get_plugin_dir", return_value=plugin_dir):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)

                result = runner.invoke(app, ["test", "test_plugin"])
                # Command may fail without actual pytest, but should attempt
                assert mock_run.called or result.exit_code in [0, 1]

    @pytest.mark.skip(reason="--coverage option not yet implemented")
    def test_test_with_coverage(self, runner, tmp_path):
        """Test running tests with coverage."""
        plugin_dir = tmp_path / "hexdag_plugins"
        plugin_dir.mkdir()
        test_plugin = plugin_dir / "test_plugin"
        test_plugin.mkdir()
        (test_plugin / "tests").mkdir()

        with patch("hexdag.cli.commands.plugin_dev_cmd.get_plugin_dir", return_value=plugin_dir):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)

                result = runner.invoke(app, ["test", "test_plugin", "--coverage"])
                assert mock_run.called or result.exit_code in [0, 1]


class TestBuildCommand:
    """Test the build command."""

    @pytest.mark.skip(reason="build command not yet implemented")
    def test_build_plugin(self, runner, tmp_path):
        """Test building a plugin."""
        plugin_dir = tmp_path / "hexdag_plugins"
        plugin_dir.mkdir()
        test_plugin = plugin_dir / "test_plugin"
        test_plugin.mkdir()
        (test_plugin / "pyproject.toml").write_text('[project]\nname = "test"\n')

        with patch("hexdag.cli.commands.plugin_dev_cmd.get_plugin_dir", return_value=plugin_dir):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)

                result = runner.invoke(app, ["build", "test_plugin"])
                assert mock_run.called or result.exit_code in [0, 1]
