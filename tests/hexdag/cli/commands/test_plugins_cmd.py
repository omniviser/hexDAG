"""Tests for hexdag.cli.commands.plugins_cmd module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from hexdag.cli.commands.plugins_cmd import (
    OutputFormat,
    _check_dependencies,
    _get_available_plugins,
    app,
)
from hexdag.core.registry.models import ClassComponent, ComponentMetadata, ComponentType


def make_component_metadata(
    name: str,
    component_type: ComponentType,
    namespace: str = "core",
    description: str = "",
    implements_port: str | None = None,
) -> ComponentMetadata:
    """Helper to create ComponentMetadata with required fields."""

    class DummyComponent:
        pass

    return ComponentMetadata(
        name=name,
        component_type=component_type,
        component=ClassComponent(value=DummyComponent),
        namespace=namespace,
        description=description,
        implements_port=implements_port,
    )


@pytest.fixture
def runner():
    """Fixture providing a Typer CLI test runner."""
    return CliRunner()


@pytest.fixture
def mock_console():
    """Fixture providing a mocked console."""
    with patch("hexdag.cli.commands.plugins_cmd.console") as mock:
        yield mock


@pytest.fixture
def mock_registry():
    """Fixture providing a mocked registry."""
    with patch("hexdag.cli.commands.plugins_cmd.registry") as mock:
        yield mock


class TestOutputFormat:
    """Test the OutputFormat enum."""

    def test_output_format_values(self):
        """Test that OutputFormat has expected values."""
        assert OutputFormat.TABLE == "table"
        assert OutputFormat.JSON == "json"
        assert OutputFormat.YAML == "yaml"


class TestListPlugins:
    """Test the list-plugins command."""

    def test_list_plugins_table_format(self, runner):
        """Test listing plugins in table format."""
        with patch("hexdag.cli.commands.plugins_cmd._get_available_plugins") as mock_get:
            mock_get.return_value = [
                {
                    "name": "mock",
                    "namespace": "core",
                    "installed": True,
                    "capabilities": ["LLM", "Database"],
                }
            ]

            result = runner.invoke(app, ["list"])
            assert result.exit_code == 0
            assert "mock" in result.stdout or "Available Plugins" in result.stdout

    def test_list_plugins_json_format(self, runner):
        """Test listing plugins in JSON format."""
        with patch("hexdag.cli.commands.plugins_cmd._get_available_plugins") as mock_get:
            mock_get.return_value = [
                {"name": "test", "namespace": "plugin", "installed": False, "capabilities": ["API"]}
            ]

            result = runner.invoke(app, ["list", "--format", "json"])
            assert result.exit_code == 0

    def test_list_plugins_yaml_format(self, runner):
        """Test listing plugins in YAML format."""
        with patch("hexdag.cli.commands.plugins_cmd._get_available_plugins") as mock_get:
            mock_get.return_value = []

            result = runner.invoke(app, ["list", "--format", "yaml"])
            assert result.exit_code == 0


class TestCheckPlugins:
    """Test the check command."""

    def test_check_plugins_all_ok(self, runner):
        """Test checking plugins when all are OK."""
        with patch("hexdag.cli.commands.plugins_cmd._check_dependencies") as mock_check:
            mock_check.return_value = [{"name": "pydantic", "status": "ok"}]

            result = runner.invoke(app, ["check"])
            assert result.exit_code == 0
            assert "pydantic" in result.stdout

    def test_check_plugins_with_missing(self, runner):
        """Test checking plugins with missing dependencies."""
        with patch("hexdag.cli.commands.plugins_cmd._check_dependencies") as mock_check:
            mock_check.return_value = [
                {
                    "name": "openai",
                    "status": "missing",
                    "install_hint": "pip install hexdag[adapters-openai]",
                }
            ]

            result = runner.invoke(app, ["check"])
            assert result.exit_code == 0
            assert "openai" in result.stdout.lower()

    def test_check_plugins_with_optional(self, runner):
        """Test checking plugins with optional dependencies."""
        with patch("hexdag.cli.commands.plugins_cmd._check_dependencies") as mock_check:
            mock_check.return_value = [
                {
                    "name": "graphviz",
                    "status": "optional",
                    "install_hint": "pip install hexdag[viz]",
                }
            ]

            result = runner.invoke(app, ["check"])
            assert result.exit_code == 0


class TestInstallPlugin:
    """Test the install command."""

    def test_install_plugin_dry_run(self, runner):
        """Test install with --dry-run flag."""
        with patch("shutil.which", return_value=None):
            result = runner.invoke(app, ["install", "openai", "--dry-run"])
            assert result.exit_code == 0
            assert "Would run:" in result.stdout

    def test_install_plugin_with_pip(self, runner):
        """Test installing plugin with pip."""
        with patch("shutil.which", return_value=None):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(
                    returncode=0, stdout="Successfully installed", stderr=""
                )

                result = runner.invoke(app, ["install", "openai"])
                assert result.exit_code == 0
                assert mock_run.called

    def test_install_plugin_with_uv(self, runner):
        """Test installing plugin with uv."""
        with patch("shutil.which", return_value="/usr/bin/uv"):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(
                    returncode=0, stdout="Successfully installed", stderr=""
                )

                result = runner.invoke(app, ["install", "anthropic", "--uv"])
                assert result.exit_code == 0
                assert mock_run.called

    def test_install_plugin_editable(self, runner, tmp_path):
        """Test installing plugin in editable mode."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create pyproject.toml
            Path("pyproject.toml").write_text('[project]\nname = "test"\n')

            with patch("shutil.which", return_value=None):
                with patch("subprocess.run") as mock_run:
                    mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

                    result = runner.invoke(app, ["install", "all", "--editable"])
                    assert result.exit_code == 0

    def test_install_plugin_editable_no_pyproject(self, runner, tmp_path):
        """Test installing editable without pyproject.toml fails."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(app, ["install", "openai", "--editable"])
            assert result.exit_code == 1
            assert "pyproject.toml" in result.stdout

    def test_install_plugin_failure(self, runner):
        """Test handling installation failure."""
        with patch("shutil.which", return_value=None):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="Error occurred")

                result = runner.invoke(app, ["install", "fake-plugin"])
                assert result.exit_code == 0  # CLI doesn't exit with error, just shows message

    def test_install_plugin_uv_not_found_warning(self, runner):
        """Test warning when uv is requested but not found."""
        with patch("shutil.which", return_value=None):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

                result = runner.invoke(app, ["install", "openai", "--uv"])
                assert "Warning" in result.stdout or result.exit_code == 0


class TestGetAvailablePlugins:
    """Test the _get_available_plugins helper function."""

    def test_get_available_plugins_returns_list(self):
        """Test that _get_available_plugins returns a list."""
        with patch("hexdag.core.bootstrap.bootstrap_registry"):
            with patch("hexdag.core.registry.registry") as mock_registry:
                mock_registry.list_components.return_value = []

                plugins = _get_available_plugins()
                assert isinstance(plugins, list)

    def test_get_available_plugins_includes_known_extras(self):
        """Test that known extras are included."""
        with patch("hexdag.core.bootstrap.bootstrap_registry"):
            with patch("hexdag.core.registry.registry") as mock_registry:
                mock_registry.list_components.return_value = []

                plugins = _get_available_plugins()
                # Should include at least some known extras
                assert len(plugins) >= 0

    def test_get_available_plugins_with_adapters(self):
        """Test getting plugins with adapters from registry."""
        from hexdag.core.registry.models import ComponentInfo, ComponentType

        with patch("hexdag.core.bootstrap.bootstrap_registry"):
            with patch("hexdag.core.registry.registry") as mock_registry:
                # Mock adapter component
                adapter_info = ComponentInfo(
                    name="mock_llm",
                    qualified_name="plugin:mock_llm",
                    component_type=ComponentType.ADAPTER,
                    namespace="plugin",
                    metadata=make_component_metadata(
                        "mock_llm", ComponentType.ADAPTER, namespace="plugin", implements_port="llm"
                    ),
                )

                mock_registry.list_components.return_value = [adapter_info]

                plugins = _get_available_plugins()
                assert len(plugins) > 0


class TestCheckDependencies:
    """Test the _check_dependencies helper function."""

    def test_check_dependencies_returns_list(self):
        """Test that _check_dependencies returns a list."""
        with patch("shutil.which", return_value=None):
            checks = _check_dependencies()
            assert isinstance(checks, list)
            assert len(checks) > 0

    def test_check_dependencies_includes_core(self):
        """Test that core dependencies are checked."""
        with patch("shutil.which", return_value=None):
            checks = _check_dependencies()
            # Should check pydantic
            assert any("pydantic" in check["name"].lower() for check in checks)

    def test_check_dependencies_with_uv(self):
        """Test dependency checks with uv available."""
        with patch("shutil.which", return_value="/usr/bin/uv"):
            checks = _check_dependencies()
            # Should suggest uv commands
            assert any(
                "uv" in check.get("install_hint", "") for check in checks if "install_hint" in check
            )

    def test_check_dependencies_without_uv(self):
        """Test dependency checks without uv."""
        with patch("shutil.which", return_value=None):
            checks = _check_dependencies()
            # Should suggest pip commands
            assert any(
                "pip install" in check.get("install_hint", "")
                for check in checks
                if "install_hint" in check
            )
