"""Tests for hexdag.cli.commands.manifest_cmd module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from hexdag.cli.commands.manifest_cmd import app


@pytest.fixture
def runner():
    """Fixture providing a Typer CLI test runner."""
    return CliRunner()


@pytest.fixture
def sample_toml_manifest(tmp_path):
    """Fixture providing a sample TOML manifest file."""
    manifest_path = tmp_path / "test-manifest.toml"
    manifest_content = """modules = [
    "hexdag.builtin.nodes",
    "hexdag.core.ports",
]

plugins = [
    "hexdag.builtin.adapters.local",
]
"""
    manifest_path.write_text(manifest_content)
    return manifest_path


@pytest.fixture
def invalid_toml_manifest(tmp_path):
    """Fixture providing an invalid TOML manifest file."""
    manifest_path = tmp_path / "invalid-manifest.toml"
    manifest_content = """modules = [
    "hexdag.builtin.nodes"
    "hexdag.core.ports",  # Missing comma
]
"""
    manifest_path.write_text(manifest_content)
    return manifest_path


class TestValidateCommand:
    """Test the validate command."""

    def test_validate_valid_manifest(self, runner, sample_toml_manifest):
        """Test validating a valid TOML manifest."""
        result = runner.invoke(app, ["validate", str(sample_toml_manifest)])
        assert result.exit_code == 0
        assert "Validating manifest:" in result.stdout
        assert "✓ Manifest validation passed" in result.stdout

    def test_validate_nonexistent_file(self, runner, tmp_path):
        """Test validating a nonexistent file."""
        nonexistent = tmp_path / "nonexistent.toml"
        result = runner.invoke(app, ["validate", str(nonexistent)])
        assert result.exit_code == 1
        assert "Manifest file not found" in result.stdout

    def test_validate_invalid_toml(self, runner, invalid_toml_manifest):
        """Test validating invalid TOML syntax."""
        result = runner.invoke(app, ["validate", str(invalid_toml_manifest)])
        assert result.exit_code == 1
        assert "TOML parsing error" in result.stdout

    def test_validate_empty_modules(self, runner, tmp_path):
        """Test validating manifest with empty module string."""
        manifest_path = tmp_path / "empty-module.toml"
        manifest_path.write_text('modules = ["", "hexdag.core.ports"]\n')
        result = runner.invoke(app, ["validate", str(manifest_path)])
        assert result.exit_code == 1
        assert "is empty" in result.stdout

    def test_validate_invalid_module_type(self, runner, tmp_path):
        """Test validating manifest with non-string module."""
        manifest_path = tmp_path / "bad-type.toml"
        manifest_path.write_text('modules = [123, "hexdag.core.ports"]\n')
        result = runner.invoke(app, ["validate", str(manifest_path)])
        assert result.exit_code == 1
        assert "must be a string" in result.stdout

    def test_validate_nonexistent_module(self, runner, tmp_path):
        """Test validating manifest with module that doesn't exist."""
        manifest_path = tmp_path / "bad-module.toml"
        manifest_path.write_text('modules = ["nonexistent.module.that.does.not.exist"]\n')
        result = runner.invoke(app, ["validate", str(manifest_path)])
        # Should pass validation but show warning
        assert result.exit_code == 0
        assert "⚠" in result.stdout or "import warning" in result.stdout.lower()


class TestBuildCommand:
    """Test the build command."""

    def test_build_default_output(self, runner, tmp_path):
        """Test building manifest with default output path."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            with patch("hexdag.core.bootstrap.bootstrap_registry"):
                with patch("hexdag.core.registry.registry") as mock_registry:
                    # Mock component metadata
                    mock_comp1 = MagicMock()
                    mock_comp1.namespace = "core"
                    mock_comp1.metadata.raw_component.__module__ = "hexdag.core.ports.llm"

                    mock_comp2 = MagicMock()
                    mock_comp2.namespace = "plugin"
                    mock_comp2.metadata.raw_component.__module__ = "hexdag.plugins.test.adapter"

                    mock_registry.list_components.return_value = [mock_comp1, mock_comp2]

                    result = runner.invoke(app, ["build"])
                    assert result.exit_code == 0
                    assert "✓ Manifest written to: hexdag-manifest.toml" in result.stdout
                    assert Path("hexdag-manifest.toml").exists()

    def test_build_custom_output(self, runner, tmp_path):
        """Test building manifest with custom output path."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            output_path = "custom-manifest.toml"
            with patch("hexdag.core.bootstrap.bootstrap_registry"):
                with patch("hexdag.core.registry.registry") as mock_registry:
                    mock_registry.list_components.return_value = []

                    result = runner.invoke(app, ["build", "--out", output_path])
                    assert result.exit_code == 0
                    assert f"✓ Manifest written to: {output_path}" in result.stdout
                    assert Path(output_path).exists()

    def test_build_groups_by_namespace(self, runner, tmp_path):
        """Test that build correctly groups components by namespace."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            with patch("hexdag.core.bootstrap.bootstrap_registry"):
                with patch("hexdag.core.registry.registry") as mock_registry:
                    # Core component
                    core_comp = MagicMock()
                    core_comp.namespace = "core"
                    core_comp.metadata.raw_component.__module__ = "hexdag.builtin.nodes.function"

                    # Plugin component
                    plugin_comp = MagicMock()
                    plugin_comp.namespace = "plugin"
                    plugin_comp.metadata.raw_component.__module__ = "hexdag.plugins.custom.tool"

                    mock_registry.list_components.return_value = [core_comp, plugin_comp]

                    result = runner.invoke(app, ["build"])
                    assert result.exit_code == 0

                    # Check that output contains both modules and plugins sections
                    manifest_content = Path("hexdag-manifest.toml").read_text()
                    assert "modules = [" in manifest_content
                    assert "plugins = [" in manifest_content


class TestDiffCommand:
    """Test the diff command."""

    def test_diff_identical_manifests(self, runner, sample_toml_manifest, tmp_path):
        """Test diffing two identical manifests."""
        # Copy the sample manifest
        manifest2 = tmp_path / "manifest2.toml"
        manifest2.write_text(sample_toml_manifest.read_text())

        result = runner.invoke(app, ["diff", str(sample_toml_manifest), str(manifest2)])
        assert result.exit_code == 0
        assert "No differences found" in result.stdout
        assert "2 → 2" in result.stdout  # Modules count (fixture has 2 modules)
        assert "1 → 1" in result.stdout  # Plugins count

    def test_diff_different_manifests(self, runner, sample_toml_manifest, tmp_path):
        """Test diffing two different manifests."""
        # Create a modified manifest
        manifest2 = tmp_path / "manifest2.toml"
        manifest2_content = """modules = [
    "hexdag.builtin.nodes",
    "hexdag.builtin.tools",
]

plugins = [
    "hexdag.builtin.adapters.local",
    "hexdag.plugins.new",
]
"""
        manifest2.write_text(manifest2_content)

        result = runner.invoke(app, ["diff", str(sample_toml_manifest), str(manifest2)])
        assert result.exit_code == 0
        assert "Added modules:" in result.stdout
        assert "hexdag.builtin.tools" in result.stdout
        assert "Removed modules:" in result.stdout
        assert "hexdag.core.ports" in result.stdout
        assert "Added plugins:" in result.stdout
        assert "hexdag.plugins.new" in result.stdout

    def test_diff_empty_manifests(self, runner, tmp_path):
        """Test diffing two empty manifests."""
        manifest1 = tmp_path / "empty1.toml"
        manifest2 = tmp_path / "empty2.toml"
        manifest1.write_text("")
        manifest2.write_text("")

        result = runner.invoke(app, ["diff", str(manifest1), str(manifest2)])
        assert result.exit_code == 0
        assert "No differences found" in result.stdout

    def test_diff_nonexistent_file(self, runner, sample_toml_manifest, tmp_path):
        """Test diffing with a nonexistent file."""
        nonexistent = tmp_path / "nonexistent.toml"
        result = runner.invoke(app, ["diff", str(sample_toml_manifest), str(nonexistent)])
        assert result.exit_code == 1
        assert "Error loading manifests" in result.stdout

    def test_diff_only_modules_differ(self, runner, tmp_path):
        """Test diff when only modules differ."""
        manifest1 = tmp_path / "m1.toml"
        manifest2 = tmp_path / "m2.toml"

        manifest1.write_text('modules = ["hexdag.core.ports"]\nplugins = ["hexdag.plugins.test"]')
        manifest2.write_text(
            'modules = ["hexdag.builtin.nodes"]\nplugins = ["hexdag.plugins.test"]'
        )

        result = runner.invoke(app, ["diff", str(manifest1), str(manifest2)])
        assert result.exit_code == 0
        assert "Added modules:" in result.stdout
        assert "Removed modules:" in result.stdout
        # Plugins should not show as changed
        assert "Added plugins:" not in result.stdout or "Removed plugins:" not in result.stdout

    def test_diff_only_plugins_differ(self, runner, tmp_path):
        """Test diff when only plugins differ."""
        manifest1 = tmp_path / "m1.toml"
        manifest2 = tmp_path / "m2.toml"

        manifest1.write_text('modules = ["hexdag.core.ports"]\nplugins = ["plugin1"]')
        manifest2.write_text('modules = ["hexdag.core.ports"]\nplugins = ["plugin2"]')

        result = runner.invoke(app, ["diff", str(manifest1), str(manifest2)])
        assert result.exit_code == 0
        assert "Added plugins:" in result.stdout
        assert "Removed plugins:" in result.stdout
        # Modules should not show as changed
        assert "Added modules:" not in result.stdout or "Removed modules:" not in result.stdout
