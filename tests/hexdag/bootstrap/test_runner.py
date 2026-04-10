"""Tests for hexdag.bootstrap.runner — self-compile execution."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from hexdag.bootstrap.runner import (
    BootstrapError,
    _resolve_deps,
    _STAGE_DEPS,
    self_compile_pipeline_path,
)


# ---------------------------------------------------------------------------
# Pipeline path
# ---------------------------------------------------------------------------


class TestSelfCompilePipelinePath:
    """Verify the self-compile pipeline YAML exists."""

    def test_path_exists(self) -> None:
        path = self_compile_pipeline_path()
        assert path.exists()
        assert path.suffix == ".yaml"

    def test_path_is_absolute(self) -> None:
        path = self_compile_pipeline_path()
        assert path.is_absolute()

    def test_yaml_is_valid(self) -> None:
        import yaml

        path = self_compile_pipeline_path()
        data = yaml.safe_load(path.read_text())
        assert data["apiVersion"] == "hexdag/v1"
        assert data["kind"] == "Pipeline"
        assert data["metadata"]["name"] == "hexdag-self-compile"
        assert len(data["spec"]["nodes"]) >= 8


# ---------------------------------------------------------------------------
# Dependency resolution
# ---------------------------------------------------------------------------


class TestResolveDeps:
    """Test transitive dependency resolution."""

    def test_no_deps(self) -> None:
        result = _resolve_deps({"lint"}, _STAGE_DEPS)
        assert result == {"lint"}

    def test_direct_dep(self) -> None:
        result = _resolve_deps({"validate_architecture"}, _STAGE_DEPS)
        assert "lint" in result
        assert "validate_architecture" in result

    def test_transitive_deps(self) -> None:
        result = _resolve_deps({"build_package"}, _STAGE_DEPS)
        # build_package -> run_tests -> lint, typecheck
        # build_package -> validate_architecture -> lint
        # build_package -> validate_self -> lint
        assert "lint" in result
        assert "typecheck" in result
        assert "run_tests" in result
        assert "validate_architecture" in result
        assert "validate_self" in result
        assert "build_package" in result

    def test_full_chain(self) -> None:
        result = _resolve_deps({"validate_package"}, _STAGE_DEPS)
        # Should include everything except format_check (no stage depends on it)
        expected = set(_STAGE_DEPS.keys()) - {"format_check"}
        assert result == expected

    def test_multiple_stages(self) -> None:
        result = _resolve_deps({"lint", "typecheck"}, _STAGE_DEPS)
        assert result == {"lint", "typecheck"}


# ---------------------------------------------------------------------------
# Stage dependency graph integrity
# ---------------------------------------------------------------------------


class TestStageDeps:
    """Verify the stage dependency graph is well-formed."""

    def test_no_cycles(self) -> None:
        """Stage dependency graph must be acyclic."""
        from hexdag.kernel.domain.dag import DirectedGraph

        # Convert to the format DirectedGraph.detect_cycle expects
        graph = {k: set(v) for k, v in _STAGE_DEPS.items()}
        assert DirectedGraph.detect_cycle(graph) is None

    def test_all_deps_exist(self) -> None:
        """Every dependency must reference an existing stage."""
        all_stages = set(_STAGE_DEPS.keys())
        for stage, deps in _STAGE_DEPS.items():
            for dep in deps:
                assert dep in all_stages, f"{stage} depends on unknown stage: {dep}"

    def test_matches_yaml(self) -> None:
        """Programmatic deps must match the YAML pipeline topology."""
        import yaml

        path = self_compile_pipeline_path()
        data = yaml.safe_load(path.read_text())
        nodes = data["spec"]["nodes"]

        yaml_deps: dict[str, list[str]] = {}
        for node in nodes:
            name = node["metadata"]["name"]
            deps = node.get("dependencies", [])
            yaml_deps[name] = sorted(deps)

        for stage_name, py_deps in _STAGE_DEPS.items():
            assert stage_name in yaml_deps, f"Stage {stage_name} not in YAML"
            assert sorted(py_deps) == yaml_deps[stage_name], (
                f"Dependency mismatch for {stage_name}: "
                f"Python={sorted(py_deps)}, YAML={yaml_deps[stage_name]}"
            )


# ---------------------------------------------------------------------------
# Programmatic runner
# ---------------------------------------------------------------------------


class TestProgrammaticRunner:
    """Test the programmatic (Stage 0) bootstrap path."""

    @pytest.mark.asyncio
    async def test_runs_all_stages(self) -> None:
        """Run all stages with mocked subprocess calls."""
        mock = AsyncMock(return_value=(0, "ok", ""))

        with patch("hexdag.bootstrap.stages._run_command", mock):
            from hexdag.bootstrap.runner import _run_programmatic

            results = await _run_programmatic(stages="lint,format_check")

        assert "lint" in results
        assert "format_check" in results

    @pytest.mark.asyncio
    async def test_fail_fast(self) -> None:
        """BootstrapError raised on first failure when fail_fast=True."""
        # Mock _run_command to return failure for all calls (lint runs first)
        mock = AsyncMock(return_value=(1, "", "E1"))

        with patch("hexdag.bootstrap.stages._run_command", mock):
            from hexdag.bootstrap.runner import _run_programmatic

            with pytest.raises(BootstrapError, match="lint"):
                await _run_programmatic(stages="lint", fail_fast=True)


# ---------------------------------------------------------------------------
# BootstrapError
# ---------------------------------------------------------------------------


class TestBootstrapError:
    """Test the BootstrapError exception."""

    def test_is_exception(self) -> None:
        assert issubclass(BootstrapError, Exception)

    def test_message(self) -> None:
        err = BootstrapError("stage failed")
        assert "stage failed" in str(err)
