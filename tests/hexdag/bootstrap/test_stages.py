"""Tests for hexdag.bootstrap.stages — self-compile build stages."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from hexdag.bootstrap import stages


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_run(rc: int = 0, stdout: str = "ok", stderr: str = "") -> AsyncMock:
    """Create a mock for _run_command that returns a fixed result."""
    return AsyncMock(return_value=(rc, stdout, stderr))


# ---------------------------------------------------------------------------
# _project_root
# ---------------------------------------------------------------------------


class TestProjectRoot:
    """Test project root detection."""

    def test_finds_root(self) -> None:
        root = stages._project_root()
        assert (root / "pyproject.toml").exists()
        assert (root / "hexdag").is_dir()

    def test_returns_path(self) -> None:
        root = stages._project_root()
        assert isinstance(root, Path)


# ---------------------------------------------------------------------------
# _stage_result
# ---------------------------------------------------------------------------


class TestStageResult:
    """Test the _stage_result helper."""

    def test_basic_result(self) -> None:
        result = stages._stage_result("lint", passed=True, output="clean")
        assert result["stage"] == "lint"
        assert result["passed"] is True
        assert result["output"] == "clean"
        assert "errors" not in result

    def test_with_errors(self) -> None:
        result = stages._stage_result("lint", passed=False, output="", errors="E1")
        assert result["errors"] == "E1"
        assert result["passed"] is False

    def test_extra_keys(self) -> None:
        result = stages._stage_result("build", passed=True, output="ok", artifacts=["a.whl"])
        assert result["artifacts"] == ["a.whl"]


# ---------------------------------------------------------------------------
# Individual stages (mocked subprocess)
# ---------------------------------------------------------------------------


class TestLint:
    """Test the lint stage."""

    @pytest.mark.asyncio
    async def test_lint_passes(self) -> None:
        with patch.object(stages, "_run_command", _fake_run(rc=0, stdout="All checks passed")):
            result = await stages.lint({})
        assert result["stage"] == "lint"
        assert result["passed"] is True

    @pytest.mark.asyncio
    async def test_lint_fails(self) -> None:
        with patch.object(stages, "_run_command", _fake_run(rc=1, stderr="E501")):
            result = await stages.lint({})
        assert result["passed"] is False
        assert "E501" in result["errors"]


class TestFormatCheck:
    """Test the format_check stage."""

    @pytest.mark.asyncio
    async def test_format_passes(self) -> None:
        with patch.object(stages, "_run_command", _fake_run(rc=0)):
            result = await stages.format_check({})
        assert result["stage"] == "format_check"
        assert result["passed"] is True

    @pytest.mark.asyncio
    async def test_format_fails(self) -> None:
        with patch.object(stages, "_run_command", _fake_run(rc=1, stderr="Would reformat")):
            result = await stages.format_check({})
        assert result["passed"] is False


class TestTypecheck:
    """Test the typecheck stage."""

    @pytest.mark.asyncio
    async def test_typecheck_passes(self) -> None:
        with patch.object(stages, "_run_command", _fake_run(rc=0, stdout="0 errors")):
            result = await stages.typecheck({})
        assert result["stage"] == "typecheck"
        assert result["passed"] is True


class TestValidateArchitecture:
    """Test the validate_architecture stage."""

    @pytest.mark.asyncio
    async def test_all_checks_pass(self) -> None:
        with patch.object(stages, "_run_command", _fake_run(rc=0, stdout="ok")):
            result = await stages.validate_architecture({})
        assert result["stage"] == "validate_architecture"
        assert result["passed"] is True
        assert "checks" in result
        assert len(result["checks"]) > 0

    @pytest.mark.asyncio
    async def test_check_failure(self) -> None:
        call_count = 0

        async def _alternating(*args: Any, **kwargs: Any) -> tuple[int, str, str]:
            nonlocal call_count
            call_count += 1
            # Fail on second check
            if call_count == 2:
                return (1, "", "boundary violation")
            return (0, "ok", "")

        with patch.object(stages, "_run_command", side_effect=_alternating):
            result = await stages.validate_architecture({})
        assert result["passed"] is False


class TestValidateSelf:
    """Test the validate_self stage (meta-recursive validation)."""

    @pytest.mark.asyncio
    async def test_pipeline_exists(self) -> None:
        from hexdag.bootstrap.runner import self_compile_pipeline_path

        path = self_compile_pipeline_path()
        assert path.exists(), f"Self-compile pipeline not found at {path}"

    @pytest.mark.asyncio
    async def test_validate_self_runs(self) -> None:
        # This test exercises the actual YAML validation path --
        # it's the meta-recursive proof that the pipeline is valid.
        result = await stages.validate_self({})
        assert result["stage"] == "validate_self"
        # If hexDAG's YAML builder can parse the self-compile pipeline,
        # the system is self-consistent.
        assert result["passed"] is True, f"Self-validation failed: {result.get('errors')}"


class TestRunTests:
    """Test the run_tests stage."""

    @pytest.mark.asyncio
    async def test_tests_pass(self) -> None:
        with patch.object(stages, "_run_command", _fake_run(rc=0, stdout="42 passed")):
            result = await stages.run_tests({})
        assert result["stage"] == "run_tests"
        assert result["passed"] is True


class TestBuildPackage:
    """Test the build_package stage."""

    @pytest.mark.asyncio
    async def test_build_passes(self) -> None:
        with (
            patch.object(stages, "_run_command", _fake_run(rc=0, stdout="Built")),
            patch("shutil.rmtree"),
        ):
            result = await stages.build_package({})
        assert result["stage"] == "build_package"


class TestValidatePackage:
    """Test the validate_package stage."""

    @pytest.mark.asyncio
    async def test_no_dist_dir(self) -> None:
        with patch.object(stages, "_project_root", return_value=Path("/nonexistent")):
            result = await stages.validate_package({})
        assert result["passed"] is False
        assert "not found" in result["errors"]


# ---------------------------------------------------------------------------
# _run_command
# ---------------------------------------------------------------------------


class TestRunCommand:
    """Test the _run_command helper."""

    @pytest.mark.asyncio
    async def test_successful_command(self) -> None:
        rc, stdout, stderr = await stages._run_command("echo", "hello")
        assert rc == 0
        assert "hello" in stdout

    @pytest.mark.asyncio
    async def test_failed_command(self) -> None:
        rc, stdout, stderr = await stages._run_command("false")
        assert rc != 0

    @pytest.mark.asyncio
    async def test_timeout(self) -> None:
        rc, stdout, stderr = await stages._run_command(
            "sleep", "10", timeout=0.1
        )
        assert rc == 1
        assert "timed out" in stderr.lower()
