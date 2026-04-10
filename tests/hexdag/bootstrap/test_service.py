"""Tests for hexdag.bootstrap.service — BootstrapService."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from hexdag.bootstrap.service import BootstrapService
from hexdag.kernel.service import _is_step, _is_tool


# ---------------------------------------------------------------------------
# Service introspection
# ---------------------------------------------------------------------------


class TestBootstrapServiceIntrospection:
    """Verify the service exposes the expected tools and steps."""

    def test_has_tools(self) -> None:
        svc = BootstrapService()
        tools = svc.get_tools()
        assert "lint" in tools
        assert "typecheck" in tools
        assert "self_compile" in tools

    def test_has_steps(self) -> None:
        svc = BootstrapService()
        steps = svc.get_steps()
        assert "lint" in steps
        assert "run_tests" in steps
        assert "validate_self" in steps

    def test_lint_is_both_tool_and_step(self) -> None:
        svc = BootstrapService()
        assert _is_tool(svc.lint)
        assert _is_step(svc.lint)

    def test_self_compile_is_tool_only(self) -> None:
        svc = BootstrapService()
        assert _is_tool(svc.self_compile)
        # self_compile is not a @step (it orchestrates the whole pipeline)
        assert not _is_step(svc.self_compile)

    def test_repr(self) -> None:
        svc = BootstrapService()
        r = repr(svc)
        assert "BootstrapService" in r


# ---------------------------------------------------------------------------
# Method delegation
# ---------------------------------------------------------------------------


class TestBootstrapServiceDelegation:
    """Verify service methods delegate to stage functions."""

    @pytest.mark.asyncio
    async def test_lint_delegates(self) -> None:
        mock = AsyncMock(return_value={"stage": "lint", "passed": True, "output": "ok"})
        svc = BootstrapService()
        with patch("hexdag.bootstrap.service.stages.lint", mock):
            result = await svc.lint()
        mock.assert_awaited_once_with({})
        assert result["passed"] is True

    @pytest.mark.asyncio
    async def test_lint_passes_input(self) -> None:
        mock = AsyncMock(return_value={"stage": "lint", "passed": True, "output": "ok"})
        svc = BootstrapService()
        data: dict[str, Any] = {"key": "value"}
        with patch("hexdag.bootstrap.service.stages.lint", mock):
            await svc.lint(input_data=data)
        mock.assert_awaited_once_with(data)

    @pytest.mark.asyncio
    async def test_typecheck_delegates(self) -> None:
        mock = AsyncMock(return_value={"stage": "typecheck", "passed": True, "output": "ok"})
        svc = BootstrapService()
        with patch("hexdag.bootstrap.service.stages.typecheck", mock):
            result = await svc.typecheck()
        mock.assert_awaited_once()
        assert result["stage"] == "typecheck"

    @pytest.mark.asyncio
    async def test_validate_self_delegates(self) -> None:
        mock = AsyncMock(return_value={"stage": "validate_self", "passed": True, "output": "ok"})
        svc = BootstrapService()
        with patch("hexdag.bootstrap.service.stages.validate_self", mock):
            result = await svc.validate_self()
        assert result["passed"] is True
