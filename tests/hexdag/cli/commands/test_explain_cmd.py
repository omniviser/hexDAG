"""Tests for hexdag.cli.commands.explain_cmd module."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import typer

from hexdag.cli.commands.explain_cmd import explain


def _run(resource: str | None = None, name: str | None = None) -> None:
    """Run explain with patched console, accepting exit code 0."""
    with patch("hexdag.cli.commands.explain_cmd.console"):
        try:
            explain(resource=resource, name=name)
        except (SystemExit, typer.Exit) as e:
            code = getattr(e, "exit_code", getattr(e, "code", 0))
            if code not in (0, None):
                raise


class TestExplainOverview:
    def test_no_args_shows_overview(self) -> None:
        _run()

    def test_unknown_resource_exits_1(self) -> None:
        with (
            patch("hexdag.cli.commands.explain_cmd.console"),
            pytest.raises((SystemExit, typer.Exit)),
        ):
            explain(resource="nonexistent_xyz")


class TestExplainNode:
    def test_list_nodes(self) -> None:
        _run("node")

    def test_specific_node(self) -> None:
        _run("node", "llm_node")

    def test_nodes_alias(self) -> None:
        _run("nodes")

    def test_unknown_node_exits_1(self) -> None:
        """Unknown node falls back to get_component_schema which errors."""
        with (
            patch("hexdag.cli.commands.explain_cmd.console"),
            pytest.raises((SystemExit, typer.Exit)),
        ):
            explain(resource="node", name="nonexistent_node_xyz")


class TestExplainAdapter:
    def test_list_adapters(self) -> None:
        _run("adapter")

    def test_adapters_alias(self) -> None:
        _run("adapters")


class TestExplainMiddleware:
    def test_list_middleware(self) -> None:
        _run("middleware")

    def test_specific_middleware(self) -> None:
        _run("middleware", "RetryWithBackoff")

    def test_unknown_middleware_exits_1(self) -> None:
        with (
            patch("hexdag.cli.commands.explain_cmd.console"),
            pytest.raises((SystemExit, typer.Exit)),
        ):
            explain(resource="middleware", name="NonexistentMW")


class TestExplainMacro:
    def test_list_macros(self) -> None:
        _run("macro")


class TestExplainTag:
    def test_list_tags(self) -> None:
        _run("tag")


class TestExplainDocs:
    def test_syntax(self) -> None:
        _run("syntax")

    def test_types(self) -> None:
        _run("types")

    def test_type_alias(self) -> None:
        _run("type")
