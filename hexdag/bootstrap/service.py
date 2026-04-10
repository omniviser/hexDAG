"""BootstrapService -- hexDAG compiles itself via the Service abstraction.

This service wraps every self-compile stage behind ``@step`` (deterministic
DAG node) and ``@tool`` (agent-callable) decorators so they can be used
both in YAML pipelines and by AI agents that manage the build.

Example (YAML)::

    spec:
      services:
        bootstrap:
          class: hexdag.bootstrap.service.BootstrapService

Example (programmatic)::

    svc = BootstrapService()
    result = await svc.lint()
"""

from __future__ import annotations

from typing import Any

from hexdag.bootstrap import stages
from hexdag.kernel.service import Service, step, tool


class BootstrapService(Service):
    """Service that exposes hexDAG's self-compile stages.

    Every public method wraps the corresponding function in
    :mod:`hexdag.bootstrap.stages` so it can participate in
    both agent tool-calling and deterministic DAG execution.
    """

    @tool
    @step
    async def lint(self, input_data: dict[str, Any] | None = None) -> dict[str, Any]:
        """Run ruff linter on hexdag source."""
        return await stages.lint(input_data or {})

    @tool
    @step
    async def format_check(self, input_data: dict[str, Any] | None = None) -> dict[str, Any]:
        """Verify code formatting with ruff."""
        return await stages.format_check(input_data or {})

    @tool
    @step
    async def typecheck(self, input_data: dict[str, Any] | None = None) -> dict[str, Any]:
        """Run pyright type checker."""
        return await stages.typecheck(input_data or {})

    @tool
    @step
    async def validate_architecture(
        self,
        input_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run architecture validation checks."""
        return await stages.validate_architecture(input_data or {})

    @tool
    @step
    async def validate_self(self, input_data: dict[str, Any] | None = None) -> dict[str, Any]:
        """Validate the self-compile pipeline using hexDAG's own validator."""
        return await stages.validate_self(input_data or {})

    @tool
    @step
    async def run_tests(self, input_data: dict[str, Any] | None = None) -> dict[str, Any]:
        """Run the hexDAG test suite."""
        return await stages.run_tests(input_data or {})

    @tool
    @step
    async def build_package(self, input_data: dict[str, Any] | None = None) -> dict[str, Any]:
        """Build Python package (wheel + sdist)."""
        return await stages.build_package(input_data or {})

    @tool
    @step
    async def validate_package(self, input_data: dict[str, Any] | None = None) -> dict[str, Any]:
        """Validate built package artifacts."""
        return await stages.validate_package(input_data or {})

    @tool
    async def self_compile(self, stages_to_run: str = "all") -> dict[str, Any]:
        """Run the full self-compile pipeline.

        Parameters
        ----------
        stages_to_run : str
            Comma-separated list of stages, or ``"all"`` for the full pipeline.
        """
        from hexdag.bootstrap.runner import run_self_compile

        return await run_self_compile(stages=stages_to_run)
