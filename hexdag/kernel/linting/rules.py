"""Lint rule protocol and runner for hexDAG pipelines."""

from __future__ import annotations

from typing import Any, Protocol

from hexdag.kernel.linting.models import LintReport, LintViolation


class LintRule(Protocol):
    """Protocol for a single lint rule."""

    rule_id: str
    severity: str
    description: str

    def check(self, config: dict[str, Any]) -> list[LintViolation]:
        """Run this rule against the config and return violations."""
        ...


def run_rules(rules: list[LintRule], config: dict[str, Any]) -> LintReport:
    """Run a list of lint rules against a pipeline config and return a report."""
    report = LintReport()
    for rule in rules:
        for violation in rule.check(config):
            report.add(violation)
    return report
