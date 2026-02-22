"""Core models for the hexDAG linting framework."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class LintViolation:
    """A single lint violation found during analysis."""

    rule_id: str
    severity: Literal["error", "warning", "info"]
    message: str
    location: str = ""
    suggestion: str | None = None


class LintReport:
    """Aggregated lint results from running rules against a pipeline."""

    __slots__ = ("_violations",)

    def __init__(self) -> None:
        """Initialize an empty lint report."""
        self._violations: list[LintViolation] = []

    def add(self, violation: LintViolation) -> None:
        """Add a violation to the report."""
        self._violations.append(violation)

    @property
    def violations(self) -> list[LintViolation]:
        """All violations."""
        return self._violations

    @property
    def errors(self) -> list[LintViolation]:
        """Violations with severity 'error'."""
        return [v for v in self._violations if v.severity == "error"]

    @property
    def warnings(self) -> list[LintViolation]:
        """Violations with severity 'warning'."""
        return [v for v in self._violations if v.severity == "warning"]

    @property
    def info(self) -> list[LintViolation]:
        """Violations with severity 'info'."""
        return [v for v in self._violations if v.severity == "info"]

    @property
    def is_clean(self) -> bool:
        """True if no violations were found."""
        return len(self._violations) == 0

    @property
    def has_errors(self) -> bool:
        """True if any error-level violations exist."""
        return any(v.severity == "error" for v in self._violations)
