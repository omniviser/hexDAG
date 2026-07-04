"""Structured compiler diagnostics.

One diagnostic model for every stage of the compiler. All user-facing
configuration problems are reported as :class:`Diagnostic` instances with a
stable ``HDnnnn`` code, a severity, and (once provenance lands) a source
location. Consumers (CLI, API, MCP, Studio) render the same objects.

Code bands (by stage):
- HD00xx  legacy/uncategorized (provisional wrappers during migration)
- HD01xx  parse / include / preprocessing
- HD02xx  manifest & node structure
- HD03xx  references, dependencies, cycles
- HD04xx  naming, expressions, templates
- HD05xx  kinds, macros, ports
- HD06xx  environments & systems
- HD09xx  deprecations
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

type Severity = Literal["error", "warning", "info"]

#: Provisional codes for string-based rules that have not yet migrated to
#: dedicated rule modules. They carry no semantic meaning beyond severity.
LEGACY_ERROR = "HD0001"
LEGACY_WARNING = "HD0002"
LEGACY_INFO = "HD0003"


@dataclass(frozen=True, slots=True)
class Location:
    """A source location: file, 1-based line/column, and structural path."""

    file: str | None = None
    line: int | None = None
    column: int | None = None
    path: tuple[str | int, ...] = ()

    def __str__(self) -> str:
        """Render as ``file:line:col``."""
        parts = [self.file or "<string>"]
        if self.line is not None:
            parts.append(str(self.line))
            if self.column is not None:
                parts.append(str(self.column))
        return ":".join(parts)


@dataclass(frozen=True, slots=True)
class Diagnostic:
    """A single user-facing compiler finding."""

    code: str
    severity: Severity
    message: str
    hint: str | None = None
    loc: Location | None = None
    stage: str = ""

    def render(self) -> str:
        """One-line human rendering: ``file:line:col: severity[code] message``."""
        prefix = f"{self.loc}: " if self.loc and self.loc.file else ""
        return f"{prefix}{self.severity}[{self.code}] {self.message}"

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly representation (API / MCP / Studio)."""
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "hint": self.hint,
            "file": self.loc.file if self.loc else None,
            "line": self.loc.line if self.loc else None,
            "column": self.loc.column if self.loc else None,
        }


@dataclass(slots=True)
class DiagnosticSink:
    """Mutable collector rules emit into; ordered, append-only."""

    diagnostics: list[Diagnostic] = field(default_factory=list)

    def emit(
        self,
        code: str,
        severity: Severity,
        message: str,
        *,
        hint: str | None = None,
        loc: Location | None = None,
        stage: str = "",
    ) -> None:
        """Append a diagnostic."""
        self.diagnostics.append(
            Diagnostic(
                code=code, severity=severity, message=message, hint=hint, loc=loc, stage=stage
            )
        )

    def error(self, code: str, message: str, **kwargs: Any) -> None:
        """Emit an error diagnostic."""
        self.emit(code, "error", message, **kwargs)

    def warning(self, code: str, message: str, **kwargs: Any) -> None:
        """Emit a warning diagnostic."""
        self.emit(code, "warning", message, **kwargs)

    def info(self, code: str, message: str, **kwargs: Any) -> None:
        """Emit an info diagnostic."""
        self.emit(code, "info", message, **kwargs)

    @property
    def has_errors(self) -> bool:
        """Whether any error-severity diagnostic was emitted."""
        return any(d.severity == "error" for d in self.diagnostics)

    def extend(self, other: DiagnosticSink | list[Diagnostic]) -> None:
        """Append diagnostics from another sink or list."""
        items = other.diagnostics if isinstance(other, DiagnosticSink) else other
        self.diagnostics.extend(items)
