"""CLI helper utilities for hexdag commands."""

from __future__ import annotations

import json
from typing import Any, Protocol

import typer
import yaml
from rich.console import Console


class ContextProtocol(Protocol):
    """Protocol for common interface between Click and Typer contexts."""

    @property
    def obj(self) -> dict[str, Any] | None: ...


console = Console()


def print_output(obj: Any, ctx: ContextProtocol | None = None) -> None:
    """Print `obj` according to `ctx.obj['output_format']`.

    If ctx is None or no format specified, pretty-print using rich.console.
    """
    fmt = None
    if ctx is not None:
        obj = getattr(ctx, "obj", None)
        if isinstance(obj, dict):  # type narrowing for mypy
            fmt = obj.get("output_format")

    if fmt == "json":
        typer.echo(json.dumps(obj, default=str, indent=2))
    elif fmt == "yaml":
        typer.echo(yaml.safe_dump(obj, sort_keys=False))
    else:
        # Fallback: pretty print (let callers format objects to strings)
        if isinstance(obj, (str, int, float)):
            typer.echo(str(obj))
        else:
            # Best-effort pretty print via rich
            console.print(obj)
