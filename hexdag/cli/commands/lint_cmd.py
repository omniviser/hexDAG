"""Pipeline linting command for HexDAG CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
import yaml
from rich.console import Console
from rich.table import Table

from hexdag.kernel.linting.pipeline_rules import ALL_PIPELINE_RULES
from hexdag.kernel.linting.rules import run_rules

if TYPE_CHECKING:
    from hexdag.kernel.linting.models import LintReport

console = Console()

_CLI_NAME = "lint"
_CLI_HELP = "Lint YAML pipeline files for best practices"
_CLI_TYPE = "command"
_CLI_FUNC = "lint"

_SEVERITY_RANK = {"error": 0, "warning": 1, "info": 2}


def lint(
    yaml_file: Annotated[
        Path,
        typer.Argument(
            help="Path to YAML pipeline file to lint",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    severity: Annotated[
        str,
        typer.Option(
            "--severity",
            "-s",
            help="Minimum severity to report (error, warning, info)",
        ),
    ] = "info",
    output_format: Annotated[
        str,
        typer.Option(
            "--format",
            "-f",
            help="Output format (text, json)",
        ),
    ] = "text",
    disable: Annotated[
        str,
        typer.Option(
            "--disable",
            "-d",
            help="Comma-separated rule IDs to skip (e.g., W200,W201)",
        ),
    ] = "",
) -> None:
    """Lint a YAML pipeline file for best practices and potential issues.

    Checks include cycle detection, unresolvable node kinds, missing
    timeout/retry on LLM nodes, unused outputs, hardcoded secrets,
    and naming conventions.

    Examples
    --------
    hexdag lint pipeline.yaml
    hexdag lint pipeline.yaml --severity warning
    hexdag lint pipeline.yaml --format json
    hexdag lint pipeline.yaml --disable W200,W201
    """
    if severity not in _SEVERITY_RANK:
        console.print(
            f"[red]Invalid severity '{severity}'.[/red] Choose from: error, warning, info"
        )
        raise typer.Exit(1)

    # Read YAML file
    try:
        with Path.open(yaml_file) as f:
            config = yaml.safe_load(f.read())
    except yaml.YAMLError as e:
        console.print(f"[red]YAML Syntax Error:[/red] {e}")
        raise typer.Exit(1) from e
    except OSError as e:
        console.print(f"[red]File Error:[/red] {e}")
        raise typer.Exit(1) from e

    if not isinstance(config, dict):
        console.print("[red]File does not contain a valid YAML mapping.[/red]")
        raise typer.Exit(1)

    # Filter rules based on --disable flag
    disabled_ids = (
        {r.strip().upper() for r in disable.split(",") if r.strip()} if disable else set()
    )

    if disabled_ids:
        known_ids = {r.rule_id for r in ALL_PIPELINE_RULES}
        unknown = disabled_ids - known_ids
        if unknown:
            console.print(
                f"[yellow]Unknown rule ID(s): {', '.join(sorted(unknown))}[/yellow]  "
                f"Known: {', '.join(sorted(known_ids))}"
            )
        active_rules = [r for r in ALL_PIPELINE_RULES if r.rule_id not in disabled_ids]
    else:
        active_rules = list(ALL_PIPELINE_RULES)

    report = run_rules(active_rules, config)

    # Filter by severity
    min_rank = _SEVERITY_RANK[severity]
    filtered = [v for v in report.violations if _SEVERITY_RANK[v.severity] <= min_rank]

    if output_format == "json":
        _print_json(filtered)
    else:
        _print_text(yaml_file, filtered, report)

    # Exit with error code if any error-level violations
    if report.has_errors:
        raise typer.Exit(1)


def _print_text(
    yaml_file: Path,
    filtered: list,
    report: LintReport,
) -> None:
    """Print lint results as rich text."""
    console.print()

    if not filtered:
        console.print(f"[green]No issues found:[/green] {yaml_file}")
        console.print()
        return

    # Summary line
    n_err = len(report.errors)
    n_warn = len(report.warnings)
    n_info = len(report.info)
    console.print(
        f"[bold]{yaml_file}[/bold]  "
        f"[red]{n_err} error(s)[/red]  "
        f"[yellow]{n_warn} warning(s)[/yellow]  "
        f"[blue]{n_info} info[/blue]"
    )
    console.print()

    # Table
    table = Table(show_header=True, border_style="dim")
    table.add_column("Rule", style="cyan", width=6)
    table.add_column("Severity", width=8)
    table.add_column("Location", style="green")
    table.add_column("Message")
    table.add_column("Suggestion", style="dim")

    severity_style = {"error": "red", "warning": "yellow", "info": "blue"}

    for v in filtered:
        style = severity_style.get(v.severity, "white")
        table.add_row(
            v.rule_id,
            f"[{style}]{v.severity}[/{style}]",
            v.location,
            v.message,
            v.suggestion or "",
        )

    console.print(table)
    console.print()


def _print_json(filtered: list) -> None:
    """Print lint results as JSON."""
    output = [
        {
            "rule_id": v.rule_id,
            "severity": v.severity,
            "location": v.location,
            "message": v.message,
            "suggestion": v.suggestion,
        }
        for v in filtered
    ]
    console.print(json.dumps(output, indent=2))
