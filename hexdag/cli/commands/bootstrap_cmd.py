"""Bootstrap command -- hexDAG compiles itself.

``hexdag bootstrap`` executes the self-compile pipeline, orchestrating
lint, type-check, architecture validation, tests, and packaging through
hexDAG's own DAG engine.  This is the proof that hexDAG is a
self-compiling operating system.
"""

import asyncio
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer()
console = Console()

_CLI_NAME = "bootstrap"
_CLI_HELP = "Self-compile hexDAG using its own pipeline engine"
_CLI_TYPE = "command"
_CLI_FUNC = "bootstrap"


@app.command()
def bootstrap(
    stages: Annotated[
        str,
        typer.Option(
            "--stages",
            "-s",
            help=(
                "Comma-separated stages to run (default: all). "
                "Available: lint, format_check, typecheck, "
                "validate_architecture, validate_self, run_tests, "
                "build_package, validate_package"
            ),
        ),
    ] = "all",
    programmatic: Annotated[
        bool,
        typer.Option(
            "--programmatic",
            "-p",
            help=(
                "Use programmatic graph construction (Stage 0 bootstrap) "
                "instead of loading the YAML pipeline."
            ),
        ),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Show detailed output for each stage.",
        ),
    ] = False,
    no_fail: Annotated[
        bool,
        typer.Option(
            "--no-fail",
            help="Continue even if a stage fails (report at the end).",
        ),
    ] = False,
) -> None:
    """Self-compile hexDAG using its own pipeline engine.

    hexDAG is a self-compiling operating system: this command uses hexDAG's
    own orchestrator to build, validate, test, and package hexDAG itself.

    The pipeline reads hexdag/bootstrap/pipelines/self_compile.yaml and
    executes it as a regular hexDAG DAG -- the same engine that runs
    user pipelines now builds the engine itself.

    Examples
    --------
    Full self-compile::

        hexdag bootstrap

    Run only lint and type-check::

        hexdag bootstrap --stages lint,typecheck

    Stage 0 bootstrap (programmatic, no YAML)::

        hexdag bootstrap --programmatic

    Verbose output::

        hexdag bootstrap --verbose
    """
    console.print(
        Panel(
            "[bold]hexDAG Self-Compile[/bold]\nBuilding hexDAG using hexDAG's own pipeline engine.",
            border_style="blue",
        )
    )

    mode = "programmatic (Stage 0)" if programmatic else "YAML pipeline (self-compiling)"
    console.print(f"[dim]Mode: {mode}[/dim]")
    console.print(f"[dim]Stages: {stages}[/dim]\n")

    from hexdag.bootstrap.runner import BootstrapError, run_self_compile

    try:
        results = asyncio.run(
            run_self_compile(
                stages=stages,
                use_yaml=not programmatic,
                fail_fast=not no_fail,
            )
        )
    except BootstrapError as e:
        console.print(f"\n[red]Bootstrap failed:[/red] {e}")
        raise typer.Exit(1) from e
    except Exception as e:
        console.print(f"\n[red]Unexpected error:[/red] {e}")
        if verbose:
            import traceback

            console.print(f"[dim]{traceback.format_exc()}[/dim]")
        raise typer.Exit(1) from e

    # Display results
    _display_results(results, verbose=verbose)

    # Check for failures
    failures = [
        name
        for name, result in results.items()
        if isinstance(result, dict) and not result.get("passed", True)
    ]

    if failures:
        console.print(f"\n[red]Failed stages: {', '.join(failures)}[/red]")
        raise typer.Exit(1)

    console.print(
        Panel(
            "[bold green]Self-compile complete.[/bold green]\nhexDAG successfully built itself.",
            border_style="green",
        )
    )


def _display_results(results: dict, *, verbose: bool = False) -> None:
    """Display bootstrap results in a rich table."""
    table = Table(
        title="Self-Compile Results",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Stage", style="white")
    table.add_column("Status", justify="center")
    table.add_column("Details", style="dim")

    for name, result in results.items():
        if isinstance(result, dict):
            passed = result.get("passed", True)
            status = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
            details = ""
            if verbose:
                output = result.get("output", "")
                errors = result.get("errors", "")
                if errors and not passed:
                    details = errors[:120]
                elif output:
                    details = output[:120]
        else:
            status = "[yellow]?[/yellow]"
            details = str(result)[:120] if verbose else ""

        table.add_row(name, status, details)

    console.print(table)
