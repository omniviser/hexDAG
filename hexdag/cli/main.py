"""HexDAG CLI - Main entrypoint."""

import sys

try:
    import typer
    from rich.console import Console
except ImportError:
    print("Error: CLI dependencies not installed.")
    print("Please install with:")
    print("  pip install hexdag[cli]")
    print("  or")
    print("  uv pip install hexdag[cli]")
    sys.exit(1)

from hexdag.cli.commands import (
    build_cmd,
    create_cmd,
    docs_cmd,
    init_cmd,
    pipeline_cmd,
    plugin_dev_cmd,
    plugins_cmd,
    studio_cmd,
    validate_cmd,
)

app = typer.Typer(
    name="hexdag",
    help="HexDAG - Lightweight DAG orchestration framework with hexagonal architecture.",
    no_args_is_help=True,
    rich_markup_mode="rich",
    pretty_exceptions_enable=False,
)

console = Console()

app.add_typer(init_cmd.app, name="init", help="Initialize a new HexDAG project")
app.add_typer(plugins_cmd.app, name="plugins", help="Manage plugins and adapters")
app.add_typer(plugin_dev_cmd.app, name="plugin", help="Plugin development commands")
app.add_typer(pipeline_cmd.app, name="pipeline", help="Pipeline validation and execution")
app.add_typer(create_cmd.app, name="create", help="Create pipeline templates from schemas")
app.add_typer(docs_cmd.app, name="docs", help="Generate and serve documentation")
app.add_typer(studio_cmd.app, name="studio", help="Visual editor for pipelines")

app.command(name="build", help="Build Docker containers for pipelines")(build_cmd.build)
app.command(name="validate", help="Validate YAML pipeline files")(validate_cmd.validate)


@app.callback(invoke_without_command=True)
def callback(
    ctx: typer.Context,
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        help="Show version and exit",
    ),
) -> None:
    """HexDAG CLI - Modular DAG orchestration framework."""
    if version:
        # Read version from package metadata
        try:
            from importlib.metadata import version as get_version

            pkg_version = get_version("hexdag")
        except Exception:
            pkg_version = "0.1.0"  # Fallback

        console.print(f"[bold blue]HexDAG[/bold blue] version [green]{pkg_version}[/green]")
        raise typer.Exit(code=0)


def main() -> None:
    """Main CLI entrypoint."""
    app()


if __name__ == "__main__":
    main()
