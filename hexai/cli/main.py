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

from hexai.cli.commands import (
    adapters_cmd,
    config_cmd,
    events_cmd,
    init_cmd,
    manifest_cmd,
    nodes_cmd,
    observers_cmd,
    pipeline_cmd,
    plugin_dev_cmd,
    plugins_cmd,
    ports_cmd,
    registry_cmd,
    run_cmd,
    runs_cmd,
    validation_cmd,
)

# Create the main Typer app
app = typer.Typer(
    name="hexdag",
    help="HexDAG - Lightweight DAG orchestration framework with hexagonal architecture.",
    no_args_is_help=True,
    rich_markup_mode="rich",
    pretty_exceptions_enable=False,
)

# Create console for rich output
console = Console()

# Add subcommands
app.add_typer(init_cmd.app, name="init", help="Initialize a new HexDAG project")
app.add_typer(config_cmd.app, name="config", help="Configuration management")
app.add_typer(plugins_cmd.app, name="plugins", help="Manage plugins and adapters")
app.add_typer(plugin_dev_cmd.app, name="plugin", help="Plugin development commands")
app.add_typer(registry_cmd.app, name="registry", help="Inspect the component registry")
app.add_typer(manifest_cmd.app, name="manifest", help="Manifest management commands")
app.add_typer(pipeline_cmd.app, name="pipeline", help="Pipeline execution commands")
app.add_typer(events_cmd.app, name="events", help="Event logging and inspection commands")
app.add_typer(runs_cmd.app, name="runs", help="Manage and inspect pipeline runs")
app.add_typer(observers_cmd.app, name="observers", help="Manage event observers")
app.add_typer(nodes_cmd.app, name="nodes", help="Manage and inspect nodes")
app.add_typer(ports_cmd.app, name="ports", help="Manage and inspect ports")
app.add_typer(adapters_cmd.app, name="adapters", help="Manage and inspect adapters")
app.add_typer(validation_cmd.app, name="validate", help="Validation commands")
app.add_typer(run_cmd.app, name="run", help="Run a pipeline (monolithic executor)")


@app.callback()
def callback(
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        help="Show version and exit",
    ),
) -> None:
    """HexDAG CLI - Modular DAG orchestration framework."""
    if version:
        console.print("[bold blue]HexDAG[/bold blue] version [green]0.1.0[/green]")
        raise typer.Exit()


def main() -> None:
    """Main CLI entrypoint."""
    app()


if __name__ == "__main__":
    main()
