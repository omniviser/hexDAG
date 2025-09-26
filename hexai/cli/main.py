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

from hexai.cli.commands import config_cmd, init_cmd, plugin_dev_cmd, plugins_cmd, registry_cmd

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
