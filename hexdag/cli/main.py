"""HexDAG CLI - Main entrypoint."""

import importlib
import pkgutil
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

import hexdag.cli.commands as _commands_pkg

app = typer.Typer(
    name="hexdag",
    help="HexDAG - Lightweight DAG orchestration framework with hexagonal architecture.",
    no_args_is_help=True,
    rich_markup_mode="rich",
    pretty_exceptions_enable=False,
)

console = Console()

# Auto-discover CLI commands from hexdag.cli.commands package.
# Each command module declares _CLI_NAME, _CLI_HELP, _CLI_TYPE,
# and optionally _CLI_FUNC (for standalone commands).
for _info in pkgutil.iter_modules(_commands_pkg.__path__):
    if _info.name.startswith("_"):
        continue
    _mod = importlib.import_module(f"hexdag.cli.commands.{_info.name}")
    _cli_type = getattr(_mod, "_CLI_TYPE", None)
    if _cli_type is None:
        continue
    _name = _mod._CLI_NAME
    _help = _mod._CLI_HELP
    if _cli_type == "typer":
        app.add_typer(_mod.app, name=_name, help=_help)
    elif _cli_type == "command":
        _func_name = _mod._CLI_FUNC
        app.command(name=_name, help=_help)(getattr(_mod, _func_name))


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
