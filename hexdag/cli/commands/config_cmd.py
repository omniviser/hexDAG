"""Configuration management commands (minimal stub).

This file exists so the main CLI can import `config_cmd` even if fuller
implementation is added later.
"""

import typer
from rich.console import Console

app = typer.Typer(help="Configuration management commands")
console = Console()


@app.command("show")
def show_config(key: str | None = None) -> None:
    """Show current configuration or a specific key (stub)."""
    config = {"project": "./", "env": "dev"}
    if key:
        console.print(config.get(key, "<not set>"))
    else:
        console.print(config)
