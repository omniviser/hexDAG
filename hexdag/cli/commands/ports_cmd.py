"""Port commands for HexDAG CLI."""

from typing import Any, cast

import typer
from rich.console import Console

from hexdag.core.registry import registry
from hexdag.core.registry.models import ComponentType

app = typer.Typer()
console = Console()


@app.command("list")
def list_ports() -> None:
    """List all available ports."""
    ports = [c for c in registry.list_components() if c.component_type == ComponentType.PORT]
    if not ports:
        console.print("[yellow]No ports found[/yellow]")
        raise typer.Exit()
    for p in ports:
        console.print(f"• {p.qualified_name}")


@app.command("test")
def test_port(
    port_name: str, config: str | None = typer.Option(None, help="YAML config file")
) -> None:
    """Test port connectivity."""
    try:
        port_info = registry.get_info(port_name)
        port_instance = registry.get(port_name, namespace=port_info.namespace)
        port_instance = cast("Any", port_instance)

        # Perform a simple smoke test
        success = port_instance.ping() if hasattr(port_instance, "ping") else True
        console.print(
            f"[green]Port '{port_name}' OK[/green]"
            if success
            else f"[red]Port '{port_name}' failed[/red]"
        )
    except Exception as e:
        console.print(f"[red]Port '{port_name}' test failed: {e}[/red]")
        raise typer.Exit(1) from e
