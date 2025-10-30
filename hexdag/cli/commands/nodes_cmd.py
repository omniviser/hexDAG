"""Node commands for HexDAG CLI."""

import typer
from rich.console import Console

from hexdag.core.registry import registry
from hexdag.core.registry.models import ComponentType

app = typer.Typer()
console = Console()


@app.command("list")
def list_nodes(
    subtype: str | None = typer.Option(None, help="Filter by subtype: function, llm, agent"),
) -> None:
    """List all registered nodes."""
    nodes = [c for c in registry.list_components() if c.component_type == ComponentType.NODE]
    if subtype:
        nodes = [n for n in nodes if getattr(n.metadata, "subtype", None) == subtype]

    if not nodes:
        console.print("[yellow]No nodes found[/yellow]")
        raise typer.Exit()

    for n in nodes:
        meta_str = f" ({n.metadata.subtype})" if getattr(n.metadata, "subtype", None) else ""
        console.print(f"• {n.qualified_name}{meta_str}")


@app.command("info")
def node_info(name: str) -> None:
    """Show IO types, parameters, and dependencies for a node."""
    try:
        node_info = registry.get_info(name)
    except KeyError:
        console.print(f"[red]Node '{name}' not found[/red]")
        raise typer.Exit(1) from None

    console.print(f"[bold]{node_info.qualified_name}[/bold]")
    console.print(f"Type: {node_info.component_type.value}")
    console.print(f"Namespace: {node_info.namespace}")
    console.print("Parameters:")
    for k, v in getattr(node_info.metadata, "parameters", {}).items():
        console.print(f"  • {k}: {v}")
    console.print("Dependencies:")
    for dep in getattr(node_info.metadata, "dependencies", []):
        console.print(f"  • {dep}")
