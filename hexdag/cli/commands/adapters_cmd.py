# hexai/cli/commands/adapters_cmd.py
"""Adapter commands for HexDAG CLI."""

import typer
from rich.console import Console

from hexdag.core.registry import registry
from hexdag.core.registry.models import ComponentType

app = typer.Typer()
console = Console()


@app.command("list")
def list_adapters() -> None:
    """List all registered adapters."""
    adapters = [c for c in registry.list_components() if c.component_type == ComponentType.ADAPTER]
    if not adapters:
        console.print("[yellow]No adapters found[/yellow]")
        raise typer.Exit()
    for a in adapters:
        console.print(f"• {a.qualified_name}")


@app.command("test")
def test_adapter(
    adapter_name: str, config: str | None = typer.Option(None, help="YAML config file")
) -> None:
    """Test adapter connectivity."""
    try:
        adapter_info = registry.get_info(adapter_name)
        adapter_instance = registry.get(adapter_name, namespace=adapter_info.namespace)

        # Smoke test
        ping_method = getattr(adapter_instance, "ping", None)
        success = ping_method() if callable(ping_method) else True

        console.print(
            f"[green]Adapter '{adapter_name}' OK[/green]"
            if success
            else f"[red]Adapter '{adapter_name}' failed[/red]"
        )
    except Exception as e:
        console.print(f"[red]Adapter '{adapter_name}' test failed: {e}[/red]")
        raise typer.Exit(1) from e
