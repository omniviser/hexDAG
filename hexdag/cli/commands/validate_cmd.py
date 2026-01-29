"""Pipeline validation command for HexDAG CLI."""

from pathlib import Path
from typing import Annotated, Any

import typer
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from hexdag.core.pipeline_builder.yaml_validator import YamlValidator

app = typer.Typer()
console = Console()


@app.command()
def validate(
    yaml_file: Annotated[
        Path,
        typer.Argument(
            help="Path to YAML pipeline file to validate",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    explain: Annotated[
        bool,
        typer.Option(
            "--explain",
            "-e",
            help="Show detailed explanation of validation process",
        ),
    ] = False,
) -> None:
    """Validate a YAML pipeline file against node schemas.

    This command validates:
    - YAML syntax and structure
    - Manifest format (apiVersion, kind, metadata, spec)
    - Node types and their schemas
    - Required fields and constraints
    - Dependency references

    Examples
    --------
    hexdag validate pipeline.yaml
    hexdag validate pipeline.yaml --explain
    """

    # Read YAML file
    try:
        with Path.open(yaml_file) as f:
            content = f.read()
            config = yaml.safe_load(content)
    except yaml.YAMLError as e:
        console.print(f"[red]✗ YAML Syntax Error:[/red] {e}")
        raise typer.Exit(1) from e
    except OSError as e:
        console.print(f"[red]✗ File Error:[/red] {e}")
        raise typer.Exit(1) from e

    # Validate using YamlValidator
    validator = YamlValidator()
    result = validator.validate(config)

    # Display results
    console.print()
    if result.is_valid:
        console.print(f"[green]✓ Validation successful:[/green] {yaml_file}")
    else:
        console.print(f"[red]✗ Validation failed:[/red] {yaml_file}")

    # Show warnings
    if result.warnings:
        console.print()
        console.print("[yellow]Warnings:[/yellow]")
        for warning in result.warnings:
            console.print(f"  [yellow]⚠[/yellow] {warning}")

    # Show errors
    if result.errors:
        console.print()
        console.print("[red]Errors:[/red]")
        for error in result.errors:
            console.print(f"  [red]✗[/red] {error}")

    # Show suggestions
    if result.suggestions:
        console.print()
        console.print("[blue]Suggestions:[/blue]")
        for suggestion in result.suggestions:
            console.print(f"  [blue]ℹ[/blue] {suggestion}")

    # Explain mode - show validation details
    if explain:
        console.print()
        _explain_validation(config, result)

    console.print()

    # Exit with error code if validation failed
    if not result.is_valid:
        raise typer.Exit(1)


def _explain_validation(config: dict, result: Any) -> None:
    """Show detailed explanation of validation process."""
    console.print(Panel("[bold]Validation Process Explanation[/bold]", border_style="blue"))

    # Show manifest structure
    console.print("\n[bold cyan]1. Manifest Structure[/bold cyan]")
    manifest_table = Table(show_header=True, border_style="cyan")
    manifest_table.add_column("Field", style="green")
    manifest_table.add_column("Value", style="white")
    manifest_table.add_column("Status", style="yellow")

    api_version = config.get("apiVersion", "missing")
    kind = config.get("kind", "missing")
    name = config.get("metadata", {}).get("name", "missing")

    manifest_table.add_row(
        "apiVersion",
        str(api_version),
        "[green]✓[/green]" if api_version != "missing" else "[red]✗[/red]",
    )
    manifest_table.add_row(
        "kind", str(kind), "[green]✓[/green]" if kind != "missing" else "[red]✗[/red]"
    )
    manifest_table.add_row(
        "metadata.name",
        str(name),
        "[green]✓[/green]" if name != "missing" else "[red]✗[/red]",
    )

    console.print(manifest_table)

    # Show node validation
    if nodes := config.get("spec", {}).get("nodes", []):
        console.print("\n[bold cyan]2. Node Validation[/bold cyan]")
        node_table = Table(show_header=True, border_style="cyan")
        node_table.add_column("Node", style="green")
        node_table.add_column("Type", style="blue")
        node_table.add_column("Schema", style="white")
        node_table.add_column("Status", style="yellow")

        for i, node in enumerate(nodes):
            node_name = node.get("metadata", {}).get("name", f"node_{i}")
            node_kind = node.get("kind", "unknown")

            # Parse namespace and type
            if ":" in node_kind:
                namespace, node_type = node_kind.split(":", 1)
            else:
                namespace = "core"
                node_type = node_kind

            # Remove _node suffix if present
            if node_type.endswith("_node"):
                node_type = node_type[:-5]

            # Check if node type is known
            from hexdag.core.pipeline_builder.yaml_validator import KNOWN_NODE_TYPES

            qualified_type = f"{namespace}:{node_type}"
            if qualified_type in KNOWN_NODE_TYPES or node_type in KNOWN_NODE_TYPES:
                schema_status = "[green]✓[/green] Known type"
            elif "." in node_kind:
                schema_status = "[blue]ℹ[/blue] Module path"
            else:
                schema_status = "[yellow]?[/yellow] Unknown type"

            node_has_error = any(node_name in str(error) for error in result.errors)
            status = "[red]✗[/red]" if node_has_error else "[green]✓[/green]"

            node_table.add_row(node_name, f"{namespace}:{node_type}", schema_status, status)

        console.print(node_table)

    # Show dependency graph
    if nodes:
        console.print("\n[bold cyan]3. Dependency Graph[/bold cyan]")
        dep_table = Table(show_header=True, border_style="cyan")
        dep_table.add_column("Node", style="green")
        dep_table.add_column("Dependencies", style="blue")
        dep_table.add_column("Status", style="yellow")

        all_node_names = {node.get("metadata", {}).get("name") for node in nodes}
        all_node_names.discard(None)

        for node in nodes:
            node_name = node.get("metadata", {}).get("name", "unknown")
            deps = node.get("spec", {}).get("dependencies", [])

            if not deps:
                dep_str = "-"
                status = "[green]✓[/green]"
            else:
                dep_str = ", ".join(str(d) for d in deps)
                if invalid_deps := [d for d in deps if d not in all_node_names]:
                    status = f"[red]✗[/red] Invalid: {', '.join(invalid_deps)}"
                else:
                    status = "[green]✓[/green]"

            dep_table.add_row(node_name, dep_str, status)

        console.print(dep_table)

    # Show validation summary
    console.print("\n[bold cyan]4. Summary[/bold cyan]")
    summary_text = f"""
Total Nodes: {len(nodes)}
Errors: {len(result.errors)}
Warnings: {len(result.warnings)}
Suggestions: {len(result.suggestions)}
Valid: {"[green]Yes[/green]" if result.is_valid else "[red]No[/red]"}
"""
    console.print(Panel(summary_text.strip(), border_style="blue"))
