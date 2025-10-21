"""Schema discovery and introspection commands for HexDAG CLI."""

import contextlib
import json
from typing import Annotated, Any

import typer
import yaml
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from hexdag.core.bootstrap import bootstrap_registry
from hexdag.core.registry import registry
from hexdag.core.registry.models import ComponentType

app = typer.Typer()
console = Console()


def _ensure_bootstrapped() -> None:
    """Ensure registry is bootstrapped."""
    with contextlib.suppress(Exception):
        bootstrap_registry()


def _generate_example_from_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Generate example values from JSON schema.

    Parameters
    ----------
    schema : dict[str, Any]
        JSON schema dictionary

    Returns
    -------
    dict[str, Any]
        Example values matching the schema
    """
    if schema.get("type") == "object":
        example = {}
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        for prop_name, prop_schema in properties.items():
            # Skip optional fields for cleaner examples
            if prop_name not in required and "default" not in prop_schema:
                continue

            # Use default if available
            if "default" in prop_schema:
                example[prop_name] = prop_schema["default"]
            elif prop_schema.get("type") == "string":
                if "enum" in prop_schema:
                    example[prop_name] = prop_schema["enum"][0]
                else:
                    example[prop_name] = f"<{prop_name}>"
            elif prop_schema.get("type") == "integer":
                example[prop_name] = 1
            elif prop_schema.get("type") == "number":
                example[prop_name] = 1.0
            elif prop_schema.get("type") == "boolean":
                example[prop_name] = True
            elif prop_schema.get("type") == "array":
                example[prop_name] = []
            elif prop_schema.get("type") == "object":
                example[prop_name] = _generate_example_from_schema(prop_schema)
            else:
                if "anyOf" in prop_schema:
                    # Use first option
                    first_option = prop_schema["anyOf"][0]
                    if first_option.get("type") == "null":
                        # Skip if first option is null
                        if len(prop_schema["anyOf"]) > 1:
                            example[prop_name] = _generate_example_from_schema({
                                "type": "object",
                                "properties": {prop_name: prop_schema["anyOf"][1]},
                            }).get(prop_name)
                    else:
                        example[prop_name] = _generate_example_from_schema({
                            "type": "object",
                            "properties": {prop_name: first_option},
                        }).get(prop_name)

        return example
    return {}


@app.command("list")
def list_schemas(
    namespace: Annotated[
        str | None,
        typer.Option(
            "--namespace",
            "-n",
            help="Filter by namespace (e.g., core, myplugin)",
        ),
    ] = None,
    format: Annotated[
        str,
        typer.Option(
            "--format",
            "-f",
            help="Output format (table, yaml, json)",
        ),
    ] = "table",
) -> None:
    """List all available node types with their schemas.

    Examples
    --------
    hexdag schema list
    hexdag schema list --namespace core
    hexdag schema list --format yaml
    """
    _ensure_bootstrapped()

    components = registry.list_components(component_type=ComponentType.NODE)

    # Apply namespace filter
    if namespace:
        components = [c for c in components if c.namespace == namespace]

    if not components:
        console.print("[yellow]No node types found[/yellow]")
        return

    if format == "table":
        table = Table(title="Available Node Types", show_header=True)
        table.add_column("Namespace", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Subtype", style="blue")
        table.add_column("Description", style="white")

        for comp in components:
            description = comp.metadata.description or ""
            # Truncate long descriptions
            if len(description) > 50:
                description = description[:47] + "..."
            table.add_row(
                comp.namespace,
                comp.name,
                comp.metadata.subtype or "-",
                escape(description),
            )

        console.print(table)

    elif format == "yaml":
        output = [
            {
                "namespace": comp.namespace,
                "name": comp.name,
                "subtype": comp.metadata.subtype,
                "description": comp.metadata.description,
            }
            for comp in components
        ]
        console.print(yaml.dump(output, default_flow_style=False))

    elif format == "json":
        output = []
        for comp in components:
            output.append({
                "namespace": comp.namespace,
                "name": comp.name,
                "subtype": comp.metadata.subtype,
                "description": comp.metadata.description,
            })
        console.print(json.dumps(output, indent=2))


@app.command("get")
def get_schema(
    node_type: Annotated[str, typer.Argument(help="Node type name (e.g., llm_node)")],
    namespace: Annotated[
        str,
        typer.Option(
            "--namespace",
            "-n",
            help="Component namespace",
        ),
    ] = "core",
    format: Annotated[
        str,
        typer.Option(
            "--format",
            "-f",
            help="Output format (yaml, json)",
        ),
    ] = "yaml",
    example: Annotated[
        bool,
        typer.Option(
            "--example",
            "-e",
            help="Show example YAML instead of schema",
        ),
    ] = False,
) -> None:
    """Get the full schema for a node type.

    Examples
    --------
    hexdag schema get llm_node
    hexdag schema get llm_node --format json
    hexdag schema get llm_node --example
    hexdag schema get custom_node --namespace myplugin
    """
    _ensure_bootstrapped()

    try:
        schema = registry.get_schema(node_type, namespace=namespace)
    except (KeyError, ValueError) as e:
        console.print(f"[red]Error:[/red] {e}")
        console.print("[yellow]Hint:[/yellow] Use 'hexdag schema list' to see available node types")
        raise typer.Exit(1) from e

    # Ensure schema is dict (get_schema can return str)
    if isinstance(schema, str):
        console.print(f"[red]Error:[/red] Schema is a string, not a dict: {schema}")
        raise typer.Exit(1)

    if example:
        # Generate example YAML
        example_spec = _generate_example_from_schema(schema)

        example_yaml = {
            "kind": f"{namespace}:{node_type}" if namespace != "core" else node_type,
            "metadata": {"name": "example_node"},
            "spec": example_spec,
        }

        if format == "yaml":
            output = yaml.dump(example_yaml, default_flow_style=False, sort_keys=False)
            syntax = Syntax(output, "yaml", theme="monokai", line_numbers=False)
            console.print(Panel(syntax, title=f"Example: {node_type}", border_style="green"))
        else:
            console.print(json.dumps(example_yaml, indent=2))
    else:
        # Output raw schema
        if format == "yaml":
            output = yaml.dump(schema, default_flow_style=False)
            syntax = Syntax(output, "yaml", theme="monokai", line_numbers=False)
            console.print(Panel(syntax, title=f"Schema: {node_type}", border_style="blue"))
        else:
            console.print(json.dumps(schema, indent=2))


@app.command("explain")
def explain_schema(
    node_type: Annotated[str, typer.Argument(help="Node type name (e.g., llm_node)")],
    namespace: Annotated[
        str,
        typer.Option(
            "--namespace",
            "-n",
            help="Component namespace",
        ),
    ] = "core",
) -> None:
    """Show human-readable explanation of a node type.

    Examples
    --------
    hexdag explain llm_node
    hexdag explain custom_node --namespace myplugin
    """
    _ensure_bootstrapped()

    try:
        schema = registry.get_schema(node_type, namespace=namespace)
        components = registry.list_components(component_type=ComponentType.NODE)
        comp_info = next(
            (c for c in components if c.name == node_type and c.namespace == namespace), None
        )
    except (KeyError, ValueError) as e:
        console.print(f"[red]Error:[/red] {e}")
        console.print("[yellow]Hint:[/yellow] Use 'hexdag schema list' to see available node types")
        raise typer.Exit(1) from e

    # Ensure schema is dict (get_schema can return str)
    if isinstance(schema, str):
        console.print(f"[red]Error:[/red] Schema is a string, not a dict: {schema}")
        raise typer.Exit(1)

    console.print(f"\n[bold cyan]{namespace}:{node_type}[/bold cyan]")

    if comp_info and comp_info.metadata.description:
        console.print(f"\n{comp_info.metadata.description}\n")

    # Show properties
    properties = schema.get("properties", {})
    required_fields = set(schema.get("required", []))

    if properties:
        table = Table(title="Parameters", show_header=True, border_style="blue")
        table.add_column("Field", style="green")
        table.add_column("Type", style="cyan")
        table.add_column("Required", style="yellow")
        table.add_column("Default", style="white")
        table.add_column("Description", style="white")

        for field_name, field_schema in properties.items():
            # Determine type
            field_type = field_schema.get("type", "any")
            if isinstance(field_type, list):
                field_type = " | ".join(str(t) for t in field_type)

            is_required = "✓" if field_name in required_fields else ""

            default = field_schema.get("default", "")
            if default == "":
                default = "-"
            else:
                default = str(default)
                if len(default) > 30:
                    default = default[:27] + "..."

            description = field_schema.get("description", "")
            if len(description) > 60:
                description = description[:57] + "..."

            if "enum" in field_schema:
                field_type = f"enum: {', '.join(str(v) for v in field_schema['enum'][:3])}"
                if len(field_schema["enum"]) > 3:
                    field_type += "..."

            if "anyOf" in field_schema:
                types = [option.get("type", "any") for option in field_schema["anyOf"][:2]]
                field_type = " | ".join(types)
                if len(field_schema["anyOf"]) > 2:
                    field_type += " | ..."

            table.add_row(
                field_name, escape(field_type), is_required, escape(default), escape(description)
            )

        console.print(table)

    # Show constraints
    additional_props = schema.get("additionalProperties", True)
    if not additional_props:
        console.print("\n[yellow]⚠ Additional properties not allowed[/yellow]")

    console.print()
