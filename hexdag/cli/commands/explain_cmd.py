"""Explain command — like ``kubectl explain`` for hexDAG YAML.

Usage::

    hexdag explain                     # overview of top-level kinds
    hexdag explain node                # list all node types
    hexdag explain node llm_node       # schema for llm_node
    hexdag explain adapter             # list all adapters
    hexdag explain adapter OpenAI      # schema for a specific adapter
    hexdag explain middleware          # list all middleware
    hexdag explain syntax              # YAML syntax reference
    hexdag explain types               # output_schema type reference
"""

from __future__ import annotations

from typing import Annotated, Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer()
console = Console()

_CLI_NAME = "explain"
_CLI_HELP = "Explain hexDAG YAML fields, node types, adapters, and syntax"
_CLI_TYPE = "command"
_CLI_FUNC = "explain"


# ---------------------------------------------------------------------------
# Main command
# ---------------------------------------------------------------------------


@app.command()
def explain(
    resource: Annotated[
        str | None,
        typer.Argument(
            help=(
                "What to explain: node, adapter, middleware, macro, tag, "
                "syntax, types, or a specific component (e.g. 'node llm_node')"
            ),
        ),
    ] = None,
    name: Annotated[
        str | None,
        typer.Argument(help="Component name (e.g. 'llm_node', 'OpenAIAdapter')"),
    ] = None,
) -> None:
    """Explain hexDAG components and YAML syntax.

    Like ``kubectl explain`` — shows documentation for nodes, adapters,
    middleware, macros, tags, and YAML syntax.

    Examples
    --------
    hexdag explain                     # overview
    hexdag explain node                # list all nodes
    hexdag explain node llm_node       # details for llm_node
    hexdag explain adapter             # list all adapters
    hexdag explain adapter --name OpenAIAdapter
    hexdag explain middleware          # list all middleware
    hexdag explain syntax              # YAML syntax reference
    hexdag explain types               # type system reference
    """
    if resource is None:
        _show_overview()
        return

    # Dispatch to the right handler
    handler = _RESOURCE_HANDLERS.get(resource)
    if handler:
        handler(name)
    else:
        # Maybe user typed a component name directly — try resolving
        _try_resolve_component(resource)


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------


def _show_overview() -> None:
    """Display an overview of available explain subcommands."""
    console.print()
    console.print(
        Panel(
            "[bold]hexdag explain[/bold] — documentation for YAML fields and components",
            border_style="blue",
        )
    )

    table = Table(show_header=True, border_style="cyan")
    table.add_column("Command", style="green", min_width=30)
    table.add_column("Description", style="white")

    table.add_row("hexdag explain node", "List all node types")
    table.add_row("hexdag explain node <name>", "Schema for a specific node")
    table.add_row("hexdag explain adapter", "List all adapters")
    table.add_row("hexdag explain adapter <name>", "Schema for a specific adapter")
    table.add_row("hexdag explain middleware", "List all middleware")
    table.add_row("hexdag explain macro", "List all macros")
    table.add_row("hexdag explain tag", "List all YAML custom tags")
    table.add_row("hexdag explain syntax", "YAML syntax reference")
    table.add_row("hexdag explain types", "Output schema type reference")

    console.print(table)
    console.print()


# ---------------------------------------------------------------------------
# Node explain
# ---------------------------------------------------------------------------


def _explain_node(name: str | None) -> None:
    """List node types or show the schema for a specific node."""
    from hexdag.api import components

    nodes = components.list_nodes()
    if name is None:
        _list_components("node", nodes, "kind")
        return

    # list_nodes() has richer schemas than get_component_schema() for nodes
    # because it tries __call__ on the MRO. Use it when available.
    for node in nodes:
        if node.get("kind") == name or node.get("name") == name:
            schema = node.get("schema", {})
            _show_schema_panel("node", name, node.get("description", ""), schema)
            return

    # Fall back to generic resolution
    _show_component_schema("node", name, components)


# ---------------------------------------------------------------------------
# Adapter explain
# ---------------------------------------------------------------------------


def _explain_adapter(name: str | None) -> None:
    """List adapters or show the schema for a specific adapter."""
    from hexdag.api import components

    if name is None:
        _list_adapters(components.list_adapters())
        return
    _show_component_schema("adapter", name, components)


# ---------------------------------------------------------------------------
# Middleware explain
# ---------------------------------------------------------------------------


def _explain_middleware(name: str | None) -> None:
    """List middleware or show detail for a specific middleware."""
    from hexdag.stdlib import middleware as mw_module

    if name is not None:
        _show_middleware_detail(name, mw_module)
        return

    console.print()
    console.print("[bold cyan]Available Middleware[/bold cyan]")
    console.print()

    table = Table(show_header=True, border_style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Type", style="blue")
    table.add_column("Description", style="white")

    for item_name in sorted(mw_module.__all__):
        obj = getattr(mw_module, item_name, None)
        if obj is None or item_name == "compose":
            continue
        doc = (getattr(obj, "__doc__", None) or "").split("\n")[0].strip()
        kind = "observer" if "Observer" in item_name else "wrapper"
        table.add_row(item_name, kind, doc or "—")

    console.print(table)
    console.print()


def _show_middleware_detail(name: str, mw_module: Any) -> None:
    """Display documentation panel for a single middleware component."""
    obj = getattr(mw_module, name, None)
    if obj is None:
        console.print(f"[red]Middleware '{name}' not found.[/red]")
        console.print(
            f"Available: {', '.join(n for n in sorted(mw_module.__all__) if n != 'compose')}"
        )
        raise typer.Exit(1)

    doc = getattr(obj, "__doc__", None) or "No documentation."
    console.print()
    console.print(Panel(f"[bold]{name}[/bold]\n\n{doc}", border_style="blue"))
    console.print()


# ---------------------------------------------------------------------------
# Macro explain
# ---------------------------------------------------------------------------


def _explain_macro(name: str | None) -> None:
    """List macros or show the schema for a specific macro."""
    from hexdag.api import components

    if name is None:
        _list_components("macro", components.list_macros(), "name")
        return
    _show_component_schema("macro", name, components)


# ---------------------------------------------------------------------------
# Tag explain
# ---------------------------------------------------------------------------


def _explain_tag(name: str | None) -> None:
    """List YAML tags or show the schema for a specific tag."""
    from hexdag.api import components

    if name is None:
        _list_tags(components.list_tags())
        return
    _show_component_schema("tag", name, components)


# ---------------------------------------------------------------------------
# Syntax & types
# ---------------------------------------------------------------------------


def _explain_syntax(_name: str | None) -> None:
    """Print the YAML syntax reference."""
    from hexdag.api import documentation

    console.print()
    console.print(documentation.get_syntax_reference())
    console.print()


def _explain_types(_name: str | None) -> None:
    """Print the output_schema type reference."""
    from hexdag.api import documentation

    console.print()
    console.print(documentation.get_type_reference())
    console.print()


# ---------------------------------------------------------------------------
# Shared display helpers
# ---------------------------------------------------------------------------


def _list_components(component_type: str, items: list[dict[str, Any]], name_key: str) -> None:
    """Render a table of available components of the given type."""
    console.print()
    console.print(f"[bold cyan]Available {component_type}s[/bold cyan]")
    console.print()

    table = Table(show_header=True, border_style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Description", style="white")

    for item in items:
        table.add_row(
            item.get(name_key, item.get("name", "?")),
            item.get("description", "—"),
        )

    console.print(table)
    console.print()
    console.print(f"[dim]Use 'hexdag explain {component_type} <name>' for details[/dim]")
    console.print()


def _list_adapters(adapters: list[dict[str, Any]]) -> None:
    """Render a table of available adapters with port information."""
    console.print()
    console.print("[bold cyan]Available Adapters[/bold cyan]")
    console.print()

    table = Table(show_header=True, border_style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Port", style="blue")
    table.add_column("Description", style="white")

    for a in adapters:
        table.add_row(
            a.get("name", "?"),
            a.get("port_type", "?"),
            a.get("description", "—"),
        )

    console.print(table)
    console.print()
    console.print("[dim]Use 'hexdag explain adapter <name>' for details[/dim]")
    console.print()


def _list_tags(tags: list[dict[str, Any]]) -> None:
    """Render a table of available YAML custom tags."""
    console.print()
    console.print("[bold cyan]Available YAML Tags[/bold cyan]")
    console.print()

    table = Table(show_header=True, border_style="cyan")
    table.add_column("Tag", style="green")
    table.add_column("Description", style="white")

    for t in tags:
        table.add_row(t.get("name", "?"), t.get("description", "—"))

    console.print(table)
    console.print()


def _show_component_schema(component_type: str, name: str, components_module: Any) -> None:
    """Fetch and display the schema panel for a named component."""
    schema = components_module.get_component_schema(component_type, name)

    if "error" in schema:
        console.print(f"[red]{schema['error']}[/red]")
        raise typer.Exit(1)

    _show_schema_panel(component_type, name, schema.get("description", ""), schema)


def _show_schema_panel(
    component_type: str,
    name: str,
    description: str,
    schema: dict[str, Any],
) -> None:
    """Render a Rich panel showing a component's field schema."""
    console.print()
    console.print(f"[bold cyan]{component_type}: {name}[/bold cyan]")

    if description:
        console.print(f"\n{description}")

    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    if properties:
        console.print()
        table = Table(show_header=True, border_style="cyan")
        table.add_column("Field", style="green")
        table.add_column("Type", style="blue")
        table.add_column("Required", style="yellow")
        table.add_column("Default", style="white")
        table.add_column("Description", style="white")

        for field_name, field_schema in properties.items():
            field_type = _schema_type_str(field_schema)
            is_required = field_name in required
            default = str(field_schema.get("default", "—"))
            desc = field_schema.get("description", "—")

            table.add_row(
                field_name,
                field_type,
                "[green]yes[/green]" if is_required else "no",
                default,
                desc,
            )

        console.print(table)
    else:
        console.print("\n[dim]No schema properties available.[/dim]")

    console.print()


def _schema_type_str(field_schema: dict[str, Any]) -> str:
    """Convert JSON Schema type info to a readable string."""
    if "anyOf" in field_schema:
        types = [_schema_type_str(t) for t in field_schema["anyOf"]]
        return " | ".join(types)
    if "enum" in field_schema:
        return f"enum({', '.join(str(v) for v in field_schema['enum'])})"
    t = field_schema.get("type", "any")
    if t == "array":
        items = field_schema.get("items", {})
        item_type = _schema_type_str(items)
        return f"list[{item_type}]"
    if t == "object":
        return "dict"
    return str(t)


def _try_resolve_component(name: str) -> None:
    """Try to resolve a name as a component and show its schema."""
    from hexdag.api import components

    for ctype in ("node", "adapter", "tool", "macro", "tag"):
        schema = components.get_component_schema(ctype, name)
        if "error" not in schema:
            _show_component_schema(ctype, name, components)
            return

    console.print(f"[red]Unknown resource or component: '{name}'[/red]")
    console.print()
    console.print("Available resources: node, adapter, middleware, macro, tag, syntax, types")
    console.print("Run 'hexdag explain' for usage.")
    raise typer.Exit(1)


# ---------------------------------------------------------------------------
# Handler dispatch table
# ---------------------------------------------------------------------------

_RESOURCE_HANDLERS: dict[str, Any] = {
    "node": _explain_node,
    "nodes": _explain_node,
    "adapter": _explain_adapter,
    "adapters": _explain_adapter,
    "middleware": _explain_middleware,
    "macro": _explain_macro,
    "macros": _explain_macro,
    "tag": _explain_tag,
    "tags": _explain_tag,
    "syntax": _explain_syntax,
    "types": _explain_types,
    "type": _explain_types,
}
