"""Registry inspection command for HexDAG CLI."""

import contextlib
from enum import Enum
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table
from rich.tree import Tree

from hexai.core.bootstrap import bootstrap_registry
from hexai.core.registry import registry
from hexai.core.registry.models import ComponentMetadata, ComponentType

app = typer.Typer()
console = Console()


class ComponentFilter(str, Enum):
    """Component type filter options."""

    ALL = "all"
    PORT = "port"
    ADAPTER = "adapter"
    NODE = "node"


@app.command("list")
def list_components(
    type_filter: Annotated[
        ComponentFilter | None,
        typer.Option(
            "--type",
            "-t",
            help="Filter by component type",
        ),
    ] = None,
    namespace: Annotated[
        str | None,
        typer.Option(
            "--namespace",
            "-n",
            help="Filter by namespace (core, plugin, dev)",
        ),
    ] = None,
    format: Annotated[
        str | None,
        typer.Option(
            "--format",
            "-f",
            help="Output format (table, json, yaml)",
        ),
    ] = None,
) -> None:
    """List all registered components."""
    # Set defaults
    if type_filter is None:
        type_filter = ComponentFilter.ALL
    if format is None:
        format = "table"

    # Bootstrap if not already done
    with contextlib.suppress(Exception):
        # Registry may already be bootstrapped or other initialization issue
        # Auto-discovery happens during bootstrap
        bootstrap_registry()

    # Get components
    components = registry.list_components()

    # Apply filters
    if type_filter != ComponentFilter.ALL:
        type_map = {
            ComponentFilter.PORT: ComponentType.PORT,
            ComponentFilter.ADAPTER: ComponentType.ADAPTER,
            ComponentFilter.NODE: ComponentType.NODE,
        }
        filter_type = type_map.get(type_filter)
        components = [c for c in components if c.component_type == filter_type]

    if namespace:
        components = [c for c in components if c.namespace == namespace]

    # Output based on format
    if format == "json":
        data = [
            {
                "name": c.name,
                "qualified_name": c.qualified_name,
                "type": c.component_type.value,
                "namespace": c.namespace,
                "metadata": c.model_dump(),
            }
            for c in components
        ]
        console.print_json(data=data)
    elif format == "yaml":
        import yaml

        data = [
            {
                "name": c.name,
                "qualified_name": c.qualified_name,
                "type": c.component_type.value,
                "namespace": c.namespace,
                "metadata": c.model_dump(),
            }
            for c in components
        ]
        console.print(yaml.dump(data, default_flow_style=False))
    else:
        # Table format
        table = Table(
            title="Component Registry",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Name", style="cyan", no_wrap=True)
        table.add_column("Type", style="green")
        table.add_column("Namespace", style="yellow")
        table.add_column("Qualified Name", style="white")

        for component in components:
            table.add_row(
                component.name,
                component.component_type.value,
                component.namespace,
                component.qualified_name,
            )

        console.print(table)
        console.print(f"\n[dim]Total: {len(components)} components[/dim]")


@app.command("show")
def show_component(
    component_name: str = typer.Argument(
        ...,
        help="Component name to inspect",
    ),
    namespace: str | None = typer.Option(
        None,
        "--namespace",
        "-n",
        help="Component namespace (searches all if not specified)",
    ),
) -> None:
    """Show detailed information about a specific component."""
    # Bootstrap if not already done
    with contextlib.suppress(Exception):
        # Registry may already be bootstrapped or other initialization issue
        bootstrap_registry()

    # Search for component across namespaces if none specified
    component_info = None

    if namespace is None:
        # Search all namespaces for the component
        all_components = registry.list_components()
        matches = [c for c in all_components if c.name == component_name]

        if not matches:
            # Component not found - show error with suggestions
            console.print(f"[red]Error: Component '{component_name}' not found[/red]")
            # Show available components
            _show_suggestions(component_name, all_components)
            raise typer.Exit(1)
        elif len(matches) == 1:
            # Single match - use it
            namespace = matches[0].namespace
        else:
            # Multiple matches - ask user to be specific
            console.print(f"[yellow]Multiple components named '{component_name}' found:[/yellow]")
            for match in matches:
                console.print(f"  • {match.namespace}:{match.name} ({match.component_type})")
            console.print(
                f"\n[dim]Specify namespace: hexdag registry show {component_name} "
                f"--namespace {matches[0].namespace}[/dim]"
            )
            raise typer.Exit(0)

    try:
        # Get component info with determined namespace
        component_info = registry.get_info(component_name, namespace=namespace)

        # Only get actual component for non-ports (ports are interfaces)
        component = None
        if component_info.component_type != ComponentType.PORT:
            try:
                component = registry.get(component_name, namespace=namespace)
            except (ImportError, TypeError, ValueError):
                # Some components can't be instantiated without config
                component = None

        console.print(f"\n[bold cyan]{component_info.qualified_name}[/bold cyan]")
        console.print(f"[dim]Type: {component_info.component_type.value}[/dim]")
        console.print(f"[dim]Namespace: {component_info.namespace}[/dim]")

        # Show metadata based on type
        if component_info.description:
            console.print(f"[dim]Description: {component_info.description}[/dim]")

        # Show methods for ports
        if component_info.component_type == ComponentType.PORT:
            console.print("\n[bold]Interface Methods:[/bold]")
            # Get the port class from registry
            import inspect

            port_class = registry._components[component_info.namespace][component_name]

            # Skip Pydantic/BaseModel methods
            skip_methods = {
                "construct",
                "copy",
                "dict",
                "from_orm",
                "json",
                "parse_file",
                "parse_obj",
                "parse_raw",
                "schema",
                "schema_json",
                "update_forward_refs",
                "validate",
                "model_construct",
                "model_copy",
                "model_dump",
                "model_dump_json",
                "model_json_schema",
                "model_parametrized_name",
                "model_post_init",
                "model_rebuild",
                "model_validate",
                "model_validate_json",
                "model_validate_strings",
            }

            for name in dir(port_class):
                if (
                    not name.startswith("_")
                    and name not in skip_methods
                    and callable(getattr(port_class, name))
                ):
                    try:
                        method = getattr(port_class, name)
                        sig = inspect.signature(method)
                        console.print(f"  • {name}{sig}")
                    except (ValueError, TypeError):
                        # Some methods may not have inspectable signatures
                        console.print(f"  • {name}(...)")

        # Show implemented port for adapters
        elif component_info.component_type == ComponentType.ADAPTER:
            if component_info.implements_port:
                port_name = component_info.implements_port
                console.print(f"\n[bold]Implements Port:[/bold] {port_name}")

                # Show methods if we have the component
                if component:
                    console.print("\n[bold]Methods:[/bold]")
                    import inspect

                    for name, method in inspect.getmembers(component, predicate=inspect.ismethod):
                        if not name.startswith("_"):
                            sig = inspect.signature(method)
                            console.print(f"  • {name}{sig}")

        # Show capabilities
        console.print("\n[bold]Capabilities:[/bold]")
        _show_capabilities(component, component_info)

    except Exception:
        if namespace:
            console.print(
                f"[red]Error: Component '{component_name}' not found "
                f"in namespace '{namespace}'[/red]"
            )
        else:
            console.print(f"[red]Error: Component '{component_name}' not found[/red]")

        # Show suggestions
        all_components = registry.list_components()
        _show_suggestions(component_name, all_components, namespace)
        raise typer.Exit(1) from None


@app.command("tree")
def show_tree() -> None:
    """Show registry structure as a tree."""
    # Bootstrap if not already done
    with contextlib.suppress(Exception):
        # Registry may already be bootstrapped or other initialization issue
        bootstrap_registry()

    components = registry.list_components()

    # Build tree structure
    tree = Tree("[bold]Component Registry[/bold]")

    # Group by namespace
    namespaces: dict[str, list[ComponentMetadata]] = {}
    for comp in components:
        if comp.namespace not in namespaces:
            namespaces[comp.namespace] = []
        namespaces[comp.namespace].append(comp)

    # Add to tree
    for ns_name, ns_components in namespaces.items():
        ns_branch = tree.add(f"[yellow]{ns_name}[/yellow]")

        # Group by type within namespace
        by_type: dict[str, list[ComponentMetadata]] = {}
        for comp in ns_components:
            type_name = comp.component_type.value
            if type_name not in by_type:
                by_type[type_name] = []
            by_type[type_name].append(comp)

        # Add to namespace branch
        for type_name, type_components in by_type.items():
            type_branch = ns_branch.add(f"[green]{type_name}[/green]")
            for comp in type_components:
                metadata_str = ""
                if comp.implements_port:
                    metadata_str = f" [dim]→ {comp.implements_port}[/dim]"
                type_branch.add(f"[cyan]{comp.name}[/cyan]{metadata_str}")

    console.print(tree)


def _show_suggestions(
    component_name: str,
    all_components: list,
    namespace: str | None = None,
) -> None:
    """Show helpful suggestions when a component is not found."""
    # Filter by namespace if specified
    if namespace:
        components = [c for c in all_components if c.namespace == namespace]
        console.print(f"\n[yellow]Available in '{namespace}' namespace:[/yellow]")
    else:
        components = all_components
        console.print("\n[yellow]Available components:[/yellow]")

    # Group by type for better display
    by_type: dict[str, list[Any]] = {}
    for comp in components:
        type_name = comp.component_type.value
        if type_name not in by_type:
            by_type[type_name] = []
        by_type[type_name].append(comp)

    # Show first few of each type
    for type_name, comps in by_type.items():
        if comps:
            console.print(f"  {type_name}s:")
            for comp in comps[:3]:  # Show first 3 of each type
                console.print(f"    • {comp.name}")
            if len(comps) > 3:
                console.print(f"    ... and {len(comps) - 3} more")

    # Try fuzzy matching for suggestions
    from difflib import get_close_matches

    comp_names = [c.name for c in components]

    # Try exact case-insensitive match first
    lower_comp_name = component_name.lower()
    exact_matches = [name for name in comp_names if name.lower() == lower_comp_name]

    if exact_matches:
        console.print("\n[yellow]Did you mean:[/yellow]")
        for name in exact_matches:
            console.print(f"  • {name} (case-sensitive)")
    else:
        # Try with common substitutions
        variations = [
            component_name.replace("-", "_"),  # dash to underscore
            component_name.replace("_", "-"),  # underscore to dash
            component_name.replace(" ", "_"),  # space to underscore
            component_name.replace(" ", "-"),  # space to dash
        ]

        # Collect all similar matches
        all_similar = set()
        for variant in [component_name] + variations:
            similar = get_close_matches(variant, comp_names, n=3, cutoff=0.5)
            all_similar.update(similar)

        # Also try case-insensitive fuzzy matching
        similar_lower = get_close_matches(
            lower_comp_name, [n.lower() for n in comp_names], n=3, cutoff=0.5
        )
        # Map back to original names
        for lower_match in similar_lower:
            for name in comp_names:
                if name.lower() == lower_match:
                    all_similar.add(name)

        if all_similar:
            console.print("\n[yellow]Did you mean:[/yellow]")
            for name in sorted(all_similar)[:5]:  # Show up to 5 suggestions
                console.print(f"  • {name}")


def _show_capabilities(component: Any, component_info: Any) -> None:
    """Show component capabilities."""
    capabilities = []

    if component_info.component_type == ComponentType.PORT:
        capabilities.append("• Defines interface contract")
        capabilities.append("• Can have multiple implementations")

    elif component_info.component_type == ComponentType.ADAPTER:
        capabilities.append("• Implements port interface")
        if component and hasattr(component, "__init__"):
            import inspect

            sig = inspect.signature(component.__init__)
            if "config" in sig.parameters:
                capabilities.append("• Configurable")

        # Check for async methods
        if component:
            for name in dir(component):
                if name.startswith("a") and callable(getattr(component, name)):
                    capabilities.append("• Async support")
                    break

    elif component_info.component_type == ComponentType.NODE:
        capabilities.append("• Creates DAG nodes")
        capabilities.append("• Handles node execution")

    for cap in capabilities:
        console.print(f"  {cap}")
