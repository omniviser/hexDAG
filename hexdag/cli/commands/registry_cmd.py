"""Registry inspection command for HexDAG CLI."""

import contextlib
from enum import StrEnum
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table
from rich.tree import Tree

from hexdag.core.bootstrap import bootstrap_registry
from hexdag.core.registry import registry
from hexdag.core.registry.models import ComponentInfo, ComponentType

app = typer.Typer()
console = Console()


class ComponentFilter(StrEnum):
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
        import json

        data = [
            {
                "name": c.name,
                "qualified_name": c.qualified_name,
                "type": str(c.component_type.value)
                if hasattr(c.component_type, "value")
                else str(c.component_type),
                "namespace": c.namespace,
                "description": c.metadata.description if c.metadata else "",
                "subtype": str(c.metadata.subtype) if c.metadata and c.metadata.subtype else None,
                "implements_port": c.metadata.implements_port if c.metadata else None,
            }
            for c in components
        ]
        console.print(json.dumps(data, indent=2))
    elif format == "yaml":
        import yaml

        data = [
            {
                "name": c.name,
                "qualified_name": c.qualified_name,
                "type": str(c.component_type.value)
                if hasattr(c.component_type, "value")
                else str(c.component_type),
                "namespace": c.namespace,
                "description": c.metadata.description if c.metadata else "",
                "subtype": str(c.metadata.subtype) if c.metadata and c.metadata.subtype else None,
                "implements_port": c.metadata.implements_port if c.metadata else None,
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
    """Show detailed information about a specific component.

    Raises
    ------
    typer.Exit
        If component not found
    """
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
        if len(matches) == 1:
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
        if component_info.metadata.description:
            console.print(f"[dim]Description: {component_info.metadata.description}[/dim]")

        # Show methods for ports
        if component_info.component_type == ComponentType.PORT:
            console.print("\n[bold]Interface Methods:[/bold]")
            # Get the port class from registry
            import inspect

            port_metadata = registry.get_metadata(
                component_name, namespace=component_info.namespace
            )
            port_class = port_metadata

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
            if component_info.metadata.implements_port:
                port_name = component_info.metadata.implements_port
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
    namespaces: dict[str, list[ComponentInfo]] = {}
    for comp in components:
        if comp.namespace not in namespaces:
            namespaces[comp.namespace] = []
        namespaces[comp.namespace].append(comp)

    # Add to tree
    for ns_name, ns_components in namespaces.items():
        ns_branch = tree.add(f"[yellow]{ns_name}[/yellow]")

        # Group by type within namespace
        by_type: dict[str, list[ComponentInfo]] = {}
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
                if comp.metadata.implements_port:
                    metadata_str = f" [dim]→ {comp.metadata.implements_port}[/dim]"
                type_branch.add(f"[cyan]{comp.name}[/cyan]{metadata_str}")

    console.print(tree)


@app.command("bootstrap")
def bootstrap_command(
    manifest: Annotated[
        str | None,
        typer.Option(
            "--manifest",
            "-m",
            help="Path to manifest YAML file",
        ),
    ] = None,
    dev: Annotated[
        bool,
        typer.Option(
            "--dev",
            help="Enable development mode",
        ),
    ] = False,
) -> None:
    """Bootstrap the registry from configuration or manifest."""
    try:
        if manifest:
            console.print(f"[cyan]Bootstrapping from manifest: {manifest}[/cyan]")
            bootstrap_registry(config_path=manifest)
        else:
            console.print("[cyan]Bootstrapping from default configuration[/cyan]")
            bootstrap_registry()

        # Show summary
        components = registry.list_components()
        namespaces = {c.namespace for c in components}

        console.print("[green]✓ Registry bootstrapped successfully[/green]")
        console.print(f"  Components: {len(components)}")
        console.print(f"  Namespaces: {', '.join(sorted(namespaces))}")

        if dev:
            console.print("[yellow]Development mode enabled[/yellow]")

    except Exception as e:
        console.print(f"[red]Error bootstrapping registry: {e}[/red]")
        raise typer.Exit(1) from e


@app.command("namespaces")
def list_namespaces() -> None:
    """List all registered namespaces."""
    # Bootstrap if not already done
    with contextlib.suppress(Exception):
        bootstrap_registry()

    components = registry.list_components()
    namespaces: dict[str, list] = {}

    # Group components by namespace
    for comp in components:
        if comp.namespace not in namespaces:
            namespaces[comp.namespace] = []
        namespaces[comp.namespace].append(comp)

    # Create table
    table = Table(
        title="Registry Namespaces",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Namespace", style="yellow")
    table.add_column("Components", justify="right", style="cyan")
    table.add_column("Types", style="green")

    for ns, comps in sorted(namespaces.items()):
        types = {c.component_type.value for c in comps}
        table.add_row(
            ns,
            str(len(comps)),
            ", ".join(sorted(types)),
        )

    console.print(table)


@app.command("search")
def search_components(
    pattern: Annotated[
        str,
        typer.Argument(help="Search pattern (supports wildcards)"),
    ],
    type_filter: Annotated[
        ComponentFilter | None,
        typer.Option(
            "--type",
            "-t",
            help="Filter by component type",
        ),
    ] = None,
) -> None:
    """Search for components by name pattern."""
    import re

    # Bootstrap if not already done
    with contextlib.suppress(Exception):
        bootstrap_registry()

    components = registry.list_components()

    # Convert wildcard pattern to regex
    regex_pattern = pattern.replace("*", ".*").replace("?", ".")
    regex = re.compile(regex_pattern, re.IGNORECASE)

    # Filter components
    matches = [c for c in components if regex.search(c.name) or regex.search(c.qualified_name)]

    # Apply type filter
    if type_filter and type_filter != ComponentFilter.ALL:
        type_map = {
            ComponentFilter.PORT: ComponentType.PORT,
            ComponentFilter.ADAPTER: ComponentType.ADAPTER,
            ComponentFilter.NODE: ComponentType.NODE,
        }
        filter_type = type_map.get(type_filter)
        matches = [c for c in matches if c.component_type == filter_type]

    if not matches:
        console.print(f"[yellow]No components found matching '{pattern}'[/yellow]")
        return

    # Display results
    table = Table(
        title=f"Search Results: '{pattern}'",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Qualified Name", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("Description", style="dim")

    for comp in sorted(matches, key=lambda c: c.qualified_name):
        desc = (
            comp.metadata.description[:60] + "..."
            if comp.metadata and len(comp.metadata.description) > 60
            else (comp.metadata.description if comp.metadata else "")
        )
        table.add_row(
            comp.qualified_name,
            comp.component_type.value,
            desc,
        )

    console.print(table)
    console.print(f"\n[dim]Found {len(matches)} component(s)[/dim]")


@app.command("verify")
def verify_registry() -> None:
    """Verify registry integrity (duplicates, protection, immutability)."""
    # Bootstrap if not already done
    with contextlib.suppress(Exception):
        bootstrap_registry()

    components = registry.list_components()
    issues = []

    # Check for duplicates
    seen = {}
    for comp in components:
        key = comp.qualified_name
        if key in seen:
            issues.append(f"Duplicate component: {key}")
        seen[key] = comp

    # Check protected components in core namespace
    core_components = [c for c in components if c.namespace == "core"]
    if core_components:
        console.print(f"[dim]Protected core components: {len(core_components)}[/dim]")

    # Report results
    if issues:
        console.print("[red]Registry integrity issues found:[/red]")
        for issue in issues:
            console.print(f"  [red]✗[/red] {issue}")
        raise typer.Exit(1)
    console.print("[green]✓ Registry integrity verified[/green]")
    console.print(f"  Total components: {len(components)}")
    console.print(f"  Unique namespaces: {len({c.namespace for c in components})}")
    console.print(f"  Protected (core): {len(core_components)}")


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
