"""Plugins management command for HexDAG CLI."""

import contextlib
from enum import StrEnum
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer()
console = Console()


class OutputFormat(StrEnum):
    """Output format options."""

    TABLE = "table"
    JSON = "json"
    YAML = "yaml"


@app.command("list")
def list_plugins(
    format: Annotated[
        OutputFormat | None,
        typer.Option(
            "--format",
            "-f",
            help="Output format",
        ),
    ] = None,
) -> None:
    """List all available plugins and adapters."""
    # Set default format if not provided
    if format is None:
        format = OutputFormat.TABLE

    # Check available extras
    available_plugins = _get_available_plugins()

    if format == OutputFormat.JSON:
        console.print_json(data=available_plugins)
    elif format == OutputFormat.YAML:
        import yaml

        console.print(yaml.dump(available_plugins, default_flow_style=False))
    else:
        # Table format
        table = Table(title="Available Plugins", show_header=True, header_style="bold magenta")
        table.add_column("Plugin", style="cyan", no_wrap=True)
        table.add_column("Namespace", style="green")
        table.add_column("Status", style="yellow")
        table.add_column("Capabilities", style="white")

        for plugin in available_plugins:
            status = "✓ Installed" if plugin["installed"] else "✗ Not installed"
            table.add_row(
                str(plugin["name"]),
                str(plugin["namespace"]),
                status,
                ", ".join(str(c) for c in plugin["capabilities"]),
            )

        console.print(table)


@app.command("check")
def check_plugins() -> None:
    """Check plugin dependencies and suggest installation commands."""
    console.print("[bold]Checking plugin dependencies...[/bold]\n")

    checks = _check_dependencies()
    has_missing = False

    for check in checks:
        if check["status"] == "ok":
            console.print(f"✓ [green]{check['name']}[/green] - OK")
        elif check["status"] == "missing":
            has_missing = True
            console.print(f"✗ [red]{check['name']}[/red] - Missing")
            if check.get("install_hint"):
                console.print(f"  → Install with: [yellow]{check['install_hint']}[/yellow]")
        elif check["status"] == "optional":
            console.print(f"○ [yellow]{check['name']}[/yellow] - Optional")
            if check.get("install_hint"):
                console.print(f"  → Install with: [dim]{check['install_hint']}[/dim]")

    if not has_missing:
        console.print("\n[green]All required dependencies are installed![/green]")
    else:
        console.print(
            "\n[yellow]Some dependencies are missing. See installation hints above.[/yellow]"
        )


@app.command("install")
def install_plugin(
    plugin_name: str = typer.Argument(
        ...,
        help="Plugin name or extra to install (e.g., 'openai', 'viz')",
    ),
    use_uv: bool = typer.Option(
        None,
        "--uv/--pip",
        help="Force use of uv or pip (auto-detects by default)",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be installed without actually installing",
    ),
    editable: bool = typer.Option(
        False,
        "--editable",
        "-e",
        help="Install in editable/development mode",
    ),
) -> None:
    """Install a plugin or adapter (wrapper around package manager).

    Raises
    ------
    typer.Exit
        If installation fails or plugin not found
    """
    # Map plugin names to extras
    plugin_map = {
        "openai": "adapters-openai",
        "anthropic": "adapters-anthropic",
        "viz": "viz",
        "visualization": "viz",
        "cli": "cli",
        "all": "all",
    }

    extra = plugin_map.get(plugin_name, plugin_name)

    # Detect package manager
    import shutil
    import subprocess  # nosec B404 - Required for package installation
    from pathlib import Path

    # Determine which package manager to use
    has_uv = shutil.which("uv") is not None
    use_uv_final = use_uv if use_uv is not None else has_uv

    if use_uv_final and not has_uv:
        console.print("[yellow]Warning: uv requested but not found. Using pip instead.[/yellow]")
        use_uv_final = False

    # Build the command - using list format for security
    if use_uv_final:
        if editable:
            # For editable install with uv, need to install from current directory
            if Path("pyproject.toml").exists():
                cmd_list = ["uv", "pip", "install", "-e", f".[{extra}]"]
                cmd_str = f"uv pip install -e .[{extra}]"
            else:
                console.print(
                    "[red]Error: Editable install requires pyproject.toml "
                    "in current directory[/red]"
                )
                raise typer.Exit(1)
        else:
            cmd_list = ["uv", "pip", "install", f"hexdag[{extra}]"]
            cmd_str = f"uv pip install hexdag[{extra}]"
    else:
        if editable:
            if Path("pyproject.toml").exists():
                cmd_list = ["pip", "install", "-e", f".[{extra}]"]
                cmd_str = f"pip install -e .[{extra}]"
            else:
                console.print(
                    "[red]Error: Editable install requires pyproject.toml "
                    "in current directory[/red]"
                )
                raise typer.Exit(1)
        else:
            cmd_list = ["pip", "install", f"hexdag[{extra}]"]
            cmd_str = f"pip install hexdag[{extra}]"

    if dry_run:
        # Use markup=False to avoid bracket interpretation
        console.print("[yellow]Would run:[/yellow] ", end="")
        console.print(cmd_str, markup=False)
        if use_uv_final:
            console.print("[dim]Using: uv package manager[/dim]")
        else:
            console.print("[dim]Using: pip package manager[/dim]")
    else:
        console.print(f"[cyan]Installing {plugin_name}...[/cyan]")
        # Use markup=False for the command
        console.print("Running: [bold]", end="")
        console.print(cmd_str, markup=False, style="bold")
        console.print("[/bold]", end="\n")

        # Run the installation - using list format without shell=True for security
        result = subprocess.run(cmd_list, capture_output=True, text=True)  # nosec B603

        if result.returncode == 0:
            console.print(f"[green]✓ Successfully installed {plugin_name}[/green]")

            # Show what was installed
            if "Successfully installed" in result.stdout:
                console.print("\n[dim]Installed packages:[/dim]")
                for line in result.stdout.split("\n"):
                    if "Successfully installed" in line:
                        packages = line.split("Successfully installed")[1].strip()
                        for pkg in packages.split():
                            console.print(f"  • {pkg}")
        else:
            console.print(f"[red]✗ Failed to install {plugin_name}[/red]")
            if result.stderr:
                console.print(f"[dim]{result.stderr}[/dim]")
            if result.stdout and "error" in result.stdout.lower():
                console.print(f"[dim]{result.stdout}[/dim]")


def _get_available_plugins() -> list[dict[str, Any]]:
    """Get list of available plugins dynamically from registry and known extras."""
    from hexdag.core.bootstrap import bootstrap_registry
    from hexdag.core.registry import registry

    # Ensure registry is bootstrapped
    from hexdag.core.registry.models import ComponentType

    with contextlib.suppress(Exception):
        # Registry may already be bootstrapped or other initialization issue
        # Auto-discovery happens during bootstrap
        bootstrap_registry()

    plugins = []

    # Get all adapters from registry
    components = registry.list_components()
    adapters = [c for c in components if c.component_type == ComponentType.ADAPTER]

    # Group adapters by their namespace
    from hexdag.core.registry.models import ComponentInfo

    by_namespace: dict[str, list[ComponentInfo]] = {}
    for adapter in adapters:
        ns = adapter.namespace
        if ns not in by_namespace:
            by_namespace[ns] = []
        by_namespace[ns].append(adapter)

    # Add discovered adapters as plugins
    for ns, ns_adapters in by_namespace.items():
        # Get unique capabilities from all adapters
        capabilities = set()
        for adapter in ns_adapters:
            if adapter.metadata.implements_port:
                port = adapter.metadata.implements_port
                # Map port names to capabilities
                capability_map = {
                    "llm": "LLM",
                    "database": "Database",
                    "memory": "Memory",
                    "tool_router": "ToolRouter",
                    "api_call": "API",
                }
                cap = capability_map.get(port, port)
                capabilities.add(cap)

        # Create a plugin entry for each namespace group
        if ns == "plugin" and ns_adapters:
            # Group adapters by their prefix (e.g., "mock" from "mock_llm", "mock_database")
            prefix_groups: dict[str, list[ComponentInfo]] = {}
            for adapter in ns_adapters:
                # Special case for in_memory_memory -> local
                if adapter.name == "in_memory_memory":
                    prefix = "local"
                elif "_" in adapter.name:
                    # Extract prefix before underscore
                    prefix = adapter.name.split("_")[0]
                else:
                    prefix = adapter.name

                if prefix not in prefix_groups:
                    prefix_groups[prefix] = []
                prefix_groups[prefix].append(adapter)

            # Create plugin entry for each prefix group
            for prefix, group_adapters in prefix_groups.items():
                # Collect capabilities from all adapters in this group
                group_capabilities = set()
                for adapter in group_adapters:
                    if adapter.metadata.implements_port:
                        port = adapter.metadata.implements_port
                        # Map port names to capabilities
                        capability_map = {
                            "llm": "LLM",
                            "database": "Database",
                            "memory": "Memory",
                            "tool_router": "ToolRouter",
                            "api_call": "API",
                        }
                        cap = capability_map.get(port, port)
                        group_capabilities.add(cap)

                caps = sorted(group_capabilities) if group_capabilities else ["Adapter"]
                plugins.append({
                    "name": prefix,
                    "namespace": ns,
                    "installed": True,  # If in registry, it's installed
                    "capabilities": caps,
                })

    # Check for known optional dependencies that may not be loaded
    known_extras = {
        "openai": {
            "module": "hexdag.adapters.openai",
            "capabilities": ["LLM", "Embeddings"],
        },
        "anthropic": {
            "module": "hexdag.adapters.anthropic",
            "capabilities": ["LLM"],
        },
        "visualization": {
            "module": "hexdag.visualization",
            "capabilities": ["DAG Visualization"],
        },
    }

    for name, info in known_extras.items():
        # Check if already in plugins
        if name not in [p["name"] for p in plugins]:
            # Try to import to check availability
            installed = False
            try:
                import importlib

                importlib.import_module(str(info["module"]))
                installed = True
            except ImportError:
                pass

            plugins.append({
                "name": name,
                "namespace": "plugin",
                "installed": installed,
                "capabilities": info["capabilities"],
            })

    # Special check for visualization
    try:
        from hexdag.visualization import GRAPHVIZ_AVAILABLE

        # Update visualization status
        for plugin in plugins:
            if plugin["name"] == "visualization":
                plugin["installed"] = GRAPHVIZ_AVAILABLE
                plugin["namespace"] = "core"
                break
    except ImportError:
        pass

    return plugins


def _check_dependencies() -> list[dict]:
    """Check plugin dependencies dynamically."""
    import importlib
    import shutil

    # Detect if uv is available for better hints
    has_uv = shutil.which("uv") is not None
    prefix = "uv pip install" if has_uv else "pip install"

    checks = []

    # Define dependency checks
    dependency_checks = {
        "pydantic": {
            "name": "pydantic (core)",
            "required": True,
            "extra": None,
        },
        "yaml": {
            "name": "PyYAML (CLI)",
            "required": False,
            "extra": "cli",
        },
        "graphviz": {
            "name": "graphviz (visualization)",
            "required": False,
            "extra": "viz",
        },
    }

    # Check each dependency
    for module_name, info in dependency_checks.items():
        try:
            importlib.import_module(module_name)
            checks.append({"name": info["name"], "status": "ok"})
        except ImportError:
            status = "missing" if info["required"] else "optional"
            hint = f"{prefix} hexdag"
            if info["extra"]:
                hint = f"{prefix} hexdag[{info['extra']}]"

            checks.append({
                "name": info["name"],
                "status": status,
                "install_hint": hint,
            })

    # Check optional adapter packages dynamically
    adapter_packages: dict[str, Any] = {
        "openai": {
            "display": "OpenAI",
            "adapter_module": "hexdag.adapters.openai",
            "extra": "adapters-openai",
        },
        "anthropic": {
            "display": "Anthropic",
            "adapter_module": "hexdag.adapters.anthropic",
            "extra": "adapters-anthropic",
        },
    }

    for sdk_name, info in adapter_packages.items():
        sdk_ok = False
        adapter_ok = False

        # Check SDK
        try:
            importlib.import_module(sdk_name)
            sdk_ok = True
        except ImportError:
            pass

        # Check adapter if SDK exists
        if sdk_ok:
            try:
                importlib.import_module(info["adapter_module"])
                adapter_ok = True
            except ImportError:
                pass

        # Add check result
        if sdk_ok and adapter_ok:
            checks.append({"name": f"{info['display']} (SDK + Adapter)", "status": "ok"})
        elif sdk_ok:
            checks.append({
                "name": f"{info['display']} (SDK only, adapter missing)",
                "status": "optional",
                "install_hint": f"{prefix} hexdag[{info['extra']}]",
            })
        else:
            checks.append({
                "name": f"{info['display']} (not installed)",
                "status": "optional",
                "install_hint": f"{prefix} hexdag[{info['extra']}]",
            })

    return checks
