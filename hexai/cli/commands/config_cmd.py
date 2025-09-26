"""Configuration management commands for HexDAG CLI."""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table

from hexai.core.bootstrap import bootstrap_registry
from hexai.core.config import load_config
from hexai.core.config.models import ManifestEntry
from hexai.core.registry import registry

app = typer.Typer(help="Configuration management commands")
console = Console()


def ensure_registry_ready() -> None:
    """Ensure the registry is bootstrapped.

    If not ready, bootstrap with default configuration.
    """
    if not registry.ready:
        # Bootstrap with default configuration
        # This will load whatever is in the config file
        bootstrap_registry(dev_mode=True)


@app.command()
def list_plugins() -> None:
    """List all available configurable plugins/adapters.

    This command discovers all registered components that implement
    the ConfigurableComponent protocol and displays their configuration options.
    """
    # Ensure registry is ready
    ensure_registry_ready()
    configurable = registry.get_configurable_components()

    if not configurable:
        console.print("[yellow]No configurable components found[/yellow]")
        return

    # Create table
    table = Table(title="Configurable Components", show_lines=True)
    table.add_column("Namespace", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("Port", style="blue")
    table.add_column("Fields", style="white")

    for namespace, info in configurable.items():
        config_class = info["config_class"]
        fields = []
        for field_name, field_info in config_class.model_fields.items():
            desc = field_info.description or ""
            if len(desc) > 30:
                desc = desc[:27] + "..."
            fields.append(f"• {field_name}: {desc}")

        table.add_row(
            namespace,
            str(info["type"]).split(".")[-1],
            info.get("port") or "N/A",
            "\n".join(fields),
        )

    console.print(table)


# Define module-level defaults to avoid B008 issues
_DEFAULT_OUTPUT = Path("hexdag.toml")
_DEFAULT_PLUGINS: list[str] = []
_DEFAULT_SHOW = False


@app.command()
def generate(
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output file path for the generated configuration",
        ),
    ] = None,
    plugins: Annotated[
        list[str] | None,
        typer.Option(
            "--plugin",
            "-p",
            help="Module paths of plugins to include (e.g., hexai.adapters.llm.openai_adapter)",
        ),
    ] = None,
    show: Annotated[
        bool | None,
        typer.Option(
            "--show",
            "-s",
            help="Show configuration on console instead of writing to file",
        ),
    ] = None,
) -> None:
    """Generate a TOML configuration template with plugin schemas.

    This command generates a complete HexDAG configuration file with:
    - Core module definitions
    - Plugin configurations with all available settings
    - Example values and documentation

    Examples:
        # Generate default config to hexdag.toml
        hexdag config generate

        # Generate config with specific plugins
        hexdag config generate --plugin openai --plugin anthropic

        # Show config on console
        hexdag config generate --show

        # Generate to custom file
        hexdag config generate -o my-config.toml
    """
    # Apply defaults
    if output is None:
        output = _DEFAULT_OUTPUT
    if plugins is None:
        plugins = _DEFAULT_PLUGINS
    if show is None:
        show = _DEFAULT_SHOW
    sections = []

    # Header
    sections.append(
        "# ============================================================================"
    )
    sections.append("# HexDAG Configuration")
    sections.append("# Generated from plugin configuration schemas")
    sections.append(
        "# ============================================================================"
    )
    sections.append("")

    # Core modules
    sections.append("# Core modules to load")
    sections.append('modules = ["hexai.core.ports", "hexai.core.application.nodes"]')
    sections.append("")

    # Plugins are provided as full module paths
    # No discovery, no hardcoding - just use what the user provides
    sections.append("# Plugins to load")
    if plugins:
        sections.append(f"plugins = {list(plugins)}")
    else:
        sections.append("plugins = []  # Add plugin module paths here")
    sections.append("")

    # Development mode
    sections.append("# Development mode - allows post-bootstrap registration")
    sections.append("dev_mode = false")
    sections.append("")

    # Global settings
    sections.append("[settings]")
    sections.append('log_level = "INFO"')
    sections.append("")

    # Auto-discover and generate plugin configurations
    # Bootstrap a temporary registry with the requested plugins
    # to get their configuration schemas

    if plugins and not registry.ready:
        # Bootstrap with the plugins we want to generate configs for
        entries = [
            ManifestEntry(namespace="core", module="hexai.core.ports"),
        ]

        # Use list comprehension for better performance (PERF401)
        entries.extend(
            ManifestEntry(namespace="core", module=module_path) for module_path in plugins
        )

        try:
            registry.bootstrap(entries, dev_mode=True)
        except Exception as e:
            console.print(f"[red]Error loading plugins: {e}[/red]")
            console.print("[dim]Make sure plugin module paths are correct[/dim]")
            raise typer.Exit(1) from e

    configurable = registry.get_configurable_components()

    # Generate configuration for each configurable component
    for component_name, info in configurable.items():
        config_class = info["config_class"]

        sections.append("")
        sections.append("# ==============================================================")
        sections.append(f"# {component_name.upper()} CONFIGURATION")
        sections.append("# ==============================================================")
        sections.append("")
        sections.append(f"[adapters.{component_name}]")

        # Generate fields from Pydantic schema
        for field_name, field_info in config_class.model_fields.items():
            description = field_info.description or f"{field_name} configuration"
            default = field_info.default

            sections.append(f"# {description}")

            if default is None:
                sections.append(f"# {field_name} = null")
            elif isinstance(default, str):
                sections.append(f'{field_name} = "{default}"')
            elif isinstance(default, bool):
                sections.append(f"{field_name} = {'true' if default else 'false'}")
            elif isinstance(default, (int, float)):
                sections.append(f"{field_name} = {default}")
            else:
                sections.append(f"# {field_name} = # TODO: Set value")
            sections.append("")

    sections.append("")

    # Join all sections
    config_content = "\n".join(sections)

    if show:
        # Display on console with syntax highlighting
        syntax = Syntax(config_content, "toml", theme="monokai", line_numbers=True)
        console.print(syntax)
    else:
        # Write to file
        output.write_text(config_content)
        console.print(f"[green]✓[/green] Configuration generated: {output}")
        console.print(f"[dim]  Plugins included: {', '.join(plugins)}[/dim]")


# Default config file for validation
_DEFAULT_CONFIG_FILE = Path("hexdag.toml")


@app.command()
def validate(
    config_file: Annotated[
        Path | None,
        typer.Argument(
            help="Path to configuration file to validate",
        ),
    ] = None,
) -> None:
    """Validate a HexDAG configuration file.

    This command checks:
    - TOML syntax validity
    - Required fields presence
    - Plugin configuration compatibility
    - Environment variable references

    Examples:
        # Validate default config
        hexdag config validate

        # Validate specific file
        hexdag config validate my-config.toml
    """
    # Apply default
    if config_file is None:
        config_file = _DEFAULT_CONFIG_FILE

    if not config_file.exists():
        console.print(f"[red]✗[/red] Configuration file not found: {config_file}")
        raise typer.Exit(1)

    try:
        # Try to load the configuration
        config = load_config(config_file)

        console.print(f"[green]✓[/green] Configuration is valid: {config_file}")
        console.print(f"[dim]  Modules: {len(config.modules)}[/dim]")
        console.print(f"[dim]  Plugins: {len(config.plugins)}[/dim]")
        console.print(f"[dim]  Dev mode: {config.dev_mode}[/dim]")

        # Check for plugin configs
        if "plugin" in config.settings:
            plugin_count = len(config.settings["plugin"])
            console.print(f"[dim]  Plugin configs: {plugin_count}[/dim]")

    except Exception as e:
        console.print("[red]✗[/red] Configuration validation failed:")
        console.print(f"[red]  {e}[/red]")
        raise typer.Exit(1) from e


@app.command()
def show(
    config_file: Annotated[
        Path | None,
        typer.Argument(
            help="Path to configuration file (auto-detects if not specified)",
        ),
    ] = None,
) -> None:
    """Show the current HexDAG configuration.

    This command displays:
    - The active configuration file location
    - Loaded modules and plugins
    - All configuration settings

    Examples:
        # Show auto-detected config
        hexdag config show

        # Show specific config file
        hexdag config show my-config.toml
    """
    try:
        from hexai.core.config import load_config

        if config_file:
            config = load_config(config_file)
            console.print(f"[blue]Configuration from:[/blue] {config_file}")
        else:
            config = load_config()
            # Try to determine which file was loaded
            search_paths = [
                Path("hexdag.toml"),
                Path("pyproject.toml"),
                Path(".hexdag.toml"),
            ]
            loaded_from = None
            for path in search_paths:
                if path.exists():
                    loaded_from = path
                    break

            if loaded_from:
                console.print(f"[blue]Configuration from:[/blue] {loaded_from}")
            else:
                console.print("[blue]Using default configuration[/blue]")

        console.print("")

        # Display configuration
        console.print("[bold]Modules:[/bold]")
        for module in config.modules:
            console.print(f"  • {module}")

        console.print("")
        console.print("[bold]Plugins:[/bold]")
        for plugin in config.plugins:
            console.print(f"  • {plugin}")

        console.print("")
        console.print(f"[bold]Dev Mode:[/bold] {config.dev_mode}")

        if config.settings:
            console.print("")
            console.print("[bold]Settings:[/bold]")
            _display_dict(config.settings, indent=2)

    except FileNotFoundError:
        console.print("[yellow]No configuration file found, using defaults[/yellow]")
    except Exception as e:
        console.print(f"[red]Error loading configuration: {e}[/red]")
        raise typer.Exit(1) from e


def _display_dict(data: dict, indent: int = 0) -> None:
    """Recursively display dictionary contents."""
    for key, value in data.items():
        prefix = " " * indent
        if isinstance(value, dict):
            console.print(f"{prefix}[cyan]{key}:[/cyan]")
            _display_dict(value, indent + 2)
        elif isinstance(value, list):
            console.print(f"{prefix}[cyan]{key}:[/cyan] {value}")
        else:
            console.print(f"{prefix}[cyan]{key}:[/cyan] {value}")


if __name__ == "__main__":
    app()
