"""Initialize command for HexDAG CLI."""

from pathlib import Path
from typing import Annotated, Any

import typer
import yaml
from rich.console import Console
from rich.prompt import Confirm

app = typer.Typer()
console = Console()

_CLI_NAME = "init"
_CLI_HELP = "Initialize a new HexDAG project"
_CLI_TYPE = "typer"


@app.callback(invoke_without_command=True)
def init(
    ctx: typer.Context,
    with_adapters: Annotated[
        str | None,
        typer.Option(
            "--with",
            help="Comma-separated list of adapters to include (e.g., openai,anthropic,local)",
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            "-f",
            help="Overwrite existing configuration",
        ),
    ] = False,
    path: Annotated[
        Path | None,
        typer.Argument(
            help="Directory to initialize (defaults to current directory)",
        ),
    ] = None,
) -> None:
    """Initialize a new HexDAG project with configuration."""
    if ctx.invoked_subcommand is not None:
        return

    if path is None:
        path = Path.cwd()

    config_path = path / "hexdag.yaml"

    if (
        config_path.exists()
        and not force
        and not Confirm.ask(f"[yellow]hexdag.yaml already exists in {path}. Overwrite?[/yellow]")
    ):
        console.print("[red]Initialization cancelled.[/red]")
        raise typer.Exit(1)

    # Parse adapters
    adapters = []
    if with_adapters:
        adapters = [a.strip() for a in with_adapters.split(",")]

    # Generate configuration
    config_content = _generate_config(adapters)

    # Ensure directory exists
    path.mkdir(parents=True, exist_ok=True)

    # Write configuration
    config_path.write_text(config_content)
    console.print(f"[green]\u2713[/green] Created hexdag.yaml in {path}")

    # Show next steps
    console.print("\n[bold]Next steps:[/bold]")
    console.print("1. Review and customize hexdag.yaml")
    console.print("2. Set any required environment variables")

    if adapters:
        console.print("\n[bold]Adapters configured:[/bold]")
        for adapter in adapters:
            _print_adapter_info(adapter)


def _generate_config(adapters: list[str]) -> str:
    """Generate hexdag.yaml configuration with kind: Config format."""
    config: dict[str, Any] = {
        "kind": "Config",
        "metadata": {
            "name": "project-defaults",
        },
        "spec": {
            "modules": [
                "hexdag.kernel.ports",
                "hexdag.stdlib.nodes",
                "hexdag.drivers.observer_manager",
                "hexdag.drivers.mock",
            ],
            "plugins": [],
            "dev_mode": True,
            "logging": {
                "level": "INFO",
                "format": "structured",
            },
            "kernel": {
                "max_concurrent_nodes": 10,
            },
        },
    }

    spec = config["spec"]

    if "openai" in adapters:
        spec["modules"].append("hexdag.drivers.openai")
    if "anthropic" in adapters:
        spec["modules"].append("hexdag.drivers.anthropic")

    if "openai" in adapters or "anthropic" in adapters or "local" in adapters or not adapters:
        spec["settings"] = {
            "log_level": "INFO",
            "enable_metrics": True,
        }

    return yaml.dump(config, default_flow_style=False, sort_keys=False)


def _print_adapter_info(adapter: str) -> None:
    """Print information about an adapter."""
    import shutil

    # Detect if uv is available
    has_uv = shutil.which("uv") is not None
    prefix = "uv pip install" if has_uv else "pip install"

    info: dict[str, dict[str, Any]] = {
        "openai": {
            "name": "OpenAI",
            "install": f"{prefix} hexdag[adapters-openai]",
            "env": "OPENAI_API_KEY",
        },
        "anthropic": {
            "name": "Anthropic Claude",
            "install": f"{prefix} hexdag[adapters-anthropic]",
            "env": "ANTHROPIC_API_KEY",
        },
        "mock": {
            "name": "Mock Adapters",
            "install": "Included in core",
            "env": None,
        },
        "local": {
            "name": "Local Adapters",
            "install": "Included in core",
            "env": None,
        },
    }

    if adapter in info:
        details = info[adapter]
        console.print(f"  \u2022 [cyan]{details['name']}[/cyan]")
        if details["install"] != "Included in core":
            console.print(f"    Install: [yellow]{details['install']}[/yellow]")
        if details["env"]:
            console.print(f"    Required: [yellow]{details['env']}[/yellow] environment variable")
