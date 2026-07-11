"""Initialize command for HexDAG CLI."""

from importlib import resources
from importlib.resources.abc import Traversable
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
    claude: Annotated[
        bool,
        typer.Option(
            "--claude",
            help="Also scaffold Claude Code assets (skills, agent, workflow) into .claude/",
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

    if claude:
        _scaffold_claude_assets(path, force=force)
        return

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


def _scaffold_claude_assets(path: Path, *, force: bool) -> None:
    """Copy the bundled Claude Code assets into ``path/.claude``.

    Reads the template tree shipped inside the wheel at
    ``hexdag.cli.templates/claude`` via :mod:`importlib.resources`, so it works from an
    installed package as well as from a source checkout.
    """
    source = resources.files("hexdag.cli.templates").joinpath("claude")
    if not source.is_dir():
        console.print("[red]Bundled Claude assets not found in this hexdag install.[/red]")
        raise typer.Exit(1)

    dest = path / ".claude"
    written, skipped = _copy_tree(source, dest, force=force)

    if not written and not skipped:
        console.print("[yellow]No Claude assets to write.[/yellow]")
        return

    console.print(f"[green]✓[/green] Scaffolded Claude Code assets into {dest}")
    for rel in written:
        console.print(f"  [green]+[/green] .claude/{rel}")
    for rel in skipped:
        console.print(
            f"  [yellow]•[/yellow] .claude/{rel} [dim](exists — use --force to overwrite)[/dim]"
        )

    console.print("\n[bold]Next steps:[/bold]")
    console.print("1. Restart Claude Code so it discovers the new skills/agent/workflow.")
    console.print("2. Try [cyan]/hexdag-pipeline[/cyan] or [cyan]/hexdag-validate[/cyan].")


def _copy_tree(
    source: Traversable,
    dest: Path,
    *,
    force: bool,
    _prefix: str = "",
) -> tuple[list[str], list[str]]:
    """Recursively copy a resource ``Traversable`` tree onto disk.

    Returns ``(written, skipped)`` lists of destination-relative paths. Existing files are
    skipped unless ``force`` is set. ``__init__.py`` markers are not copied.
    """
    written: list[str] = []
    skipped: list[str] = []
    dest.mkdir(parents=True, exist_ok=True)

    for entry in source.iterdir():
        rel = f"{_prefix}{entry.name}"
        target = dest / entry.name
        if entry.is_dir():
            sub_written, sub_skipped = _copy_tree(entry, target, force=force, _prefix=f"{rel}/")
            written.extend(sub_written)
            skipped.extend(sub_skipped)
            continue
        if entry.name == "__init__.py":
            continue
        if target.exists() and not force:
            skipped.append(rel)
            continue
        target.write_text(entry.read_text(encoding="utf-8"), encoding="utf-8")
        written.append(rel)

    return written, skipped


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
            "limits": {
                "max_llm_calls": 100,
                "max_cost_usd": 10.0,
                "warning_threshold": 0.8,
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
