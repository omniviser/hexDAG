"""CLI commands for MkDocs documentation."""

import subprocess
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

app = typer.Typer(help="Generate and serve documentation with MkDocs")
console = Console()


@app.command("build")
def build_docs(
    clean: bool = typer.Option(
        False,
        "--clean",
        "-c",
        help="Clean site directory before building",
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        "-s",
        help="Enable strict mode (warnings as errors)",
    ),
    config: Path | None = typer.Option(
        None,
        "--config",
        help="Path to TOML configuration file (pyproject.toml or hexdag.toml)",
    ),
) -> None:
    """Build documentation site with MkDocs.

    Generates static HTML documentation from markdown files and the component
    registry. The registry hooks automatically generate reference documentation
    during the build process.

    You can specify a custom TOML configuration to document your project's
    specific components, plugins, and adapters.

    Examples:
        hexdag docs build
        hexdag docs build --clean
        hexdag docs build --strict
        hexdag docs build --config /path/to/myproject/pyproject.toml
    """
    import os

    try:
        if config:
            if not config.exists():
                console.print(f"[red]✗[/red] Config file not found: {config}")
                raise typer.Exit(1)
            os.environ["HEXDAG_CONFIG_PATH"] = str(config.resolve())
            console.print(
                Panel.fit(
                    "[bold cyan]Building Documentation[/bold cyan]\n"
                    "Using MkDocs with registry hooks\n"
                    f"Config: {config}",
                    border_style="cyan",
                )
            )
        else:
            console.print(
                Panel.fit(
                    "[bold cyan]Building Documentation[/bold cyan]\n"
                    "Using MkDocs with registry hooks",
                    border_style="cyan",
                )
            )

        cmd = ["mkdocs", "build"]
        if clean:
            cmd.append("--clean")
        if strict:
            cmd.append("--strict")

        result = subprocess.run(cmd, capture_output=True, text=True, env=os.environ.copy())  # nosec B603

        if result.returncode == 0:
            console.print("\n[green]✓[/green] Documentation built successfully!")
            console.print("\n[dim]Site output: ./site/[/dim]")
            console.print("[dim]View locally: hexdag docs serve[/dim]")
        else:
            console.print("\n[red]✗[/red] Build failed:")
            console.print(result.stderr)
            raise typer.Exit(1)

    except FileNotFoundError:
        console.print("\n[red]✗[/red] MkDocs not found. Install with:")
        console.print("  [cyan]pip install hexdag[docs][/cyan]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"\n[red]✗[/red] Error building documentation: {e}")
        raise typer.Exit(1)


@app.command("serve")
def serve_docs(
    port: int = typer.Option(
        8000,
        "--port",
        "-p",
        help="Port to serve on",
    ),
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        "-h",
        help="Host to bind to",
    ),
    watch_theme: bool = typer.Option(
        False,
        "--watch-theme",
        help="Watch theme directory for changes",
    ),
    config: Path | None = typer.Option(
        None,
        "--config",
        help="Path to TOML configuration file (pyproject.toml or hexdag.toml)",
    ),
) -> None:
    """Serve documentation with live reload.

    Starts MkDocs development server with live reload. The server watches for
    changes in markdown files and automatically regenerates registry documentation.

    You can specify a custom TOML configuration to document your project's
    specific components, plugins, and adapters.

    Examples:
        hexdag docs serve
        hexdag docs serve --port 8080
        hexdag docs serve --watch-theme
        hexdag docs serve --config /path/to/myproject/pyproject.toml
    """
    import os

    try:
        if config:
            if not config.exists():
                console.print(f"[red]✗[/red] Config file not found: {config}")
                raise typer.Exit(1)
            os.environ["HEXDAG_CONFIG_PATH"] = str(config.resolve())
            console.print(
                Panel.fit(
                    "[bold cyan]MkDocs Development Server[/bold cyan]\n"
                    f"URL: http://{host}:{port}\n"
                    "Live reload: Enabled\n"
                    f"Config: {config}",
                    border_style="cyan",
                )
            )
        else:
            console.print(
                Panel.fit(
                    "[bold cyan]MkDocs Development Server[/bold cyan]\n"
                    f"URL: http://{host}:{port}\n"
                    "Live reload: Enabled",
                    border_style="cyan",
                )
            )

        cmd = ["mkdocs", "serve", "--dev-addr", f"{host}:{port}"]
        if watch_theme:
            cmd.append("--watch-theme")

        console.print(f"\n[green]✓[/green] Starting server at [cyan]http://{host}:{port}[/cyan]")
        console.print("\nPress [red]Ctrl+C[/red] to stop the server\n")

        # Run mkdocs serve (blocking)
        subprocess.run(cmd, env=os.environ.copy())  # nosec B603

    except FileNotFoundError:
        console.print("\n[red]✗[/red] MkDocs not found. Install with:")
        console.print("  [cyan]pip install hexdag[docs][/cyan]")
        raise typer.Exit(1)
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Server stopped[/yellow]")
    except Exception as e:
        console.print(f"\n[red]✗[/red] Error starting server: {e}")
        raise typer.Exit(1)


@app.command("deploy")
def deploy_docs(
    remote_branch: str = typer.Option(
        "gh-pages",
        "--remote-branch",
        "-b",
        help="Remote branch to deploy to",
    ),
    message: str | None = typer.Option(
        None,
        "--message",
        "-m",
        help="Commit message for deploy",
    ),
) -> None:
    """Deploy documentation to GitHub Pages.

    Builds and deploys documentation to the specified remote branch (default: gh-pages).
    Requires git repository with GitHub remote configured.

    Examples:
        hexdag docs deploy
        hexdag docs deploy --message "Update docs"
        hexdag docs deploy --remote-branch main
    """
    try:
        console.print(
            Panel.fit(
                f"[bold cyan]Deploying Documentation[/bold cyan]\nRemote branch: {remote_branch}",
                border_style="cyan",
            )
        )

        cmd = ["mkdocs", "gh-deploy", "--remote-branch", remote_branch]
        if message:
            cmd.extend(["--message", message])

        result = subprocess.run(cmd, capture_output=True, text=True)  # nosec B603

        if result.returncode == 0:
            console.print("\n[green]✓[/green] Documentation deployed successfully!")
        else:
            console.print("\n[red]✗[/red] Deploy failed:")
            console.print(result.stderr)
            raise typer.Exit(1)

    except FileNotFoundError:
        console.print("\n[red]✗[/red] MkDocs not found. Install with:")
        console.print("  [cyan]pip install hexdag[docs][/cyan]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"\n[red]✗[/red] Error deploying documentation: {e}")
        raise typer.Exit(1)


@app.command("new")
def new_page(
    path: str = typer.Argument(..., help="Path for new page (e.g., guide/my-guide.md)"),
    title: str | None = typer.Option(None, "--title", "-t", help="Page title"),
) -> None:
    """Create a new documentation page.

    Creates a new markdown file with basic frontmatter and adds it to the
    documentation structure.

    Examples:
        hexdag docs new guide/custom-nodes.md
        hexdag docs new guide/advanced.md --title "Advanced Usage"
    """
    docs_dir = Path("docs")
    if not docs_dir.exists():
        console.print(f"[red]✗[/red] Documentation directory not found: {docs_dir}")
        raise typer.Exit(1)

    file_path = docs_dir / path
    if file_path.exists():
        console.print(f"[red]✗[/red] File already exists: {file_path}")
        raise typer.Exit(1)

    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate title from filename if not provided
    if not title:
        title = file_path.stem.replace("-", " ").replace("_", " ").title()

    content = f"""# {title}

Add your content here.

## Overview

...

## Examples

...

## Next Steps

...
"""

    file_path.write_text(content)

    console.print(f"[green]✓[/green] Created new page: [cyan]{file_path}[/cyan]")
    console.print("\nRemember to add it to [cyan]mkdocs.yml[/cyan] navigation:")
    console.print(f"  - {title}: {path}")
