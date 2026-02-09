"""hexdag-studio CLI - Standalone visual editor for hexDAG pipelines.

Usage:
    hexdag-studio ./pipelines/
    hexdag-studio --port 8080
    hexdag-studio --plugin ./hexdag_plugins
"""

import threading
import time
import webbrowser
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

app = typer.Typer(name="hexdag-studio", help="Visual Studio UI for hexDAG pipelines")
console = Console()


def main() -> None:
    """Entry point for the hexdag-studio CLI."""
    app()


@app.callback(invoke_without_command=True)
def studio(
    ctx: typer.Context,
    path: Annotated[
        Path,
        typer.Argument(
            help="Directory containing pipeline YAML files",
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = Path(),
    host: Annotated[
        str,
        typer.Option(
            "--host",
            "-h",
            help="Host to bind to",
        ),
    ] = "127.0.0.1",
    port: Annotated[
        int,
        typer.Option(
            "--port",
            "-p",
            help="Port to bind to",
        ),
    ] = 3141,
    no_browser: Annotated[
        bool,
        typer.Option(
            "--no-browser",
            help="Don't open browser automatically",
        ),
    ] = False,
    plugin: Annotated[
        list[Path],
        typer.Option(
            "--plugin",
            help="Plugin directory path (can be used multiple times)",
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = None,
    with_subdirs: Annotated[
        bool,
        typer.Option(
            "--with-subdirs",
            help="Treat each subdirectory of --plugin as a separate plugin",
        ),
    ] = False,
    install_plugin_deps: Annotated[
        bool,
        typer.Option(
            "--install-plugin-deps",
            help="Install plugin dependencies from pyproject.toml before loading",
        ),
    ] = False,
) -> None:
    """Start the hexdag-studio visual editor.

    Opens a browser-based editor that reads/writes YAML files directly.
    No cloud, no accounts - just local files that work with git.

    Examples:
        hexdag-studio ./pipelines/
        hexdag-studio --port 8080
        hexdag-studio . --no-browser
        hexdag-studio . --plugin ./hexdag_plugins/azure
        hexdag-studio . --plugin ./hexdag_plugins --with-subdirs
        hexdag-studio . --plugin ./hexdag_plugins/azure --install-plugin-deps
    """
    # Resolve paths
    if plugin is None:
        plugin = []
    workspace = path.resolve()
    plugin_paths = [p.resolve() for p in plugin]

    # Print startup banner
    console.print()
    console.print("[bold blue]hexdag-studio[/bold blue] v0.1.0")
    console.print()
    console.print(f"  [dim]Workspace:[/dim]  {workspace}")
    if plugin_paths:
        console.print(f"  [dim]Plugins:[/dim]    {', '.join(str(p) for p in plugin_paths)}")
        if with_subdirs:
            console.print("  [dim]Mode:[/dim]       with-subdirs")
    console.print(
        f"  [dim]Local:[/dim]      [link=http://{host}:{port}]http://{host}:{port}[/link]"
    )
    console.print()
    console.print("  [dim]Press Ctrl+C to stop[/dim]")
    console.print()

    # Open browser
    if not no_browser:

        def open_browser() -> None:
            time.sleep(1)  # Wait for server to start
            webbrowser.open(f"http://{host}:{port}")

        threading.Thread(target=open_browser, daemon=True).start()

    # Start server
    from hexdag_studio.server.main import run_server

    # Install plugin dependencies if requested
    if install_plugin_deps and plugin_paths:
        from hexdag_studio.server.routes.plugins import install_plugin_dependencies

        install_plugin_dependencies(plugin_paths, with_subdirs=with_subdirs)

    run_server(
        workspace, host=host, port=port, plugin_paths=plugin_paths, with_subdirs=with_subdirs
    )


if __name__ == "__main__":
    main()
