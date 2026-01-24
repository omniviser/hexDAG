"""hexdag studio - Local-first visual editor for pipelines.

Usage:
    hexdag studio ./pipelines/
    hexdag studio --port 8080
"""

from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(name="studio", help="Local-first visual editor for pipelines")
console = Console()


@app.callback(invoke_without_command=True)
def studio(
    ctx: typer.Context,
    path: Path = typer.Argument(
        Path(),
        help="Directory containing pipeline YAML files",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        "-h",
        help="Host to bind to",
    ),
    port: int = typer.Option(
        3141,
        "--port",
        "-p",
        help="Port to bind to",
    ),
    no_browser: bool = typer.Option(
        False,
        "--no-browser",
        help="Don't open browser automatically",
    ),
) -> None:
    """Start the hexdag studio visual editor.

    Opens a browser-based editor that reads/writes YAML files directly.
    No cloud, no accounts - just local files that work with git.

    Examples:
        hexdag studio ./pipelines/
        hexdag studio --port 8080
        hexdag studio . --no-browser
    """
    # Check for studio dependencies
    try:
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401
    except ImportError:
        console.print(
            "[red]Error:[/red] Studio dependencies not installed.\n"
            "Please install with:\n"
            "  [cyan]pip install hexdag[studio][/cyan]\n"
            "  or\n"
            "  [cyan]uv pip install hexdag[studio][/cyan]"
        )
        raise typer.Exit(code=1)

    # Resolve path
    workspace = path.resolve()

    # Print startup banner
    console.print()
    console.print("[bold blue]hexdag studio[/bold blue] v0.1.0")
    console.print()
    console.print(f"  [dim]Workspace:[/dim]  {workspace}")
    console.print(
        f"  [dim]Local:[/dim]      [link=http://{host}:{port}]http://{host}:{port}[/link]"
    )
    console.print()
    console.print("  [dim]Press Ctrl+C to stop[/dim]")
    console.print()

    # Open browser
    if not no_browser:
        import threading
        import webbrowser

        def open_browser() -> None:
            import time

            time.sleep(1)  # Wait for server to start
            webbrowser.open(f"http://{host}:{port}")

        threading.Thread(target=open_browser, daemon=True).start()

    # Start server
    from hexdag.studio.server.main import run_server

    run_server(workspace, host=host, port=port)
