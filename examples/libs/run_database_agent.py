#!/usr/bin/env python
"""hexDAG System Libraries Demo: Database Agent Tools.

Demonstrates the DatabaseTools lib — agent-callable SQL query tools
backed by the MockDatabaseAdapter (no external database needed).

Shows:
- Auto-discovered tools from the lib
- OpenAI-compatible tool schemas (auto-generated from signatures)
- Running SQL queries, listing tables, and describing schemas

Run:
    uv run python examples/libs/run_database_agent.py
"""

import asyncio
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from hexdag.kernel.lib_base import get_lib_tool_schemas
from hexdag.stdlib.adapters.mock import MockDatabaseAdapter
from hexdag.stdlib.lib import DatabaseTools

console = Console()


# ── 1. Setup ─────────────────────────────────────────────────────────


def setup_database_tools() -> DatabaseTools:
    """Create DatabaseTools with a mock adapter containing sample data."""
    adapter = MockDatabaseAdapter(enable_sample_data=True)
    return DatabaseTools(store=adapter)


# ── 2. Tool discovery ────────────────────────────────────────────────


def show_tools(db_tools: DatabaseTools) -> None:
    """Display auto-discovered tools and their schemas."""
    console.print(
        Panel(
            "DatabaseTools exposes 3 agent-callable tools.\n"
            "Each public async method starting with 'a' auto-becomes a tool.",
            title="Step 1: Tool Discovery",
        )
    )

    # Tool names
    tools = db_tools.get_tools()
    table = Table(title="DatabaseTools.get_tools()")
    table.add_column("Tool Name", style="green")
    table.add_column("Type", style="dim")
    for name, fn in tools.items():
        table.add_row(name, "async" if asyncio.iscoroutinefunction(fn) else "sync")
    console.print(table)
    console.print()

    # Tool schemas (OpenAI-compatible)
    schemas = get_lib_tool_schemas(db_tools)
    schema_json = json.dumps(schemas, indent=2)
    console.print(
        Panel(
            Syntax(schema_json, "json", theme="monokai"),
            title="OpenAI-Compatible Tool Schemas (via get_lib_tool_schemas())",
            subtitle="These schemas are sent to LLMs for function calling",
        )
    )
    console.print()


# ── 3. Execute tools ─────────────────────────────────────────────────


async def execute_tools(db_tools: DatabaseTools) -> None:
    """Call each tool directly and display results."""
    console.print(
        Panel(
            "Call each tool as an agent would during pipeline execution.",
            title="Step 2: Execute Tools",
        )
    )

    # Get table schemas (via the adapter directly, since alist_tables uses
    # information_schema which the mock doesn't simulate)
    adapter = db_tools._store
    schemas = await adapter.aget_table_schemas()
    table = Table(title="Available Tables (from MockDatabaseAdapter)")
    table.add_column("Table Name", style="green")
    table.add_column("Columns", style="dim")
    for table_name, schema_info in schemas.items():
        cols = ", ".join(schema_info.get("columns", {}).keys())
        table.add_row(table_name, cols)
    console.print(table)
    console.print()

    # Query customers
    console.print("  [cyan]adatabase_query('SELECT * FROM customers')[/cyan]")
    results = await db_tools.adatabase_query("SELECT * FROM customers")
    table = Table(title="Query Results: customers")
    if results:
        for col in results[0]:
            table.add_column(col, style="green")
        for row in results:
            table.add_row(*[str(v) for v in row.values()])
    console.print(table)
    console.print()

    # Query orders
    console.print("  [cyan]adatabase_query('SELECT * FROM orders')[/cyan]")
    results = await db_tools.adatabase_query("SELECT * FROM orders")
    table = Table(title="Query Results: orders")
    if results:
        for col in results[0]:
            table.add_column(col, style="green")
        for row in results:
            table.add_row(*[str(v) for v in row.values()])
    console.print(table)
    console.print()


# ── Main ─────────────────────────────────────────────────────────────


async def main() -> None:
    console.print(
        Panel.fit(
            "[bold cyan]hexDAG System Libraries Demo[/bold cyan]\n"
            "DatabaseTools — Agent-callable SQL query tools",
            border_style="bold",
        )
    )
    console.print()

    db_tools = setup_database_tools()
    show_tools(db_tools)
    await execute_tools(db_tools)

    console.print(
        Panel.fit(
            "[green]DatabaseTools demonstrated successfully.[/green]\n"
            "The MockDatabaseAdapter provides sample data for customers, orders,\n"
            "products, and order_items — no external database needed.",
            border_style="green",
        )
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Demo interrupted.[/yellow]")
