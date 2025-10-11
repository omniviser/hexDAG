"""Pipeline management commands for HexDAG CLI."""

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer()
console = Console()


@app.command("validate")
def validate_pipeline(
    pipeline_path: Annotated[
        Path,
        typer.Argument(help="Path to pipeline YAML file"),
    ],
) -> None:
    """Validate pipeline file (schema + DAG validation)."""
    import yaml

    if not pipeline_path.exists():
        console.print(f"[red]Error: Pipeline file not found: {pipeline_path}[/red]")
        raise typer.Exit(1)

    try:
        # Load YAML
        with open(pipeline_path) as f:
            pipeline_data = yaml.safe_load(f)

        console.print(f"[cyan]Validating pipeline: {pipeline_path}[/cyan]")

        issues = []
        warnings = []

        # Basic structure validation
        if not isinstance(pipeline_data, dict):
            issues.append("Pipeline must be a dictionary")
            console.print("[red]✗ Invalid pipeline structure[/red]")
            raise typer.Exit(1)

        # Check required fields
        required_fields = ["name", "nodes"]
        issues.extend(
            f"'{field}' is required" for field in required_fields if field not in pipeline_data
        )

        # Validate nodes section
        if "nodes" in pipeline_data and not issues:
            nodes = pipeline_data["nodes"]
            if not isinstance(nodes, list):
                issues.append("'nodes' must be a list")
            else:
                console.print(f"[dim]Found {len(nodes)} node(s)[/dim]")

                # Validate each node
                node_ids = set()
                for i, node in enumerate(nodes):
                    if not isinstance(node, dict):
                        issues.append(f"Node {i} must be a dictionary")
                        continue

                    # Check node ID
                    if "id" not in node:
                        issues.append(f"Node {i} missing 'id' field")
                    else:
                        node_id = node["id"]
                        if node_id in node_ids:
                            issues.append(f"Duplicate node ID: '{node_id}'")
                        node_ids.add(node_id)

                        console.print(
                            f"  [green]✓[/green] {node_id} ({node.get('type', 'unknown')})"
                        )

                    # Check node type
                    if "type" not in node:
                        warnings.append(f"Node {node.get('id', i)} missing 'type' field")

                    # Validate depends_on references
                    if "depends_on" in node:
                        depends_on = node["depends_on"]
                        if not isinstance(depends_on, list):
                            issues.append(f"Node {node.get('id', i)}: 'depends_on' must be a list")
                        # Defer dependency validation until all node IDs are collected

                # Validate dependency references (after all node IDs collected)
                for i, node in enumerate(nodes):
                    if isinstance(node, dict) and "depends_on" in node:
                        depends_on = node["depends_on"]
                        if isinstance(depends_on, list):
                            issues.extend(
                                f"Node '{node.get('id', i)}' dependency '{dep}' not found"
                                for dep in depends_on
                                if dep not in node_ids
                            )

                # DAG validation - check for cycles
                if not issues:
                    console.print("\n[cyan]Checking DAG for cycles...[/cyan]")
                    cycles = _detect_cycles(nodes)
                    if cycles:
                        issues.append(f"DAG contains cycle(s): {cycles}")
                        console.print(f"[red]✗ Cycle detected: {' -> '.join(cycles)}[/red]")
                    else:
                        console.print("[green]✓ No cycles detected[/green]")

        # Report results
        if issues:
            console.print("\n[red]Validation errors:[/red]")
            for issue in issues:
                console.print(f"  [red]✗[/red] {issue}")
            raise typer.Exit(1)

        if warnings:
            console.print("\n[yellow]Warnings:[/yellow]")
            for warning in warnings:
                console.print(f"  [yellow]⚠[/yellow] {warning}")

        console.print("\n[green]✓ Pipeline validation passed[/green]")
        console.print(f"  Name: {pipeline_data.get('name', 'unnamed')}")
        console.print(f"  Nodes: {len(pipeline_data.get('nodes', []))}")

    except yaml.YAMLError as e:
        console.print(f"[red]YAML parsing error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Validation error: {e}[/red]")
        raise typer.Exit(1)


@app.command("graph")
def generate_graph(
    pipeline_path: Annotated[
        Path,
        typer.Argument(help="Path to pipeline YAML file"),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--out",
            "-o",
            help="Output file path (supports .svg, .png, .dot)",
        ),
    ] = None,
) -> None:
    """Generate visual graph of pipeline DAG."""
    import yaml

    if not pipeline_path.exists():
        console.print(f"[red]Error: Pipeline file not found: {pipeline_path}[/red]")
        raise typer.Exit(1)

    try:
        # Check if graphviz is available
        try:
            import graphviz
        except ImportError:
            console.print("[red]Error: graphviz not installed[/red]")
            console.print("Install with: uv pip install graphviz")
            raise typer.Exit(1)

        # Load pipeline
        with open(pipeline_path) as f:
            pipeline_data = yaml.safe_load(f)

        console.print(f"[cyan]Generating graph for: {pipeline_path}[/cyan]")

        # Create graph
        pipeline_name = pipeline_data.get("name", "pipeline")
        dot = graphviz.Digraph(
            name=pipeline_name,
            comment=f"Pipeline: {pipeline_name}",
            format="svg" if not output else output.suffix[1:],
        )

        # Configure graph appearance
        dot.attr(rankdir="TB")  # Top to bottom
        dot.attr("node", shape="box", style="rounded,filled", fillcolor="lightblue")

        nodes = pipeline_data.get("nodes", [])

        # Add nodes
        for node in nodes:
            node_id = node.get("id", "unknown")
            node_type = node.get("type", "unknown")
            label = f"{node_id}\\n({node_type})"

            # Color by type
            color = "lightblue"
            if node_type in ["agent", "agent_node"]:
                color = "lightgreen"
            elif node_type in ["llm", "llm_node"]:
                color = "lightyellow"
            elif node_type in ["function", "function_node"]:
                color = "lightgray"
            elif node_type in ["conditional", "conditional_node"]:
                color = "orange"

            dot.node(node_id, label=label, fillcolor=color)

        # Add edges
        for node in nodes:
            node_id = node.get("id")
            depends_on = node.get("depends_on", [])
            for dep in depends_on:
                dot.edge(dep, node_id)

        # Determine output path
        output_path = output or pipeline_path.with_suffix(".svg")

        # Render graph
        output_base = str(output_path.with_suffix(""))
        dot.render(output_base, cleanup=True)

        console.print(f"[green]✓ Graph generated: {output_path}[/green]")
        console.print(f"  Nodes: {len(nodes)}")
        console.print(f"  Edges: {sum(len(n.get('depends_on', [])) for n in nodes)}")

    except Exception as e:
        console.print(f"[red]Error generating graph: {e}[/red]")
        raise typer.Exit(1)


@app.command("plan")
def plan_execution(
    pipeline_path: Annotated[
        Path,
        typer.Argument(help="Path to pipeline YAML file"),
    ],
) -> None:
    """Show execution plan (waves, concurrency, expected I/O)."""
    import yaml

    if not pipeline_path.exists():
        console.print(f"[red]Error: Pipeline file not found: {pipeline_path}[/red]")
        raise typer.Exit(1)

    try:
        # Load pipeline
        with open(pipeline_path) as f:
            pipeline_data = yaml.safe_load(f)

        console.print(f"[cyan]Execution plan for: {pipeline_data.get('name', 'unnamed')}[/cyan]\n")

        nodes = pipeline_data.get("nodes", [])

        # Calculate execution waves (topological sort)
        waves = _calculate_waves(nodes)

        # Display waves
        console.print("[bold]Execution Waves:[/bold]")
        for wave_num, wave_nodes in enumerate(waves, 1):
            console.print(f"\n[yellow]Wave {wave_num}:[/yellow] (parallel execution)")
            for node_id in wave_nodes:
                # Find node details
                node: dict = next((n for n in nodes if n.get("id") == node_id), {})
                node_type = node.get("type", "unknown")
                console.print(f"  • {node_id} [{node_type}]")

        # Summary
        console.print("\n[bold]Summary:[/bold]")
        console.print(f"  Total nodes: {len(nodes)}")
        console.print(f"  Execution waves: {len(waves)}")
        console.print(f"  Max concurrency: {max(len(wave) for wave in waves) if waves else 0}")
        console.print(f"  Estimated steps: {len(waves)}")

        # Calculate I/O expectations
        io_nodes = [n for n in nodes if n.get("type") in ["agent", "agent_node", "llm", "llm_node"]]
        console.print(f"  Expected LLM calls: {len(io_nodes)}")

    except Exception as e:
        console.print(f"[red]Error generating plan: {e}[/red]")
        raise typer.Exit(1)


def _detect_cycles(nodes: list) -> list[str] | None:
    """Detect cycles in DAG using DFS."""
    # Build adjacency list
    graph = {}
    for node in nodes:
        node_id = node.get("id")
        graph[node_id] = node.get("depends_on", [])

    # DFS to detect cycles
    visited = set()
    rec_stack = set()
    path = []

    def dfs(node_id: str) -> list[str] | None:
        visited.add(node_id)
        rec_stack.add(node_id)
        path.append(node_id)

        for neighbor in graph.get(node_id, []):
            if neighbor not in visited:
                result = dfs(neighbor)
                if result is not None:
                    return result
            elif neighbor in rec_stack:
                # Found cycle
                cycle_start = path.index(neighbor)
                return path[cycle_start:] + [neighbor]

        path.pop()
        rec_stack.remove(node_id)
        return None

    for node_id in graph:
        if node_id not in visited:
            result = dfs(node_id)
            if result is not None:
                return result

    return None


def _calculate_waves(nodes: list) -> list[list[str]]:
    """Calculate execution waves using topological sort."""
    # Build dependency graph
    graph = {}
    in_degree = {}

    for node in nodes:
        node_id = node.get("id")
        graph[node_id] = node.get("depends_on", [])
        in_degree[node_id] = 0

    # Calculate in-degrees
    for node_id, deps in graph.items():
        for dep in deps:
            if dep in in_degree:
                in_degree[node_id] += 1

    waves = []
    remaining = set(graph.keys())

    while remaining:
        # Find nodes with no dependencies
        wave = [node_id for node_id in remaining if in_degree[node_id] == 0]

        if not wave:
            # Shouldn't happen if DAG is valid
            break

        waves.append(wave)

        # Remove processed nodes and update in-degrees
        for node_id in wave:
            remaining.remove(node_id)
            # Update dependents
            for other_id in remaining:
                if node_id in graph[other_id]:
                    in_degree[other_id] -= 1

    return waves


@app.command("run")
def run_pipeline(
    pipeline_path: Annotated[
        Path,
        typer.Argument(help="Path to pipeline YAML file"),
    ],
    input_data: Annotated[
        str | None,
        typer.Option(
            "--input",
            "-i",
            help='Input data as JSON string (e.g., \'{"key": "value"}\')',
        ),
    ] = None,
    input_file: Annotated[
        Path | None,
        typer.Option(
            "--input-file",
            "-f",
            help="Input data from JSON file",
        ),
    ] = None,
    output_file: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Save output to JSON file",
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Show detailed execution information",
        ),
    ] = False,
) -> None:
    """Execute a pipeline with optional input data."""
    import asyncio

    if not pipeline_path.exists():
        console.print(f"[red]Error: Pipeline file not found: {pipeline_path}[/red]")
        raise typer.Exit(1)

    try:
        # Parse input data
        inputs = {}
        if input_data:
            try:
                inputs = json.loads(input_data)
            except json.JSONDecodeError as e:
                console.print(f"[red]Error: Invalid JSON in --input: {e}[/red]")
                raise typer.Exit(1)
        elif input_file:
            if not input_file.exists():
                console.print(f"[red]Error: Input file not found: {input_file}[/red]")
                raise typer.Exit(1)
            try:
                with open(input_file) as f:
                    inputs = json.load(f)
            except json.JSONDecodeError as e:
                console.print(f"[red]Error: Invalid JSON in input file: {e}[/red]")
                raise typer.Exit(1)

        # Import hexdag components
        from hexdag import Orchestrator, YamlPipelineBuilder

        # Build pipeline
        if verbose:
            console.print(f"[cyan]Loading pipeline: {pipeline_path}[/cyan]")

        builder = YamlPipelineBuilder()
        graph, pipeline_config = builder.build_from_yaml_file(str(pipeline_path))

        if verbose:
            console.print(f"[dim]Pipeline: {pipeline_config.metadata.get('name', 'unnamed')}[/dim]")
            console.print(f"[dim]Nodes: {len(graph.nodes)}[/dim]\n")

        # Execute pipeline
        console.print("[cyan]Executing pipeline...[/cyan]")

        orchestrator = Orchestrator()
        result = asyncio.run(orchestrator.run(graph, inputs))

        # Display results
        if verbose:
            console.print("\n[green]✓ Pipeline execution completed[/green]\n")

            # Show results in a table
            table = Table(title="Pipeline Results", show_header=True, header_style="bold magenta")
            table.add_column("Node", style="cyan")
            table.add_column("Status", style="green")
            table.add_column("Output", style="dim")

            for node_id, node_result in result.items():
                status = "✓" if node_result else "✗"
                output_preview = (
                    str(node_result)[:50] + "..."
                    if len(str(node_result)) > 50
                    else str(node_result)
                )
                table.add_row(node_id, status, output_preview)

            console.print(table)
        else:
            console.print("[green]✓ Pipeline execution completed[/green]")

        # Save output if requested
        if output_file:
            # Convert result to JSON-serializable format
            output_data = {}
            for k, v in result.items():
                try:
                    # Try to serialize directly
                    json.dumps({k: v})
                    output_data[k] = v
                except (TypeError, ValueError):
                    # Fall back to string representation
                    output_data[k] = str(v)

            with open(output_file, "w") as f:
                json.dump(output_data, f, indent=2)
            console.print(f"[dim]Output saved to: {output_file}[/dim]")

        # Print final result
        if not verbose and not output_file:
            console.print("\n[bold]Results:[/bold]")
            for node_id, node_result in result.items():
                console.print(f"  {node_id}: {node_result}")

    except ImportError as e:
        console.print(f"[red]Error: Missing dependency - {e}[/red]")
        console.print("Install with: uv pip install hexdag[all]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error executing pipeline: {e}[/red]")
        if verbose:
            import traceback

            console.print(f"[dim]{traceback.format_exc()}[/dim]")
        raise typer.Exit(1)
