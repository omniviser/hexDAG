"""Pipeline creation commands for HexDAG CLI."""

import contextlib
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml
from rich.console import Console
from rich.prompt import Confirm, Prompt

from hexai.core.bootstrap import bootstrap_registry
from hexai.core.registry import registry

app = typer.Typer()
console = Console()


def _ensure_bootstrapped() -> None:
    """Ensure registry is bootstrapped."""
    with contextlib.suppress(Exception):
        bootstrap_registry()


def _generate_minimal_example(schema: dict[str, Any]) -> dict[str, Any]:
    """Generate minimal example with only required fields."""
    example = {}
    properties = schema.get("properties", {})
    required = schema.get("required", [])

    for prop_name in required:
        if prop_name not in properties:
            continue

        prop_schema = properties[prop_name]

        # Use default if available
        if "default" in prop_schema:
            example[prop_name] = prop_schema["default"]
        elif prop_schema.get("type") == "string":
            if "enum" in prop_schema:
                example[prop_name] = prop_schema["enum"][0]
            else:
                example[prop_name] = f"<{prop_name}>"
        elif prop_schema.get("type") == "integer":
            example[prop_name] = 1
        elif prop_schema.get("type") == "number":
            example[prop_name] = 1.0
        elif prop_schema.get("type") == "boolean":
            example[prop_name] = True
        elif prop_schema.get("type") == "array":
            example[prop_name] = []
        elif prop_schema.get("type") == "object":
            example[prop_name] = {}

    return example


@app.command("pipeline")
def create_pipeline(
    name: Annotated[
        str | None,
        typer.Option(
            "--name",
            "-n",
            help="Pipeline name",
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output file path (default: <name>.yaml)",
        ),
    ] = None,
    template: Annotated[
        str,
        typer.Option(
            "--template",
            "-t",
            help="Template type (minimal, example, full)",
        ),
    ] = "minimal",
) -> None:
    """Create a new pipeline YAML file from a template.

    Examples
    --------
    hexdag create pipeline
    hexdag create pipeline --name my-pipeline
    hexdag create pipeline --template example --output pipeline.yaml
    """
    _ensure_bootstrapped()

    # Interactive mode if name not provided
    if name is None:
        name = Prompt.ask("[cyan]Pipeline name[/cyan]", default="my-pipeline")

    # Determine output path
    if output is None:
        output = Path(f"{name}.yaml")

    # Check if file exists
    if output.exists():
        if not Confirm.ask(f"[yellow]File {output} already exists. Overwrite?[/yellow]"):
            console.print("[red]Aborted.[/red]")
            raise typer.Exit(1)

    # Create pipeline template
    if template == "minimal":
        pipeline = {
            "apiVersion": "v1",
            "kind": "Pipeline",
            "metadata": {
                "name": name,
                "description": "TODO: Add description",
            },
            "spec": {
                "nodes": [
                    {
                        "kind": "function_node",
                        "metadata": {"name": "step1"},
                        "spec": {
                            "fn": "my_function",
                            "input_mapping": {"x": "input.value"},
                            "dependencies": [],
                        },
                    }
                ]
            },
        }
    elif template == "example":
        pipeline = {
            "apiVersion": "v1",
            "kind": "Pipeline",
            "metadata": {
                "name": name,
                "description": "Example multi-step pipeline",
                "version": "1.0",
                "tags": ["example", "demo"],
            },
            "spec": {
                "nodes": [
                    {
                        "kind": "function_node",
                        "metadata": {
                            "name": "input_processor",
                            "annotations": {"description": "Process input data"},
                        },
                        "spec": {
                            "fn": "process_input",
                            "input_mapping": {"data": "input.raw_data"},
                            "dependencies": [],
                        },
                    },
                    {
                        "kind": "llm_node",
                        "metadata": {
                            "name": "analyzer",
                            "annotations": {"description": "Analyze processed data"},
                        },
                        "spec": {
                            "template": "Analyze this data: {{input_processor}}",
                            "dependencies": ["input_processor"],
                        },
                    },
                    {
                        "kind": "function_node",
                        "metadata": {
                            "name": "output_formatter",
                            "annotations": {"description": "Format final output"},
                        },
                        "spec": {
                            "fn": "format_output",
                            "input_mapping": {"result": "analyzer"},
                            "dependencies": ["analyzer"],
                        },
                    },
                ]
            },
        }
    else:  # full
        pipeline = {
            "apiVersion": "v1",
            "kind": "Pipeline",
            "metadata": {
                "name": name,
                "description": "Full-featured pipeline template",
                "version": "1.0",
                "author": "Your Name",
                "tags": ["production", "template"],
            },
            "spec": {
                "input_schema": {
                    "type": "object",
                    "properties": {"data": {"type": "string"}},
                    "required": ["data"],
                },
                "common_field_mappings": {"default_mapping": {"value": "input.data"}},
                "nodes": [
                    {
                        "kind": "function_node",
                        "metadata": {
                            "name": "validator",
                            "annotations": {
                                "description": "Validate input",
                                "timeout": "5s",
                            },
                        },
                        "spec": {
                            "fn": "validate_input",
                            "input_mapping": {"data": "input.data"},
                            "output_schema": {
                                "type": "object",
                                "properties": {"valid": {"type": "boolean"}},
                            },
                            "dependencies": [],
                        },
                    },
                    {
                        "kind": "llm_node",
                        "metadata": {"name": "processor"},
                        "spec": {
                            "template": "Process: {{validator.result}}",
                            "output_schema": {
                                "type": "object",
                                "properties": {"processed": {"type": "string"}},
                            },
                            "dependencies": ["validator"],
                        },
                    },
                ],
            },
        }

    # Write to file
    with open(output, "w") as f:
        yaml.dump(pipeline, f, default_flow_style=False, sort_keys=False)

    console.print(f"[green]✓[/green] Created pipeline: {output}")
    console.print("\n[cyan]Next steps:[/cyan]")
    console.print(f"  1. Edit {output} to customize your pipeline")
    console.print(f"  2. Validate: [bold]hexdag validate {output}[/bold]")
    console.print("  3. Use [bold]hexdag schema list[/bold] to see available node types")


@app.command("from-schema")
def create_from_schema(
    node_type: Annotated[str, typer.Argument(help="Node type to create template from")],
    namespace: Annotated[
        str,
        typer.Option(
            "--namespace",
            "-n",
            help="Component namespace",
        ),
    ] = "core",
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output file path",
        ),
    ] = None,
    name: Annotated[
        str | None,
        typer.Option(
            "--name",
            help="Pipeline name",
        ),
    ] = None,
) -> None:
    """Create a pipeline template from a specific node schema.

    Examples
    --------
    hexdag create from-schema llm_node
    hexdag create from-schema agent_node --output agent-pipeline.yaml
    hexdag create from-schema custom_node --namespace myplugin
    """
    _ensure_bootstrapped()

    # Get schema
    try:
        schema = registry.get_schema(node_type, namespace=namespace)
    except (KeyError, ValueError) as e:
        console.print(f"[red]Error:[/red] {e}")
        console.print("[yellow]Hint:[/yellow] Use 'hexdag schema list' to see available node types")
        raise typer.Exit(1) from e

    # Interactive prompts
    if name is None:
        name = Prompt.ask(
            "[cyan]Pipeline name[/cyan]",
            default=f"{node_type.replace('_node', '')}-pipeline",
        )

    if output is None:
        output = Path(f"{name}.yaml")

    # Check if file exists
    if output.exists():
        if not Confirm.ask(f"[yellow]File {output} already exists. Overwrite?[/yellow]"):
            console.print("[red]Aborted.[/red]")
            raise typer.Exit(1)

    # Ensure schema is dict (get_schema can return str)
    if isinstance(schema, str):
        console.print(f"[red]Error:[/red] Schema is a string, not a dict: {schema}")
        raise typer.Exit(1)

    # Generate example spec from schema
    example_spec = _generate_minimal_example(schema)

    # Create pipeline
    node_name = f"{node_type.replace('_node', '')}_1"
    pipeline = {
        "apiVersion": "v1",
        "kind": "Pipeline",
        "metadata": {
            "name": name,
            "description": f"Pipeline using {namespace}:{node_type}",
        },
        "spec": {
            "nodes": [
                {
                    "kind": f"{namespace}:{node_type}" if namespace != "core" else node_type,
                    "metadata": {"name": node_name},
                    "spec": example_spec,
                }
            ]
        },
    }

    # Write to file
    with open(output, "w") as f:
        yaml.dump(pipeline, f, default_flow_style=False, sort_keys=False)

    console.print(f"[green]✓[/green] Created pipeline from {namespace}:{node_type} schema")
    console.print(f"[green]✓[/green] Output: {output}")
    console.print("\n[cyan]Next steps:[/cyan]")
    console.print(f"  1. Edit {output} to fill in placeholder values")
    console.print(
        f"  2. Use [bold]hexdag schema explain {node_type}[/bold] for field documentation"
    )
    console.print(f"  3. Validate: [bold]hexdag validate {output}[/bold]")
