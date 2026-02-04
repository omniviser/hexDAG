"""Generate typed stubs from YAML pipelines for IDE autocomplete support."""

from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer()
console = Console()


def _python_type_to_str(annotation: Any) -> str:
    """Convert a Python type annotation to a string representation.

    Args
    ----
        annotation: The type annotation to convert

    Returns
    -------
        String representation of the type
    """
    if annotation is None:
        return "Any"

    # Handle None type
    if annotation is type(None):
        return "None"

    # Handle string annotations (forward references)
    if isinstance(annotation, str):
        return annotation

    # Get the type name
    origin = getattr(annotation, "__origin__", None)
    args = getattr(annotation, "__args__", ())

    # Handle Union types (X | Y)
    is_union_type = (
        hasattr(annotation, "__class__") and annotation.__class__.__name__ == "UnionType"
    )
    if origin is type(None) or is_union_type:
        # Python 3.10+ union syntax
        if hasattr(annotation, "__args__"):
            arg_strs = [_python_type_to_str(arg) for arg in annotation.__args__]
            return " | ".join(arg_strs)
        return "Any"

    # Handle typing.Union
    if origin is not None:
        origin_name = getattr(origin, "__name__", str(origin))
        if origin_name == "Union":
            arg_strs = [_python_type_to_str(arg) for arg in args]
            return " | ".join(arg_strs)

        # Handle generic types (list[str], dict[str, int], etc.)
        if args:
            arg_strs = [_python_type_to_str(arg) for arg in args]
            return f"{origin_name}[{', '.join(arg_strs)}]"
        return origin_name

    # Handle basic types
    if hasattr(annotation, "__name__"):
        return annotation.__name__

    # Fallback
    return str(annotation).replace("typing.", "").replace("<class '", "").replace("'>", "")


def _export_model_stub(model_name: str, fields: dict[str, Any], output_path: Path) -> None:
    """Generate a .pyi stub file for a Pydantic model.

    Args
    ----
        model_name: Name of the model class
        fields: Dictionary of field names to their type annotations
        output_path: Path to write the stub file
    """
    from datetime import datetime as dt_type

    # Determine which imports are needed
    needs_datetime = False
    for field_type in fields.values():
        is_datetime = hasattr(field_type, "__name__") and field_type.__name__ == "datetime"
        if field_type is dt_type or is_datetime:
            needs_datetime = True
            break

    lines = [
        "# Auto-generated type stub from hexdag pipeline",
        "# Do not edit manually - regenerate with: hexdag generate-types <yaml>",
        "",
    ]

    if needs_datetime:
        lines.append("from datetime import datetime")
    lines.append("from typing import Any")
    lines.append("from pydantic import BaseModel")
    lines.append("")
    lines.append("")
    lines.append(f"class {model_name}(BaseModel):")

    if not fields:
        lines.append("    pass")
    else:
        for field_name, field_type in fields.items():
            type_str = _python_type_to_str(field_type)
            lines.append(f"    {field_name}: {type_str}")

    lines.append("")  # Trailing newline
    output_path.write_text("\n".join(lines))


def _infer_type_from_mapping(source_path: str) -> Any:
    """Infer a type from a mapping source path.

    Args
    ----
        source_path: The source path string (e.g., "$input.name", "node.field")

    Returns
    -------
        Inferred type (defaults to Any)
    """
    # Check for expression patterns that have known return types
    from hexdag.core.expression_parser import ALLOWED_FUNCTIONS

    for func_name in ALLOWED_FUNCTIONS:
        if f"{func_name}(" in source_path:
            # Infer return type from function
            if func_name in {"len", "int", "abs", "round"}:
                return int
            if func_name in {"float", "sum"}:
                return float
            if func_name in {"str", "upper", "lower", "strip", "join"}:
                return str
            if func_name in {"bool", "all", "any"}:
                return bool
            if func_name in {"list", "sorted", "split"}:
                return list
            if func_name in {"now", "utcnow"}:
                from datetime import datetime

                return datetime
            return Any

    # Default to Any for simple field paths
    return Any


@app.command()
def generate_types(
    yaml_path: Annotated[
        Path,
        typer.Argument(
            help="Path to YAML pipeline file",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            "-o",
            help="Output directory for stub files (default: current directory)",
        ),
    ] = Path(),
    prefix: Annotated[
        str,
        typer.Option(
            "--prefix",
            "-p",
            help="Prefix for generated file names",
        ),
    ] = "",
) -> None:
    """Generate typed stubs from YAML pipeline definitions.

    This command analyzes YAML pipelines and generates .pyi stub files
    for nodes with input_mapping. This enables IDE autocomplete for
    the input_data parameter in node functions.

    Examples
    --------
    hexdag generate-types pipeline.yaml
    hexdag generate-types pipeline.yaml -o ./types
    hexdag generate-types pipeline.yaml --prefix pipeline_
    """
    import yaml

    # Read YAML file
    try:
        with Path.open(yaml_path) as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        console.print(f"[red]Error:[/red] Invalid YAML syntax: {e}")
        raise typer.Exit(1) from e
    except OSError as e:
        console.print(f"[red]Error:[/red] Cannot read file: {e}")
        raise typer.Exit(1) from e

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract nodes with input_mapping
    nodes = config.get("spec", {}).get("nodes", [])
    generated_files: list[tuple[str, Path]] = []

    for node in nodes:
        node_name = node.get("metadata", {}).get("name", "")
        spec = node.get("spec", {})
        input_mapping = spec.get("input_mapping")

        if not input_mapping or not node_name:
            continue

        # Create model name
        model_name = f"{node_name.replace('-', '_').title().replace('_', '')}Input"

        # Infer types from mapping
        fields: dict[str, Any] = {}
        for target_field, source_path in input_mapping.items():
            fields[target_field] = _infer_type_from_mapping(source_path)

        # Generate stub file
        file_name = f"{prefix}{node_name.replace('-', '_')}_types.pyi"
        stub_path = output_dir / file_name

        _export_model_stub(model_name, fields, stub_path)
        generated_files.append((node_name, stub_path))

    # Display results
    if not generated_files:
        console.print("[yellow]No nodes with input_mapping found in pipeline.[/yellow]")
        raise typer.Exit(0)

    console.print(f"\n[green]Generated {len(generated_files)} stub file(s):[/green]\n")

    table = Table(show_header=True, border_style="green")
    table.add_column("Node", style="cyan")
    table.add_column("Stub File", style="white")

    for node_name, stub_path in generated_files:
        table.add_row(node_name, str(stub_path))

    console.print(table)
    console.print()
