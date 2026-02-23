"""Export API for hexdag studio.

Generates complete standalone Python projects from pipelines.
"""

import re
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from yaml import safe_load

router = APIRouter(prefix="/export", tags=["export"])


class ExportRequest(BaseModel):
    """Request to export a pipeline as a standalone project."""

    content: str
    project_name: str | None = None
    include_docker: bool = False
    python_version: str = "3.12"


class ExportedFile(BaseModel):
    """A single exported file."""

    path: str
    content: str


class ExportResponse(BaseModel):
    """Complete exported project."""

    success: bool
    project_name: str
    files: list[ExportedFile]
    error: str | None = None


def slugify(name: str) -> str:
    """Convert name to valid Python package name."""
    slug = re.sub(r"[^a-zA-Z0-9_]", "_", name.lower())
    slug = re.sub(r"_+", "_", slug)
    return slug.strip("_") or "pipeline"


def extract_env_vars(content: str) -> list[str]:
    """Extract environment variable references from YAML."""
    # Match ${VAR_NAME} patterns
    pattern = r"\$\{([A-Z_][A-Z0-9_]*)\}"
    return list(set(re.findall(pattern, content)))


def detect_adapters(pipeline: dict[str, Any]) -> dict[str, list[str]]:
    """Detect which adapters are used in the pipeline."""
    adapters: dict[str, list[str]] = {
        "llm": [],
        "memory": [],
        "database": [],
    }

    ports = pipeline.get("spec", {}).get("ports", {})
    for port_type, config in ports.items():
        if isinstance(config, dict):
            adapter_name = config.get("adapter", "")
            if port_type in adapters:
                adapters[port_type].append(adapter_name)

    # Also check nodes for implicit adapter usage
    nodes = pipeline.get("spec", {}).get("nodes", [])
    for node in nodes:
        kind = node.get("kind", "")
        if kind in ("llm_node", "raw_llm_node", "agent_node") and not adapters["llm"]:
            adapters["llm"].append("openai")  # Default
        if kind == "agent_node" and not adapters["memory"]:
            adapters["memory"].append("in_memory")

    return adapters


def generate_pyproject(
    project_name: str,
    pipeline: dict[str, Any],
    python_version: str,
) -> str:
    """Generate pyproject.toml content."""
    adapters = detect_adapters(pipeline)

    # Base dependencies
    deps = [
        '"hexdag>=0.1.0"',
        '"python-dotenv>=1.0.0"',
    ]

    # Add adapter-specific dependencies
    if "openai" in adapters.get("llm", []):
        deps.append('"openai>=1.0.0"')
    if "anthropic" in adapters.get("llm", []):
        deps.append('"anthropic>=0.18.0"')

    deps_str = ",\n    ".join(deps)

    return f'''[project]
name = "{project_name}"
version = "0.1.0"
description = "Auto-generated hexdag pipeline project"
requires-python = ">={python_version}"
dependencies = [
    {deps_str}
]

[project.scripts]
{project_name} = "{project_name}.main:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["{project_name}"]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "UP"]
'''


def generate_readme(
    project_name: str,
    pipeline: dict[str, Any],
    env_vars: list[str],
) -> str:
    """Generate README.md content."""
    pipeline_name = pipeline.get("metadata", {}).get("name", project_name)
    description = pipeline.get("metadata", {}).get("description", "A hexdag pipeline project.")

    nodes = pipeline.get("spec", {}).get("nodes", [])
    node_list = "\n".join([
        f"- **{n.get('metadata', {}).get('name', 'unnamed')}** (`{n.get('kind', 'unknown')}`)"
        for n in nodes
    ])

    env_section = ""
    if env_vars:
        env_list = "\n".join([f"- `{var}`" for var in sorted(env_vars)])
        env_section = f"""
## Environment Variables

The following environment variables are required:

{env_list}

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```
"""

    return f"""# {pipeline_name}

{description}

## Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate

# Install dependencies
pip install -e .
```
{env_section}
## Usage

### Run the pipeline

```bash
# Using the CLI
python -m {project_name}.main

# Or using the installed script
{project_name}
```

### Run with custom input

```python
import asyncio
from {project_name}.main import run_pipeline

result = asyncio.run(run_pipeline({{"input": "your data here"}}))
print(result)
```

## Pipeline Structure

{node_list}

## Development

```bash
# Run tests
pytest

# Format code
ruff format .

# Lint
ruff check .
```

---

Generated with [hexdag studio](https://github.com/hexdag/hexdag)
"""


def generate_env_example(env_vars: list[str]) -> str:
    """Generate .env.example content."""
    if not env_vars:
        return "# No environment variables required\n"

    lines = ["# Environment variables for this pipeline", ""]
    for var in sorted(env_vars):
        if "KEY" in var or "SECRET" in var or "TOKEN" in var:
            lines.append(f"{var}=your-secret-here")
        elif "URL" in var:
            lines.append(f"{var}=https://api.example.com")
        else:
            lines.append(f"{var}=")

    return "\n".join(lines) + "\n"


def generate_main_py(project_name: str, pipeline: dict[str, Any]) -> str:
    """Generate main.py runner script."""
    adapters = detect_adapters(pipeline)

    # Build imports based on adapters
    imports = [
        "import asyncio",
        "import os",
        "from pathlib import Path",
        "",
        "from dotenv import load_dotenv",
        "",
        "from hexdag.compiler import YamlPipelineBuilder",
        "from hexdag.kernel.orchestration.orchestrator import Orchestrator",
    ]

    # Add adapter imports
    adapter_imports = []
    if adapters.get("llm"):
        adapter_imports.append("from hexdag.stdlib.adapters.openai import OpenAIAdapter")
    if adapters.get("memory"):
        adapter_imports.append("from hexdag.stdlib.adapters.memory import InMemoryMemory")

    if adapter_imports:
        imports.append("")
        imports.extend(adapter_imports)

    imports_str = "\n".join(imports)

    # Build port configuration
    ports_config = []
    if adapters.get("llm"):
        ports_config.append('        "llm": OpenAIAdapter(),')
    if adapters.get("memory"):
        ports_config.append('        "memory": InMemoryMemory(),')

    ports_str = "\n".join(ports_config) if ports_config else "        # Add your adapters here"

    return f'''{imports_str}


# Load environment variables
load_dotenv()

# Path to the pipeline YAML
PIPELINE_PATH = Path(__file__).parent / "pipeline.yaml"


async def run_pipeline(inputs: dict | None = None) -> dict:
    """Run the pipeline with the given inputs.

    Args:
        inputs: Input data for the pipeline. Defaults to empty dict.

    Returns:
        Pipeline execution results.
    """
    inputs = inputs or {{}}

    # Build the pipeline from YAML
    builder = YamlPipelineBuilder()
    graph, config = builder.build_from_yaml_file(str(PIPELINE_PATH))

    # Create orchestrator with adapters
    orchestrator = Orchestrator(
        ports={{
{ports_str}
        }}
    )

    # Run the pipeline
    result = await orchestrator.run(graph, inputs)

    return result


def main():
    """CLI entry point."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Run the {project_name} pipeline")
    parser.add_argument(
        "--input", "-i",
        type=str,
        help="JSON input data for the pipeline",
        default="{{}}"
    )
    parser.add_argument(
        "--input-file", "-f",
        type=str,
        help="Path to JSON file with input data"
    )

    args = parser.parse_args()

    # Parse input
    if args.input_file:
        with open(args.input_file) as f:
            inputs = json.load(f)
    else:
        inputs = json.loads(args.input)

    # Run pipeline
    result = asyncio.run(run_pipeline(inputs))

    # Output result
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
'''


def generate_init_py(project_name: str) -> str:
    """Generate __init__.py content."""
    return f'''"""{project_name} - Auto-generated hexdag pipeline."""

__version__ = "0.1.0"
'''


def generate_gitignore() -> str:
    """Generate .gitignore content."""
    return """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
.venv/
venv/
ENV/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Environment
.env
.env.local

# Testing
.pytest_cache/
.coverage
htmlcov/

# OS
.DS_Store
Thumbs.db
"""


def generate_dockerfile(project_name: str, python_version: str) -> str:
    """Generate Dockerfile content."""
    return f'''FROM python:{python_version}-slim

WORKDIR /app

# Install dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

# Copy application
COPY . .

# Run the pipeline
CMD ["python", "-m", "{project_name}.main"]
'''


@router.post("", response_model=ExportResponse)
async def export_pipeline(request: ExportRequest) -> ExportResponse:
    """Export a pipeline as a complete standalone Python project.

    Generates:
    - pyproject.toml with dependencies
    - README.md with usage instructions
    - .env.example with required environment variables
    - main.py runner script
    - pipeline.yaml (the original pipeline)
    - .gitignore
    - Optionally: Dockerfile
    """
    try:
        # Parse the YAML
        pipeline = safe_load(request.content)
        if not pipeline:
            return ExportResponse(
                success=False, project_name="", files=[], error="Invalid YAML content"
            )

        # Determine project name
        yaml_name = pipeline.get("metadata", {}).get("name", "pipeline")
        project_name = slugify(request.project_name or yaml_name)

        # Extract environment variables
        env_vars = extract_env_vars(request.content)

        # Generate files
        files: list[ExportedFile] = []

        # pyproject.toml
        files.append(
            ExportedFile(
                path="pyproject.toml",
                content=generate_pyproject(project_name, pipeline, request.python_version),
            )
        )

        # README.md
        files.append(
            ExportedFile(
                path="README.md", content=generate_readme(project_name, pipeline, env_vars)
            )
        )

        # .env.example
        files.append(ExportedFile(path=".env.example", content=generate_env_example(env_vars)))

        # .gitignore
        files.append(ExportedFile(path=".gitignore", content=generate_gitignore()))

        # Package directory
        files.append(
            ExportedFile(path=f"{project_name}/__init__.py", content=generate_init_py(project_name))
        )

        # main.py
        files.append(
            ExportedFile(
                path=f"{project_name}/main.py", content=generate_main_py(project_name, pipeline)
            )
        )

        # pipeline.yaml
        files.append(ExportedFile(path=f"{project_name}/pipeline.yaml", content=request.content))

        # Optional: Dockerfile
        if request.include_docker:
            files.append(
                ExportedFile(
                    path="Dockerfile",
                    content=generate_dockerfile(project_name, request.python_version),
                )
            )

        return ExportResponse(success=True, project_name=project_name, files=files)

    except Exception as e:
        return ExportResponse(success=False, project_name="", files=[], error=str(e))


@router.post("/download")
async def download_project(request: ExportRequest):
    """Export and download as a ZIP file."""
    import io
    import traceback
    import zipfile

    from fastapi.responses import StreamingResponse

    # Validate content
    if not request.content or not request.content.strip():
        return JSONResponse(status_code=400, content={"error": "No YAML content provided"})

    # Generate the export
    try:
        export_result = await export_pipeline(request)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": f"Export failed: {str(e)}"})

    if not export_result.success:
        return JSONResponse(
            status_code=400, content={"error": export_result.error or "Unknown error"}
        )

    # Create ZIP file in memory
    try:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for file in export_result.files:
                zf.writestr(f"{export_result.project_name}/{file.path}", file.content)

        zip_buffer.seek(0)

        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={
                "Content-Disposition": f"attachment; filename={export_result.project_name}.zip"
            },
        )
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": f"ZIP creation failed: {str(e)}"})
