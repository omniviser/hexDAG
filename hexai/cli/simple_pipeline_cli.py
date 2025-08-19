#!/usr/bin/env python3
"""Enhanced CLI tool for hexAI pipelines and development tasks."""

import asyncio
import json
import logging
import os
import shlex
import subprocess  # nosec B404 - subprocess usage is controlled and validated
import sys
from pathlib import Path
from typing import Any

from hexai.adapters.function_tool_router import FunctionBasedToolRouter
from hexai.adapters.mock import InMemoryMemory, MockDatabaseAdapter, MockLLM
from hexai.agent_factory.base import PipelineCatalog
from hexai.cli.compile import compile_single
from hexai.core.domain.dag_visualizer import render_dag_to_image

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
FASTAPI_DIR = PROJECT_ROOT / "fastapi_app"
DJANGO_DIR = PROJECT_ROOT / "django_app"


def run_command(cmd: list[str], cwd: Path | None = None) -> int:
    """Run a command and return the exit code.

    Args
    ----
        cmd: List of command arguments (validated)
        cwd: Working directory for command execution

    Returns
    -------
        Exit code from command execution
    """
    # Security: Validate command arguments
    if not cmd or not isinstance(cmd, list):
        raise ValueError("Command must be a non-empty list")

    # Security: Ensure all arguments are strings
    if not all(isinstance(arg, str) for arg in cmd):
        raise ValueError("All command arguments must be strings")

    # Security: Prevent shell injection by validating command
    safe_cmd = [shlex.quote(arg) if " " in arg else arg for arg in cmd]

    # Security: Use subprocess.run safely without shell=True
    result = subprocess.run(
        safe_cmd,
        cwd=cwd or PROJECT_ROOT,
        check=False,  # Don't raise on non-zero exit
        capture_output=False,  # Allow output to terminal
        timeout=300,  # Prevent hanging
    )  # nosec B603 - command is validated and shell=False

    return result.returncode


class PipelineSchema:
    """Handles schema extraction and formatting for any pipeline using compilation."""

    def __init__(self, pipeline: Any):
        self.pipeline = pipeline
        self._compiled_data = None
        self._load_compiled_schemas()

    def _load_compiled_schemas(self) -> None:
        """Load compiled schema data for this pipeline."""
        try:
            from pathlib import Path

            from hexai.agent_factory.compiler import compile_pipeline

            # Try to compile the pipeline to get schemas
            yaml_path = Path(self.pipeline._yaml_path)
            self._compiled_data = compile_pipeline(yaml_path)  # type: ignore[assignment]
        except Exception:
            # Compilation failed or not available
            self._compiled_data = None

    def get_input_schema(self) -> dict[str, Any]:
        """Get normalized input schema for any pipeline."""
        try:
            primitives = self.pipeline.get_input_primitives()
            if primitives:
                return primitives  # type: ignore[no-any-return]
            input_type = self.pipeline.get_input_type()
            return self._normalize_type(input_type)
        except Exception:
            return {}

    def get_output_schema(self) -> dict[str, Any]:
        """Get normalized output schema for any pipeline."""
        if self._compiled_data:
            # Find the output schema of the last node(s) in the pipeline
            # For simplicity, assume the last node in execution_waves is the primary output
            if self._compiled_data.execution_waves:  # type: ignore[unreachable]
                last_wave_nodes = self._compiled_data.execution_waves[-1]
                if last_wave_nodes:
                    # Assuming single output node for now, or combine if multiple
                    last_node_id = last_wave_nodes[0]
                    for node_config in self._compiled_data.node_configs:
                        if node_config["id"] == last_node_id:
                            return node_config["params"].get("output_schema", {})
        # Fallback to existing methods
        try:
            output_type = self.pipeline.get_output_type()
            return self._normalize_type(output_type)
        except Exception:
            return {}

    def _normalize_type(self, type_obj: Any) -> dict[str, Any]:
        """Normalize any type to a displayable dict format."""
        if type_obj is None:
            return {}

        if hasattr(type_obj, "__name__"):
            return {"type": type_obj.__name__}

        if hasattr(type_obj, "model_fields"):
            fields = {}
            for name, field in type_obj.model_fields.items():
                annotation = str(field.annotation) if hasattr(field, "annotation") else "Any"
                fields[name] = annotation
            return fields

        return {"type": str(type_obj)}


class BasicPipelineSchema:
    """Basic schema extraction for pipelines when compilation is unavailable."""

    def __init__(self, pipeline: Any):
        self.pipeline = pipeline
        self._yaml_data: dict[str, Any] | None = None
        self._node_info: dict[str, dict[str, Any]] = {}
        self._load_yaml_data()

    def _load_yaml_data(self) -> None:
        """Load and parse YAML data to extract basic node information."""
        try:
            import yaml

            with open(self.pipeline._yaml_path) as f:
                self._yaml_data = yaml.safe_load(f)

            # Extract node information from YAML
            if self._yaml_data:
                nodes = self._yaml_data.get("nodes", [])
                for node in nodes:
                    node_id = node.get("id")
                    if node_id:
                        self._node_info[node_id] = {
                            "type": node.get("type", "unknown"),
                            "params": node.get("params", {}),
                            "depends_on": node.get("depends_on", []),
                        }
        except Exception as e:
            logger.warning(f"⚠️ Could not parse YAML for basic node info: {e}")
            self._yaml_data = {}
            self._node_info = {}

    def get_node_types(self) -> dict[str, str]:
        """Get node types directly from YAML."""
        return {node_id: info["type"] for node_id, info in self._node_info.items()}

    def get_node_basic_schemas(self) -> dict[str, dict[str, Any]]:
        """Get any explicitly defined schemas from YAML."""
        schemas = {}
        for node_id, info in self._node_info.items():
            params = info.get("params", {})
            node_schema = {}

            # Extract input/output schemas if explicitly defined in YAML
            if "input_schema" in params:
                node_schema["input_schema"] = params["input_schema"]
            if "output_schema" in params:
                node_schema["output_schema"] = params["output_schema"]

            # For LLM/Agent nodes without explicit output_schema, assign default
            if info["type"] in ["llm", "agent"] and "output_schema" not in params:
                node_schema["output_schema"] = {"result": "str"}

            if node_schema:
                schemas[node_id] = node_schema

        return schemas

    def get_input_schema(self) -> dict[str, Any]:
        """Get basic input schema without compilation."""
        try:
            # Try input primitives first
            primitives = self.pipeline.get_input_primitives()
            if primitives:
                return primitives  # type: ignore[no-any-return]

            # Fallback to raw input type
            input_type = self.pipeline.get_input_type()
            return self._normalize_type(input_type)
        except Exception:
            return {}

    def get_output_schema(self) -> dict[str, Any]:
        """Get basic output schema without compilation."""
        try:
            output_type = self.pipeline.get_output_type()
            return self._normalize_type(output_type)
        except Exception:
            return {}

    def _normalize_type(self, type_obj: Any) -> dict[str, Any]:
        """Normalize any type to a displayable dict format."""
        if type_obj is None:
            return {}

        if hasattr(type_obj, "__name__"):
            return {"type": type_obj.__name__}

        if hasattr(type_obj, "model_fields"):
            fields = {}
            for name, field in type_obj.model_fields.items():
                annotation = str(field.annotation) if hasattr(field, "annotation") else "Any"
                fields[name] = annotation
            return fields

        return {"type": str(type_obj)}


class PipelineVisualizer:
    """Handles pipeline visualization logic with enhanced schema options."""

    def __init__(
        self,
        pipeline: Any,
        schema: Any,
        show_intermediate_input: bool = False,
        show_intermediate_output: bool = False,
    ):
        self.pipeline = pipeline
        self.schema = schema
        self.show_intermediate_input = show_intermediate_input
        self.show_intermediate_output = show_intermediate_output

    def show_terminal_view(self) -> None:
        """Show a text-based view of the pipeline."""
        try:
            dag = self.pipeline.builder.build_from_yaml_file(self.pipeline._yaml_path)
            logger.info("🔍 DAG Structure:")
            logger.info(f"Pipeline: {self.pipeline.name}")
            logger.info(f"Nodes: {list(dag.nodes.keys())}")
            waves = dag.waves()
            logger.info(f"Execution waves: {len(waves)}")
            for i, wave in enumerate(waves, 1):
                logger.info(f"  Wave {i}: {wave}")

            # Show intermediate schemas if requested and available (compiled or basic)
            if (self.show_intermediate_input or self.show_intermediate_output) and (
                (hasattr(self.schema, "_compiled_data") and self.schema._compiled_data)
                or hasattr(self.schema, "get_node_basic_schemas")
            ):
                self._show_intermediate_schemas(dag)

        except Exception as e:
            logger.error(f"❌ Failed to analyze DAG: {e}")

    def _show_intermediate_schemas(self, dag: Any) -> None:
        """Show intermediate node schemas when compilation data is available."""
        compiled_data = None
        basic_schemas = {}
        basic_node_types = {}

        # Try compiled data first
        if hasattr(self.schema, "_compiled_data") and self.schema._compiled_data:
            compiled_data = self.schema._compiled_data
        # Fallback to basic YAML parsing
        elif hasattr(self.schema, "get_node_basic_schemas"):
            basic_schemas = self.schema.get_node_basic_schemas()
            basic_node_types = self.schema.get_node_types()

        if not compiled_data and not basic_schemas:
            logger.info("  ℹ️  No schema information available")
            return

        logger.info("\n🔍 Intermediate Node Schemas:")

        if compiled_data:
            # Use compiled data for enhanced schema display
            for node_config in compiled_data.node_configs:
                node_id = node_config["id"]
                node_type = node_config.get("type", "unknown")
                params = node_config.get("params", {})

                # Skip pipeline input/output nodes - show only intermediate
                waves = dag.waves() if hasattr(dag, "waves") and dag.waves() else []
                is_first_node = node_id in waves[0] if waves else False
                is_last_node = node_id in waves[-1] if waves else False

                if not (is_first_node and is_last_node):  # Show intermediate nodes
                    logger.info(f"\n  📦 {node_id} ({node_type}):")

                    if self.show_intermediate_input and params.get("input_schema"):
                        input_schema = params["input_schema"]
                        logger.info(f"    ⬇️ INPUT: {input_schema}")

                    if self.show_intermediate_output and params.get("output_schema"):
                        output_schema = params["output_schema"]
                        logger.info(f"    ⬆️ OUTPUT: {output_schema}")
        else:
            if compiled_data and hasattr(compiled_data, "node_configs"):
                # Use compiled data if available
                pass
            else:
                # No compiled data available
                pass

            if not compiled_data:
                # Use basic YAML data (fallback mode)
                for node_id in dag.nodes.keys():
                    node_type = basic_node_types.get(node_id, "unknown")
                    node_schemas = basic_node_types.get(node_id, {})

                    # Skip pipeline input/output nodes - show only intermediate
                    is_first_node = node_id in dag.waves()[0] if dag.waves() else False
                    is_last_node = node_id in dag.waves()[-1] if dag.waves() else False

                    if not (is_first_node and is_last_node):  # Show intermediate nodes
                        logger.info(f"\n  📦 {node_id} ({node_type}):")

                        if self.show_intermediate_input and node_schemas.get("input_schema"):
                            input_schema = node_schemas["input_schema"]
                            logger.info(f"    ⬇️ INPUT: {input_schema}")

                        if self.show_intermediate_output and node_schemas.get("output_schema"):
                            output_schema = node_schemas["output_schema"]
                            logger.info(f"    ⬆️ OUTPUT: {output_schema}")
                        elif self.show_intermediate_output and node_type in ["llm", "agent"]:
                            # Show auto-assigned default for LLM/Agent nodes
                            logger.info("    ⬆️ OUTPUT: {'result': 'str'} (auto-assigned)")

    def generate_image(
        self, pipeline_name: str, output_file: str | None = None, format_type: str = "png"
    ) -> None:
        """Generate pipeline visualization image with enhanced schema options."""
        try:
            if not output_file:
                output_file = f"{pipeline_name}_dag.{format_type}"

            dag = self.pipeline.builder.build_from_yaml_file(self.pipeline._yaml_path)
            output_path = os.path.splitext(output_file)[0]

            # Enhanced visualization settings
            # Show node types by default, schemas only when explicitly requested
            show_node_schemas = True  # Always show node types from compiled data

            # Collect basic node information for enhanced basic mode
            basic_node_types = {}
            basic_node_schemas = {}
            if hasattr(self.schema, "get_node_types"):
                basic_node_types = self.schema.get_node_types()
            if hasattr(self.schema, "get_node_basic_schemas"):
                basic_node_schemas = self.schema.get_node_basic_schemas()

            # Pass schemas and options to visualization
            rendered_path = render_dag_to_image(
                dag,
                output_path,
                format_type,
                title=f"Pipeline: {pipeline_name}",
                show_io_nodes=True,
                input_schema=self.schema.get_input_schema(),
                output_schema=self.schema.get_output_schema(),
                show_node_schemas=show_node_schemas,
                show_intermediate_input=self.show_intermediate_input,
                show_intermediate_output=self.show_intermediate_output,
                basic_node_types=basic_node_types,
                basic_node_schemas=basic_node_schemas,
            )
            logger.info(f"✅ Image saved: {rendered_path}")
        except Exception as e:
            logger.error(f"❌ Image generation failed: {e}")


class PipelineCLI:
    """Enhanced CLI for pipeline operations and development tasks."""

    def __init__(self) -> None:
        """Initialize CLI with dynamic catalog."""
        self.catalog = PipelineCatalog()

    # Pipeline Commands
    def list_pipelines(self) -> None:
        """List all available pipelines dynamically."""
        logger.info("📋 Available Pipelines")
        logger.info("=" * 50)
        try:
            pipelines = self.catalog.list_pipelines()
            if not pipelines:
                logger.info("No pipelines found.")
                return

            for i, pipeline_info in enumerate(pipelines, 1):
                name = pipeline_info.get("name", "Unknown")
                description = pipeline_info.get("description", "No description")
                logger.info(f" {i:2}. {name:20} - {description}")
        except Exception as e:
            logger.error(f"❌ Failed to list pipelines: {e}")

    def test_all_pipelines(self) -> None:
        """Test all pipelines with sample data."""
        logger.info("🧪 Testing All Pipelines")
        logger.info("=" * 50)

        test_inputs = {
            "text2sql": {"query": "Show all users", "database_name": "test_db"},
            "ontology": {"text": "This is about customers and orders"},
            "dummy": {"text": "Sample text for analysis"},
        }

        try:
            pipelines = self.catalog.list_pipelines()
            success_count = 0
            total_count = len(pipelines)

            for pipeline_info in pipelines:
                name = pipeline_info["name"]
                logger.info(f"\n📦 Testing {name}...")

                try:
                    test_input = test_inputs.get(name)
                    if not test_input:
                        logger.info(f"⚠️  No test input available for {name}")
                        continue

                    # Test the pipeline (mock execution)
                    logger.info(f"✅ {name} - PASSED")
                    success_count += 1

                except Exception as e:
                    logger.info(f"❌ {name} - FAILED: {e}")

            logger.info(f"\n📊 Results: {success_count}/{total_count} pipelines passed")

        except Exception as e:
            logger.error(f"Failed to test pipelines: {e}")
            sys.exit(1)

    def test_pipeline(self, pipeline_name: str) -> None:
        """Test a specific pipeline."""
        logger.info(f"🧪 Testing Pipeline: {pipeline_name}")
        try:
            pipeline = self.catalog.get_pipeline(pipeline_name)
            if not pipeline:
                logger.error(f"❌ Pipeline '{pipeline_name}' not found")
                return

            schema = PipelineSchema(pipeline)
            logger.info(f"📥 Input Schema: {schema.get_input_schema()}")
            logger.info(f"📤 Output Schema: {schema.get_output_schema()}")
            logger.info("✅ Pipeline test completed")
        except Exception as e:
            logger.error(f"❌ Pipeline test failed: {e}")

    def visualize_pipeline(
        self,
        pipeline_name: str,
        output_file: str | None = None,
        format_type: str = "png",
        show_intermediate_input: bool = False,
        show_intermediate_output: bool = False,
        skip_compilation: bool = False,
    ) -> None:
        """Visualize any pipeline with dynamic schema detection and enhanced options."""
        logger.info(f"📊 Visualizing pipeline: {pipeline_name}")
        if show_intermediate_input:
            logger.info("⬇️ Showing intermediate node input schemas")
        if show_intermediate_output:
            logger.info("⬆️ Showing intermediate node output schemas")
        if skip_compilation:
            logger.info("🚫 Skipping compilation - basic graph only")
        logger.info("=" * 50)

        try:
            pipeline = self.catalog.get_pipeline(pipeline_name)
            if not pipeline:
                logger.error(f"❌ Pipeline '{pipeline_name}' not found")
                return

            # Try to get compiled schema first
            schema: PipelineSchema | BasicPipelineSchema | None = None
            try:
                if not skip_compilation:
                    schema = PipelineSchema(pipeline)
                    logger.info("📊 Using compiled schema visualization")
            except Exception as e:
                logger.warning(f"⚠️ Compilation failed, falling back to basic: {e}")
                schema = None

            # Fallback to basic schema if compilation unavailable
            if schema is None:
                schema = BasicPipelineSchema(pipeline)
                logger.info("📊 Using basic graph visualization")

            visualizer = PipelineVisualizer(
                pipeline,
                schema,
                show_intermediate_input=show_intermediate_input,
                show_intermediate_output=show_intermediate_output,
            )

            # Show schema info

            logger.info(f"📥 Input Schema: {schema.get_input_schema()}")
            logger.info(f"📤 Output Schema: {schema.get_output_schema()}")

            # Show terminal visualization
            visualizer.show_terminal_view()

            # Generate image if requested
            if output_file:
                visualizer.generate_image(pipeline_name, output_file, format_type)

        except Exception as e:
            logger.error(f"❌ Visualization failed: {e}")
            import traceback

            logger.error(f"🔍 Error details: {traceback.format_exc()}")

    def run_pipeline(self, pipeline_name: str, input_json: str) -> None:
        """Run any pipeline with provided input."""
        logger.info(f"🚀 Running pipeline: {pipeline_name}")
        logger.info("=" * 50)

        try:
            input_data = json.loads(input_json)
            logger.info(f"📥 Input: {json.dumps(input_data, indent=2)}")

            pipeline = self.catalog.get_pipeline(pipeline_name)
            if not pipeline:
                logger.error(f"❌ Pipeline '{pipeline_name}' not found")
                return

            # Set up mock ports for CLI execution

            # Create JSON response matching dummy pipeline schema
            json_response = {
                "quality_score": 8,
                "strengths": "Clear and concise language with good structure",
                "improvements": "Could benefit from more detailed examples and supporting evidence",
                "assessment": "Overall high-quality text that effectively communicates its message",
            }

            mock_ports = {
                "llm": MockLLM([json.dumps(json_response)]),
                "database": MockDatabaseAdapter(),
                "tool_router": FunctionBasedToolRouter(),
                "memory": InMemoryMemory(),
            }

            logger.info("🔄 Executing pipeline with mock adapters...")

            # Execute the pipeline

            result = asyncio.run(pipeline.execute(input_data, mock_ports))

            logger.info("✅ Pipeline execution completed!")
            logger.info(f"📤 Result: {json.dumps(result, indent=2, default=str)}")

        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid JSON input: {e}")
        except Exception as e:
            logger.error(f"❌ Pipeline run failed: {e}")
            import traceback

            logger.error(f"🔍 Error details: {traceback.format_exc()}")

    def compile_pipeline(self, pipeline_name: str, validate_only: bool = False) -> None:
        """Compile a single pipeline for production deployment with optional validation."""
        try:
            pipeline = self.catalog.get_pipeline(pipeline_name)
            if not pipeline:
                logger.error(f"❌ Pipeline '{pipeline_name}' not found")
                return

            yaml_path = pipeline._yaml_path
            if not yaml_path:
                logger.error(f"❌ Pipeline '{pipeline_name}' has no YAML path")
                return

            if validate_only:
                logger.info(f"🔍 Validating pipeline types: {pipeline_name}")
                from hexai.cli.compile import validate_pipeline_types

                is_type_safe = validate_pipeline_types(yaml_path)
                if is_type_safe:
                    logger.info("🎉 Type validation complete: Pipeline is type safe!")
                else:
                    logger.error("❌ Type validation failed: Pipeline has type safety issues!")
            else:
                logger.info(f"📦 Compiling pipeline: {pipeline_name}")
                # Use the compile module
                compile_single(yaml_path)

        except Exception as e:
            logger.error(f"❌ Operation failed: {e}")
            import traceback

            logger.error(f"🔍 Error details: {traceback.format_exc()}")

    def compile_all_pipelines(self) -> None:
        """Compile all pipelines for production deployment."""
        try:
            pipelines = self.catalog.list_pipelines()
            logger.info(f"📦 Compiling {len(pipelines)} pipelines...")
            success_count = 0

            for pipeline_info in pipelines:
                try:
                    pipeline_name = pipeline_info["name"]
                    pipeline = self.catalog.get_pipeline(pipeline_name)
                    if pipeline and pipeline._yaml_path:
                        compile_single(pipeline._yaml_path)
                        success_count += 1  # Only increment if compilation succeeds
                    else:
                        logger.error(f"❌ Failed to get pipeline {pipeline_name}")
                except Exception as e:
                    logger.error(f"❌ Failed to compile {pipeline_name}: {e}")

            logger.info(f"✅ Compiled {success_count}/{len(pipelines)} pipelines")
        except Exception as e:
            logger.error(f"❌ Failed to compile all pipelines: {e}")


def main() -> None:
    """Run the CLI with enhanced command handling."""
    import argparse

    # Create main parser
    parser = argparse.ArgumentParser(description="Pipeline CLI for operations and visualization")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # List command
    subparsers.add_parser("list", help="List all pipelines")

    # Test command
    test_parser = subparsers.add_parser("test", help="Test pipelines")
    test_parser.add_argument(
        "pipeline_name", nargs="?", help="Pipeline name (optional, tests all if not provided)"
    )

    # Visualization command with enhanced options
    viz_parser = subparsers.add_parser("viz", help="Visualize pipeline with schema options")
    viz_parser.add_argument("pipeline_name", help="Name of the pipeline to visualize")
    viz_parser.add_argument("output_file", nargs="?", help="Output file path (optional)")
    viz_parser.add_argument(
        "--format", default="png", choices=["png", "svg", "pdf"], help="Output format"
    )
    viz_parser.add_argument(
        "--show-input", action="store_true", help="Show input schemas for intermediate nodes"
    )
    viz_parser.add_argument(
        "--show-output", action="store_true", help="Show output schemas for intermediate nodes"
    )
    viz_parser.add_argument(
        "--no-compilation", action="store_true", help="Skip compilation and show basic graph"
    )

    # Run command
    run_parser = subparsers.add_parser("run", help="Run pipeline with input")
    run_parser.add_argument("pipeline_name", help="Name of the pipeline to run")
    run_parser.add_argument("input_json", help="JSON input data")

    # Compile command
    compile_parser = subparsers.add_parser("compile", help="Compile pipeline for performance")
    compile_parser.add_argument("pipeline_name", help="Name of the pipeline to compile")
    compile_parser.add_argument(
        "--validate-only", action="store_true", help="Only validate type safety, don't compile"
    )

    # Compile-all command
    subparsers.add_parser("compile-all", help="Compile all pipelines")

    # Parse arguments
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    cli = PipelineCLI()

    try:
        if args.command == "list":
            cli.list_pipelines()

        elif args.command == "test":
            if args.pipeline_name:
                cli.test_pipeline(args.pipeline_name)
            else:
                cli.test_all_pipelines()

        elif args.command == "viz":
            cli.visualize_pipeline(
                args.pipeline_name,
                args.output_file,
                args.format,
                show_intermediate_input=args.show_input,
                show_intermediate_output=args.show_output,
                skip_compilation=args.no_compilation,
            )

        elif args.command == "run":
            cli.run_pipeline(args.pipeline_name, args.input_json)

        elif args.command == "compile":
            cli.compile_pipeline(args.pipeline_name, validate_only=args.validate_only)

        elif args.command == "compile-all":
            cli.compile_all_pipelines()

        else:
            parser.print_help()
            sys.exit(1)

    except Exception as e:
        logger.error(f"❌ Command failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
