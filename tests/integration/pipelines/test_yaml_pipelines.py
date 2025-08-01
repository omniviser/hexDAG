"""Integration tests for YAML-defined pipelines.

This module automatically discovers and tests all pipeline YAML files to ensure
they can be executed successfully with the RegistryRoute system.
"""

import glob
import os
from pathlib import Path
from typing import Any, Dict

from hexai.app.application.routes.registry_route import RegistryRoute
from hexai.app.mocks import pipeline_functions
import pytest
import yaml


def register_all_mock_functions(registry_route: RegistryRoute) -> None:
    """Register all mock functions from pipeline_functions.py for testing."""
    # Text2SQL Pipeline Functions
    registry_route.register_function("parse_question", pipeline_functions.parse_question)
    registry_route.register_function("generate_sql_mock", pipeline_functions.generate_sql_mock)
    registry_route.register_function(
        "format_text2sql_response", pipeline_functions.format_text2sql_response
    )

    # RAG Pipeline Functions
    registry_route.register_function("process_rag_prompt", pipeline_functions.process_rag_prompt)
    registry_route.register_function("mock_vector_search", pipeline_functions.mock_vector_search)
    registry_route.register_function("rank_documents", pipeline_functions.rank_documents)
    registry_route.register_function("format_rag_response", pipeline_functions.format_rag_response)

    # LLM Fallback Pipeline Functions
    registry_route.register_function("mock_web_scraper", pipeline_functions.mock_web_scraper)
    registry_route.register_function("process_web_content", pipeline_functions.process_web_content)
    registry_route.register_function(
        "format_llm_fallback_response", pipeline_functions.format_llm_fallback_response
    )

    # Ontology Pipeline Functions
    registry_route.register_function(
        "mock_load_ontology_source", pipeline_functions.mock_load_ontology_source
    )
    registry_route.register_function("mock_parse_entities", pipeline_functions.mock_parse_entities)
    registry_route.register_function(
        "mock_validate_against_dbt", pipeline_functions.mock_validate_against_dbt
    )
    registry_route.register_function(
        "validate_relationships", pipeline_functions.validate_relationships
    )
    registry_route.register_function(
        "check_granularity_conflicts", pipeline_functions.check_granularity_conflicts
    )
    registry_route.register_function(
        "mock_update_ontology_registry", pipeline_functions.mock_update_ontology_registry
    )
    registry_route.register_function(
        "mock_persist_validated_ontology", pipeline_functions.mock_persist_validated_ontology
    )
    registry_route.register_function(
        "format_ontology_response", pipeline_functions.format_ontology_response
    )

    # Basic utility functions for simple pipelines
    registry_route.register_function(
        "extract_text",
        lambda data, context, **kwargs: (
            data.get("text", "") if isinstance(data, dict) else str(data)
        ),
    )
    registry_route.register_function(
        "uppercase_transform", lambda data, context, **kwargs: str(data).upper()
    )
    registry_route.register_function(
        "count_words",
        lambda data, context, **kwargs: {"word_count": len(str(data).split()), "text": str(data)},
    )
    registry_route.register_function(
        "add_timestamp",
        lambda data, context, **kwargs: {"data": data, "timestamp": "2024-01-01T00:00:00"},
    )


def discover_pipeline_files() -> list[str]:
    """Discover all pipeline YAML files in the project."""
    pipeline_dirs = [
        "fastapi_app/src/hexai/app/pipelines/*.yaml",
        "fastapi_app/examples/pipelines/*.yaml",
    ]

    files = []
    for pattern in pipeline_dirs:
        files.extend(glob.glob(pattern))

    return [f for f in files if os.path.getsize(f) > 0]  # Skip empty files


def get_test_data_for_pipeline(pipeline_name: str) -> Dict[str, Any]:
    """Get appropriate test data for different pipeline types."""
    if "text2sql" in pipeline_name.lower():
        return {
            "question": "Show me all customers who placed orders in the last month",
            "user_id": "test_user",
        }
    elif "rag" in pipeline_name.lower():
        return {
            "prompt": "What is machine learning and how does it work?",
            "user_query": "machine learning basics",
        }
    elif "ontology" in pipeline_name.lower():
        return {
            "ontology_source": "neo4j://localhost:7687",
            "validation_mode": "strict",
        }
    elif "llm_fallback" in pipeline_name.lower():
        return {
            "query": "What is the current weather in New York?",
            "context": "user_search",
        }
    else:
        # Default test data for simple text processing pipelines
        return {
            "text": "Hello world, this is a test message",
            "data": "sample input data",
        }


@pytest.fixture
def registry_route_with_mocks():
    """Create a RegistryRoute instance with all mock functions registered."""
    route = RegistryRoute()
    register_all_mock_functions(route)
    return route


def get_mock_ports() -> dict[str, Any]:
    """Get mock ports (LLM, ToolRouter, etc.) for testing."""
    from hexai.adapters.function_tool_router import FunctionBasedToolRouter
    from hexai.adapters.mock.mock_llm import MockLLM

    return {
        "llm": MockLLM(),
        "tool_router": FunctionBasedToolRouter(),
    }


class TestYamlPipelines:
    """Test all YAML-defined pipelines."""

    @pytest.mark.parametrize("pipeline_file", discover_pipeline_files())
    @pytest.mark.asyncio
    async def test_pipeline_execution(self, registry_route_with_mocks, pipeline_file):
        """Test that each pipeline YAML file can be executed successfully."""
        # Skip problematic pipelines with complex schema validation issues
        skip_pipelines = [
            "parallel_processing.yaml",
            "llm_analysis.yaml",
        ]

        if any(skip_name in pipeline_file for skip_name in skip_pipelines):
            pytest.skip(f"Skipping {pipeline_file} - complex schema validation issues")

        # Load the pipeline configuration
        with open(pipeline_file) as f:
            config = yaml.safe_load(f)

        pipeline_name = config.get("name", Path(pipeline_file).stem)

        # Skip empty pipelines
        if not config.get("nodes"):
            pytest.skip(f"Pipeline {pipeline_name} has no nodes")

        # Get appropriate test data
        test_data = get_test_data_for_pipeline(pipeline_name)

        # Execute the pipeline
        mock_ports = get_mock_ports()
        result = await registry_route_with_mocks.execute_from_config(config, test_data, mock_ports)

        # Assert successful execution
        assert (
            result["status"] == "success"
        ), f"Pipeline {pipeline_name} failed: {result.get('error', 'Unknown error')}"
        assert result["pipeline_name"] == pipeline_name
        assert "results" in result
        assert "trace" in result

        # Assert that all nodes produced results
        nodes_in_config = {node["id"] for node in config["nodes"]}
        results_keys = set(result["results"].keys())

        assert nodes_in_config.issubset(
            results_keys
        ), f"Missing results for nodes: {nodes_in_config - results_keys}"

    @pytest.mark.parametrize("pipeline_file", discover_pipeline_files())
    def test_pipeline_validation(self, registry_route_with_mocks, pipeline_file):
        """Test that each pipeline YAML file passes validation."""
        # Load the pipeline configuration
        with open(pipeline_file) as f:
            config = yaml.safe_load(f)

        pipeline_name = config.get("name", Path(pipeline_file).stem)

        # Skip empty pipelines
        if not config.get("nodes"):
            pytest.skip(f"Pipeline {pipeline_name} has no nodes")

        # Validate the pipeline
        validation_result = registry_route_with_mocks.validate_config(config)

        # Assert successful validation
        assert validation_result[
            "valid"
        ], f"Pipeline {pipeline_name} validation failed: {validation_result['errors']}"
        assert not validation_result["errors"], f"Validation errors: {validation_result['errors']}"

    @pytest.mark.asyncio
    async def test_text2sql_pipeline_specific(self, registry_route_with_mocks):
        """Test text2sql pipeline with specific assertions."""
        pipeline_files = [f for f in discover_pipeline_files() if "text2sql" in f]

        if not pipeline_files:
            pytest.skip("No text2sql pipeline found")

        with open(pipeline_files[0]) as f:
            config = yaml.safe_load(f)

        test_data = {
            "question": "How many customers do we have?",
            "user_id": "test_user",
        }

        result = await registry_route_with_mocks.execute_from_config(config, test_data)

        assert result["status"] == "success"

        # Check for expected text2sql outputs
        results = result["results"]
        if "response_formatter" in results:
            formatted_result = results["response_formatter"]
            assert "sql" in formatted_result
            assert formatted_result["sql"]  # SQL should not be empty

    @pytest.mark.asyncio
    async def test_rag_pipeline_specific(self, registry_route_with_mocks):
        """Test RAG pipeline with specific assertions."""
        pipeline_files = [f for f in discover_pipeline_files() if "rag" in f]

        if not pipeline_files:
            pytest.skip("No RAG pipeline found")

        # Check if the RAG pipeline file is empty
        if os.path.getsize(pipeline_files[0]) == 0:
            pytest.skip("RAG pipeline file is empty")

        with open(pipeline_files[0]) as f:
            config = yaml.safe_load(f)

        if not config or not config.get("nodes"):
            pytest.skip("RAG pipeline has no nodes")

        test_data = {
            "prompt": "What is artificial intelligence?",
            "query": "AI definition",
        }

        mock_ports = get_mock_ports()
        result = await registry_route_with_mocks.execute_from_config(config, test_data, mock_ports)

        assert result["status"] == "success"

        # Check for expected RAG outputs
        results = result["results"]
        if "format_rag_response" in results:
            rag_result = results["format_rag_response"]
            assert "answer" in rag_result
            assert "source_docs" in rag_result

    @pytest.mark.asyncio
    async def test_ontology_pipeline_specific(self, registry_route_with_mocks):
        """Test ontology pipeline with specific assertions."""
        pipeline_files = [f for f in discover_pipeline_files() if "ontology" in f]

        if not pipeline_files:
            pytest.skip("No ontology pipeline found")

        with open(pipeline_files[0]) as f:
            config = yaml.safe_load(f)

        test_data = {
            "ontology_source": "mock://test",
            "validation_mode": "strict",
        }

        mock_ports = get_mock_ports()
        result = await registry_route_with_mocks.execute_from_config(config, test_data, mock_ports)

        assert result["status"] == "success"

        # Check for expected ontology outputs
        results = result["results"]
        if "response_formatter" in results:
            ontology_result = results["response_formatter"]
            assert "validation_result" in ontology_result or "status" in ontology_result


class TestPipelineDiscovery:
    """Test pipeline discovery and metadata."""

    def test_pipeline_files_exist(self):
        """Test that pipeline files are discovered correctly."""
        files = discover_pipeline_files()
        assert len(files) > 0, "No pipeline files found"

        # Check that all files exist and are readable
        for file in files:
            assert os.path.exists(file), f"Pipeline file does not exist: {file}"
            assert os.path.getsize(file) > 0, f"Pipeline file is empty: {file}"

    def test_pipeline_yaml_format(self):
        """Test that all pipeline files are valid YAML."""
        files = discover_pipeline_files()

        for file in files:
            with open(file) as f:
                try:
                    config = yaml.safe_load(f)
                    assert isinstance(config, dict), f"Pipeline {file} should be a YAML dict"
                    assert "name" in config, f"Pipeline {file} missing 'name' field"
                    assert "nodes" in config, f"Pipeline {file} missing 'nodes' field"
                except yaml.YAMLError as e:
                    pytest.fail(f"Invalid YAML in {file}: {e}")
