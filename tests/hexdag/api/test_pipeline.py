"""Tests for hexdag.api.pipeline module."""

from hexdag.api.pipeline import (
    add_node,
    init,
    list_nodes,
    remove_node,
    update_node,
)


class TestInit:
    """Tests for init function."""

    def test_init_creates_valid_yaml(self):
        """Test init creates valid YAML structure."""
        result = init("test-pipeline")
        assert result["success"] is True
        assert "yaml_content" in result
        yaml_content = result["yaml_content"]
        assert "apiVersion" in yaml_content
        assert "kind: Pipeline" in yaml_content
        assert "test-pipeline" in yaml_content

    def test_init_with_description(self):
        """Test init with description."""
        result = init("my-pipeline", description="A test pipeline")
        assert result["success"] is True
        assert "A test pipeline" in result["yaml_content"]

    def test_init_message(self):
        """Test init returns success message."""
        result = init("test")
        assert "message" in result
        assert "test" in result["message"]


class TestAddNode:
    """Tests for add_node function."""

    def test_add_node_success(self):
        """Test adding a node successfully."""
        yaml_content = init("test")["yaml_content"]
        result = add_node(
            yaml_content,
            kind="llm_node",
            name="analyzer",
            spec={"prompt_template": "Analyze: {{input}}"},
            dependencies=[],
        )
        assert result["success"] is True
        assert result["node_count"] == 1
        assert "analyzer" in result["yaml_content"]

    def test_add_node_with_dependencies(self):
        """Test adding node with dependencies."""
        yaml_content = init("test")["yaml_content"]
        # Add first node
        result = add_node(
            yaml_content,
            kind="data_node",
            name="source",
            spec={"output": {"value": 1}},
            dependencies=[],
        )
        yaml_content = result["yaml_content"]

        # Add dependent node
        result = add_node(
            yaml_content,
            kind="function_node",
            name="processor",
            spec={"fn": "json.dumps"},
            dependencies=["source"],
        )
        assert result["success"] is True
        assert result["node_count"] == 2
        assert "processor" in result["yaml_content"]
        assert "source" in result["yaml_content"]

    def test_add_node_duplicate_name_fails(self):
        """Test adding node with duplicate name fails."""
        yaml_content = init("test")["yaml_content"]
        result = add_node(
            yaml_content,
            kind="data_node",
            name="node1",
            spec={"output": {"value": 1}},
            dependencies=[],
        )
        yaml_content = result["yaml_content"]

        # Try to add with same name
        result = add_node(
            yaml_content,
            kind="data_node",
            name="node1",
            spec={"output": {"value": 2}},
            dependencies=[],
        )
        assert result["success"] is False
        assert "already exists" in result["error"]

    def test_add_node_warns_missing_dependency(self):
        """Test adding node warns about missing dependencies."""
        yaml_content = init("test")["yaml_content"]
        result = add_node(
            yaml_content,
            kind="function_node",
            name="processor",
            spec={"fn": "test"},
            dependencies=["nonexistent"],
        )
        # Should succeed but with warning
        assert result["success"] is True
        assert result["warnings"] is not None
        assert len(result["warnings"]) > 0
        assert "nonexistent" in result["warnings"][0]

    def test_add_node_invalid_yaml_fails(self):
        """Test adding node to invalid YAML fails."""
        result = add_node(
            "invalid: yaml: :",
            kind="data_node",
            name="test",
            spec={},
            dependencies=[],
        )
        assert result["success"] is False
        assert "error" in result


class TestRemoveNode:
    """Tests for remove_node function."""

    def test_remove_node_success(self):
        """Test removing a node successfully."""
        yaml_content = init("test")["yaml_content"]
        result = add_node(
            yaml_content,
            kind="data_node",
            name="to_remove",
            spec={"output": {"value": 1}},
            dependencies=[],
        )
        yaml_content = result["yaml_content"]

        result = remove_node(yaml_content, "to_remove")
        assert result["success"] is True
        assert result["removed"] is True
        assert result["node_count"] == 0
        assert "to_remove" not in result["yaml_content"]

    def test_remove_node_not_found(self):
        """Test removing nonexistent node fails."""
        yaml_content = init("test")["yaml_content"]
        result = remove_node(yaml_content, "nonexistent")
        assert result["success"] is False
        assert "not found" in result["error"]

    def test_remove_node_warns_dependents(self):
        """Test removing node warns about dependents."""
        yaml_content = init("test")["yaml_content"]
        # Add two nodes where second depends on first
        result = add_node(
            yaml_content,
            kind="data_node",
            name="source",
            spec={"output": {"value": 1}},
            dependencies=[],
        )
        yaml_content = result["yaml_content"]

        result = add_node(
            yaml_content,
            kind="function_node",
            name="consumer",
            spec={"fn": "test"},
            dependencies=["source"],
        )
        yaml_content = result["yaml_content"]

        # Remove source node
        result = remove_node(yaml_content, "source")
        assert result["success"] is True
        assert result["warnings"] is not None
        assert any("consumer" in w for w in result["warnings"])


class TestUpdateNode:
    """Tests for update_node function."""

    def test_update_node_spec(self):
        """Test updating node spec."""
        yaml_content = init("test")["yaml_content"]
        result = add_node(
            yaml_content,
            kind="llm_node",
            name="analyzer",
            spec={"prompt_template": "Old prompt"},
            dependencies=[],
        )
        yaml_content = result["yaml_content"]

        result = update_node(
            yaml_content,
            "analyzer",
            spec={"prompt_template": "New prompt: {{input}}"},
        )
        assert result["success"] is True
        assert "New prompt" in result["yaml_content"]

    def test_update_node_dependencies(self):
        """Test updating node dependencies."""
        yaml_content = init("test")["yaml_content"]
        result = add_node(
            yaml_content,
            kind="data_node",
            name="source",
            spec={"output": {"value": 1}},
            dependencies=[],
        )
        yaml_content = result["yaml_content"]

        result = add_node(
            yaml_content,
            kind="function_node",
            name="processor",
            spec={"fn": "test"},
            dependencies=[],
        )
        yaml_content = result["yaml_content"]

        result = update_node(yaml_content, "processor", dependencies=["source"])
        assert result["success"] is True
        # Check that dependency is in YAML
        assert "source" in result["yaml_content"]

    def test_update_node_kind_warns(self):
        """Test updating node kind issues warning."""
        yaml_content = init("test")["yaml_content"]
        result = add_node(
            yaml_content,
            kind="data_node",
            name="node1",
            spec={"output": {"value": 1}},
            dependencies=[],
        )
        yaml_content = result["yaml_content"]

        result = update_node(yaml_content, "node1", kind="function_node")
        assert result["success"] is True
        assert result["warnings"] is not None
        assert any("Changed node type" in w for w in result["warnings"])

    def test_update_node_not_found(self):
        """Test updating nonexistent node fails."""
        yaml_content = init("test")["yaml_content"]
        result = update_node(yaml_content, "nonexistent", spec={"key": "value"})
        assert result["success"] is False
        assert "not found" in result["error"]


class TestListNodes:
    """Tests for list_nodes function (pipeline module version)."""

    def test_list_nodes_empty_pipeline(self):
        """Test listing nodes in empty pipeline."""
        yaml_content = init("test")["yaml_content"]
        result = list_nodes(yaml_content)
        assert result["success"] is True
        assert result["node_count"] == 0
        assert result["nodes"] == []

    def test_list_nodes_with_nodes(self):
        """Test listing nodes in pipeline with nodes."""
        yaml_content = init("test")["yaml_content"]
        result = add_node(
            yaml_content,
            kind="data_node",
            name="node_a",
            spec={"output": {"value": 1}},
            dependencies=[],
        )
        yaml_content = result["yaml_content"]

        result = add_node(
            yaml_content,
            kind="data_node",
            name="node_b",
            spec={"output": {"value": 2}},
            dependencies=["node_a"],
        )
        yaml_content = result["yaml_content"]

        result = list_nodes(yaml_content)
        assert result["success"] is True
        assert result["node_count"] == 2
        assert len(result["nodes"]) == 2

    def test_list_nodes_returns_node_info(self):
        """Test that list_nodes returns detailed node info."""
        yaml_content = init("test")["yaml_content"]
        result = add_node(
            yaml_content,
            kind="llm_node",
            name="analyzer",
            spec={"prompt_template": "Test"},
            dependencies=[],
        )
        yaml_content = result["yaml_content"]

        result = list_nodes(yaml_content)
        node_info = result["nodes"][0]
        assert "name" in node_info
        assert "kind" in node_info
        assert "dependencies" in node_info
        assert "dependents" in node_info
        assert node_info["name"] == "analyzer"
        assert node_info["kind"] == "llm_node"

    def test_list_nodes_returns_execution_order(self):
        """Test that list_nodes returns execution order."""
        yaml_content = init("test")["yaml_content"]
        result = add_node(
            yaml_content,
            kind="data_node",
            name="a",
            spec={"output": {"value": 1}},
            dependencies=[],
        )
        yaml_content = result["yaml_content"]

        result = add_node(
            yaml_content,
            kind="data_node",
            name="b",
            spec={"output": {"value": 2}},
            dependencies=["a"],
        )
        yaml_content = result["yaml_content"]

        result = list_nodes(yaml_content)
        assert "execution_order" in result
        # a should come before b
        order = result["execution_order"]
        assert order.index("a") < order.index("b")

    def test_list_nodes_returns_dependents(self):
        """Test that list_nodes returns reverse dependencies."""
        yaml_content = init("test")["yaml_content"]
        result = add_node(
            yaml_content,
            kind="data_node",
            name="source",
            spec={"output": {"value": 1}},
            dependencies=[],
        )
        yaml_content = result["yaml_content"]

        result = add_node(
            yaml_content,
            kind="function_node",
            name="consumer",
            spec={"fn": "test"},
            dependencies=["source"],
        )
        yaml_content = result["yaml_content"]

        result = list_nodes(yaml_content)
        source_node = next(n for n in result["nodes"] if n["name"] == "source")
        assert "consumer" in source_node["dependents"]

    def test_list_nodes_returns_pipeline_name(self):
        """Test that list_nodes returns pipeline name."""
        yaml_content = init("my-pipeline")["yaml_content"]
        result = list_nodes(yaml_content)
        assert result["pipeline_name"] == "my-pipeline"
