"""Tests for hexdag.api.documentation module."""

from hexdag.api.documentation import (
    explain_yaml_structure,
    get_custom_adapter_guide,
    get_custom_node_guide,
    get_custom_tool_guide,
    get_extension_guide,
    get_syntax_reference,
)


class TestGetSyntaxReference:
    """Tests for get_syntax_reference function."""

    def test_returns_string(self):
        """Test that syntax reference returns a string."""
        result = get_syntax_reference()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_input_reference(self):
        """Test that syntax reference documents $input."""
        result = get_syntax_reference()
        assert "$input" in result

    def test_contains_env_var_reference(self):
        """Test that syntax reference documents environment variables."""
        result = get_syntax_reference()
        assert "${" in result or "ENV" in result

    def test_contains_jinja_template(self):
        """Test that syntax reference documents Jinja templates."""
        result = get_syntax_reference()
        assert "{{" in result or "template" in result.lower()


class TestGetCustomAdapterGuide:
    """Tests for get_custom_adapter_guide function."""

    def test_returns_string(self):
        """Test that adapter guide returns a string."""
        result = get_custom_adapter_guide()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_adapter_decorator(self):
        """Test that guide documents @adapter decorator."""
        result = get_custom_adapter_guide()
        assert "@adapter" in result or "adapter" in result.lower()

    def test_contains_secrets_info(self):
        """Test that guide documents secrets handling."""
        result = get_custom_adapter_guide()
        assert "secret" in result.lower()

    def test_contains_code_example(self):
        """Test that guide contains code examples."""
        result = get_custom_adapter_guide()
        assert "class" in result or "def" in result


class TestGetCustomNodeGuide:
    """Tests for get_custom_node_guide function."""

    def test_returns_string(self):
        """Test that node guide returns a string."""
        result = get_custom_node_guide()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_node_decorator(self):
        """Test that guide documents @node decorator."""
        result = get_custom_node_guide()
        assert "@node" in result or "node" in result.lower()

    def test_contains_factory_pattern(self):
        """Test that guide documents factory pattern."""
        result = get_custom_node_guide()
        assert "factory" in result.lower() or "__call__" in result

    def test_contains_yaml_usage(self):
        """Test that guide shows YAML usage."""
        result = get_custom_node_guide()
        assert "yaml" in result.lower() or "kind:" in result


class TestGetCustomToolGuide:
    """Tests for get_custom_tool_guide function."""

    def test_returns_string(self):
        """Test that tool guide returns a string."""
        result = get_custom_tool_guide()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_function_example(self):
        """Test that guide contains function examples."""
        result = get_custom_tool_guide()
        assert "def " in result

    def test_contains_agent_reference(self):
        """Test that guide references agents."""
        result = get_custom_tool_guide()
        assert "agent" in result.lower()


class TestGetExtensionGuide:
    """Tests for get_extension_guide function."""

    def test_returns_overview_by_default(self):
        """Test that no argument returns overview."""
        result = get_extension_guide()
        assert isinstance(result, str)
        # Should mention multiple component types
        assert "adapter" in result.lower()
        assert "node" in result.lower()
        assert "tool" in result.lower()

    def test_adapter_type_returns_adapter_guide(self):
        """Test that 'adapter' returns adapter guide."""
        result = get_extension_guide("adapter")
        assert "@adapter" in result or "adapter" in result.lower()
        # Should be same as get_custom_adapter_guide
        assert result == get_custom_adapter_guide()

    def test_node_type_returns_node_guide(self):
        """Test that 'node' returns node guide."""
        result = get_extension_guide("node")
        assert "@node" in result or "node" in result.lower()
        assert result == get_custom_node_guide()

    def test_tool_type_returns_tool_guide(self):
        """Test that 'tool' returns tool guide."""
        result = get_extension_guide("tool")
        assert "def " in result
        assert result == get_custom_tool_guide()

    def test_unknown_type_returns_overview(self):
        """Test that unknown type returns overview."""
        result = get_extension_guide("unknown")
        # Should return overview
        assert "adapter" in result.lower()
        assert "node" in result.lower()


class TestExplainYamlStructure:
    """Tests for explain_yaml_structure function."""

    def test_returns_string(self):
        """Test that YAML structure explanation returns a string."""
        result = explain_yaml_structure()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_api_version(self):
        """Test that explanation mentions apiVersion."""
        result = explain_yaml_structure()
        assert "apiVersion" in result

    def test_contains_kind(self):
        """Test that explanation mentions kind."""
        result = explain_yaml_structure()
        assert "kind" in result or "Pipeline" in result

    def test_contains_metadata(self):
        """Test that explanation mentions metadata."""
        result = explain_yaml_structure()
        assert "metadata" in result

    def test_contains_spec(self):
        """Test that explanation mentions spec."""
        result = explain_yaml_structure()
        assert "spec" in result

    def test_contains_nodes(self):
        """Test that explanation mentions nodes."""
        result = explain_yaml_structure()
        assert "nodes" in result

    def test_contains_dependencies(self):
        """Test that explanation mentions dependencies."""
        result = explain_yaml_structure()
        assert "depend" in result.lower()

    def test_contains_ports_info(self):
        """Test that explanation mentions ports."""
        result = explain_yaml_structure()
        assert "port" in result.lower()

    def test_contains_yaml_example(self):
        """Test that explanation contains YAML example."""
        result = explain_yaml_structure()
        # Should have code block with YAML
        assert "```" in result or "yaml" in result.lower()
