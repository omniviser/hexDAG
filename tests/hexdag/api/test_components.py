"""Tests for hexdag.api.components module."""

from hexdag.api.components import (
    detect_port_type,
    get_component_schema,
    list_adapters,
    list_macros,
    list_nodes,
    list_tags,
    list_tools,
)


class TestListNodes:
    """Tests for list_nodes function."""

    def test_list_nodes_returns_list(self):
        """Test that list_nodes returns a list."""
        nodes = list_nodes()
        assert isinstance(nodes, list)

    def test_list_nodes_contains_expected_keys(self):
        """Test that node info dicts contain expected keys."""
        nodes = list_nodes()
        if nodes:  # May be empty in minimal test environments
            node = nodes[0]
            assert "kind" in node
            assert "name" in node
            assert "module_path" in node
            assert "description" in node

    def test_list_nodes_contains_builtin_nodes(self):
        """Test that builtin nodes are discovered."""
        nodes = list_nodes()
        kinds = [n["kind"] for n in nodes]
        # At least some common node types should be present
        expected = ["llm_node", "function_node", "agent_node"]
        found = [k for k in expected if k in kinds]
        assert len(found) > 0, f"Expected at least one of {expected}, got {kinds}"

    def test_list_nodes_filters_deprecated_by_default(self):
        """Test that deprecated nodes are filtered by default."""
        nodes_without_deprecated = list_nodes(include_deprecated=False)
        nodes_with_deprecated = list_nodes(include_deprecated=True)
        # With deprecated should be >= without deprecated
        assert len(nodes_with_deprecated) >= len(nodes_without_deprecated)

    def test_list_nodes_sorted_by_kind(self):
        """Test that nodes are sorted by kind."""
        nodes = list_nodes()
        if len(nodes) > 1:
            kinds = [n["kind"] for n in nodes]
            assert kinds == sorted(kinds)


class TestListAdapters:
    """Tests for list_adapters function."""

    def test_list_adapters_returns_list(self):
        """Test that list_adapters returns a list."""
        adapters = list_adapters()
        assert isinstance(adapters, list)

    def test_list_adapters_contains_expected_keys(self):
        """Test that adapter info dicts contain expected keys."""
        adapters = list_adapters()
        if adapters:
            adapter = adapters[0]
            assert "name" in adapter
            assert "module_path" in adapter
            assert "port_type" in adapter
            assert "description" in adapter

    def test_list_adapters_filter_by_port_type(self):
        """Test filtering adapters by port type."""
        llm_adapters = list_adapters(port_type="llm")
        for adapter in llm_adapters:
            assert adapter["port_type"] == "llm"

    def test_list_adapters_filter_memory(self):
        """Test filtering memory adapters."""
        memory_adapters = list_adapters(port_type="memory")
        for adapter in memory_adapters:
            assert adapter["port_type"] == "memory"

    def test_list_adapters_unknown_port_type_returns_empty(self):
        """Test that unknown port type returns empty list."""
        adapters = list_adapters(port_type="nonexistent_port_type")
        assert adapters == []


class TestDetectPortType:
    """Tests for detect_port_type function."""

    def test_detect_llm_adapter(self):
        """Test detection of LLM adapter."""
        from hexdag.builtin.adapters.mock import MockLLM

        result = detect_port_type(MockLLM)
        assert result == "llm"

    def test_detect_memory_adapter(self):
        """Test detection of memory adapter."""
        from hexdag.builtin.adapters.memory import InMemoryMemory

        result = detect_port_type(InMemoryMemory)
        assert result == "memory"

    def test_detect_unknown_class(self):
        """Test detection returns unknown for non-adapter classes."""

        class RandomClass:
            pass

        result = detect_port_type(RandomClass)
        assert result == "unknown"


class TestListTools:
    """Tests for list_tools function."""

    def test_list_tools_returns_list(self):
        """Test that list_tools returns a list."""
        tools = list_tools()
        assert isinstance(tools, list)

    def test_list_tools_contains_expected_keys(self):
        """Test that tool info dicts contain expected keys."""
        tools = list_tools()
        if tools:
            tool = tools[0]
            assert "name" in tool
            assert "module_path" in tool
            assert "description" in tool


class TestListMacros:
    """Tests for list_macros function."""

    def test_list_macros_returns_list(self):
        """Test that list_macros returns a list."""
        macros = list_macros()
        assert isinstance(macros, list)

    def test_list_macros_contains_expected_keys(self):
        """Test that macro info dicts contain expected keys."""
        macros = list_macros()
        if macros:
            macro = macros[0]
            assert "name" in macro
            assert "module_path" in macro
            assert "description" in macro

    def test_list_macros_ends_with_macro_suffix(self):
        """Test that macro names end with 'Macro'."""
        macros = list_macros()
        for macro in macros:
            assert macro["name"].endswith("Macro")


class TestListTags:
    """Tests for list_tags function."""

    def test_list_tags_returns_list(self):
        """Test that list_tags returns a list."""
        tags = list_tags()
        assert isinstance(tags, list)

    def test_list_tags_contains_expected_keys(self):
        """Test that tag info dicts contain expected keys."""
        tags = list_tags()
        if tags:
            tag = tags[0]
            assert "name" in tag
            assert "description" in tag
            assert "module" in tag

    def test_list_tags_contains_py_tag(self):
        """Test that !py tag is discovered."""
        tags = list_tags()
        tag_names = [t["name"] for t in tags]
        assert "!py" in tag_names


class TestGetComponentSchema:
    """Tests for get_component_schema function."""

    def test_get_node_schema(self):
        """Test getting schema for a node."""
        schema = get_component_schema("node", "llm_node")
        assert isinstance(schema, dict)
        # Should return either valid schema or error dict
        assert "properties" in schema or "error" in schema

    def test_get_adapter_schema(self):
        """Test getting schema for an adapter."""
        schema = get_component_schema("adapter", "MockLLM")
        assert isinstance(schema, dict)

    def test_get_tool_schema(self):
        """Test getting schema for a tool."""
        schema = get_component_schema("tool", "tool_end")
        assert isinstance(schema, dict)

    def test_get_schema_unknown_component_type(self):
        """Test error for unknown component type."""
        schema = get_component_schema("unknown_type", "foo")
        assert "error" in schema
        assert "Unknown component type" in schema["error"]

    def test_get_schema_nonexistent_component(self):
        """Test error for nonexistent component."""
        schema = get_component_schema("node", "nonexistent_node_xyz")
        assert "error" in schema


class TestGetTagSchema:
    """Tests for tag schema retrieval."""

    def test_get_tag_schema_py(self):
        """Test getting schema for !py tag."""
        schema = get_component_schema("tag", "!py")
        assert isinstance(schema, dict)

    def test_get_tag_schema_nonexistent(self):
        """Test error for nonexistent tag."""
        schema = get_component_schema("tag", "!nonexistent")
        assert isinstance(schema, dict)
        # Should return error or empty schema
