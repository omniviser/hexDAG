"""Tests for MCP server (hexdag.mcp_server)."""

import json

import pytest

# Skip all tests in this module if mcp is not installed
mcp = pytest.importorskip("mcp", reason="MCP package not installed")


class TestComponentDiscoveryTools:
    """Tests for the direct component discovery MCP tools."""

    @pytest.mark.asyncio
    async def test_list_nodes_returns_json(self) -> None:
        from hexdag.mcp_server import list_nodes

        result = json.loads(list_nodes())
        assert isinstance(result, list)
        kinds = [n["kind"] for n in result]
        assert "llm_node" in kinds

    @pytest.mark.asyncio
    async def test_list_adapters_returns_json(self) -> None:
        from hexdag.mcp_server import list_adapters

        result = json.loads(list_adapters())
        assert isinstance(result, list)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_list_adapters_filter_by_port(self) -> None:
        from hexdag.mcp_server import list_adapters

        result = json.loads(list_adapters(port_type="llm"))
        assert all(a["port_type"] == "llm" for a in result)

    @pytest.mark.asyncio
    async def test_list_tools_returns_json(self) -> None:
        from hexdag.mcp_server import list_tools

        result = json.loads(list_tools())
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_list_macros_returns_json(self) -> None:
        from hexdag.mcp_server import list_macros

        result = json.loads(list_macros())
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_list_tags_returns_json(self) -> None:
        from hexdag.mcp_server import list_tags

        result = json.loads(list_tags())
        assert isinstance(result, list)
        names = [t["name"] for t in result]
        assert "!py" in names

    @pytest.mark.asyncio
    async def test_get_component_schema_node(self) -> None:
        from hexdag.mcp_server import get_component_schema

        result = json.loads(get_component_schema("tag", "!py"))
        assert "name" in result or "error" not in result

    @pytest.mark.asyncio
    async def test_get_component_schema_unknown(self) -> None:
        from hexdag.mcp_server import get_component_schema

        result = json.loads(get_component_schema("node", "nonexistent_xyz"))
        assert "error" in result
