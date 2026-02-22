"""Tests for MCP server (hexdag.mcp_server)."""

import json

import pytest

# Skip all tests in this module if mcp is not installed
mcp = pytest.importorskip("mcp", reason="MCP package not installed")


class TestVFSToolsRead:
    """Tests for the vfs_read MCP tool."""

    @pytest.mark.asyncio
    async def test_read_tags_returns_json(self) -> None:
        """vfs_read /lib/tags should return valid JSON list."""
        from hexdag.mcp_server import vfs_read

        result = await vfs_read("/lib/tags")
        parsed = json.loads(result)
        assert isinstance(parsed, list)

    @pytest.mark.asyncio
    async def test_read_tags_contains_py(self) -> None:
        """Reading /lib/tags should include !py tag data."""
        from hexdag.mcp_server import vfs_read

        result = json.loads(await vfs_read("/lib/tags"))
        names = [t["name"] for t in result]
        assert "!py" in names

    @pytest.mark.asyncio
    async def test_read_specific_tag(self) -> None:
        """Reading /lib/tags/!py should return tag detail."""
        from hexdag.mcp_server import vfs_read

        result = json.loads(await vfs_read("/lib/tags/!py"))
        assert result["name"] == "!py"
        assert "description" in result
        assert "security_warning" in result

    @pytest.mark.asyncio
    async def test_read_tag_schema(self) -> None:
        """Reading /lib/tags/!py/schema should return tag schema."""
        from hexdag.mcp_server import vfs_read

        result = json.loads(await vfs_read("/lib/tags/!py/schema"))
        assert result["name"] == "!py"
        assert result["type"] == "yaml_tag"
        assert "schema" in result
        assert "yaml_example" in result

    @pytest.mark.asyncio
    async def test_read_include_tag_schema(self) -> None:
        """Reading /lib/tags/!include/schema should return schema."""
        from hexdag.mcp_server import vfs_read

        result = json.loads(await vfs_read("/lib/tags/!include/schema"))
        assert result["name"] == "!include"
        assert result["type"] == "yaml_tag"

    @pytest.mark.asyncio
    async def test_read_nodes(self) -> None:
        """Reading /lib/nodes should return node list."""
        from hexdag.mcp_server import vfs_read

        result = json.loads(await vfs_read("/lib/nodes"))
        assert isinstance(result, list)
        kinds = [n["kind"] for n in result]
        assert "llm_node" in kinds

    @pytest.mark.asyncio
    async def test_read_adapters(self) -> None:
        """Reading /lib/adapters should return adapter list."""
        from hexdag.mcp_server import vfs_read

        result = json.loads(await vfs_read("/lib/adapters"))
        assert isinstance(result, list)
        assert len(result) > 0


class TestVFSToolsList:
    """Tests for the vfs_list MCP tool."""

    @pytest.mark.asyncio
    async def test_list_lib_root(self) -> None:
        """Listing /lib/ should return entity type directories."""
        from hexdag.mcp_server import vfs_list

        result = json.loads(await vfs_list("/lib/"))
        names = [e["name"] for e in result]
        assert "nodes" in names
        assert "adapters" in names
        assert "tags" in names

    @pytest.mark.asyncio
    async def test_list_nodes(self) -> None:
        """Listing /lib/nodes/ should return node entries."""
        from hexdag.mcp_server import vfs_list

        result = json.loads(await vfs_list("/lib/nodes/"))
        names = [e["name"] for e in result]
        assert "llm_node" in names

    @pytest.mark.asyncio
    async def test_list_tags(self) -> None:
        """Listing /lib/tags/ should return tag entries."""
        from hexdag.mcp_server import vfs_list

        result = json.loads(await vfs_list("/lib/tags/"))
        names = [e["name"] for e in result]
        assert "!py" in names
        assert "!include" in names


class TestVFSToolsStat:
    """Tests for the vfs_stat MCP tool."""

    @pytest.mark.asyncio
    async def test_stat_lib(self) -> None:
        """Stat /lib should return directory metadata."""
        from hexdag.mcp_server import vfs_stat

        result = json.loads(await vfs_stat("/lib"))
        assert result["entry_type"] == "directory"
        assert result["child_count"] > 0

    @pytest.mark.asyncio
    async def test_stat_node(self) -> None:
        """Stat /lib/nodes/llm_node should return node metadata."""
        from hexdag.mcp_server import vfs_stat

        result = json.loads(await vfs_stat("/lib/nodes/llm_node"))
        assert result["entry_type"] == "file"
        assert result["entity_type"] == "node"

    @pytest.mark.asyncio
    async def test_stat_adapter(self) -> None:
        """Stat on an adapter should include port_type tag."""
        from hexdag.mcp_server import vfs_read, vfs_stat

        # Find an adapter name first
        adapters = json.loads(await vfs_read("/lib/adapters"))
        if adapters:
            name = adapters[0]["name"]
            result = json.loads(await vfs_stat(f"/lib/adapters/{name}"))
            assert result["entity_type"] == "adapter"
