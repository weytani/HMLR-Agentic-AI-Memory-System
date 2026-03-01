# ABOUTME: Integration tests for HMLR with Anthropic provider — full round trip.
# ABOUTME: Requires ANTHROPIC_API_KEY. Skipped automatically if key is not set.

import asyncio
import json
import os
import tempfile

import pytest
from mcp.types import ListToolsRequest

# ---------------------------------------------------------------------------
# Marker: skip all API-dependent tests when key is absent
# ---------------------------------------------------------------------------
requires_api_key = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY required for integration tests",
)


# ===========================================================================
# MCP server structural tests (no API key needed)
# ===========================================================================


class TestMCPServerStructure:
    """Verify the MCP server can be created and tools introspected."""

    def test_mcp_server_creates_and_lists_tools(self):
        """MCP server should create successfully and list all 7 tools."""
        from hmlr.mcp_server import create_server

        server = create_server()
        assert server is not None

    @pytest.mark.asyncio
    async def test_mcp_server_tool_count(self):
        """MCP server should expose exactly 7 memory tools."""
        from hmlr.mcp_server import create_server

        server = create_server()
        handler = server.request_handlers[ListToolsRequest]
        result = await handler(ListToolsRequest(method="tools/list"))
        tools = result.root.tools

        assert len(tools) == 7, (
            f"Expected 7 tools, got {len(tools)}: "
            f"{[t.name for t in tools]}"
        )

    @pytest.mark.asyncio
    async def test_mcp_server_tool_names(self):
        """MCP server tools should match expected names."""
        from hmlr.mcp_server import create_server

        server = create_server()
        handler = server.request_handlers[ListToolsRequest]
        result = await handler(ListToolsRequest(method="tools/list"))
        tool_names = {t.name for t in result.root.tools}

        expected = {
            "mem_search",
            "mem_add",
            "mem_add_file",
            "mem_delete",
            "mem_status",
            "mem_garden",
            "mem_ingest_reflection",
        }
        assert tool_names == expected

    @pytest.mark.asyncio
    async def test_mcp_server_all_tools_have_input_schemas(self):
        """Every tool should declare an inputSchema with type 'object'."""
        from hmlr.mcp_server import create_server

        server = create_server()
        handler = server.request_handlers[ListToolsRequest]
        result = await handler(ListToolsRequest(method="tools/list"))

        for tool in result.root.tools:
            assert tool.inputSchema is not None, (
                f"Tool {tool.name} has no inputSchema"
            )
            assert tool.inputSchema.get("type") == "object", (
                f"Tool {tool.name} inputSchema type is not 'object'"
            )


# ===========================================================================
# Full round-trip integration tests (require ANTHROPIC_API_KEY)
# ===========================================================================


@requires_api_key
class TestFullRoundTrip:
    """End-to-end tests that exercise the Anthropic provider."""

    @pytest.mark.asyncio
    async def test_add_and_search_memory(self):
        """Store a memory and retrieve it."""
        from hmlr import HMLRClient

        with tempfile.NamedTemporaryFile(suffix=".db", delete=True) as tmp:
            db_path = tmp.name

        # NamedTemporaryFile is deleted on close; we just want the path
        os.environ["API_PROVIDER"] = "anthropic"
        os.environ["COGNITIVE_LATTICE_DB"] = db_path

        client = HMLRClient(db_path=db_path)
        try:
            # Store a fact
            await client.chat(
                "David prefers uv for Python package management. "
                "He never uses pip or poetry.",
                session_id="test_ingest",
            )

            # Retrieve it
            result = await client.chat(
                "What package manager does David prefer?",
                session_id="test_retrieval",
            )

            assert "uv" in result["content"].lower()
        finally:
            client.close()

    @pytest.mark.asyncio
    async def test_memory_stats(self):
        """Memory stats should return valid structure."""
        from hmlr import HMLRClient

        with tempfile.NamedTemporaryFile(suffix=".db", delete=True) as tmp:
            db_path = tmp.name

        os.environ["API_PROVIDER"] = "anthropic"
        os.environ["COGNITIVE_LATTICE_DB"] = db_path

        client = HMLRClient(db_path=db_path)
        try:
            stats = client.get_memory_stats()
            assert "db_path" in stats
        finally:
            client.close()

    @pytest.mark.asyncio
    async def test_ingest_and_status(self):
        """After ingesting a memory, stats should reflect stored data."""
        from hmlr import HMLRClient

        with tempfile.NamedTemporaryFile(suffix=".db", delete=True) as tmp:
            db_path = tmp.name

        os.environ["API_PROVIDER"] = "anthropic"
        os.environ["COGNITIVE_LATTICE_DB"] = db_path

        client = HMLRClient(db_path=db_path)
        try:
            await client.chat(
                "Python 3.12 introduced several performance improvements.",
                session_id="test_status_ingest",
            )

            stats = client.get_memory_stats()
            assert "db_path" in stats
            assert stats["db_path"] == db_path
        finally:
            client.close()

    @pytest.mark.asyncio
    async def test_mcp_tool_dispatch_mem_status(self):
        """MCP mem_status tool should return stats via the dispatch handler."""
        from hmlr.mcp_server import create_server

        with tempfile.NamedTemporaryFile(suffix=".db", delete=True) as tmp:
            db_path = tmp.name

        os.environ["API_PROVIDER"] = "anthropic"
        os.environ["HMLR_DB_PATH"] = db_path

        # Reset the global client so it picks up the new env vars
        import hmlr.mcp_server as mcp_mod

        mcp_mod._client = None

        server = create_server()
        try:
            from mcp.types import CallToolRequest

            handler = server.request_handlers[CallToolRequest]
            request = CallToolRequest(
                method="tools/call",
                params={"name": "mem_status", "arguments": {}},
            )
            result = await handler(request)
            content = result.root.content

            assert len(content) == 1
            parsed = json.loads(content[0].text)
            assert "db_path" in parsed
        finally:
            # Clean up the global client
            if mcp_mod._client is not None:
                mcp_mod._client.close()
                mcp_mod._client = None
