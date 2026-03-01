# ABOUTME: Tests for HMLR MCP server tool definitions and structure.
# ABOUTME: Verifies server creation, tool registration, and tool schema correctness.

import pytest
import asyncio
import json

from mcp.types import (
    CallToolRequest,
    ListToolsRequest,
    TextContent,
)


def test_mcp_server_module_imports():
    """MCP server module should be importable."""
    from hmlr import mcp_server

    assert hasattr(mcp_server, "create_server")


def test_server_creation_returns_server_instance():
    """create_server should return an MCP Server instance."""
    from hmlr.mcp_server import create_server
    from mcp.server import Server

    server = create_server()
    assert isinstance(server, Server)


def test_server_has_correct_name():
    """Server should be named 'hmlr-memory'."""
    from hmlr.mcp_server import create_server

    server = create_server()
    assert server.name == "hmlr-memory"


# ---------- helper to list tools from a server ----------


async def _get_tools(server):
    """Call the server's list_tools handler and return tool list."""
    handler = server.request_handlers[ListToolsRequest]
    result = await handler(ListToolsRequest(method="tools/list"))
    return result.root.tools


async def _call_tool(server, name, arguments=None):
    """Call a tool through the server's call_tool handler."""
    handler = server.request_handlers[CallToolRequest]
    request = CallToolRequest(
        method="tools/call",
        params={"name": name, "arguments": arguments or {}},
    )
    result = await handler(request)
    return result.root.content


# ---------- tool registration tests ----------


@pytest.mark.asyncio
async def test_server_registers_all_required_tools():
    """Server should register all seven required memory tools."""
    from hmlr.mcp_server import create_server

    server = create_server()
    tools = await _get_tools(server)

    tool_names = {t.name for t in tools}

    expected_tools = {
        "mem_search",
        "mem_add",
        "mem_add_file",
        "mem_delete",
        "mem_status",
        "mem_garden",
        "mem_ingest_reflection",
    }
    assert tool_names == expected_tools, (
        f"Missing tools: {expected_tools - tool_names}, "
        f"Extra tools: {tool_names - expected_tools}"
    )


@pytest.mark.asyncio
async def test_all_tools_have_descriptions():
    """Every registered tool should have a non-empty description."""
    from hmlr.mcp_server import create_server

    server = create_server()
    tools = await _get_tools(server)

    for tool in tools:
        assert tool.description, f"Tool {tool.name} is missing a description"
        assert len(tool.description) > 10, (
            f"Tool {tool.name} description is too short: {tool.description!r}"
        )


# ---------- schema validation tests ----------


@pytest.mark.asyncio
async def test_mem_search_tool_schema():
    """mem_search tool should have correct input schema."""
    from hmlr.mcp_server import create_server

    server = create_server()
    tools = await _get_tools(server)
    tools_map = {t.name: t for t in tools}

    search_tool = tools_map["mem_search"]
    schema = search_tool.inputSchema
    assert schema["type"] == "object"
    assert "query" in schema["properties"]
    assert "limit" in schema["properties"]
    assert "query" in schema["required"]


@pytest.mark.asyncio
async def test_mem_add_tool_schema():
    """mem_add tool should require text, with optional tags."""
    from hmlr.mcp_server import create_server

    server = create_server()
    tools = await _get_tools(server)
    tools_map = {t.name: t for t in tools}

    add_tool = tools_map["mem_add"]
    schema = add_tool.inputSchema
    assert "text" in schema["properties"]
    assert "tags" in schema["properties"]
    assert "text" in schema["required"]
    assert schema["properties"]["tags"]["type"] == "array"


@pytest.mark.asyncio
async def test_mem_add_file_tool_schema():
    """mem_add_file tool should require path, with optional tags."""
    from hmlr.mcp_server import create_server

    server = create_server()
    tools = await _get_tools(server)
    tools_map = {t.name: t for t in tools}

    file_tool = tools_map["mem_add_file"]
    schema = file_tool.inputSchema
    assert "path" in schema["properties"]
    assert "tags" in schema["properties"]
    assert "path" in schema["required"]


@pytest.mark.asyncio
async def test_mem_delete_tool_schema():
    """mem_delete tool should require memory_id."""
    from hmlr.mcp_server import create_server

    server = create_server()
    tools = await _get_tools(server)
    tools_map = {t.name: t for t in tools}

    delete_tool = tools_map["mem_delete"]
    schema = delete_tool.inputSchema
    assert "memory_id" in schema["properties"]
    assert "memory_id" in schema["required"]


@pytest.mark.asyncio
async def test_mem_status_tool_schema():
    """mem_status tool should have no required inputs."""
    from hmlr.mcp_server import create_server

    server = create_server()
    tools = await _get_tools(server)
    tools_map = {t.name: t for t in tools}

    status_tool = tools_map["mem_status"]
    schema = status_tool.inputSchema
    assert schema["type"] == "object"
    # No required field, or empty required list
    required = schema.get("required")
    assert required is None or required == []


@pytest.mark.asyncio
async def test_mem_garden_tool_schema():
    """mem_garden tool should have optional target parameter."""
    from hmlr.mcp_server import create_server

    server = create_server()
    tools = await _get_tools(server)
    tools_map = {t.name: t for t in tools}

    garden_tool = tools_map["mem_garden"]
    schema = garden_tool.inputSchema
    assert "target" in schema["properties"]
    # target is optional — no required field or not in required
    required = schema.get("required")
    assert required is None or "target" not in required


@pytest.mark.asyncio
async def test_mem_ingest_reflection_tool_schema():
    """mem_ingest_reflection should require reflection, optional session_id."""
    from hmlr.mcp_server import create_server

    server = create_server()
    tools = await _get_tools(server)
    tools_map = {t.name: t for t in tools}

    reflection_tool = tools_map["mem_ingest_reflection"]
    schema = reflection_tool.inputSchema
    assert "reflection" in schema["properties"]
    assert "session_id" in schema["properties"]
    assert "reflection" in schema["required"]


# ---------- handler registration tests ----------


@pytest.mark.asyncio
async def test_call_tool_handler_registered():
    """Server should have a call_tool handler registered."""
    from hmlr.mcp_server import create_server

    server = create_server()
    assert CallToolRequest in server.request_handlers, (
        "tools/call handler not registered"
    )


@pytest.mark.asyncio
async def test_call_tool_unknown_tool_returns_error():
    """Calling an unknown tool should return an error response."""
    from hmlr.mcp_server import create_server

    server = create_server()
    content = await _call_tool(server, "nonexistent_tool", {})

    assert len(content) == 1
    assert content[0].type == "text"
    parsed = json.loads(content[0].text)
    assert "error" in parsed


@pytest.mark.asyncio
async def test_mem_add_file_nonexistent_returns_error():
    """mem_add_file with nonexistent path should return file not found error."""
    from hmlr.mcp_server import create_server

    server = create_server()
    content = await _call_tool(
        server,
        "mem_add_file",
        {"path": "/nonexistent/path/to/file.txt"},
    )

    assert len(content) == 1
    parsed = json.loads(content[0].text)
    assert "error" in parsed
    assert "not found" in parsed["error"].lower()


# ---------- module-level tests ----------


def test_server_has_main_function():
    """Module should have a main() coroutine for stdio execution."""
    from hmlr.mcp_server import main

    assert asyncio.iscoroutinefunction(main)


def test_lazy_client_initialization():
    """_get_client should not be called until a tool is invoked."""
    from hmlr import mcp_server

    # The global _client should be None at import time
    assert mcp_server._client is None
