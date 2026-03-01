# ABOUTME: MCP server that exposes HMLR memory system as tools for Claude Code.
# ABOUTME: Wraps HMLRClient with stdio transport for integration via mcp.json.

import asyncio
import json
import os
import logging
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

logger = logging.getLogger(__name__)

_client = None


def _get_client():
    """Get or create the HMLR client singleton.

    Lazy initialization — the client is not created until the first tool call.
    This avoids loading heavy dependencies (sentence-transformers, torch) at
    server startup time.
    """
    global _client
    if _client is None:
        from hmlr import HMLRClient

        db_path = os.getenv(
            "HMLR_DB_PATH",
            str(Path.home() / ".hmlr" / "memory.db"),
        )
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        _client = HMLRClient(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            db_path=db_path,
        )
    return _client


def _json_response(data: Any) -> list[TextContent]:
    """Normalize any data to a single-element list of TextContent (JSON)."""
    return [TextContent(type="text", text=json.dumps(data, indent=2, default=str))]


def create_server() -> Server:
    """Create and configure the MCP server with memory tools."""
    server = Server("hmlr-memory")

    # ------------------------------------------------------------------
    # Tool definitions
    # ------------------------------------------------------------------

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="mem_search",
                description=(
                    "Search memories using HMLR's retrieval pipeline "
                    "(vector search + Governor filtering). Returns relevant "
                    "memories with context."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language search query",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum results to return",
                            "default": 10,
                        },
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="mem_add",
                description=(
                    "Add a memory to HMLR. Ingested through ChunkEngine, "
                    "FactScrubber, and Scribe. Tags help with retrieval."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The memory content to store",
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Tags for categorization",
                        },
                    },
                    "required": ["text"],
                },
            ),
            Tool(
                name="mem_add_file",
                description=(
                    "Ingest a file's contents as a memory. The file is read "
                    "and processed through HMLR's ingestion pipeline."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Absolute path to the file to ingest",
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Tags for categorization",
                        },
                    },
                    "required": ["path"],
                },
            ),
            Tool(
                name="mem_delete",
                description="Delete a specific memory by ID.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "memory_id": {
                            "type": "string",
                            "description": "The ID of the memory to delete",
                        },
                    },
                    "required": ["memory_id"],
                },
            ),
            Tool(
                name="mem_status",
                description=(
                    "Get statistics about the memory system: total memories, "
                    "bridge blocks, dossiers, last gardener run."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            Tool(
                name="mem_garden",
                description=(
                    "Manually trigger the gardener to promote bridge blocks "
                    "to long-term memory and create dossiers."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "target": {
                            "type": "string",
                            "description": (
                                "'all' to process all blocks, or a specific "
                                "block ID"
                            ),
                            "default": "all",
                        },
                    },
                },
            ),
            Tool(
                name="mem_ingest_reflection",
                description=(
                    "Ingest session-reflection-analysis output as a tagged "
                    "memory. Automatically tagged as a session reflection for "
                    "gardener processing."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "reflection": {
                            "type": "string",
                            "description": "The session-reflection-analysis output text",
                        },
                        "session_id": {
                            "type": "string",
                            "description": "Optional session identifier for tracking",
                        },
                    },
                    "required": ["reflection"],
                },
            ),
        ]

    # ------------------------------------------------------------------
    # Tool dispatch
    # ------------------------------------------------------------------

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        try:
            return await _dispatch_tool(name, arguments)
        except Exception as exc:
            logger.exception("Tool %s raised an exception", name)
            return _json_response({"error": str(exc)})

    return server


# ------------------------------------------------------------------
# Individual tool handlers
# ------------------------------------------------------------------


async def _dispatch_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Route a tool call to the appropriate handler."""
    if name == "mem_search":
        return await _handle_mem_search(arguments)
    elif name == "mem_add":
        return await _handle_mem_add(arguments)
    elif name == "mem_add_file":
        return await _handle_mem_add_file(arguments)
    elif name == "mem_delete":
        return await _handle_mem_delete(arguments)
    elif name == "mem_status":
        return await _handle_mem_status(arguments)
    elif name == "mem_garden":
        return await _handle_mem_garden(arguments)
    elif name == "mem_ingest_reflection":
        return await _handle_mem_ingest_reflection(arguments)
    else:
        return _json_response({"error": f"Unknown tool: {name}"})


async def _handle_mem_search(arguments: dict[str, Any]) -> list[TextContent]:
    client = _get_client()
    result = await client.chat(
        message=arguments["query"],
        session_id="mcp_search",
        force_intent="retrieval",
    )
    return _json_response(result)


async def _handle_mem_add(arguments: dict[str, Any]) -> list[TextContent]:
    client = _get_client()
    text = arguments["text"]
    tags = arguments.get("tags", [])
    tag_prefix = f"[Tags: {', '.join(tags)}] " if tags else ""
    result = await client.chat(
        message=f"{tag_prefix}{text}",
        session_id="mcp_ingest",
    )
    return _json_response({"stored": True, "response": result})


async def _handle_mem_add_file(arguments: dict[str, Any]) -> list[TextContent]:
    file_path = Path(arguments["path"]).expanduser()
    if not file_path.exists():
        return _json_response({"error": f"File not found: {file_path}"})

    content = file_path.read_text()
    client = _get_client()
    tags = arguments.get("tags", [])
    tag_prefix = (
        f"[Tags: {', '.join(tags)}] [Source: {file_path.name}] "
        if tags
        else f"[Source: {file_path.name}] "
    )
    result = await client.chat(
        message=f"{tag_prefix}{content}",
        session_id="mcp_file_ingest",
    )
    return _json_response({"stored": True, "file": str(file_path)})


async def _handle_mem_delete(arguments: dict[str, Any]) -> list[TextContent]:
    client = _get_client()
    memory_id = arguments["memory_id"]
    try:
        # Delete from the daily_ledger (bridge block archive)
        cursor = client.components.storage.conn.cursor()
        cursor.execute(
            "DELETE FROM daily_ledger WHERE block_id = ?",
            (memory_id,),
        )
        client.components.storage.conn.commit()
        deleted = cursor.rowcount > 0
        return _json_response({"deleted": deleted, "id": memory_id})
    except Exception as exc:
        return _json_response({"error": str(exc)})


async def _handle_mem_status(arguments: dict[str, Any]) -> list[TextContent]:
    client = _get_client()
    stats = client.get_memory_stats()
    return _json_response(stats)


async def _handle_mem_garden(arguments: dict[str, Any]) -> list[TextContent]:
    client = _get_client()
    from hmlr.memory.gardener.manual_gardener import ManualGardener

    gardener = ManualGardener(
        storage=client.components.storage,
        embedding_storage=client.components.embedding_storage,
        llm_client=(
            client.components.governor.api_client
            if client.components.governor
            else None
        ),
        dossier_governor=getattr(client.components, "dossier_governor", None),
        dossier_storage=getattr(client.components, "dossier_storage", None),
    )
    target = arguments.get("target", "all")
    blocks = client.components.storage.get_active_bridge_blocks()

    if not blocks:
        return _json_response({"message": "No active bridge blocks to garden"})

    if target == "all":
        block_ids = [b.get("block_id") for b in blocks]
    else:
        block_ids = [target]

    results = []
    for block_id in block_ids:
        try:
            result = await gardener.process_bridge_block(block_id)
            results.append({"block_id": block_id, "status": "processed", **result})
        except Exception as exc:
            results.append(
                {"block_id": block_id, "status": "error", "error": str(exc)}
            )

    return _json_response({"gardened": results})


async def _handle_mem_ingest_reflection(
    arguments: dict[str, Any],
) -> list[TextContent]:
    client = _get_client()
    reflection = arguments["reflection"]
    session_id = arguments.get("session_id", "reflection")
    result = await client.chat(
        message=(
            "[Tags: session-reflection, learning, pattern] "
            f"[Type: reflection-analysis] {reflection}"
        ),
        session_id=f"reflection_{session_id}",
    )
    return _json_response({"stored": True, "type": "reflection"})


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------


async def main():
    """Run the MCP server via stdio transport."""
    server = create_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
