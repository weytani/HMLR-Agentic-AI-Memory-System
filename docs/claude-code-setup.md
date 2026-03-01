# ABOUTME: Instructions for configuring HMLR Memory as a Claude Code MCP server.
# ABOUTME: Documents the mcp.json entry and required environment setup.

# Claude Code MCP Setup

## Prerequisites
- HMLR Memory installed at `~/code/hmlr-memory`
- `uv` available on PATH
- `ANTHROPIC_API_KEY` set in your environment

## mcp.json Configuration

Add to `~/.claude/mcp.json` under `mcpServers`:

```json
"hmlr-memory": {
    "type": "stdio",
    "command": "uv",
    "args": ["run", "--directory", "/Users/weytani/code/hmlr-memory", "python", "-m", "hmlr.mcp_server"],
    "env": {
        "API_PROVIDER": "anthropic",
        "HMLR_DEFAULT_MODEL": "claude-sonnet-4-6",
        "HMLR_MAIN_MODEL": "claude-opus-4-6",
        "HMLR_SYNTHESIS_MODEL": "claude-opus-4-6",
        "HMLR_THINKING_BUDGET": "10000",
        "HMLR_DB_PATH": "/Users/weytani/.hmlr/memory.db"
    }
}
```

The `ANTHROPIC_API_KEY` is intentionally omitted from the config. It should be set in
your shell environment (e.g., in `~/.zshrc` or `~/.zprofile`) so it is inherited by
the MCP server process at runtime.

## Storage Directory

Create the database storage directory:

```bash
mkdir -p ~/.hmlr
```

The SQLite database will be created automatically at `~/.hmlr/memory.db` on first use.

## Verification

Start a new Claude Code session and check that `mem_status` tool is available.
You can test it by asking Claude to run `mem_status` -- it should return information
about the memory system's state and database location.
