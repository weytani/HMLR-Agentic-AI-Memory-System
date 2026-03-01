# HMLR MCP Server Implementation Plan

> **For Claude:** Use `@skills/collaboration/executing-plans/SKILL.md` to implement this plan task-by-task.

**Goal:** Wrap the HMLR memory system in an MCP server so Claude Code can use it for persistent cross-session memory, knowledge retrieval, and agent self-improvement via session-reflection-analysis.

**Architecture:** Fork of HMLR with Anthropic as the LLM provider (Opus 4.6 for Governor/Gardener, Sonnet 4.6 for workers). Local `sentence-transformers` embeddings (no OpenAI dependency). MCP server exposes memory tools via stdio transport. Cron-based gardener promotes short-term → long-term memory nightly.

**Tech Stack:** Python 3.12, uv, `mcp` Python SDK, `anthropic` SDK, `sentence-transformers`, SQLite

**Research docs:** `research/brainstorm-summary.md`, `research/hmlr-architecture.md`, `research/codebase-audit.md`

---

## Git Workflow

**Every task gets its own branch.** This is non-negotiable.

### Repository Setup

- **Origin:** `https://github.com/weytani/HMLR-Agentic-AI-Memory-System.git` (our fork)
- **Upstream:** `https://github.com/Sean-V-Dev/HMLR-Agentic-AI-Memory-System.git` (original HMLR)
- **Default branch:** `master`
- **Work branch base:** Create a `develop` branch from `master` as the integration branch

### Branch Naming Convention

Each task gets a branch off `develop`:

```
develop                          ← integration branch
  ├── task/1-uv-dependencies
  ├── task/2-extended-thinking
  ├── task/3-env-config
  ├── task/4-mcp-server
  ├── task/5-gardener-cron
  ├── task/6-claude-code-config
  ├── task/7-remember-skill
  ├── task/8-integration-tests
  └── task/9-reflection-hook
```

### Per-Task Git Process

1. **Start:** `git checkout develop && git pull && git checkout -b task/N-short-name`
2. **During:** Commit early and often. Every step that passes gets a commit.
3. **Finish:** Push branch, merge to `develop` (or PR if you prefer review).
4. **Between tasks:** Always start the next task from fresh `develop`.

### Commit Message Format

Follow conventional commits:
- `feat:` — new functionality
- `build:` — dependency/packaging changes
- `config:` — configuration changes
- `test:` — test additions
- `docs:` — documentation changes

### First-Session Setup

Before starting Task 1, create the `develop` branch:

```bash
cd ~/code/hmlr-memory
git checkout -b develop
git push -u origin develop
```

---

### Task 1: Convert Project to uv and Add Dependencies

**Files:**
- Modify: `pyproject.toml`
- Delete: `setup.py`, `requirements.txt`, `requirements-core.txt`, `requirements-dev.txt`

**Step 1: Initialize uv and update pyproject.toml**

Replace the existing `pyproject.toml` build system with uv-native config. Add `anthropic` and `mcp` as dependencies. Remove `openai` from required dependencies (move to optional).

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "hmlr-memory"
version = "0.2.0"
description = "HMLR memory system with MCP server interface for Claude Code"
readme = "README.md"
requires-python = ">=3.12"
license = "MIT"

dependencies = [
    "anthropic>=0.42.0",
    "mcp>=1.0.0",
    "sentence-transformers>=2.2.0",
    "numpy>=1.24.0",
    "torch>=2.0.0",
]

[project.optional-dependencies]
openai = ["openai>=1.0.0"]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "ruff>=0.4.0",
]
```

**Step 2: Remove legacy packaging files**

```bash
rm setup.py requirements.txt requirements-core.txt requirements-dev.txt MANIFEST.in
```

**Step 3: Sync dependencies**

```bash
cd ~/code/hmlr-memory && uv sync
```

**Step 4: Verify import works**

```bash
uv run python -c "import hmlr; print('HMLR import OK')"
```

Expected: `HMLR import OK` (may warn about missing embedding model on first run — that's fine, it downloads on first use)

**Step 5: Commit**

```bash
git add pyproject.toml uv.lock .python-version
git rm setup.py requirements.txt requirements-core.txt requirements-dev.txt MANIFEST.in
git commit -m "build: convert to uv, add anthropic and mcp dependencies"
```

---

### Task 2: Add Extended Thinking Support to Anthropic Client

HMLR's `ExternalAPIClient` already has Anthropic support but doesn't pass extended thinking parameters. The Governor and Gardener need Opus with extended thinking.

**Files:**
- Modify: `hmlr/core/external_api_client.py`
- Modify: `hmlr/core/model_config.py`
- Create: `tests/test_anthropic_extended_thinking.py`

**Step 1: Write the failing test**

```python
# tests/test_anthropic_extended_thinking.py
"""Tests for Anthropic extended thinking support in ExternalAPIClient."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from hmlr.core.model_config import ModelConfig


def test_model_config_has_thinking_budget():
    """ModelConfig should have THINKING_BUDGET_TOKENS setting."""
    assert hasattr(ModelConfig, 'THINKING_BUDGET_TOKENS')
    assert isinstance(ModelConfig.THINKING_BUDGET_TOKENS, int)
    assert ModelConfig.THINKING_BUDGET_TOKENS > 0


def test_model_config_thinking_models():
    """ModelConfig should identify which operations use extended thinking."""
    assert hasattr(ModelConfig, 'MAIN_USES_THINKING')
    assert hasattr(ModelConfig, 'SYNTHESIS_USES_THINKING')
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_anthropic_extended_thinking.py -v
```

Expected: FAIL — `THINKING_BUDGET_TOKENS` attribute doesn't exist

**Step 3: Add thinking config to model_config.py**

In `hmlr/core/model_config.py`, add to the `ModelConfig` class after the `DEFAULT_PRESENCE_PENALTY` block:

```python
# ===== EXTENDED THINKING (Anthropic-specific) =====
# Budget for extended thinking on supported models (Opus)
# Set to 0 to disable extended thinking
THINKING_BUDGET_TOKENS: int = int(os.getenv("HMLR_THINKING_BUDGET", "10000"))

# Which operations use extended thinking (only meaningful for Anthropic Opus)
MAIN_USES_THINKING: bool = os.getenv("HMLR_MAIN_USES_THINKING", "True").lower() == "true"
SYNTHESIS_USES_THINKING: bool = os.getenv("HMLR_SYNTHESIS_USES_THINKING", "True").lower() == "true"
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_anthropic_extended_thinking.py -v
```

Expected: PASS

**Step 5: Update _call_anthropic_api_async to support extended thinking**

In `hmlr/core/external_api_client.py`, modify `_call_anthropic_api_async`:

After the `params` dict is built (around line 635), add thinking support:

```python
# Add extended thinking if configured
from .model_config import model_config
if (model_config.THINKING_BUDGET_TOKENS > 0 and
    "opus" in model.lower()):
    params["thinking"] = {
        "type": "enabled",
        "budget_tokens": model_config.THINKING_BUDGET_TOKENS
    }
    # Extended thinking requires temperature=1
    params["temperature"] = 1
    # Increase max_tokens to accommodate thinking output
    params["max_tokens"] = max(max_tokens, model_config.THINKING_BUDGET_TOKENS + max_tokens)
```

Also update the sync version `_call_anthropic_api` with the same logic.

**Step 6: Commit**

```bash
git add hmlr/core/model_config.py hmlr/core/external_api_client.py tests/test_anthropic_extended_thinking.py
git commit -m "feat: add extended thinking support for Anthropic Opus models"
```

---

### Task 3: Create Environment Configuration

**Files:**
- Create: `hmlr/.env.example`
- Create: `hmlr/env_config.py`

**Step 1: Create .env.example**

```bash
# HMLR Memory System - Anthropic Configuration
API_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-your-key-here

# Model Tiers
HMLR_DEFAULT_MODEL=claude-sonnet-4-6
HMLR_MAIN_MODEL=claude-opus-4-6
HMLR_SYNTHESIS_MODEL=claude-opus-4-6
HMLR_NANO_MODEL=claude-sonnet-4-6
HMLR_LATTICE_MODEL=claude-sonnet-4-6

# Extended Thinking
HMLR_THINKING_BUDGET=10000
HMLR_MAIN_USES_THINKING=true
HMLR_SYNTHESIS_USES_THINKING=true

# Database
COGNITIVE_LATTICE_DB=~/.hmlr/memory.db

# Embedding (local, no API needed)
HMLR_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
HMLR_EMBEDDING_DIM=384
```

**Step 2: Create actual .env for local use (DO NOT COMMIT)**

Copy `.env.example` to `.env` and fill in real API key. Add `.env` to `.gitignore`.

**Step 3: Commit**

```bash
echo ".env" >> .gitignore
git add .env.example .gitignore
git commit -m "config: add environment configuration for Anthropic provider"
```

---

### Task 4: Build the MCP Server

This is the core deliverable — an MCP server that exposes HMLR's memory system as tools Claude Code can call.

**Files:**
- Create: `hmlr/mcp_server.py`
- Create: `tests/test_mcp_server.py`

**Step 1: Write the failing test**

```python
# tests/test_mcp_server.py
"""Tests for HMLR MCP server tool definitions."""
import pytest
import asyncio


def test_mcp_server_module_imports():
    """MCP server module should be importable."""
    from hmlr import mcp_server
    assert hasattr(mcp_server, 'create_server')


def test_server_has_required_tools():
    """Server should register all required memory tools."""
    from hmlr.mcp_server import create_server
    server = create_server()

    # Get registered tool names
    tool_names = [tool.name for tool in server.list_tools()]

    required_tools = [
        'mem_search',
        'mem_add',
        'mem_add_file',
        'mem_delete',
        'mem_status',
        'mem_garden',
        'mem_ingest_reflection',
    ]

    for tool_name in required_tools:
        assert tool_name in tool_names, f"Missing tool: {tool_name}"
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_mcp_server.py -v
```

Expected: FAIL — module doesn't exist

**Step 3: Implement the MCP server**

```python
# hmlr/mcp_server.py
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

# Lazy-initialized client
_client = None


def _get_client():
    """Get or create the HMLR client singleton."""
    global _client
    if _client is None:
        from hmlr import HMLRClient
        db_path = os.getenv("HMLR_DB_PATH", str(Path.home() / ".hmlr" / "memory.db"))

        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        _client = HMLRClient(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            db_path=db_path,
        )
    return _client


def create_server() -> Server:
    """Create and configure the MCP server with memory tools."""
    server = Server("hmlr-memory")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="mem_search",
                description="Search memories using HMLR's retrieval pipeline (vector search + Governor filtering). Returns relevant memories with context.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language search query"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum results to return",
                            "default": 10
                        }
                    },
                    "required": ["query"]
                }
            ),
            Tool(
                name="mem_add",
                description="Add a memory to HMLR. Ingested through ChunkEngine, FactScrubber, and Scribe. Tags help with retrieval.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The memory content to store"
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Tags for categorization (e.g., 'preference', 'architecture', 'gotcha')"
                        }
                    },
                    "required": ["text"]
                }
            ),
            Tool(
                name="mem_add_file",
                description="Ingest a file's contents as a memory. The file is read and processed through HMLR's ingestion pipeline.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Absolute path to the file to ingest"
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Tags for categorization"
                        }
                    },
                    "required": ["path"]
                }
            ),
            Tool(
                name="mem_delete",
                description="Delete a specific memory by ID.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "memory_id": {
                            "type": "string",
                            "description": "The ID of the memory to delete"
                        }
                    },
                    "required": ["memory_id"]
                }
            ),
            Tool(
                name="mem_status",
                description="Get statistics about the memory system: total memories, bridge blocks, dossiers, last gardener run.",
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            ),
            Tool(
                name="mem_garden",
                description="Manually trigger the gardener to promote bridge blocks to long-term memory and create dossiers.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "target": {
                            "type": "string",
                            "description": "'all' to process all blocks, or a specific block ID",
                            "default": "all"
                        }
                    }
                }
            ),
            Tool(
                name="mem_ingest_reflection",
                description="Ingest session-reflection-analysis output as a tagged memory. Automatically tagged as a session reflection for gardener processing.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "reflection": {
                            "type": "string",
                            "description": "The session-reflection-analysis output text"
                        },
                        "session_id": {
                            "type": "string",
                            "description": "Optional session identifier for tracking"
                        }
                    },
                    "required": ["reflection"]
                }
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        client = _get_client()

        if name == "mem_search":
            result = await client.chat(
                message=arguments["query"],
                session_id="mcp_search",
                force_intent="retrieval"
            )
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "mem_add":
            text = arguments["text"]
            tags = arguments.get("tags", [])
            tag_prefix = f"[Tags: {', '.join(tags)}] " if tags else ""
            result = await client.chat(
                message=f"{tag_prefix}{text}",
                session_id="mcp_ingest"
            )
            return [TextContent(type="text", text=json.dumps({"stored": True, "response": result}, indent=2))]

        elif name == "mem_add_file":
            file_path = Path(arguments["path"]).expanduser()
            if not file_path.exists():
                return [TextContent(type="text", text=json.dumps({"error": f"File not found: {file_path}"}))]
            content = file_path.read_text()
            tags = arguments.get("tags", [])
            tag_prefix = f"[Tags: {', '.join(tags)}] [Source: {file_path.name}] " if tags else f"[Source: {file_path.name}] "
            result = await client.chat(
                message=f"{tag_prefix}{content}",
                session_id="mcp_file_ingest"
            )
            return [TextContent(type="text", text=json.dumps({"stored": True, "file": str(file_path)}, indent=2))]

        elif name == "mem_delete":
            # Direct storage operation
            memory_id = arguments["memory_id"]
            try:
                client.components.storage.delete_memory(memory_id)
                return [TextContent(type="text", text=json.dumps({"deleted": True, "id": memory_id}))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}))]

        elif name == "mem_status":
            stats = client.get_memory_stats()
            return [TextContent(type="text", text=json.dumps(stats, indent=2))]

        elif name == "mem_garden":
            from hmlr.core.component_factory import ComponentFactory
            from hmlr.memory.gardener.manual_gardener import ManualGardener

            gardener = ManualGardener(
                storage=client.components.storage,
                embedding_storage=client.components.embedding_storage,
                llm_client=client.components.governor.api_client if client.components.governor else None,
                dossier_governor=getattr(client.components, 'dossier_governor', None),
                dossier_storage=getattr(client.components, 'dossier_storage', None)
            )

            target = arguments.get("target", "all")
            blocks = client.components.storage.get_active_bridge_blocks()

            if not blocks:
                return [TextContent(type="text", text=json.dumps({"message": "No active bridge blocks to garden"}))]

            results = []
            block_ids = [b.get('block_id') for b in blocks] if target == "all" else [target]

            for block_id in block_ids:
                try:
                    result = await gardener.process_bridge_block(block_id)
                    results.append({"block_id": block_id, "status": "processed", **result})
                except Exception as e:
                    results.append({"block_id": block_id, "status": "error", "error": str(e)})

            return [TextContent(type="text", text=json.dumps({"gardened": results}, indent=2))]

        elif name == "mem_ingest_reflection":
            reflection = arguments["reflection"]
            session_id = arguments.get("session_id", "reflection")
            result = await client.chat(
                message=f"[Tags: session-reflection, learning, pattern] [Type: reflection-analysis] {reflection}",
                session_id=f"reflection_{session_id}"
            )
            return [TextContent(type="text", text=json.dumps({"stored": True, "type": "reflection"}, indent=2))]

        else:
            return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]

    return server


async def main():
    """Run the MCP server via stdio."""
    server = create_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_mcp_server.py -v
```

Expected: PASS

**Step 5: Verify MCP server starts**

```bash
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"capabilities":{}}}' | uv run python -m hmlr.mcp_server
```

Expected: JSON-RPC response with server capabilities

**Step 6: Commit**

```bash
git add hmlr/mcp_server.py tests/test_mcp_server.py
git commit -m "feat: add MCP server wrapping HMLR memory tools"
```

---

### Task 5: Create Gardener Cron Script

A standalone script for cron that runs the gardener without needing Claude Code.

**Files:**
- Create: `scripts/garden-cron.sh`

**Step 1: Write the cron script**

```bash
#!/usr/bin/env bash
# ABOUTME: Cron wrapper for HMLR memory gardener.
# ABOUTME: Promotes bridge blocks to long-term memory and creates dossiers.

set -euo pipefail

HMLR_DIR="${HMLR_DIR:-$HOME/code/hmlr-memory}"
LOG_DIR="${HOME}/.hmlr/logs"
LOG_FILE="${LOG_DIR}/gardener-$(date +%Y-%m-%d).log"

mkdir -p "$LOG_DIR"

echo "=== Gardener run: $(date) ===" >> "$LOG_FILE"

cd "$HMLR_DIR"

# Run gardener with 'all' to process all active bridge blocks
uv run python -m hmlr.run_gardener all >> "$LOG_FILE" 2>&1

echo "=== Gardener complete: $(date) ===" >> "$LOG_FILE"
```

**Step 2: Make executable**

```bash
chmod +x scripts/garden-cron.sh
```

**Step 3: Document cron setup (but don't install — that's user's choice)**

Add a comment at the top of the script:
```bash
# Install with: crontab -e
# Run nightly at midnight:
# 0 0 * * * /Users/weytani/code/hmlr-memory/scripts/garden-cron.sh
```

**Step 4: Commit**

```bash
git add scripts/garden-cron.sh
git commit -m "feat: add gardener cron script for nightly memory promotion"
```

---

### Task 6: Configure MCP Server in Claude Code

Wire the MCP server into Claude Code's config so it's available in all sessions.

**Files:**
- Modify: `~/.claude/mcp.json`

**Step 1: Add hmlr-memory server to mcp.json**

Add the `hmlr-memory` entry alongside the existing `roam-research` entry:

```json
{
  "mcpServers": {
    "roam-research": {
      "...existing config..."
    },
    "hmlr-memory": {
      "type": "stdio",
      "command": "uv",
      "args": ["run", "--directory", "/Users/weytani/code/hmlr-memory", "python", "-m", "hmlr.mcp_server"],
      "env": {
        "API_PROVIDER": "anthropic",
        "ANTHROPIC_API_KEY": "<key>",
        "HMLR_DEFAULT_MODEL": "claude-sonnet-4-6",
        "HMLR_MAIN_MODEL": "claude-opus-4-6",
        "HMLR_SYNTHESIS_MODEL": "claude-opus-4-6",
        "HMLR_THINKING_BUDGET": "10000",
        "HMLR_DB_PATH": "/Users/weytani/.hmlr/memory.db"
      }
    }
  }
}
```

**Step 2: Create storage directory**

```bash
mkdir -p ~/.hmlr
```

**Step 3: Test MCP server from Claude Code**

Start a new Claude Code session and verify the `mem_status` tool is available and callable.

**Step 4: Commit mcp.json (without API key)**

NOTE: Do NOT commit with real API key. Use a placeholder or rely on env vars.

---

### Task 7: Update Remember Skill to Reference MCP Tools

Update the existing remember skill to document the MCP-backed tools instead of the unimplemented `mem_*` functions.

**Files:**
- Modify: `~/.claude/skills/remember/skill.md`

**Step 1: Update skill to reference MCP tools**

The skill should document that these tools are backed by the `hmlr-memory` MCP server and explain the retrieval pipeline (Governor with Opus extended thinking, vector search, dossier system).

Key changes:
- Remove references to unimplemented function signatures
- Document that `mem_search` goes through HMLR's full pipeline
- Document that `mem_ingest_reflection` is for session-reflection-analysis output
- Document the gardener and dossier system
- Keep the tag strategy and "search before create" principle

**Step 2: Commit**

```bash
git add ~/.claude/skills/remember/skill.md
git commit -m "docs: update remember skill to reference HMLR MCP backend"
```

---

### Task 8: Integration Test — Full Round Trip

End-to-end verification that the complete pipeline works.

**Files:**
- Create: `tests/test_integration_mcp.py`

**Step 1: Write integration test**

```python
# tests/test_integration_mcp.py
"""Integration test: full round trip through HMLR MCP server."""
import pytest
import asyncio
import os
import json

pytestmark = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY required for integration tests"
)


@pytest.mark.asyncio
async def test_add_and_search_memory():
    """Store a memory and retrieve it."""
    from hmlr import HMLRClient
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".db", delete=True) as tmp:
        os.environ["API_PROVIDER"] = "anthropic"
        os.environ["COGNITIVE_LATTICE_DB"] = tmp.name

        client = HMLRClient(db_path=tmp.name)

        # Store a fact
        await client.chat(
            "David prefers uv for Python package management. He never uses pip or poetry.",
            session_id="test_ingest"
        )

        # Retrieve it
        result = await client.chat(
            "What package manager does David prefer?",
            session_id="test_retrieval"
        )

        assert "uv" in result["content"].lower()
        client.close()


@pytest.mark.asyncio
async def test_memory_stats():
    """Memory stats should return valid structure."""
    from hmlr import HMLRClient
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".db", delete=True) as tmp:
        os.environ["API_PROVIDER"] = "anthropic"
        os.environ["COGNITIVE_LATTICE_DB"] = tmp.name

        client = HMLRClient(db_path=tmp.name)
        stats = client.get_memory_stats()

        assert "db_path" in stats
        client.close()
```

**Step 2: Run integration tests**

```bash
uv run pytest tests/test_integration_mcp.py -v -s
```

Expected: PASS (requires `ANTHROPIC_API_KEY` set)

**Step 3: Commit**

```bash
git add tests/test_integration_mcp.py
git commit -m "test: add integration tests for HMLR with Anthropic provider"
```

---

### Task 9: Wire Session-Reflection-Analysis Hook

Create a hook that feeds session-reflection-analysis output into HMLR after each session.

**Files:**
- Create: `scripts/ingest-reflection.py`

**Step 1: Write the ingestion script**

This script takes reflection text as stdin or argument and calls the HMLR MCP server's `mem_ingest_reflection` tool.

```python
#!/usr/bin/env python3
# ABOUTME: Ingests session-reflection-analysis output into HMLR memory.
# ABOUTME: Called by Claude Code hook or manually after a session.

import asyncio
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


async def ingest_reflection(reflection_text: str, session_id: str = "manual"):
    """Ingest a reflection into HMLR."""
    from hmlr import HMLRClient

    db_path = os.getenv("HMLR_DB_PATH", str(Path.home() / ".hmlr" / "memory.db"))
    client = HMLRClient(db_path=db_path)

    result = await client.chat(
        message=f"[Tags: session-reflection, learning, pattern] [Type: reflection-analysis] {reflection_text}",
        session_id=f"reflection_{session_id}"
    )

    print(f"Reflection ingested: {result.get('status', 'unknown')}")
    client.close()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        text = sys.stdin.read()

    if not text.strip():
        print("No reflection text provided")
        sys.exit(1)

    asyncio.run(ingest_reflection(text.strip()))
```

**Step 2: Make executable**

```bash
chmod +x scripts/ingest-reflection.py
```

**Step 3: Commit**

```bash
git add scripts/ingest-reflection.py
git commit -m "feat: add session reflection ingestion script for HMLR"
```

---

## Execution Order

Tasks 1-3 are prerequisites (project setup).
Tasks 4-5 are the core build (MCP server + gardener).
Tasks 6-7 are integration (wire into Claude Code).
Tasks 8-9 are validation and hooks.

Dependencies:
- Task 2 depends on Task 1 (uv must be set up)
- Task 4 depends on Tasks 1-3 (dependencies + config)
- Task 5 depends on Task 1
- Task 6 depends on Task 4
- Task 7 depends on Task 4
- Task 8 depends on Tasks 4 + 6
- Task 9 depends on Task 4

## Per-Task Branch Checklist

Every task MUST follow this exact sequence:

```bash
# 1. Start from develop
git checkout develop && git pull origin develop

# 2. Create task branch
git checkout -b task/N-short-name

# 3. Do the work (commit at every passing step)
# ... write test, commit ...
# ... write implementation, commit ...
# ... refactor, commit ...

# 4. Push task branch
git push -u origin task/N-short-name

# 5. Merge to develop
git checkout develop
git merge task/N-short-name
git push origin develop

# 6. Start next task from fresh develop
```

**Do not skip branches. Do not work directly on develop. Do not squash — keep the commit history.**
