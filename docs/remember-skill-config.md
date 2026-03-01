# ABOUTME: Documents how the ~/.claude/skills/remember skill integrates with HMLR MCP.
# ABOUTME: Maps skill-level tool names to HMLR pipeline components and data flow.

# Remember Skill Integration with HMLR MCP Backend

The `remember` skill (at `~/.claude/skills/remember/skill.md`) teaches Claude Code how to
manage persistent memory. It references MCP tool names that are served by the `hmlr-memory`
MCP server defined in this repository.

## Tool-to-Pipeline Mapping

Each MCP tool exposed by `hmlr/mcp_server.py` maps to specific HMLR pipeline components:

| MCP Tool | Handler | HMLR Pipeline |
|----------|---------|---------------|
| `mem_search` | `_handle_mem_search` | HMLRClient.chat (force_intent=retrieval) -> LatticeCrawler vector search -> Governor filtering -> dossier enrichment via ContextHydrator |
| `mem_add` | `_handle_mem_add` | HMLRClient.chat -> ChunkEngine -> FactScrubber -> Scribe user profile update |
| `mem_add_file` | `_handle_mem_add_file` | File read -> same as mem_add pipeline |
| `mem_delete` | `_handle_mem_delete` | Direct SQL delete from daily_ledger (bridge block archive) |
| `mem_status` | `_handle_mem_status` | HMLRClient.get_memory_stats() |
| `mem_garden` | `_handle_mem_garden` | ManualGardener -> fact classification (3 heuristics) -> meta tagging -> semantic grouping -> DossierGovernor |
| `mem_ingest_reflection` | `_handle_mem_ingest_reflection` | HMLRClient.chat with auto-tags: session-reflection, learning, pattern |

## Data Flow

```
mem_add / mem_add_file
    |
    v
ChunkEngine (split + embed)
    |
    v
FactScrubber (key-value extraction)
    |
    v
Scribe (user profile update)
    |
    v
Bridge Block (short-term, topic-scoped)
    |
    v  [mem_garden or cron]
ManualGardener
    |-- Classify facts (environment, constraint, definition)
    |-- Apply sticky meta tags
    |-- Group remaining facts semantically
    |-- Route to DossierGovernor
    v
Dossier (long-term, synthesized)

mem_search
    |
    v
LatticeCrawler (vector similarity)
    |
    v
Governor (relevance filtering)
    |
    v
ContextHydrator (dossier enrichment)
    |
    v
Results
```

## Configuration

The MCP server is configured in `~/.claude/mcp.json`. See `docs/claude-code-setup.md`
for the full mcp.json entry and environment variable setup.

Key environment variables that affect memory operations:
- `HMLR_DB_PATH` -- SQLite database location (default: `~/.hmlr/memory.db`)
- `HMLR_DEFAULT_MODEL` -- Model for worker operations (ChunkEngine, FactScrubber)
- `HMLR_MAIN_MODEL` -- Model for Governor decisions
- `HMLR_SYNTHESIS_MODEL` -- Model for DossierGovernor synthesis
- `API_PROVIDER` -- Must be `anthropic` for this fork

## Skill File Location

The skill file lives outside this repository at:
```
~/.claude/skills/remember/skill.md
```

Changes to the skill file should be reflected in this document. Changes to the MCP
tool signatures in `hmlr/mcp_server.py` should be reflected in the skill file.
