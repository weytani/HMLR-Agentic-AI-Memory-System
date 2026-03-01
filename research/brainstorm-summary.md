# ABOUTME: Summary of brainstorming session for adopting HMLR as our memory system.
# ABOUTME: Captures decisions, architecture direction, and design rationale.

# HMLR Memory System — Brainstorm Summary

## Problem Statement

Three interrelated problems with the current Claude Code setup:

1. **Cross-session persistence** — Context lost between sessions. Preferences, decisions, gotchas, project state don't carry over.
2. **Knowledge retrieval** — Information scattered across Roam, conversations, files. Claude can't find and use it when relevant.
3. **Agent self-improvement** — No mechanism for Claude to learn from past sessions — what worked, what didn't, patterns to repeat or avoid.

## Current State

### Existing Systems (None Fully Connected)

| System | Status | What It Does | Gap |
|--------|--------|-------------|-----|
| Remember Skill (`~/.claude/skills/remember/`) | Protocol only | Defines `mem_search`, `mem_add_text`, etc. | No backend implementation |
| Conversation Search (clank `remembering-conversations`) | Working | Auto-indexes session transcripts into SQLite with embeddings | Session-level, not fact-level |
| Roam MCP | Working | External knowledge graph | Not wired into agent memory |

### Key Insight

The remember skill defines a clean interface, but the backend doesn't exist. HMLR provides the backend architecture. An MCP server bridges them.

## Architecture Decisions

### Direction: Fork & Adapt HMLR (Approach A)

- Clone HMLR, swap LLM from `gpt-4.1-mini` to Anthropic models
- Wrap in an MCP server for Claude Code integration
- Stay as close to HMLR 1:1 as possible — trust their proven architecture

### Why Not Other Approaches

- **Clean-room rewrite (B):** More work, risk of diverging from HMLR's proven behavior
- **HMLR as library (C):** Dual LLM bills, no customization, black-box dependency

### Model Strategy: Tiered

| Component | Model | Rationale |
|-----------|-------|-----------|
| Governor | Opus 4.6 + extended thinking | Reasoning-heavy: temporal conflicts, multi-hop, routing. Worth the cost. |
| Gardener | Opus 4.6 + extended thinking | Runs once daily via cron. Quality over speed for dossier creation. |
| FactScrubber | Sonnet 4.6 | Extraction task. Structured, not reasoning-intensive. |
| Scribe | Sonnet 4.6 | Profile updates. Structured output. |
| ContextHydrator | Sonnet 4.6 | Assembly/formatting. Mechanical. |
| Embeddings | OpenAI `text-embedding-3-small` | Commodity. Proven with HMLR. Least-risk swap. |

### Pipeline Stays Unified

Harper Reed's approach (separate MCP servers per concern) solves a deployment/maintenance problem. HMLR's unified pipeline is what produces high benchmark scores. The Governor needs to cross-reference facts, profile, and conversation history during retrieval.

**Decision:** One HMLR-powered MCP server as the core memory brain. Existing services (Roam, conversation archive) stay independent.

### Staleness Handling

HMLR handles temporal conflicts well (Governor resolves contradictions, policy supersession). The gardener promotes short-term → long-term + creates dossiers. Currently manual (`run_gardener.py`), no TTL or decay.

**Decision:** Trust HMLR's approach. Automate the gardener via cron (`uv run python gardener.py`).

## MCP Server Design

### Tools Exposed

| Tool | Maps To | Description |
|------|---------|-------------|
| `mem_search` | HMLR retrieval pipeline | Query → ChunkEngine → LatticeCrawler → Governor (Opus) → results |
| `mem_add` | HMLR ingestion pipeline | Text + tags → ChunkEngine → FactScrubber + Scribe (Sonnet) → stored |
| `mem_add_file` | Read file + ingest | Path → read → same ingestion pipeline |
| `mem_link` | Relation creation | Connect two memories with a typed relation |
| `mem_delete` | Remove memory | Explicit deletion by ID |
| `mem_status` | Stats query | Memory counts, bridge blocks, dossiers, last gardener run |
| `mem_garden` | Trigger gardener | Manual gardener invocation (vs waiting for cron) |
| `mem_ingest_reflection` | Reflection → ingestion | Session-reflection-analysis output ingested as tagged memories |

### MCP Config

```json
{
  "mcpServers": {
    "hmlr-memory": {
      "type": "stdio",
      "command": "uv",
      "args": ["run", "--directory", "~/code/hmlr-memory", "python", "-m", "hmlr.mcp_server"],
      "env": {
        "ANTHROPIC_API_KEY": "...",
        "OPENAI_API_KEY": "...",
        "HMLR_DB_PATH": "~/.hmlr/memory.db"
      }
    }
  }
}
```

### Storage Layout

```
~/.hmlr/
  ├── memory.db           # SQLite: facts, bridge blocks, long-term, vectors
  ├── user_profile.json   # Scribe-maintained agent profile
  └── config.json         # Model tiers, gardener schedule
```

## Session Reflection Integration

- Uses **session-reflection-analysis** skill (not session-reflection)
- After each session, reflection analysis output feeds into HMLR via `mem_ingest_reflection`
- FactScrubber extracts structured learnings
- Scribe updates agent profile with session patterns
- Gardener promotes session learnings to long-term memory + dossiers
- Dossiers accumulate cross-session wisdom over time

## Gardener Automation

- Direct Python script via cron: `uv run python gardener.py`
- No Claude Code overhead — just HMLR logic + Anthropic API calls (Opus for gardener)
- Runs nightly (or user's preferred schedule)
- Promotes bridge blocks → long-term memory
- Creates/updates dossiers

## Implementation Scope

### Core Work (Fork Adaptation)

1. Replace OpenAI LLM calls → Anthropic API (Sonnet 4.6 / Opus 4.6 per tier)
2. Update prompt formats for Anthropic message schema (tool_use blocks)
3. Build MCP server wrapper using `mcp` Python SDK
4. Add `mem_ingest_reflection` tool
5. Gardener cron script

### Integration Work

6. Update remember skill to reference MCP tools
7. Wire session-reflection-analysis output to `mem_ingest_reflection`
8. Add MCP server to `~/.claude/mcp.json`

### What We Don't Touch

- HMLR's pipeline architecture (ChunkEngine → LatticeCrawler → Governor → ContextHydrator)
- SQLite schema
- Dossier system logic
- Bridge block management
- Embedding approach (keep OpenAI `text-embedding-3-small`)
