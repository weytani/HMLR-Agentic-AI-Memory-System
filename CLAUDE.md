# HMLR Memory — MCP Server Fork

> **Baja Blastoise** (that's me) and **Lil Harp Nasty** (that's David) are turning HMLR into a memory backend for Claude Code.

## Project Overview

HMLR (Hierarchical Memory Lookup & Routing) is a structured long-term memory system for AI agents. This fork wraps it in an MCP server so Claude Code can use it for persistent cross-session memory, knowledge retrieval, and session reflection ingestion.

**Upstream:** `Sean-V-Dev/HMLR-Agentic-AI-Memory-System` (original, OpenAI-only)
**Fork goal:** Anthropic provider (Opus for Governor/Gardener, Sonnet for workers), MCP server interface, local embeddings.

## Architecture

```
User Query → ChunkEngine → [Scribe, FactScrubber, LatticeCrawler, Governor] → ContextHydrator → LLM Response
```

Key components:
- **ChunkEngine** (`hmlr/memory/chunking/`) — splits input into chunks + embeddings
- **Governor** (`hmlr/core/conversation_engine.py`) — router, filter, brain
- **FactScrubber** (`hmlr/memory/fact_scrubber.py`) — key-value fact extraction
- **Scribe** (`hmlr/memory/synthesis/scribe.py`) — user profile updates
- **LatticeCrawler** (`hmlr/memory/retrieval/crawler.py`) — vector search
- **Gardener** (`hmlr/memory/gardener/`) — promotes bridge blocks → long-term memory + dossiers
- **DossierGovernor** (`hmlr/memory/synthesis/dossier_governor.py`) — synthesizes dossiers from facts
- **ExternalAPIClient** (`hmlr/core/external_api_client.py`) — multi-provider LLM client
- **ModelConfig** (`hmlr/core/model_config.py`) — centralized model/token/temp config via env vars
- **HMLRClient** (`hmlr/client.py`) — public API wrapper

## Tech Stack

- Python 3.12, uv (NO pip/poetry)
- `anthropic` SDK, `mcp` Python SDK
- `sentence-transformers` (local embeddings, BAAI/bge-small-en-v1.5)
- SQLite for storage
- hatchling build backend

## Development

```bash
cd ~/code/hmlr-memory
uv sync                              # install deps
uv run pytest tests/ -v --tb=short   # run tests
uv run python main.py                # interactive console
```

## Git Workflow

- **Default branch:** `master` (upstream)
- **Integration branch:** `develop` (our work)
- **Every task gets its own branch** off `develop`: `task/N-short-name`
- Conventional commits: `feat:`, `build:`, `config:`, `test:`, `docs:`
- Never work directly on `develop`
- Never squash — keep commit history

## Plan & Progress

Full plan: `docs/plans/2026-02-28-hmlr-mcp-server.md`
Reflections: `docs/reflections/`
Research: `research/`

### Completed
- Task 1: Convert to uv, add anthropic/mcp deps, remove legacy packaging

### Remaining (in order)
- Task 2: Extended thinking support for Anthropic Opus
- Task 3: Environment configuration (.env.example)
- Task 4: MCP server (core deliverable)
- Task 5: Gardener cron script
- Task 6: Wire MCP into Claude Code config
- Task 7: Update remember skill
- Task 8: Integration tests
- Task 9: Session reflection hook
- Task 10: Final code review + finishing

## Key Patterns

- All API responses normalized to OpenAI format (`choices[0].message.content`)
- Config via env vars with hierarchical fallback (operation-specific → DEFAULT)
- Bridge blocks = short-term topic-scoped memory containers
- Dossiers = long-term synthesized knowledge (created by gardener)
- `ComponentFactory` wires everything together

## Execution Method

Using **subagent-driven-development**: fresh subagent per task, code review after every task. Tighter quality gates for a forked codebase.

## Environment Variables

See `docs/plans/2026-02-28-hmlr-mcp-server.md` Task 3 for full list. Key ones:
- `API_PROVIDER=anthropic`
- `ANTHROPIC_API_KEY=sk-ant-...`
- `HMLR_DEFAULT_MODEL`, `HMLR_MAIN_MODEL`, `HMLR_SYNTHESIS_MODEL`
- `HMLR_THINKING_BUDGET`, `HMLR_MAIN_USES_THINKING`
- `COGNITIVE_LATTICE_DB` (SQLite path)
