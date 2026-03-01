# ABOUTME: Audit of the forked HMLR codebase — what exists, what we need to build.
# ABOUTME: Key finding: multi-provider LLM support and local embeddings already exist.

# HMLR Codebase Audit

## Key Finding: LLM Swap is Config, Not Surgery

HMLR already supports multiple LLM providers including Anthropic. The swap is environment variables, not code changes.

### Multi-Provider Support (Already Built)

**`hmlr/core/external_api_client.py`** — `ExternalAPIClient` class:
- `api_provider` parameter: `"openai"`, `"gemini"`, `"grok"`, `"anthropic"`
- `_call_anthropic_api()` — sync Anthropic API call (already implemented)
- `_call_anthropic_api_async()` — async Anthropic API call (already implemented)
- Uses `anthropic.Anthropic` and `anthropic.AsyncAnthropic` clients
- Handles system message separation (Anthropic requires it separate from messages array)
- Normalizes response to OpenAI-compatible format (all providers do this)

### Tiered Model Config (Already Built)

**`hmlr/core/model_config.py`** — `ModelConfig` class:

| Config | Env Var | Our Value | Purpose |
|--------|---------|-----------|---------|
| `DEFAULT_MODEL` | `HMLR_DEFAULT_MODEL` | `claude-sonnet-4-6` | Base model for all operations |
| `MAIN_MODEL` | `HMLR_MAIN_MODEL` | `claude-opus-4-6` | Governor (user-facing responses) |
| `NANO_MODEL` | `HMLR_NANO_MODEL` | `claude-sonnet-4-6` | Metadata extraction |
| `LATTICE_MODEL` | `HMLR_LATTICE_MODEL` | `claude-sonnet-4-6` | Topic classification |
| `SYNTHESIS_MODEL` | `HMLR_SYNTHESIS_MODEL` | `claude-opus-4-6` | Dossier synthesis, gardener |

### Local Embeddings (No API Needed)

**`hmlr/memory/embeddings/embedding_manager.py`** — Uses `sentence_transformers`:
- Model: `BAAI/bge-small-en-v1.5` (384 dimensions)
- Runs locally on CPU/GPU via PyTorch
- No OpenAI or external API calls for embeddings
- Stores embeddings as pickled numpy arrays in SQLite

### Config for Anthropic Setup

```bash
export API_PROVIDER="anthropic"
export ANTHROPIC_API_KEY="sk-ant-..."
export HMLR_DEFAULT_MODEL="claude-sonnet-4-6"
export HMLR_MAIN_MODEL="claude-opus-4-6"
export HMLR_SYNTHESIS_MODEL="claude-opus-4-6"
```

## What Needs to Be Built

### 1. MCP Server (hmlr/mcp_server.py)
Does not exist. Need to create a stdio MCP server wrapping `HMLRClient`.

### 2. Extended Thinking Support
The existing `_call_anthropic_api_async` doesn't pass extended thinking parameters.
For Opus Governor and Gardener, we need to add `thinking` parameter support.

Anthropic extended thinking API:
```python
response = client.messages.create(
    model="claude-opus-4-6",
    max_tokens=16000,
    thinking={
        "type": "enabled",
        "budget_tokens": 10000
    },
    messages=[...]
)
```

The `_call_anthropic_api_async` method needs a way to optionally enable this.

### 3. Session Reflection Ingestion
Custom tool: `mem_ingest_reflection` — takes session-reflection-analysis output, tags it appropriately, feeds through ingestion pipeline.

### 4. Gardener Cron Script
Wrapper script for cron that:
- Sets env vars
- Runs `uv run python -m hmlr.run_gardener`
- Logs output

### 5. Integration
- Update remember skill to reference MCP tools
- Add MCP server to `~/.claude/mcp.json`
- Hook session-reflection-analysis to feed into memory

## File Structure

```
hmlr/
├── __init__.py                        # Exports HMLRClient
├── client.py                          # Main client class
├── core/
│   ├── background_tasks.py            # Async task management
│   ├── component_factory.py           # Component initialization
│   ├── config.py                      # API provider config (API_PROVIDER env)
│   ├── conversation_engine.py         # Main conversation loop
│   ├── exceptions.py                  # Custom exceptions
│   ├── external_api_client.py         # LLM API calls (multi-provider!)
│   ├── model_config.py                # Model names, tokens, temps (tiered!)
│   ├── models/
│   │   ├── __init__.py
│   │   └── conversation_response.py   # Response models
│   └── prompts.py                     # System prompts for all components
├── integrations/
│   ├── __init__.py
│   └── langgraph/                     # LangGraph integration
│       ├── __init__.py
│       ├── client.py
│       ├── nodes.py
│       └── state.py
├── memory/
│   ├── __init__.py
│   ├── bridge_models/
│   │   ├── __init__.py
│   │   └── bridge_block.py            # Short-term memory blocks
│   ├── chunking/
│   │   ├── __init__.py
│   │   ├── chunk_engine.py            # Text chunking
│   │   └── chunk_storage.py           # Chunk persistence
│   ├── conversation_manager.py        # Conversation state
│   ├── dossier_storage.py             # Dossier persistence
│   ├── embeddings/
│   │   ├── __init__.py
│   │   └── embedding_manager.py       # LOCAL sentence-transformers (no API!)
│   ├── fact_scrubber.py               # Fact extraction
│   ├── gardener/
│   │   └── manual_gardener.py         # Memory gardening (short→long)
│   ├── id_generator.py                # ID generation
│   ├── metadata_extractor.py          # Metadata extraction
│   ├── models.py                      # Memory data models
│   ├── persistence/
│   │   ├── dossier_store.py           # Dossier DB operations
│   │   ├── ledger_store.py            # Ledger DB operations
│   │   └── schema.py                  # SQLite schema
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── context_assembler.py       # Context assembly
│   │   ├── context_hydrator.py        # Final prompt assembly
│   │   ├── crawler.py                 # Memory crawling
│   │   ├── dossier_retriever.py       # Dossier retrieval
│   │   ├── hmlr_hydrator.py           # HMLR-specific hydration
│   │   └── lattice.py                 # Vector search (LatticeCrawler)
│   ├── sliding_window.py              # Sliding window for context
│   ├── storage.py                     # Core SQLite storage
│   └── synthesis/
│       ├── __init__.py
│       ├── dossier_governor.py        # Dossier creation logic
│       ├── scribe.py                  # User profile updates
│       ├── synthesis_engine.py        # Main synthesis orchestration
│       └── user_profile_manager.py    # User profile management
└── run_gardener.py                    # Gardener entry point
```
