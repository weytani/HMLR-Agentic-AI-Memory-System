# ABOUTME: Technical analysis of HMLR's architecture from the upstream repo.
# ABOUTME: Reference for understanding what we're forking and adapting.

# HMLR Architecture Analysis

## Pipeline Overview

```
User Query
  → ChunkEngine (chunk & embed)
  → Parallel Fan-Out:
      Task 1: Scribe Agent (fire-and-forget → update User Profile JSON)
      Task 2: FactScrubber (async → extract key-value facts → Fact Store SQL)
      Task 3: LatticeCrawler (retrieval → vector search → Raw Candidates)
      Task 4: Governor (waits for candidates → context filter → routing)
  → Governor produces: Validated Memories + Routing Decision
  → ContextHydrator assembles: Validated Memories + Fact Store + User Profile → Final Prompt
  → LLM Response Generation
```

## Component Details

### ChunkEngine
- Segments incoming messages into chunks
- Generates embeddings for each chunk
- Entry point for all ingestion

### Scribe Agent
- Fire-and-forget async task
- Updates the User Profile JSON with new information
- Extracts preferences, personal facts, stated constraints

### FactScrubber
- Extracts structured key-value facts from messages
- Stores in SQL (Fact Store)
- Key-value format enables natural overwrite on update

### LatticeCrawler
- Performs vector similarity search against stored memories
- Produces "Raw Candidates" — semantically similar but unfiltered
- This is "Key 1" of the dual-key retrieval

### Governor (The Brain)
- The most important component
- Receives raw candidates from LatticeCrawler
- Applies "Key 2" — context filtering:
  - Temporal ordering (newer supersedes older)
  - Policy supersession tracking
  - Cross-topic constraint persistence
  - Relevance validation
- Makes routing decisions:
  - Active topic → Resume existing Bridge Block
  - New topic → Create new Bridge Block
- Produces "Truly Relevant Memories"

### ContextHydrator
- Assembles the final prompt from:
  - Validated memories (from Governor)
  - Fact Store contents
  - User Profile
  - Current Bridge Block state
- Formats everything for LLM consumption

### Bridge Blocks
- Short-term conversation memory
- Represent active session/topic state
- Governor decides when to resume vs create new blocks
- Get promoted to long-term memory by the Gardener

## Dossier System (v0.1.2)

Dossiers are persistent, cross-topic summaries created during gardening:

- When `run_gardener.py` executes:
  1. Transfers bridge blocks → long-term memory
  2. Takes the day's facts and creates dossiers
  3. Dossiers persist across days and topics
- On new queries, system pulls dossiers AND long-term memories
- Enables reconstruction of causal chains from past into present
- Critical for the Hydra benchmark: all context comes from long-term retrieval

## Storage

| Store | Type | Contents |
|-------|------|----------|
| User Profile | JSON file | Preferences, personal facts, constraints |
| Fact Store | SQL (SQLite) | Structured key-value facts |
| Bridge Blocks | Database | Short-term conversation state |
| Long-Term Memory | Database | Gardened memories |
| Dossiers | Database | Cross-topic persistent summaries |
| Embeddings | Database | Vector representations for similarity search |

## Benchmark Capabilities

| Capability | How It Works |
|-----------|-------------|
| Temporal Conflict Resolution | Governor tracks policy versions, applies latest valid rule (handles reverts) |
| Multi-Hop Reasoning | Governor traces transitive identity chains across memories |
| Policy Enforcement | Persistent constraints survive topic changes and adversarial prompts |
| Zero-Keyword Semantic Recall | LatticeCrawler's vector search finds semantically similar content without keyword overlap |
| Constraint Persistence | User-stated invariants (e.g., "strict vegetarian") resist override attempts |

## Dependencies

- Python 3.10+
- OpenAI API (gpt-4.1-mini for LLM, embeddings for vectors)
- SQLite for storage
- LangGraph (optional, for agent integration)

## Key Files (Upstream)

```
hmlr/
  ├── integrations/
  │   └── langgraph/          # LangGraph drop-in component
  ├── (core modules)          # ChunkEngine, Governor, LatticeCrawler, etc.
  └── __init__.py             # Exports HMLRClient

examples/
  └── simple_agent.py         # LangGraph usage example
  └── simple_usage.py         # Direct API usage

tests/
  ├── ragas_test_7b_vegetarian.py    # Constraint test
  ├── test_12_hydra_e2e.py           # Full Hydra benchmark
  └── (other test files)

run_gardener.py               # Memory gardener script
main.py                       # Entry point
```
