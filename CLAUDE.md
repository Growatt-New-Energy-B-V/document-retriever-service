# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Document-retriever-service extracts structured JSON from uploaded documents (PDF, DOCX, TXT, MD, HTML) using Claude as an extraction agent with custom MCP tools. The agent uses a **search-then-read pattern** — it never sees the full document. Instead it calls `doc_search` to find relevant chunks, then `doc_read` to read only those chunks.

## Build and Run Commands

```bash
# Initialize submodules (required after clone)
git submodule update --init --recursive

# Build all services
docker compose build

# Run all services (requires .env with ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN)
docker compose up

# Run extraction with client script (no host Python required)
./scripts/client.sh http://localhost:8000 "path/to/document.pdf" path/to/schema.json
```

## Testing Commands

```bash
# Unit tests (indexer/chunker, no API key needed)
docker run --rm -v "$(pwd)/mcp_tools:/app" -w /app python:3.11-slim \
  sh -c "pip install -q pytest && python -m pytest tests/ -v"

# API smoke tests (mocked agent, no API key needed)
docker run --rm -v "$(pwd)/api:/app" -w /app python:3.11-slim \
  sh -c "pip install -q fastapi uvicorn aiosqlite ulid-py python-multipart httpx pydantic pydantic-settings pytest 'pytest-asyncio<1.0' && python -m pytest tests/test_smoke.py -v"

# E2E test (requires all services running + API key)
docker compose run --rm -e ANTHROPIC_API_KEY document-retriever-service \
  pytest tests/test_e2e.py -v --timeout=300

# Run single test file (example for indexer)
docker run --rm -v "$(pwd)/mcp_tools:/app" -w /app python:3.11-slim \
  sh -c "pip install -q pytest && python -m pytest tests/test_indexer.py::test_chunker_tables -v"
```

## Architecture

Three Docker services sharing `/data` volume:

| Service | Port | Role |
|---------|------|------|
| `document-retriever-service` | 8000 | FastAPI, SQLite task queue, Claude agent via `claude-agent-sdk` |
| `mcp-tools` | 8001 | FastMCP SSE server with 4 document tools |
| `markitdown` | 8002 | Document-to-Markdown conversion (git submodule) |

### Data Flow

1. Client POSTs file + schema to `/tasks` → returns `task_id`
2. Background worker runs Claude agent connected to MCP tools
3. Agent calls `doc_index_build` → markitdown converts to MD → indexer chunks and builds TF-IDF index
4. Agent iterates schema: `doc_search` → `doc_read` → extract values
5. Agent calls `json_validate_and_normalize` → client polls for result

### Key Directories

- `api/` — document-retriever-service (FastAPI + worker + agent)
- `mcp_tools/` — MCP tools service (FastMCP + indexer + markitdown client)
- `markitdown-vision-service/` — git submodule for document conversion
- `data/` — shared volume (tasks.db, uploads/, cache/)

## MCP Tools

Four tools exposed at `http://mcp-tools:8001/sse`:

- `doc_index_build(file_path)` — converts doc via markitdown, chunks markdown, builds TF-IDF index
- `doc_search(index_id, query, top_k, scope)` — keyword search returning locators + snippets
- `doc_read(index_id, locator, max_chars)` — reads specific chunk by locator
- `json_validate_and_normalize(data, schema)` — validates extraction against schema

### Chunk Locator Scheme

| Pattern | Meaning |
|---------|---------|
| `h{n}` | n-th heading in document |
| `h{n}-c{m}` | m-th text chunk under heading n |
| `h{n}-t{m}` | m-th table under heading n |

## Agent Integration

The extraction agent (`api/app/agent.py`) uses `claude-agent-sdk` with:
- System prompt enforcing search-then-read protocol
- 4 allowed MCP tools (doc_index_build, doc_search, doc_read, json_validate_and_normalize)
- max_turns=50, permission_mode="acceptEdits"

Agent output format: two JSON objects `{"data": {...}, "evidence": {...}}`

## Configuration

All via environment variables. Either `ANTHROPIC_API_KEY` or `ANTHROPIC_AUTH_TOKEN` required.

Key service config:
- `MCP_TOOLS_URL` — MCP endpoint (default: `http://mcp-tools:8001/sse`)
- `MARKITDOWN_URL` — markitdown service (default: `http://markitdown:8000`)
- `DATA_DIR` — shared data directory (default: `/data`)
