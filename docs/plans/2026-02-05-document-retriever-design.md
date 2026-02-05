# Document Retriever — MVP Design

## 1. Overview

A containerized microservice that extracts structured data from uploaded documents
(PDF, DOCX, TXT, MD, HTML) into a JSON object conforming to a user-provided schema.
Uses Claude as an extraction agent with custom MCP tools that enforce a
search-then-read pattern — the model never sees the full document.

This is a first-test MVP. The architecture is minimal but cleanly extensible.

---

## 2. Architecture

Three Docker services orchestrated by docker-compose:

```
┌──────────────────┐    SSE/MCP     ┌──────────────────┐     HTTP      ┌──────────────────────┐
│ document-retriever│ ────────────→  │    mcp-tools     │ ───────────→  │ markitdown-vision-   │
│     (API)        │                │   (4 MCP tools)  │               │     service          │
│     :8000        │                │     :8001        │               │      :8002           │
└────────┬─────────┘                └────────┬─────────┘               └──────────┬───────────┘
         │                                   │                                    │
         └─────────── /data (shared) ────────┘                                    │
                                                                          /data/markitdown
```

| Service | Port | Responsibility |
|---------|------|----------------|
| `document-retriever` | 8000 | HTTP API, task queue, SQLite persistence, Claude agent (via `claude-agent-sdk`) |
| `mcp-tools` | 8001 | FastMCP SSE server exposing 4 document tools; chunking, indexing, search |
| `markitdown` | 8002 | Document-to-Markdown conversion (PDF, DOCX, HTML, TXT, MD) via MarkItDown |

### Data flow for a single task

1. Client POSTs file + schema to `document-retriever` at `:8000/tasks`.
2. API saves file to `/data/uploads/{task_id}/`, persists task in SQLite, returns `task_id`.
3. Background worker picks up task, creates Claude agent with MCP client pointing to `mcp-tools`.
4. Agent calls `doc_index_build(file_path=...)`.
   - `mcp-tools` POSTs file to `markitdown` service at `:8002/tasks`.
   - Polls until conversion completes, downloads `.md` output.
   - Parses markdown into chunks, computes IDF scores, caches index at `/data/cache/{sha256}/`.
   - Returns `index_id`, `toc`, `chunk_count`.
5. Agent iterates schema fields:
   - `doc_search(index_id, query="max DC voltage", top_k=5)` — keyword-scored search.
   - `doc_read(index_id, locator="h3-c2", max_chars=2000)` — reads matching chunk.
6. Agent builds `data` + `evidence` objects.
7. Agent calls `json_validate_and_normalize(data, schema)` — ensures all keys present, normalizes formatting.
8. Worker persists result to SQLite, marks task `succeeded`.
9. Client polls `GET /tasks/{task_id}`, then fetches `GET /tasks/{task_id}/result`.

### Shared volume layout

```
./data/
  tasks.db                          # SQLite (owned by document-retriever)
  uploads/{task_id}/original.ext    # uploaded files
  cache/{sha256}/                   # document indexes (owned by mcp-tools)
    index.json
    chunks.json
    idf.json
  markitdown/                       # markitdown-vision-service working dir
```

---

## 3. Repo Structure

```
document-retriever/
├── docker-compose.yml
├── README.md
├── .env.example
│
├── api/                              # Service: document-retriever
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                   # FastAPI app + lifespan (start worker)
│   │   ├── worker.py                 # Task queue + background processing
│   │   ├── store.py                  # SQLite task persistence (aiosqlite)
│   │   ├── agent.py                  # Claude Agent SDK client + system prompt
│   │   └── models.py                 # Pydantic request/response models
│   └── tests/
│       ├── __init__.py
│       ├── test_smoke.py             # API endpoint tests (mocked agent)
│       └── test_e2e.py               # Full end-to-end test with real PDF
│
├── mcp_tools/                        # Service: mcp-tools
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── server/
│   │   ├── __init__.py
│   │   ├── main.py                   # FastMCP SSE server entrypoint
│   │   ├── tools.py                  # 4 tool implementations
│   │   ├── indexer.py                # Markdown chunking, TF-IDF search
│   │   └── markitdown_client.py      # HTTP client for markitdown service
│   └── tests/
│       ├── __init__.py
│       └── test_tools.py             # Unit tests for indexer + tools
│
├── markitdown-vision-service/        # Git submodule (existing Growatt service)
│
├── scripts/
│   └── client.sh                     # Dockerized client (curl + python, no host deps)
│
├── fixtures/                         # Test fixtures
│   ├── MOD 8~15KTL3-X2(Pro).pdf
│   └── datasheet_schema.json
│
└── docs/
    └── plans/
        └── 2026-02-05-document-retriever-design.md
```

---

## 4. Service API

### POST /tasks

Multipart/form-data upload.

| Field | Type | Description |
|-------|------|-------------|
| `file` | file | Uploaded document (PDF, DOCX, TXT, MD, HTML) |
| `schema` | string | JSON string — the target schema skeleton |

Response `202`:
```json
{ "task_id": "01JKABC123..." }
```

### GET /tasks/{task_id}

```json
{
  "task_id": "01JKABC123...",
  "status": "queued|running|succeeded|failed",
  "created_at": "2026-02-05T12:00:00Z",
  "updated_at": "2026-02-05T12:00:05Z",
  "error": null
}
```

When `status` is `failed`:
```json
{
  "error": {
    "message": "Unsupported file format: .pptx",
    "details": ["Only PDF, DOCX, TXT, MD, HTML are supported"]
  }
}
```

### GET /tasks/{task_id}/result

When `succeeded`:
```json
{
  "task_id": "01JKABC123...",
  "data": { ... },
  "evidence": { ... },
  "warnings": ["Field 'thd' not found in document"],
  "timing": {
    "total_seconds": 42.3,
    "index_seconds": 8.1,
    "extraction_seconds": 34.2
  }
}
```

When `failed`:
```json
{
  "task_id": "01JKABC123...",
  "status": "failed",
  "error": { "message": "...", "details": [...] }
}
```

---

## 5. MCP Tools (FastMCP SSE Server)

The `mcp-tools` service exposes exactly 4 tools via MCP over SSE at `:8001/sse`.
No bash, no file write, no web access. Read-only document extraction only.

### Tool A: `doc_index_build`

**Args:**
```json
{ "file_path": "string", "cache_key": "string|null" }
```

**Returns:**
```json
{
  "index_id": "string (sha256 fingerprint)",
  "doc_fingerprint": "string",
  "doc_type": "pdf|docx|txt|md|html",
  "page_count": "int|null",
  "chunk_count": "int",
  "toc": [{"locator": "h3", "title": "PV DC Input"}],
  "notes": ["Converted via markitdown-vision-service"]
}
```

**Behavior:**
1. Compute sha256 of file bytes → `fingerprint`.
2. If `/data/cache/{fingerprint}/index.json` exists → return cached.
3. POST file to markitdown service (`http://markitdown:8002/tasks`).
4. Poll until completed, download `.md` output.
5. Parse markdown into chunks (see chunking strategy below).
6. Compute IDF scores across all chunks.
7. Write `index.json`, `chunks.json`, `idf.json` to `/data/cache/{fingerprint}/`.
8. Return index metadata.

### Tool B: `doc_search`

**Args:**
```json
{ "index_id": "string", "query": "string", "top_k": "int (default 5)", "scope": "text|tables|all" }
```

**Returns:**
```json
[
  {
    "locator": "h3-c2",
    "kind": "text",
    "snippet": "Max. DC voltage: 1100V...",
    "score": 0.82
  }
]
```

**Behavior:**
1. Load `chunks.json` and `idf.json` from cache.
2. Tokenize query into terms.
3. Score each chunk: `score = sum(tf(term, chunk) * idf(term))`.
4. Filter by `scope` (`text`, `tables`, or `all`).
5. Return `top_k` results sorted by score descending.
6. Each result includes a 200-char snippet.

### Tool C: `doc_read`

**Args:**
```json
{ "index_id": "string", "locator": "string", "max_chars": "int (default 3000)" }
```

**Returns:**
```json
{
  "locator": "h3-c2",
  "kind": "text",
  "content": "Full chunk content...",
  "meta": {
    "heading_path": "PV DC Input > Specifications",
    "original_heading": "PV DC Input",
    "chunk_index": 2
  }
}
```

**Behavior:**
1. Load `chunks.json` from cache.
2. Find chunk by locator (exact match).
3. Truncate content to `max_chars`.
4. Return chunk with metadata.

### Tool D: `json_validate_and_normalize`

**Args:**
```json
{ "data": "object", "schema": "object" }
```

**Returns:**
```json
{
  "ok": true,
  "errors": [],
  "warnings": ["Field 'thd' is null"],
  "normalized": { ... }
}
```

**Behavior:**
1. Walk schema skeleton recursively.
2. For each key in schema: if missing in data → add as `null`, record warning.
3. Normalize formatting: whitespace cleanup, range style (`1000~1500V` → `1000-1500V`).
4. For per-model dicts: ensure `dict` type, warn if flat value found.
5. Return validation result.

---

## 6. Chunking Strategy

The MCP tools service receives Markdown from the markitdown service and chunks it
into searchable units. All formats are handled uniformly since markitdown converts
everything to Markdown first.

### Locator scheme

| Pattern | Meaning | Example |
|---------|---------|---------|
| `h{n}` | Heading (section boundary) | `h3` = 3rd heading |
| `h{n}-c{m}` | Text chunk under heading | `h3-c2` = 2nd paragraph under 3rd heading |
| `h{n}-t{m}` | Table under heading | `h3-t1` = 1st table under 3rd heading |

### Chunk data structure

```python
@dataclass
class Chunk:
    locator: str        # stable ID
    kind: str           # "text" or "table"
    content: str        # raw text or markdown table
    meta: dict          # heading_path, original_heading, chunk_index
```

### Chunking rules

- Walk the markdown line by line. Track current heading stack.
- Each `##` / `###` / etc. heading starts a new section and gets an `h{n}` locator.
- Paragraphs (non-empty text blocks separated by blank lines) under a heading get `h{n}-c{m}`.
- Markdown tables (`| ... |` blocks) get `h{n}-t{m}`.
- Max chunk size: 1500 chars. If exceeded, split at sentence boundaries.
- Min chunk size: 50 chars. Merge tiny chunks with the next one.
- Tables are never split — one table = one chunk regardless of size.
- Every chunk stores its heading ancestry in `meta`.

### Search scoring (MVP)

TF-IDF keyword scoring:
- Tokenize: lowercase, split on non-alphanumeric, remove stopwords.
- TF: term frequency in chunk (count / total_terms).
- IDF: log(total_chunks / chunks_containing_term).
- Score: sum of TF * IDF for each query term.

Precomputed IDF values are cached in `idf.json` during indexing.

---

## 7. Claude Agent SDK Wiring

The `document-retriever` service uses `claude-agent-sdk` (v0.1.29+) to run the
extraction agent. The agent connects to the external `mcp-tools` SSE server.

### Agent configuration

```python
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions

options = ClaudeAgentOptions(
    system_prompt=SYSTEM_PROMPT,
    model="claude-sonnet-4-20250514",
    mcp_servers={
        "doc-tools": {
            "type": "sse",
            "url": "http://mcp-tools:8001/sse",
        }
    },
    allowed_tools=[
        "mcp__doc-tools__doc_index_build",
        "mcp__doc-tools__doc_search",
        "mcp__doc-tools__doc_read",
        "mcp__doc-tools__json_validate_and_normalize",
    ],
    max_turns=50,
)
```

### System prompt (embedded in code)

```
You are a document data extraction agent. You extract structured data from
documents using the provided tools. You MUST follow this protocol exactly:

STEP 1 — INDEX
Call doc_index_build with the provided file_path. Note the index_id,
chunk_count, and toc.

STEP 2 — EXTRACT FIELD BY FIELD
For each field in the target schema:
  a) Call doc_search with a query describing the field. Use top_k=5.
  b) Review the returned snippets and scores.
  c) For promising hits, call doc_read with the locator to get full content.
  d) Extract the value. Record the locator and exact snippet as evidence.

RULES:
- NEVER call doc_read without first calling doc_search for that field.
- NEVER guess or hallucinate values. If not found, set the value to null.
- For every extracted value, record evidence: { locator, snippet }.
- For per-model values (tables with multiple models), return a dict keyed
  by model name.
- For lists, return arrays.
- Preserve original units. Standardize formatting only (e.g. "1000V" not
  "1000 V" or "1kV").

STEP 3 — VALIDATE
Call json_validate_and_normalize with your extracted data and the target
schema. Fix any errors reported.

STEP 4 — RESPOND
Return exactly two JSON objects, nothing else:
{"data": { ... }, "evidence": { ... }}

The "evidence" object mirrors the "data" structure but each leaf value is:
{"locator": "h3-c2", "snippet": "exact text from document"}

If a field is null, its evidence should be:
{"locator": null, "snippet": null, "warning": "Not found in document"}

No prose. No explanation. Only the two JSON objects.
```

### Invocation (in worker)

```python
async def run_extraction(file_path: str, schema: dict) -> dict:
    prompt = f"Extract data from: {file_path}\nTarget schema:\n{json.dumps(schema, indent=2)}"

    async with ClaudeSDKClient(options=options) as client:
        await client.query(prompt)
        final_text = ""
        async for msg in client.receive_response():
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        final_text += block.text

    return parse_agent_output(final_text)
```

### Retry & timeout

- Agent invocation: 5-minute timeout.
- On failure: 1 retry with the same prompt.
- Individual tool calls: 60-second timeout (markitdown conversion may take longer;
  `doc_index_build` gets a 120-second timeout).

---

## 8. Background Execution Model

### Task lifecycle

```
POST /tasks → queued → running → succeeded
                               → failed
```

### Internal work queue

- `queue.Queue` with worker threads started on app startup (1 worker for MVP).
- Worker thread pulls `task_id`, runs async processing via `asyncio.run_coroutine_threadsafe`.
- Pattern matches the proven approach in markitdown-vision-service.

### SQLite persistence

- Database: `/data/tasks.db` via `aiosqlite`.
- Table: `tasks` with columns: `task_id`, `status`, `original_filename`, `schema_json`,
  `created_at`, `updated_at`, `started_at`, `finished_at`, `result_json`, `error_json`.
- All state changes are persisted before acknowledgment.
- Tasks survive container restarts (volume-mounted).

### File storage

- Uploaded files: `/data/uploads/{task_id}/original.{ext}`
- No automatic cleanup in MVP (add janitor later).

---

## 9. Docker Compose

```yaml
services:
  document-retriever:
    build: ./api
    ports: ["8000:8000"]
    environment:
      # LLM config — pass through all Anthropic-compatible env vars
      - ANTHROPIC_API_KEY
      - ANTHROPIC_AUTH_TOKEN
      - ANTHROPIC_BASE_URL
      - ANTHROPIC_DEFAULT_HAIKU_MODEL
      - ANTHROPIC_DEFAULT_SONNET_MODEL
      - ANTHROPIC_DEFAULT_OPUS_MODEL
      - API_TIMEOUT_MS
      # Service config
      - MCP_TOOLS_URL=http://mcp-tools:8001/sse
    volumes: ["./data:/data"]
    depends_on:
      mcp-tools:
        condition: service_healthy
    restart: unless-stopped

  mcp-tools:
    build: ./mcp_tools
    ports: ["8001:8001"]
    environment:
      - MARKITDOWN_URL=http://markitdown:8002
    volumes: ["./data:/data"]
    depends_on:
      markitdown:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8001/health')"]
      interval: 5s
      timeout: 5s
      retries: 3
    restart: unless-stopped

  markitdown:
    build:
      context: ./markitdown-vision-service
    ports: ["8002:8002"]
    volumes: ["./data/markitdown:/data"]
    environment:
      - PORT=8002
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8002/health"]
      interval: 5s
      timeout: 5s
      retries: 3
    restart: unless-stopped
```

### Build & run

```bash
# Clone markitdown-vision-service into repo
git submodule add https://github.com/Growatt-New-Energy-B-V/markitdown-vision-service.git

# Build all services
docker compose build

# Run (pass API key via .env or inline)
ANTHROPIC_API_KEY=sk-ant-... docker compose up
```

---

## 10. Client Script

`scripts/client.sh` — requires only Docker on the host.

```bash
#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:?Usage: client.sh <base_url> <file_path> <schema_path>}"
FILE_PATH="$(cd "$(dirname "${2:?}")" && pwd)/$(basename "$2")"
SCHEMA_PATH="$(cd "$(dirname "${3:?}")" && pwd)/$(basename "$3")"
FILE_NAME="$(basename "$FILE_PATH")"

# POST file + schema
RESPONSE=$(docker run --rm --network host \
  -v "$FILE_PATH:/upload/$FILE_NAME" \
  -v "$SCHEMA_PATH:/upload/schema.json" \
  curlimages/curl -s -X POST "$BASE_URL/tasks" \
    -F "file=@/upload/$FILE_NAME" \
    -F "schema=</upload/schema.json")

TASK_ID=$(echo "$RESPONSE" | docker run --rm -i python:3.11-alpine \
  python3 -c "import sys,json; print(json.load(sys.stdin)['task_id'])")

echo "Task created: $TASK_ID"
echo "Polling..."

while true; do
  sleep 10
  STATUS_JSON=$(docker run --rm --network host \
    curlimages/curl -s "$BASE_URL/tasks/$TASK_ID")

  STATUS=$(echo "$STATUS_JSON" | docker run --rm -i python:3.11-alpine \
    python3 -c "import sys,json; print(json.load(sys.stdin)['status'])")

  case "$STATUS" in
    succeeded)
      echo "--- RESULT ---"
      docker run --rm --network host \
        curlimages/curl -s "$BASE_URL/tasks/$TASK_ID/result" | \
        docker run --rm -i python:3.11-alpine \
        python3 -c "import sys,json; json.dump(json.load(sys.stdin),sys.stdout,indent=2)"
      echo ""
      exit 0
      ;;
    failed)
      echo "--- FAILED ---" >&2
      echo "$STATUS_JSON" >&2
      exit 1
      ;;
    *)
      echo "  status: $STATUS"
      ;;
  esac
done
```

Usage:
```bash
chmod +x scripts/client.sh
./scripts/client.sh http://localhost:8000 "./MOD 8~15KTL3-X2(Pro).pdf" ./datasheet_schema.json
```

---

## 11. Testing

### Unit tests: MCP tools (`mcp_tools/tests/test_tools.py`)

Run inside the `mcp-tools` container. No Claude API calls needed.

| Test | What it verifies |
|------|-----------------|
| `test_chunker_headings` | Feed known markdown with `##` headings → verify chunk locators are `h1-c1`, `h1-c2`, `h2-c1`, etc. |
| `test_chunker_tables` | Markdown with `\| col \|` tables → verify table chunks get `h{n}-t{m}` locators, `kind="table"` |
| `test_chunker_max_size` | Chunk > 1500 chars → verify split at sentence boundary |
| `test_chunker_min_merge` | Chunk < 50 chars → verify merged with next chunk |
| `test_search_ranking` | Index with known chunks, query "DC voltage" → verify top result contains "DC voltage" |
| `test_search_scope_filter` | Scope `tables` → verify only table chunks returned |
| `test_read_locator` | Read by locator → verify exact chunk content returned |
| `test_read_max_chars` | `max_chars=100` → verify truncation |
| `test_validate_missing_keys` | Data missing schema keys → verify warnings list includes them, `normalized` has nulls |
| `test_validate_type_mismatch` | String where dict expected → verify error reported |

### Smoke tests: API (`api/tests/test_smoke.py`)

Run against the API with the agent mocked out.

| Test | What it verifies |
|------|-----------------|
| `test_post_task` | POST multipart with file + schema → 202 with `task_id` |
| `test_get_task_status` | GET `/tasks/{id}` → valid status response |
| `test_get_task_not_found` | GET `/tasks/bad-id` → 404 |
| `test_result_structure` | Mock agent returns known JSON → GET `/tasks/{id}/result` has `data`, `evidence`, `warnings`, `timing` |
| `test_all_schema_keys_present` | Mock agent returns partial data → verify all schema keys exist in result (nulls for missing) |
| `test_unsupported_format` | POST a `.pptx` file → task fails with clear error message |

### End-to-end test: Real extraction (`api/tests/test_e2e.py`)

Full pipeline test using the actual PDF and schema. Requires all 3 services running
and a valid `ANTHROPIC_API_KEY`.

```
Test: test_e2e_datasheet_extraction

1. Read fixtures/MOD 8~15KTL3-X2(Pro).pdf and fixtures/datasheet_schema.json
2. POST /tasks with multipart upload (file + schema content)
3. Assert response is 202, parse task_id
4. Poll GET /tasks/{task_id} every 5 seconds, max 5 minutes
5. Assert final status is "succeeded"
6. GET /tasks/{task_id}/result
7. Validate result structure:
   a) result["data"] is a dict
   b) result["evidence"] is a dict
   c) result["warnings"] is a list
   d) result["timing"] is a dict with "total_seconds" > 0
8. Validate schema coverage:
   a) All top-level keys from schema exist in result["data"]:
      - product_identity, main_application, pv_dc_input, ac_grid_output,
        battery_interface, efficiency, protection_and_safety,
        monitoring_and_communication, environment_and_installation,
        compliance_and_certifications, key_marketing_features
   b) For each top-level key, all nested keys exist (may be null)
9. Validate known ground-truth values (spot checks):
   a) product_identity.series_name contains "MOD" and "KTL3"
   b) product_identity.models is a list with length > 1
   c) pv_dc_input.max_dc_voltage contains "1100" (known from datasheet)
   d) pv_dc_input.number_of_mppts is 2 (known from datasheet)
   e) ac_grid_output.grid_type contains "3" (three-phase)
   f) environment_and_installation.protection_class contains "IP66" or "IP65"
10. Validate evidence:
    a) For every non-null leaf in result["data"], a corresponding entry
       exists in result["evidence"] with non-null locator and snippet
    b) Evidence snippets are non-empty strings
    c) Evidence locators match the h{n}-c{m} or h{n}-t{m} pattern
11. Print summary: extracted fields count, null fields count, warnings count
```

Run command:
```bash
docker compose run --rm -e ANTHROPIC_API_KEY document-retriever \
  pytest tests/test_e2e.py -v --timeout=300
```

---

## 12. Output Contract

### Result JSON structure

```json
{
  "task_id": "01JKABC123...",
  "data": {
    "original_file": "MOD 8~15KTL3-X2(Pro).pdf",
    "product_identity": {
      "product_type": "Three-phase hybrid solar inverter",
      "series_name": "MOD 8~15KTL3-X2(Pro)",
      "models": ["MOD 8KTL3-X2(Pro)", "MOD 10KTL3-X2(Pro)", "MOD 12KTL3-X2(Pro)", "MOD 15KTL3-X2(Pro)"]
    },
    "pv_dc_input": {
      "max_dc_voltage": "1100V",
      "number_of_mppts": 2,
      "max_recommended_pv_power_per_model": {
        "MOD 8KTL3-X2(Pro)": "12000W",
        "MOD 10KTL3-X2(Pro)": "15000W"
      }
    }
  },
  "evidence": {
    "product_identity": {
      "product_type": {"locator": "h1-c1", "snippet": "Three-Phase Hybrid Solar Inverter"},
      "series_name": {"locator": "h1-c1", "snippet": "MOD 8~15KTL3-X2(Pro)"},
      "models": {"locator": "h2-t1", "snippet": "MOD 8KTL3-X2(Pro) | MOD 10KTL3-X2(Pro) | ..."}
    },
    "pv_dc_input": {
      "max_dc_voltage": {"locator": "h3-t1", "snippet": "Max. DC Voltage: 1100V"},
      "number_of_mppts": {"locator": "h3-t1", "snippet": "No. of MPP Trackers: 2"}
    }
  },
  "warnings": [
    "Field 'thd' not found in document"
  ],
  "timing": {
    "total_seconds": 42.3,
    "index_seconds": 8.1,
    "extraction_seconds": 34.2
  }
}
```

### Guarantees

- Every key from the input schema exists in `data` (null if not found).
- Every non-null value in `data` has a corresponding entry in `evidence`.
- Evidence entries always include `locator` and `snippet`.
- Null values have evidence: `{"locator": null, "snippet": null, "warning": "..."}`.
- `warnings` lists all fields that could not be extracted.

---

## 13. Dependencies per Service

### document-retriever (`api/requirements.txt`)

```
fastapi
uvicorn[standard]
aiosqlite
claude-agent-sdk>=0.1.29
python-multipart
ulid-py
pydantic
pydantic-settings
httpx
pytest
pytest-asyncio
pytest-timeout
```

### mcp-tools (`mcp_tools/requirements.txt`)

```
mcp>=1.0
httpx
fastapi
uvicorn[standard]
```

### markitdown (existing — no changes)

Uses its own `Dockerfile` and `pyproject.toml` from the submodule.

---

## 14. Configuration

All configuration via environment variables.

### LLM Configuration

The service supports any Anthropic-compatible API endpoint (direct Anthropic,
proxy services, or alternative providers that expose an Anthropic-compatible API).
All Claude Agent SDK env vars are passed through to the CLI.

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | (none) | Standard Anthropic API key |
| `ANTHROPIC_AUTH_TOKEN` | (none) | Alternative auth token (e.g. for proxy APIs) |
| `ANTHROPIC_BASE_URL` | (none) | Custom API endpoint (e.g. `https://api.z.ai/api/anthropic`) |
| `ANTHROPIC_DEFAULT_SONNET_MODEL` | (none) | Override model name for sonnet-tier requests |
| `ANTHROPIC_DEFAULT_HAIKU_MODEL` | (none) | Override model name for haiku-tier requests |
| `ANTHROPIC_DEFAULT_OPUS_MODEL` | (none) | Override model name for opus-tier requests |
| `API_TIMEOUT_MS` | (none) | API call timeout in milliseconds |

**Either `ANTHROPIC_API_KEY` or `ANTHROPIC_AUTH_TOKEN` must be set.** All other
LLM variables are optional and only needed for proxy/alternative provider setups.

### Service Configuration

| Variable | Service | Default | Description |
|----------|---------|---------|-------------|
| `MCP_TOOLS_URL` | document-retriever | `http://mcp-tools:8001/sse` | MCP tools SSE endpoint |
| `MARKITDOWN_URL` | mcp-tools | `http://markitdown:8002` | Markitdown service base URL |
| `DATA_DIR` | both | `/data` | Shared data directory |
| `DB_PATH` | document-retriever | `/data/tasks.db` | SQLite database path |
| `MAX_CONCURRENT_TASKS` | document-retriever | `1` | Worker thread count |
| `AGENT_TIMEOUT` | document-retriever | `300` | Agent timeout in seconds |
| `AGENT_MAX_RETRIES` | document-retriever | `1` | Retries on agent failure |

---

## 15. Implementation Order

1. **markitdown submodule** — clone, verify `docker compose up markitdown` works, test `/health`.
2. **mcp-tools service** — implement `indexer.py` + `tools.py` + `markitdown_client.py`, write unit tests, verify SSE server starts.
3. **document-retriever API** — implement `store.py` + `models.py` + `main.py` + `worker.py`, write smoke tests with mocked agent.
4. **Agent wiring** — implement `agent.py`, connect to MCP tools, verify tool calls work.
5. **Integration** — `docker compose up`, run client.sh with test PDF.
6. **E2E test** — run `test_e2e.py` against real PDF + schema, validate ground-truth values.
7. **Client script** — finalize `client.sh`, test from host.
