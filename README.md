# document-retriever-service

Extract structured data from uploaded documents (PDF, DOCX, TXT, MD, HTML) into
JSON using Claude as an extraction agent with custom MCP tools.

## Quick Start

### Prerequisites

- Docker + Docker Compose
- Anthropic API key

### 1. Clone and initialize

```bash
git clone <this-repo>
cd document-retriever-service
git submodule update --init --recursive
```

### 2. Set your API key

```bash
cp .env.example .env
# Edit .env — set EITHER direct Anthropic key OR proxy configuration
```

**Option A: Direct Anthropic API**
```bash
echo "ANTHROPIC_API_KEY=sk-ant-your-key-here" >> .env
```

**Option B: Proxy / alternative provider**
```bash
cat >> .env << 'EOF'
ANTHROPIC_AUTH_TOKEN=your-token-here
ANTHROPIC_BASE_URL=https://api.z.ai/api/anthropic
ANTHROPIC_DEFAULT_SONNET_MODEL=glm-4.7
ANTHROPIC_DEFAULT_HAIKU_MODEL=glm-4.7
ANTHROPIC_DEFAULT_OPUS_MODEL=glm-4.7
API_TIMEOUT_MS=3000000
EOF
```

### 3. Build all services

```bash
docker compose build
```

### 4. Run

```bash
docker compose up
```

Wait for all 3 services to report healthy:
- `document-retriever-service` on http://localhost:8000
- `mcp-tools` on http://localhost:8001
- `markitdown` on http://localhost:8002

### 5. Extract data from a document

Using the client script (no host Python required):

```bash
./scripts/client.sh http://localhost:8000 \
  "fixtures/MOD 8~15KTL3-X2(Pro).pdf" \
  fixtures/datasheet_schema.json
```

Or with curl directly:

```bash
# Create task
curl -s -X POST http://localhost:8000/tasks \
  -F "file=@fixtures/MOD 8~15KTL3-X2(Pro).pdf" \
  -F "schema=<fixtures/datasheet_schema.json"

# Poll status (replace TASK_ID)
curl -s http://localhost:8000/tasks/TASK_ID

# Get result
curl -s http://localhost:8000/tasks/TASK_ID/result | python3 -m json.tool
```

## Architecture

Three services:

| Service | Port | Role |
|---------|------|------|
| `document-retriever-service` | 8000 | API + task queue + Claude agent |
| `mcp-tools` | 8001 | MCP server with 4 document tools |
| `markitdown` | 8002 | Document -> Markdown conversion |

The agent uses a **search-then-read** pattern: it never sees the full document.
Instead it calls `doc_search` to find relevant chunks, then `doc_read` to read
only those chunks.

## API

### POST /tasks -- Create extraction task
### GET /tasks/{task_id} -- Check status
### GET /tasks/{task_id}/result -- Get extraction result

See [design doc](docs/plans/2026-02-05-document-retriever-design.md) for details.

## Testing

```bash
# Unit tests (indexer, no API key needed)
docker run --rm -v "$(pwd)/mcp_tools:/app" -w /app python:3.11-slim \
  sh -c "pip install -q pytest && python -m pytest tests/ -v"

# API smoke tests (no API key needed)
docker run --rm -v "$(pwd)/api:/app" -w /app python:3.11-slim \
  sh -c "pip install -q fastapi uvicorn aiosqlite ulid-py python-multipart httpx pydantic pydantic-settings pytest 'pytest-asyncio<1.0' && python -m pytest tests/test_smoke.py -v"

# E2E test (requires all services running + API key)
docker compose run --rm -e ANTHROPIC_API_KEY document-retriever-service \
  pytest tests/test_e2e.py -v --timeout=300
```
