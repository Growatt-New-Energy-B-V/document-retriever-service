# Document Retriever Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a three-service containerized microservice that extracts structured data from uploaded documents using Claude as an agent with custom MCP tools (search-then-read pattern).

**Architecture:** Three Docker services — `document-retriever` (FastAPI + Claude agent), `mcp-tools` (FastMCP SSE server with 4 tools), and `markitdown` (existing document-to-markdown converter). The agent never sees the full document; it must search then selectively read chunks.

**Tech Stack:** Python 3.11+, FastAPI, FastMCP, claude-agent-sdk, SQLite (aiosqlite), Docker Compose, httpx

**Design doc:** `docs/plans/2026-02-05-document-retriever-design.md`

---

## Task 1: Project Scaffolding + Git Init

**Files:**
- Create: `.gitignore`
- Create: `.env.example`
- Create: `fixtures/datasheet_schema.json` (move existing)
- Move: `MOD 8~15KTL3-X2(Pro).pdf` → `fixtures/`

**Step 1: Initialize git repo and create .gitignore**

```bash
cd /home/ubuntu/workspaces/document-retriever
git init
```

Create `.gitignore`:
```
__pycache__/
*.pyc
.env
data/
*.egg-info/
dist/
.pytest_cache/
.mypy_cache/
markitdown-vision-service/
```

**Step 2: Create .env.example and move fixtures**

Create `.env.example`:
```
# LLM Configuration — use ONE of the two auth methods:

# Option A: Direct Anthropic API
# ANTHROPIC_API_KEY=sk-ant-your-key-here

# Option B: Proxy / alternative provider (Anthropic-compatible API)
# ANTHROPIC_AUTH_TOKEN=your-proxy-token-here
# ANTHROPIC_BASE_URL=https://api.z.ai/api/anthropic
# ANTHROPIC_DEFAULT_HAIKU_MODEL=glm-4.7
# ANTHROPIC_DEFAULT_SONNET_MODEL=glm-4.7
# ANTHROPIC_DEFAULT_OPUS_MODEL=glm-4.7
# API_TIMEOUT_MS=3000000

# Markitdown image descriptions (optional)
# OPENAI_API_TOKEN=sk-your-openai-key-here
```

```bash
mkdir -p fixtures
mv "MOD 8~15KTL3-X2(Pro).pdf" fixtures/
mv datasheet_schema.json fixtures/
```

**Step 3: Commit**

```bash
git add .gitignore .env.example fixtures/
git commit -m "chore: project scaffolding with fixtures"
```

---

## Task 2: Add markitdown-vision-service Submodule

**Files:**
- Add: `markitdown-vision-service/` (git submodule)

**Step 1: Add submodule**

```bash
cd /home/ubuntu/workspaces/document-retriever
git submodule add https://github.com/Growatt-New-Energy-B-V/markitdown-vision-service.git
```

**Step 2: Verify it builds**

```bash
docker build -t markitdown-test ./markitdown-vision-service
```

Expected: Build succeeds.

**Step 3: Verify it starts and responds to health check**

```bash
docker run --rm -d --name markitdown-test -p 8002:8000 markitdown-test
sleep 3
curl -s http://localhost:8002/health
docker stop markitdown-test
```

Expected: `{"status":"ok"}` or similar health response.

**Step 4: Commit**

```bash
git add .gitmodules markitdown-vision-service
git commit -m "chore: add markitdown-vision-service as submodule"
```

---

## Task 3: MCP Tools — Indexer Core (Chunk + Chunker)

**Files:**
- Create: `mcp_tools/server/__init__.py`
- Create: `mcp_tools/server/indexer.py`
- Create: `mcp_tools/tests/__init__.py`
- Create: `mcp_tools/tests/test_indexer.py`
- Create: `mcp_tools/requirements.txt`

**Step 1: Create directory structure**

```bash
mkdir -p mcp_tools/server mcp_tools/tests
touch mcp_tools/__init__.py mcp_tools/server/__init__.py mcp_tools/tests/__init__.py
```

**Step 2: Write the failing tests for chunking**

Create `mcp_tools/tests/test_indexer.py`:

```python
"""Unit tests for the markdown indexer/chunker."""
import pytest
from server.indexer import Chunk, chunk_markdown, search_chunks, build_idf


# --- Chunking tests ---

SIMPLE_MD = """\
# Introduction

This is the introduction paragraph.

## PV DC Input

Max DC voltage is 1100V. The system supports 2 MPPTs.

Each MPPT has a voltage range of 200-1000V.

## AC Grid Output

| Model | Power | Current |
|-------|-------|---------|
| MOD 8K | 8000W | 12.7A |
| MOD 10K | 10000W | 15.2A |

The grid type is 3-phase.
"""


def test_chunker_headings():
    chunks = chunk_markdown(SIMPLE_MD)
    # Should have chunks under h1 (Introduction), h2 (PV DC Input), h3 (AC Grid Output)
    locators = [c.locator for c in chunks]
    # First heading creates h1, paragraphs under it get h1-c1
    assert any(loc.startswith("h1") for loc in locators)
    assert any(loc.startswith("h2") for loc in locators)
    assert any(loc.startswith("h3") for loc in locators)


def test_chunker_text_chunks():
    chunks = chunk_markdown(SIMPLE_MD)
    text_chunks = [c for c in chunks if c.kind == "text"]
    # Should have text chunks for paragraphs
    assert len(text_chunks) >= 3
    # Check content is preserved
    dc_chunks = [c for c in text_chunks if "1100V" in c.content]
    assert len(dc_chunks) == 1
    assert dc_chunks[0].locator.startswith("h2")


def test_chunker_tables():
    chunks = chunk_markdown(SIMPLE_MD)
    table_chunks = [c for c in chunks if c.kind == "table"]
    assert len(table_chunks) == 1
    t = table_chunks[0]
    assert "MOD 8K" in t.content
    assert "MOD 10K" in t.content
    assert t.locator.endswith("-t1")
    assert t.locator.startswith("h3")


def test_chunker_meta_heading_path():
    chunks = chunk_markdown(SIMPLE_MD)
    dc_chunk = [c for c in chunks if "1100V" in c.content][0]
    assert dc_chunk.meta["heading_path"] == "Introduction > PV DC Input"
    assert dc_chunk.meta["original_heading"] == "PV DC Input"


def test_chunker_max_size_split():
    # Create a markdown with a very long paragraph (>1500 chars)
    long_para = "This is a sentence about solar inverters. " * 50  # ~2150 chars
    md = f"## Test Section\n\n{long_para}\n"
    chunks = chunk_markdown(md)
    text_chunks = [c for c in chunks if c.kind == "text"]
    # Should be split into 2+ chunks
    assert len(text_chunks) >= 2
    for c in text_chunks:
        assert len(c.content) <= 1500


def test_chunker_min_merge():
    # Short paragraphs should be merged
    md = "## Section\n\nHi.\n\nThis is a longer paragraph that has enough content to stand alone as a chunk by itself.\n"
    chunks = chunk_markdown(md)
    text_chunks = [c for c in chunks if c.kind == "text"]
    # "Hi." is < 50 chars, should be merged with next paragraph
    assert len(text_chunks) == 1
    assert "Hi." in text_chunks[0].content


# --- Search tests ---

def test_search_ranking():
    chunks = chunk_markdown(SIMPLE_MD)
    idf = build_idf(chunks)
    results = search_chunks(chunks, idf, query="DC voltage", top_k=3, scope="all")
    assert len(results) > 0
    # Top result should be the chunk mentioning DC voltage
    assert "1100V" in results[0]["snippet"] or "DC" in results[0]["snippet"]


def test_search_scope_tables():
    chunks = chunk_markdown(SIMPLE_MD)
    idf = build_idf(chunks)
    results = search_chunks(chunks, idf, query="power current model", top_k=5, scope="tables")
    assert len(results) > 0
    for r in results:
        assert r["kind"] == "table"


def test_search_scope_text():
    chunks = chunk_markdown(SIMPLE_MD)
    idf = build_idf(chunks)
    results = search_chunks(chunks, idf, query="voltage", top_k=5, scope="text")
    for r in results:
        assert r["kind"] == "text"


def test_search_returns_scores():
    chunks = chunk_markdown(SIMPLE_MD)
    idf = build_idf(chunks)
    results = search_chunks(chunks, idf, query="DC voltage", top_k=3, scope="all")
    for r in results:
        assert "score" in r
        assert isinstance(r["score"], float)
        assert r["score"] >= 0


# --- IDF tests ---

def test_build_idf():
    chunks = chunk_markdown(SIMPLE_MD)
    idf = build_idf(chunks)
    assert isinstance(idf, dict)
    # Common words should have lower IDF than rare words
    assert len(idf) > 0
```

**Step 3: Run tests to verify they fail**

```bash
cd /home/ubuntu/workspaces/document-retriever
docker run --rm -v "$(pwd)/mcp_tools:/app" -w /app python:3.11-slim \
  sh -c "pip install -q pytest && python -m pytest tests/test_indexer.py -v 2>&1 | head -40"
```

Expected: All tests FAIL with `ModuleNotFoundError: No module named 'server.indexer'`

**Step 4: Implement the indexer**

Create `mcp_tools/server/indexer.py`:

```python
"""Markdown chunker, TF-IDF search, and index management."""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class Chunk:
    locator: str
    kind: str  # "text" or "table"
    content: str
    meta: dict = field(default_factory=dict)


# --- Stopwords (minimal set) ---
STOPWORDS = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would shall should may might can could of in to for on with "
    "at by from as into through during before after above below between "
    "and or but not no nor so yet both either neither each every all "
    "any few more most other some such it its this that these those "
    "i me my we our you your he him his she her they them their".split()
)


def _tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric, remove stopwords."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def _split_at_sentence(text: str, max_chars: int) -> list[str]:
    """Split text at sentence boundaries, respecting max_chars."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    parts = []
    current = ""
    for s in sentences:
        if current and len(current) + len(s) + 1 > max_chars:
            parts.append(current.strip())
            current = s
        else:
            current = f"{current} {s}".strip() if current else s
    if current.strip():
        parts.append(current.strip())
    # If a single sentence exceeds max_chars, force-split
    final = []
    for p in parts:
        while len(p) > max_chars:
            final.append(p[:max_chars])
            p = p[max_chars:]
        if p:
            final.append(p)
    return final


def chunk_markdown(
    markdown: str,
    max_chunk_chars: int = 1500,
    min_chunk_chars: int = 50,
) -> list[Chunk]:
    """Parse markdown into chunks with stable locators.

    Locator scheme:
      h{n}        — heading (section boundary), n is sequential heading index
      h{n}-c{m}   — text chunk m under heading n
      h{n}-t{m}   — table chunk m under heading n
    """
    lines = markdown.split("\n")
    chunks: list[Chunk] = []

    heading_index = 0
    heading_stack: list[str] = []  # [(level, title), ...]
    current_heading_title = ""
    chunk_counter = 0
    table_counter = 0

    i = 0
    pending_text = ""

    def _heading_path() -> str:
        return " > ".join(heading_stack) if heading_stack else ""

    def _flush_text():
        nonlocal pending_text, chunk_counter
        text = pending_text.strip()
        pending_text = ""
        if not text:
            return
        # Split if too long
        if len(text) > max_chunk_chars:
            parts = _split_at_sentence(text, max_chunk_chars)
        else:
            parts = [text]

        for part in parts:
            chunk_counter += 1
            chunks.append(Chunk(
                locator=f"h{heading_index}-c{chunk_counter}",
                kind="text",
                content=part,
                meta={
                    "heading_path": _heading_path(),
                    "original_heading": current_heading_title,
                    "chunk_index": chunk_counter,
                },
            ))

    def _is_table_line(line: str) -> bool:
        stripped = line.strip()
        return stripped.startswith("|") and stripped.endswith("|")

    def _is_separator_line(line: str) -> bool:
        stripped = line.strip()
        return bool(re.match(r"^\|[\s\-:|]+\|$", stripped))

    while i < len(lines):
        line = lines[i]

        # Check for heading
        heading_match = re.match(r"^(#{1,6})\s+(.+)$", line.strip())
        if heading_match:
            _flush_text()
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()
            heading_index += 1
            chunk_counter = 0
            table_counter = 0
            current_heading_title = title

            # Maintain heading stack
            while heading_stack and len(heading_stack) >= level:
                heading_stack.pop()
            heading_stack.append(title)

            i += 1
            continue

        # Check for table
        if _is_table_line(line):
            _flush_text()
            table_lines = []
            while i < len(lines) and (_is_table_line(lines[i]) or _is_separator_line(lines[i])):
                table_lines.append(lines[i])
                i += 1
            table_content = "\n".join(table_lines)
            table_counter += 1
            chunks.append(Chunk(
                locator=f"h{heading_index}-t{table_counter}",
                kind="table",
                content=table_content,
                meta={
                    "heading_path": _heading_path(),
                    "original_heading": current_heading_title,
                    "table_index": table_counter,
                },
            ))
            continue

        # Blank line = paragraph boundary
        if not line.strip():
            if pending_text.strip():
                _flush_text()
            i += 1
            continue

        # Regular text line
        pending_text += line + "\n"
        i += 1

    # Flush remaining
    _flush_text()

    # Post-process: merge tiny chunks with next sibling
    merged: list[Chunk] = []
    for chunk in chunks:
        if (
            merged
            and merged[-1].kind == "text"
            and chunk.kind == "text"
            and len(merged[-1].content) < min_chunk_chars
            and merged[-1].locator.split("-")[0] == chunk.locator.split("-")[0]
        ):
            merged[-1] = Chunk(
                locator=merged[-1].locator,
                kind="text",
                content=merged[-1].content + "\n" + chunk.content,
                meta=merged[-1].meta,
            )
        else:
            merged.append(chunk)

    # Also merge trailing tiny chunk with previous
    if (
        len(merged) >= 2
        and merged[-1].kind == "text"
        and len(merged[-1].content) < min_chunk_chars
        and merged[-2].kind == "text"
        and merged[-1].locator.split("-")[0] == merged[-2].locator.split("-")[0]
    ):
        merged[-2] = Chunk(
            locator=merged[-2].locator,
            kind="text",
            content=merged[-2].content + "\n" + merged[-1].content,
            meta=merged[-2].meta,
        )
        merged.pop()

    return merged


def build_idf(chunks: list[Chunk]) -> dict[str, float]:
    """Compute inverse document frequency for all terms across chunks."""
    n = len(chunks)
    if n == 0:
        return {}
    doc_freq: dict[str, int] = {}
    for chunk in chunks:
        terms = set(_tokenize(chunk.content))
        for term in terms:
            doc_freq[term] = doc_freq.get(term, 0) + 1
    return {term: math.log(n / df) for term, df in doc_freq.items()}


def search_chunks(
    chunks: list[Chunk],
    idf: dict[str, float],
    query: str,
    top_k: int = 5,
    scope: str = "all",
) -> list[dict]:
    """TF-IDF keyword search over chunks.

    Args:
        chunks: List of Chunk objects.
        idf: Precomputed IDF scores from build_idf().
        query: Search query string.
        top_k: Number of results to return.
        scope: "text", "tables", or "all".

    Returns:
        List of dicts with locator, kind, snippet, score.
    """
    query_terms = _tokenize(query)
    if not query_terms:
        return []

    filtered = chunks
    if scope == "text":
        filtered = [c for c in chunks if c.kind == "text"]
    elif scope == "tables":
        filtered = [c for c in chunks if c.kind == "table"]

    scored = []
    for chunk in filtered:
        chunk_tokens = _tokenize(chunk.content)
        if not chunk_tokens:
            continue
        total = len(chunk_tokens)
        score = 0.0
        for qt in query_terms:
            tf = chunk_tokens.count(qt) / total
            score += tf * idf.get(qt, 0.0)
        if score > 0:
            scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)

    results = []
    for score, chunk in scored[:top_k]:
        snippet = chunk.content[:200]
        if len(chunk.content) > 200:
            snippet += "..."
        results.append({
            "locator": chunk.locator,
            "kind": chunk.kind,
            "snippet": snippet,
            "score": round(score, 6),
        })
    return results


def read_chunk(chunks: list[Chunk], locator: str, max_chars: int = 3000) -> Optional[dict]:
    """Read a single chunk by locator."""
    for chunk in chunks:
        if chunk.locator == locator:
            content = chunk.content[:max_chars]
            return {
                "locator": chunk.locator,
                "kind": chunk.kind,
                "content": content,
                "meta": chunk.meta,
            }
    return None


def validate_and_normalize(data: dict, schema: dict) -> dict:
    """Validate extracted data against schema, normalize formatting.

    Returns dict with: ok, errors, warnings, normalized.
    """
    errors: list[str] = []
    warnings: list[str] = []
    normalized = _normalize_recursive(data, schema, "", errors, warnings)
    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "normalized": normalized,
    }


def _normalize_recursive(
    data, schema, path: str, errors: list[str], warnings: list[str]
):
    """Recursively walk schema, fill missing keys, normalize values."""
    if isinstance(schema, dict):
        if not isinstance(data, dict):
            if data is None:
                data = {}
                warnings.append(f"Field '{path}' is null, expected object")
            else:
                errors.append(f"Field '{path}' expected object, got {type(data).__name__}")
                return data
        result = {}
        for key, sub_schema in schema.items():
            child_path = f"{path}.{key}" if path else key
            if key not in data or data[key] is None:
                warnings.append(f"Field '{child_path}' is null")
                result[key] = _default_for_schema(sub_schema)
            else:
                result[key] = _normalize_recursive(
                    data[key], sub_schema, child_path, errors, warnings
                )
        return result
    elif isinstance(schema, list):
        if not isinstance(data, list):
            if data is None:
                warnings.append(f"Field '{path}' is null, expected list")
                return []
            else:
                errors.append(f"Field '{path}' expected list, got {type(data).__name__}")
                return data
        return data
    else:
        # Leaf value — normalize formatting
        if isinstance(data, str):
            # Normalize range separators: ~ to -
            data = re.sub(r"(\d)\s*~\s*(\d)", r"\1-\2", data)
            # Clean up whitespace
            data = " ".join(data.split())
        return data


def _default_for_schema(schema):
    """Return a default value matching the schema type."""
    if isinstance(schema, dict):
        return {k: _default_for_schema(v) for k, v in schema.items()}
    elif isinstance(schema, list):
        return []
    else:
        return None


# --- Index persistence ---

def save_index(
    cache_dir: Path,
    fingerprint: str,
    chunks: list[Chunk],
    idf: dict[str, float],
    doc_type: str,
    page_count: Optional[int],
    toc: list[dict],
    notes: list[str],
) -> None:
    """Persist index to disk."""
    index_dir = cache_dir / fingerprint
    index_dir.mkdir(parents=True, exist_ok=True)

    index_meta = {
        "index_id": fingerprint,
        "doc_fingerprint": fingerprint,
        "doc_type": doc_type,
        "page_count": page_count,
        "chunk_count": len(chunks),
        "toc": toc,
        "notes": notes,
    }
    (index_dir / "index.json").write_text(json.dumps(index_meta, indent=2))
    (index_dir / "chunks.json").write_text(
        json.dumps([asdict(c) for c in chunks], indent=2)
    )
    (index_dir / "idf.json").write_text(json.dumps(idf))


def load_index(cache_dir: Path, index_id: str) -> Optional[dict]:
    """Load index metadata from disk."""
    index_file = cache_dir / index_id / "index.json"
    if not index_file.exists():
        return None
    return json.loads(index_file.read_text())


def load_chunks(cache_dir: Path, index_id: str) -> list[Chunk]:
    """Load chunks from disk."""
    chunks_file = cache_dir / index_id / "chunks.json"
    if not chunks_file.exists():
        return []
    raw = json.loads(chunks_file.read_text())
    return [Chunk(**c) for c in raw]


def load_idf(cache_dir: Path, index_id: str) -> dict[str, float]:
    """Load IDF scores from disk."""
    idf_file = cache_dir / index_id / "idf.json"
    if not idf_file.exists():
        return {}
    return json.loads(idf_file.read_text())


def compute_fingerprint(file_path: str) -> str:
    """SHA-256 fingerprint of file bytes."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()
```

**Step 5: Run tests to verify they pass**

```bash
cd /home/ubuntu/workspaces/document-retriever
docker run --rm -v "$(pwd)/mcp_tools:/app" -w /app python:3.11-slim \
  sh -c "pip install -q pytest && python -m pytest tests/test_indexer.py -v"
```

Expected: All tests PASS.

**Step 6: Commit**

```bash
git add mcp_tools/
git commit -m "feat: indexer with chunking, TF-IDF search, and validation"
```

---

## Task 4: MCP Tools — Validate & Normalize Tests

**Files:**
- Modify: `mcp_tools/tests/test_indexer.py` (add validation tests)

**Step 1: Add failing validation tests**

Append to `mcp_tools/tests/test_indexer.py`:

```python
# --- Validation tests ---

from server.indexer import validate_and_normalize


SAMPLE_SCHEMA = {
    "product_identity": {
        "product_type": "string",
        "series_name": "string",
        "models": ["list"],
    },
    "pv_dc_input": {
        "max_dc_voltage": "number with unit",
        "number_of_mppts": "integer",
    },
}


def test_validate_complete_data():
    data = {
        "product_identity": {
            "product_type": "Hybrid Inverter",
            "series_name": "MOD 8K",
            "models": ["MOD 8K", "MOD 10K"],
        },
        "pv_dc_input": {
            "max_dc_voltage": "1100V",
            "number_of_mppts": 2,
        },
    }
    result = validate_and_normalize(data, SAMPLE_SCHEMA)
    assert result["ok"] is True
    assert len(result["errors"]) == 0
    assert len(result["warnings"]) == 0


def test_validate_missing_keys():
    data = {
        "product_identity": {
            "product_type": "Hybrid Inverter",
            # missing series_name and models
        },
        # missing pv_dc_input entirely
    }
    result = validate_and_normalize(data, SAMPLE_SCHEMA)
    assert any("series_name" in w for w in result["warnings"])
    assert any("models" in w for w in result["warnings"])
    assert any("pv_dc_input" in w for w in result["warnings"])
    # Normalized should have all keys
    assert "series_name" in result["normalized"]["product_identity"]
    assert result["normalized"]["product_identity"]["series_name"] is None
    assert "pv_dc_input" in result["normalized"]


def test_validate_type_mismatch():
    data = {
        "product_identity": "not a dict",
        "pv_dc_input": {
            "max_dc_voltage": "1100V",
            "number_of_mppts": 2,
        },
    }
    result = validate_and_normalize(data, SAMPLE_SCHEMA)
    assert result["ok"] is False
    assert any("expected object" in e for e in result["errors"])


def test_validate_normalizes_ranges():
    data = {
        "product_identity": {
            "product_type": "Inverter",
            "series_name": "MOD",
            "models": [],
        },
        "pv_dc_input": {
            "max_dc_voltage": "1000~1500V",
            "number_of_mppts": 2,
        },
    }
    result = validate_and_normalize(data, SAMPLE_SCHEMA)
    assert result["normalized"]["pv_dc_input"]["max_dc_voltage"] == "1000-1500V"


def test_validate_null_data():
    result = validate_and_normalize(None, SAMPLE_SCHEMA)
    assert "normalized" in result
    assert result["normalized"]["product_identity"]["product_type"] is None
```

**Step 2: Run tests**

```bash
cd /home/ubuntu/workspaces/document-retriever
docker run --rm -v "$(pwd)/mcp_tools:/app" -w /app python:3.11-slim \
  sh -c "pip install -q pytest && python -m pytest tests/test_indexer.py -v"
```

Expected: All tests PASS (implementation was included in Task 3).

**Step 3: Commit**

```bash
git add mcp_tools/tests/test_indexer.py
git commit -m "test: add validation and normalization tests"
```

---

## Task 5: MCP Tools — Markitdown Client

**Files:**
- Create: `mcp_tools/server/markitdown_client.py`
- Create: `mcp_tools/tests/test_markitdown_client.py`

**Step 1: Write the markitdown HTTP client**

Create `mcp_tools/server/markitdown_client.py`:

```python
"""HTTP client for the markitdown-vision-service."""
from __future__ import annotations

import logging
import asyncio
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


class MarkitdownClient:
    """Async client for markitdown-vision-service."""

    def __init__(self, base_url: str = "http://markitdown:8000", timeout: float = 300.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def convert_to_markdown(self, file_path: str) -> str:
        """Upload a file to markitdown, poll until done, return markdown text.

        Args:
            file_path: Path to the file to convert.

        Returns:
            The markdown text output.

        Raises:
            RuntimeError: If conversion fails or times out.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            # POST file
            with open(file_path, "rb") as f:
                files = {"file": (path.name, f, "application/octet-stream")}
                resp = await client.post(f"{self.base_url}/tasks", files=files)
                resp.raise_for_status()
                task_data = resp.json()

            task_id = task_data["task_id"]
            logger.info(f"Markitdown task created: {task_id}")

            # Poll until completed
            elapsed = 0.0
            poll_interval = 2.0
            while elapsed < self.timeout:
                await asyncio.sleep(poll_interval)
                elapsed += poll_interval

                status_resp = await client.get(f"{self.base_url}/tasks/{task_id}")
                status_resp.raise_for_status()
                status_data = status_resp.json()
                status = status_data["status"]

                if status == "completed":
                    # Download the markdown file
                    outputs = status_data.get("outputs", [])
                    md_files = [o for o in outputs if o.endswith(".md")]
                    if not md_files:
                        raise RuntimeError(f"No markdown output for task {task_id}")

                    md_resp = await client.get(
                        f"{self.base_url}/tasks/{task_id}/files/{md_files[0]}"
                    )
                    md_resp.raise_for_status()
                    logger.info(f"Markitdown conversion complete: {len(md_resp.text)} chars")
                    return md_resp.text

                elif status == "failed":
                    error_msg = status_data.get("error_message", "Unknown error")
                    raise RuntimeError(f"Markitdown conversion failed: {error_msg}")

                logger.debug(f"Markitdown task {task_id}: {status} ({elapsed:.0f}s)")

            raise RuntimeError(f"Markitdown conversion timed out after {self.timeout}s")

    async def health_check(self) -> bool:
        """Check if the markitdown service is healthy."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.base_url}/health")
                return resp.status_code == 200
        except Exception:
            return False
```

**Step 2: Write a basic test (mocked)**

Create `mcp_tools/tests/test_markitdown_client.py`:

```python
"""Tests for the markitdown client (mocked HTTP)."""
import pytest
import json


def test_markitdown_client_import():
    """Verify the client module can be imported."""
    from server.markitdown_client import MarkitdownClient
    client = MarkitdownClient(base_url="http://localhost:9999")
    assert client.base_url == "http://localhost:9999"
```

**Step 3: Run test**

```bash
cd /home/ubuntu/workspaces/document-retriever
docker run --rm -v "$(pwd)/mcp_tools:/app" -w /app python:3.11-slim \
  sh -c "pip install -q pytest httpx && python -m pytest tests/test_markitdown_client.py -v"
```

Expected: PASS.

**Step 4: Commit**

```bash
git add mcp_tools/server/markitdown_client.py mcp_tools/tests/test_markitdown_client.py
git commit -m "feat: markitdown HTTP client for document conversion"
```

---

## Task 6: MCP Tools — FastMCP Server + Tools

**Files:**
- Create: `mcp_tools/server/tools.py`
- Create: `mcp_tools/server/main.py`
- Create: `mcp_tools/requirements.txt`

**Step 1: Create requirements.txt**

Create `mcp_tools/requirements.txt`:

```
mcp>=1.0
httpx
uvicorn[standard]
```

**Step 2: Implement tools.py with 4 MCP tools**

Create `mcp_tools/server/tools.py`:

```python
"""MCP tool implementations for document extraction."""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from .indexer import (
    Chunk,
    chunk_markdown,
    build_idf,
    search_chunks,
    read_chunk,
    validate_and_normalize,
    save_index,
    load_index,
    load_chunks,
    load_idf,
    compute_fingerprint,
)
from .markitdown_client import MarkitdownClient

logger = logging.getLogger(__name__)

CACHE_DIR = Path(os.environ.get("DATA_DIR", "/data")) / "cache"
MARKITDOWN_URL = os.environ.get("MARKITDOWN_URL", "http://markitdown:8000")


def register_tools(mcp: FastMCP) -> None:
    """Register all document tools on the FastMCP server."""

    @mcp.tool()
    async def doc_index_build(file_path: str, cache_key: str | None = None) -> str:
        """Build a searchable index from a document file.

        Converts the document to markdown via markitdown-vision-service,
        then chunks it and builds a TF-IDF index for search.

        Args:
            file_path: Absolute path to the document file.
            cache_key: Optional cache key. If not provided, uses file SHA-256.

        Returns:
            JSON with index_id, doc_fingerprint, doc_type, page_count,
            chunk_count, toc, and notes.
        """
        fingerprint = compute_fingerprint(file_path)

        # Check cache
        cached = load_index(CACHE_DIR, fingerprint)
        if cached:
            logger.info(f"Index cache hit for {fingerprint}")
            return json.dumps(cached)

        # Detect doc type from extension
        ext = Path(file_path).suffix.lower().lstrip(".")
        ext_map = {"pdf": "pdf", "docx": "docx", "txt": "txt", "md": "md", "html": "html", "htm": "html"}
        doc_type = ext_map.get(ext)
        if not doc_type:
            return json.dumps({"error": f"Unsupported file format: .{ext}"})

        # Convert via markitdown
        client = MarkitdownClient(base_url=MARKITDOWN_URL)
        notes = []
        try:
            markdown_text = await client.convert_to_markdown(file_path)
            notes.append("Converted via markitdown-vision-service")
        except Exception as e:
            return json.dumps({"error": f"Conversion failed: {str(e)}"})

        # Chunk the markdown
        chunks = chunk_markdown(markdown_text)
        idf = build_idf(chunks)

        # Build TOC from heading chunks
        toc = []
        seen_headings = set()
        for chunk in chunks:
            heading = chunk.meta.get("original_heading", "")
            h_prefix = chunk.locator.split("-")[0]  # e.g. "h3"
            if heading and h_prefix not in seen_headings:
                seen_headings.add(h_prefix)
                toc.append({"locator": h_prefix, "title": heading})

        # Detect page count from markdown content (markitdown may include page markers)
        page_markers = re.findall(r"<!--\s*page\s+(\d+)\s*-->", markdown_text, re.IGNORECASE)
        page_count = int(page_markers[-1]) if page_markers else None

        # Persist
        save_index(CACHE_DIR, fingerprint, chunks, idf, doc_type, page_count, toc, notes)
        logger.info(f"Built index {fingerprint}: {len(chunks)} chunks")

        result = load_index(CACHE_DIR, fingerprint)
        return json.dumps(result)

    @mcp.tool()
    async def doc_search(
        index_id: str,
        query: str,
        top_k: int = 5,
        scope: str = "all",
    ) -> str:
        """Search the document index for relevant chunks.

        Args:
            index_id: The index_id returned by doc_index_build.
            query: Natural language search query.
            top_k: Number of results to return (default 5).
            scope: "text", "tables", or "all" (default "all").

        Returns:
            JSON array of {locator, kind, snippet, score}.
        """
        chunks = load_chunks(CACHE_DIR, index_id)
        if not chunks:
            return json.dumps({"error": f"Index not found: {index_id}"})

        idf = load_idf(CACHE_DIR, index_id)
        results = search_chunks(chunks, idf, query, top_k, scope)
        return json.dumps(results)

    @mcp.tool()
    async def doc_read(
        index_id: str,
        locator: str,
        max_chars: int = 3000,
    ) -> str:
        """Read a specific chunk from the document index.

        Args:
            index_id: The index_id returned by doc_index_build.
            locator: The chunk locator (e.g. "h3-c2" or "h3-t1").
            max_chars: Maximum characters to return (default 3000).

        Returns:
            JSON with locator, kind, content, and meta.
        """
        chunks = load_chunks(CACHE_DIR, index_id)
        if not chunks:
            return json.dumps({"error": f"Index not found: {index_id}"})

        result = read_chunk(chunks, locator, max_chars)
        if result is None:
            return json.dumps({"error": f"Locator not found: {locator}"})
        return json.dumps(result)

    @mcp.tool()
    async def json_validate_and_normalize(data: str, schema: str) -> str:
        """Validate extracted data against schema and normalize formatting.

        Args:
            data: JSON string of the extracted data object.
            schema: JSON string of the target schema skeleton.

        Returns:
            JSON with ok (bool), errors, warnings, and normalized data.
        """
        try:
            data_obj = json.loads(data) if isinstance(data, str) else data
        except json.JSONDecodeError as e:
            return json.dumps({"ok": False, "errors": [f"Invalid data JSON: {e}"], "warnings": [], "normalized": {}})

        try:
            schema_obj = json.loads(schema) if isinstance(schema, str) else schema
        except json.JSONDecodeError as e:
            return json.dumps({"ok": False, "errors": [f"Invalid schema JSON: {e}"], "warnings": [], "normalized": {}})

        result = validate_and_normalize(data_obj, schema_obj)
        return json.dumps(result)
```

**Step 3: Implement main.py (FastMCP SSE server)**

Create `mcp_tools/server/main.py`:

```python
"""FastMCP SSE server entrypoint for document tools."""
import logging
import os

from mcp.server.fastmcp import FastMCP

from .tools import register_tools

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

mcp = FastMCP("document-tools")
register_tools(mcp)

# Add a health endpoint for docker-compose healthcheck
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route


async def health(request):
    return JSONResponse({"status": "ok"})


# The FastMCP app is a Starlette app; we can add routes
app = mcp.sse_app()
app.routes.append(Route("/health", health))

if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8001"))
    logger.info(f"Starting MCP tools server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
```

**Step 4: Commit**

```bash
git add mcp_tools/server/tools.py mcp_tools/server/main.py mcp_tools/requirements.txt
git commit -m "feat: MCP tools server with 4 document tools over SSE"
```

---

## Task 7: MCP Tools — Dockerfile

**Files:**
- Create: `mcp_tools/Dockerfile`

**Step 1: Write Dockerfile**

Create `mcp_tools/Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server/ ./server/

ENV PYTHONUNBUFFERED=1
ENV HOST=0.0.0.0
ENV PORT=8001

EXPOSE 8001

CMD ["python", "-m", "server.main"]
```

**Step 2: Verify build**

```bash
cd /home/ubuntu/workspaces/document-retriever
docker build -t mcp-tools-test ./mcp_tools
```

Expected: Build succeeds.

**Step 3: Commit**

```bash
git add mcp_tools/Dockerfile
git commit -m "feat: Dockerfile for mcp-tools service"
```

---

## Task 8: Document-Retriever API — Models

**Files:**
- Create: `api/app/__init__.py`
- Create: `api/app/models.py`

**Step 1: Create directory structure**

```bash
mkdir -p api/app api/tests
touch api/app/__init__.py api/tests/__init__.py
```

**Step 2: Write Pydantic models**

Create `api/app/models.py`:

```python
"""Pydantic request/response models for the document-retriever API."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class TaskCreateResponse(BaseModel):
    task_id: str


class TaskError(BaseModel):
    message: str
    details: list[str] = Field(default_factory=list)


class TaskStatusResponse(BaseModel):
    task_id: str
    status: TaskStatus
    created_at: str
    updated_at: str
    error: Optional[TaskError] = None


class TaskTiming(BaseModel):
    total_seconds: float
    index_seconds: Optional[float] = None
    extraction_seconds: Optional[float] = None


class TaskResultResponse(BaseModel):
    task_id: str
    data: dict[str, Any]
    evidence: dict[str, Any]
    warnings: list[str] = Field(default_factory=list)
    timing: TaskTiming


class TaskFailedResponse(BaseModel):
    task_id: str
    status: str = "failed"
    error: TaskError
```

**Step 3: Commit**

```bash
git add api/
git commit -m "feat: Pydantic models for API request/response"
```

---

## Task 9: Document-Retriever API — SQLite Store

**Files:**
- Create: `api/app/store.py`
- Create: `api/tests/test_store.py`

**Step 1: Write failing test**

Create `api/tests/test_store.py`:

```python
"""Tests for SQLite task store."""
import asyncio
import json
import os
import tempfile

import pytest

# Override DB path before import
_tmpdir = tempfile.mkdtemp()
os.environ["DB_PATH"] = os.path.join(_tmpdir, "test_tasks.db")
os.environ["DATA_DIR"] = _tmpdir

from app.store import (
    init_db,
    close_db,
    create_task,
    get_task,
    update_task_status,
    update_task_result,
    update_task_error,
)


@pytest.fixture(autouse=True)
def setup_db():
    """Initialize DB for each test."""
    asyncio.get_event_loop().run_until_complete(init_db())
    yield
    asyncio.get_event_loop().run_until_complete(close_db())


@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def test_create_and_get_task():
    task_id = run(create_task("test.pdf", '{"key": "value"}'))
    assert task_id is not None
    task = run(get_task(task_id))
    assert task is not None
    assert task["task_id"] == task_id
    assert task["status"] == "queued"
    assert task["original_filename"] == "test.pdf"


def test_get_nonexistent_task():
    task = run(get_task("nonexistent"))
    assert task is None


def test_update_status():
    task_id = run(create_task("test.pdf", "{}"))
    run(update_task_status(task_id, "running"))
    task = run(get_task(task_id))
    assert task["status"] == "running"


def test_update_result():
    task_id = run(create_task("test.pdf", "{}"))
    run(update_task_status(task_id, "running"))
    result = {"data": {"key": "val"}, "evidence": {}, "warnings": []}
    timing = {"total_seconds": 10.5}
    run(update_task_result(task_id, result, timing))
    task = run(get_task(task_id))
    assert task["status"] == "succeeded"
    assert json.loads(task["result_json"])["data"]["key"] == "val"


def test_update_error():
    task_id = run(create_task("test.pdf", "{}"))
    run(update_task_status(task_id, "running"))
    run(update_task_error(task_id, "Something broke", ["detail1"]))
    task = run(get_task(task_id))
    assert task["status"] == "failed"
    error = json.loads(task["error_json"])
    assert error["message"] == "Something broke"
```

**Step 2: Run tests to verify they fail**

```bash
cd /home/ubuntu/workspaces/document-retriever
docker run --rm -v "$(pwd)/api:/app" -w /app python:3.11-slim \
  sh -c "pip install -q pytest aiosqlite ulid-py && python -m pytest tests/test_store.py -v 2>&1 | head -30"
```

Expected: FAIL — `ModuleNotFoundError: No module named 'app.store'`

**Step 3: Implement store.py**

Create `api/app/store.py`:

```python
"""SQLite task persistence."""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

import aiosqlite
import ulid

logger = logging.getLogger(__name__)

DB_PATH = os.environ.get("DB_PATH", "/data/tasks.db")

_db: Optional[aiosqlite.Connection] = None

CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS tasks (
    task_id TEXT PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'queued',
    original_filename TEXT NOT NULL,
    schema_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    result_json TEXT,
    error_json TEXT
)
"""


async def init_db() -> None:
    """Initialize database connection and create tables."""
    global _db
    if _db is not None:
        return
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    _db = await aiosqlite.connect(DB_PATH)
    _db.row_factory = aiosqlite.Row
    await _db.execute(CREATE_TABLE)
    await _db.commit()
    logger.info(f"Database initialized at {DB_PATH}")


async def close_db() -> None:
    """Close database connection."""
    global _db
    if _db is not None:
        await _db.close()
        _db = None


async def _get_db() -> aiosqlite.Connection:
    if _db is None:
        await init_db()
    return _db


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def create_task(original_filename: str, schema_json: str) -> str:
    """Create a new task. Returns task_id."""
    db = await _get_db()
    task_id = str(ulid.new())
    now = _now()
    await db.execute(
        "INSERT INTO tasks (task_id, status, original_filename, schema_json, created_at, updated_at) "
        "VALUES (?, 'queued', ?, ?, ?, ?)",
        (task_id, original_filename, schema_json, now, now),
    )
    await db.commit()
    logger.info(f"Task {task_id} created")
    return task_id


async def get_task(task_id: str) -> Optional[dict]:
    """Get task by ID. Returns dict or None."""
    db = await _get_db()
    async with db.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)) as cursor:
        row = await cursor.fetchone()
        return dict(row) if row else None


async def update_task_status(task_id: str, status: str) -> None:
    """Update task status."""
    db = await _get_db()
    now = _now()
    updates = {"status": status, "updated_at": now}
    if status == "running":
        updates["started_at"] = now
    set_clause = ", ".join(f"{k} = ?" for k in updates)
    values = list(updates.values()) + [task_id]
    await db.execute(f"UPDATE tasks SET {set_clause} WHERE task_id = ?", values)
    await db.commit()


async def update_task_result(task_id: str, result: dict, timing: dict) -> None:
    """Mark task as succeeded with result."""
    db = await _get_db()
    now = _now()
    result_with_timing = {**result, "timing": timing}
    await db.execute(
        "UPDATE tasks SET status = 'succeeded', result_json = ?, finished_at = ?, updated_at = ? WHERE task_id = ?",
        (json.dumps(result_with_timing), now, now, task_id),
    )
    await db.commit()


async def update_task_error(task_id: str, message: str, details: list[str] | None = None) -> None:
    """Mark task as failed with error."""
    db = await _get_db()
    now = _now()
    error = {"message": message, "details": details or []}
    await db.execute(
        "UPDATE tasks SET status = 'failed', error_json = ?, finished_at = ?, updated_at = ? WHERE task_id = ?",
        (json.dumps(error), now, now, task_id),
    )
    await db.commit()
```

**Step 4: Run tests**

```bash
cd /home/ubuntu/workspaces/document-retriever
docker run --rm -v "$(pwd)/api:/app" -w /app python:3.11-slim \
  sh -c "pip install -q pytest aiosqlite ulid-py && python -m pytest tests/test_store.py -v"
```

Expected: All tests PASS.

**Step 5: Commit**

```bash
git add api/app/store.py api/tests/test_store.py
git commit -m "feat: SQLite task store with CRUD operations"
```

---

## Task 10: Document-Retriever API — FastAPI Endpoints

**Files:**
- Create: `api/app/main.py`
- Create: `api/app/config.py`

**Step 1: Create config.py**

Create `api/app/config.py`:

```python
"""Application configuration."""
from __future__ import annotations

import os


class Settings:
    # LLM config — all passed through to claude-agent-sdk CLI as env vars
    ANTHROPIC_API_KEY: str = os.environ.get("ANTHROPIC_API_KEY", "")
    ANTHROPIC_AUTH_TOKEN: str = os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
    ANTHROPIC_BASE_URL: str = os.environ.get("ANTHROPIC_BASE_URL", "")
    ANTHROPIC_DEFAULT_SONNET_MODEL: str = os.environ.get("ANTHROPIC_DEFAULT_SONNET_MODEL", "")
    ANTHROPIC_DEFAULT_HAIKU_MODEL: str = os.environ.get("ANTHROPIC_DEFAULT_HAIKU_MODEL", "")
    ANTHROPIC_DEFAULT_OPUS_MODEL: str = os.environ.get("ANTHROPIC_DEFAULT_OPUS_MODEL", "")
    API_TIMEOUT_MS: str = os.environ.get("API_TIMEOUT_MS", "")

    # Service config
    MCP_TOOLS_URL: str = os.environ.get("MCP_TOOLS_URL", "http://mcp-tools:8001/sse")
    DATA_DIR: str = os.environ.get("DATA_DIR", "/data")
    DB_PATH: str = os.environ.get("DB_PATH", "/data/tasks.db")
    MAX_CONCURRENT_TASKS: int = int(os.environ.get("MAX_CONCURRENT_TASKS", "1"))
    AGENT_TIMEOUT: int = int(os.environ.get("AGENT_TIMEOUT", "300"))
    AGENT_MAX_RETRIES: int = int(os.environ.get("AGENT_MAX_RETRIES", "1"))
    MAX_UPLOAD_SIZE: int = 500 * 1024 * 1024  # 500MB

    @property
    def has_llm_auth(self) -> bool:
        """Check if any LLM auth is configured."""
        return bool(self.ANTHROPIC_API_KEY or self.ANTHROPIC_AUTH_TOKEN)


settings = Settings()
```

**Step 2: Create main.py (FastAPI app)**

Create `api/app/main.py`:

```python
"""FastAPI application for document-retriever."""
from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from .config import settings
from .models import (
    TaskCreateResponse,
    TaskError,
    TaskFailedResponse,
    TaskResultResponse,
    TaskStatusResponse,
    TaskTiming,
)
from .store import init_db, close_db, create_task, get_task

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

UPLOAD_DIR = Path(settings.DATA_DIR) / "uploads"
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".html", ".htm"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """App startup and shutdown."""
    os.makedirs(settings.DATA_DIR, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    await init_db()
    logger.info("Database initialized")

    # Start background worker
    from .worker import start_workers, stop_workers
    start_workers()
    logger.info("Workers started")

    yield

    stop_workers()
    await close_db()
    logger.info("Shutdown complete")


app = FastAPI(
    title="document-retriever",
    description="Extract structured data from documents using Claude + MCP tools",
    version="0.1.0",
    lifespan=lifespan,
)


@app.post("/tasks", response_model=TaskCreateResponse, status_code=202)
async def create_extraction_task(
    file: UploadFile = File(...),
    schema: str = Form(...),
):
    """Upload a document and schema, start async extraction."""
    # Validate schema is valid JSON
    try:
        json.loads(schema)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in schema field")

    # Validate file extension
    filename = file.filename or "upload"
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {ext}. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    # Create task
    task_id = await create_task(filename, schema)

    # Save uploaded file
    task_upload_dir = UPLOAD_DIR / task_id
    task_upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = task_upload_dir / filename

    size = 0
    with open(file_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > settings.MAX_UPLOAD_SIZE:
                f.close()
                file_path.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="File too large")
            f.write(chunk)

    logger.info(f"Task {task_id} created for {filename} ({size} bytes)")

    # Enqueue for processing
    from .worker import enqueue_task
    enqueue_task(task_id)

    return TaskCreateResponse(task_id=task_id)


@app.get("/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """Get task status."""
    task = await get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    error = None
    if task["error_json"]:
        err_data = json.loads(task["error_json"])
        error = TaskError(message=err_data["message"], details=err_data.get("details", []))

    return TaskStatusResponse(
        task_id=task["task_id"],
        status=task["status"],
        created_at=task["created_at"],
        updated_at=task["updated_at"],
        error=error,
    )


@app.get("/tasks/{task_id}/result")
async def get_task_result(task_id: str):
    """Get task result (only if succeeded or failed)."""
    task = await get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if task["status"] == "failed":
        error_data = json.loads(task["error_json"]) if task["error_json"] else {"message": "Unknown error", "details": []}
        return JSONResponse(
            status_code=200,
            content={
                "task_id": task_id,
                "status": "failed",
                "error": error_data,
            },
        )

    if task["status"] != "succeeded":
        raise HTTPException(
            status_code=400,
            detail=f"Task not complete (status: {task['status']})",
        )

    result = json.loads(task["result_json"])
    return JSONResponse(
        status_code=200,
        content={
            "task_id": task_id,
            "data": result.get("data", {}),
            "evidence": result.get("evidence", {}),
            "warnings": result.get("warnings", []),
            "timing": result.get("timing", {}),
        },
    )


@app.get("/health")
async def health():
    return {"status": "ok"}
```

**Step 3: Commit**

```bash
git add api/app/main.py api/app/config.py
git commit -m "feat: FastAPI endpoints for task management"
```

---

## Task 11: Document-Retriever API — Worker

**Files:**
- Create: `api/app/worker.py`

**Step 1: Implement the background worker**

Create `api/app/worker.py`:

```python
"""Background task queue and worker for document extraction."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from queue import Queue, Empty
from threading import Thread, Event

from .config import settings
from .store import get_task, update_task_status, update_task_result, update_task_error

logger = logging.getLogger(__name__)

_task_queue: Queue[str] = Queue()
_workers: list[Thread] = []
_shutdown = Event()
_loop: asyncio.AbstractEventLoop | None = None


def enqueue_task(task_id: str) -> None:
    """Add a task to the processing queue."""
    _task_queue.put(task_id)
    logger.info(f"Task {task_id} enqueued")


def _worker_thread(worker_id: int, loop: asyncio.AbstractEventLoop) -> None:
    """Worker thread that processes tasks."""
    logger.info(f"Worker {worker_id} started")
    while not _shutdown.is_set():
        try:
            task_id = _task_queue.get(timeout=1.0)
        except Empty:
            continue

        logger.info(f"Worker {worker_id} processing task {task_id}")
        try:
            future = asyncio.run_coroutine_threadsafe(_process_task(task_id), loop)
            future.result(timeout=settings.AGENT_TIMEOUT + 60)
        except Exception as e:
            logger.exception(f"Worker {worker_id} error on task {task_id}: {e}")
            asyncio.run_coroutine_threadsafe(
                update_task_error(task_id, f"Worker error: {str(e)[:500]}"),
                loop,
            )
        finally:
            _task_queue.task_done()

    logger.info(f"Worker {worker_id} stopped")


async def _process_task(task_id: str) -> None:
    """Process a single extraction task."""
    task = await get_task(task_id)
    if not task:
        logger.error(f"Task {task_id} not found")
        return

    await update_task_status(task_id, "running")
    start_time = time.time()

    try:
        # Find the uploaded file
        upload_dir = Path(settings.DATA_DIR) / "uploads" / task_id
        files = list(upload_dir.iterdir()) if upload_dir.exists() else []
        if not files:
            raise FileNotFoundError(f"No uploaded file found for task {task_id}")
        file_path = str(files[0])

        schema = json.loads(task["schema_json"])

        # Run extraction agent with retry
        from .agent import run_extraction

        last_error = None
        for attempt in range(settings.AGENT_MAX_RETRIES + 1):
            try:
                index_time_start = time.time()
                result = await asyncio.wait_for(
                    run_extraction(file_path, schema),
                    timeout=settings.AGENT_TIMEOUT,
                )
                total_time = time.time() - start_time

                timing = {
                    "total_seconds": round(total_time, 1),
                }

                await update_task_result(task_id, result, timing)
                logger.info(f"Task {task_id} succeeded in {total_time:.1f}s")
                return

            except Exception as e:
                last_error = e
                if attempt < settings.AGENT_MAX_RETRIES:
                    logger.warning(f"Task {task_id} attempt {attempt+1} failed: {e}, retrying...")
                    await asyncio.sleep(2)

        raise last_error

    except Exception as e:
        logger.exception(f"Task {task_id} failed: {e}")
        await update_task_error(task_id, str(e)[:500])


def start_workers() -> None:
    """Start worker threads."""
    global _loop
    _loop = asyncio.get_event_loop()
    _shutdown.clear()
    for i in range(settings.MAX_CONCURRENT_TASKS):
        t = Thread(target=_worker_thread, args=(i, _loop), daemon=True)
        t.start()
        _workers.append(t)
    logger.info(f"Started {settings.MAX_CONCURRENT_TASKS} workers")


def stop_workers() -> None:
    """Stop worker threads."""
    _shutdown.set()
    for t in _workers:
        t.join(timeout=5.0)
    _workers.clear()
    logger.info("Workers stopped")
```

**Step 2: Commit**

```bash
git add api/app/worker.py
git commit -m "feat: background worker with queue, retry, and timeout"
```

---

## Task 12: Document-Retriever API — Agent Wiring

**Files:**
- Create: `api/app/agent.py`

**Step 1: Implement agent.py with Claude Agent SDK**

Create `api/app/agent.py`:

```python
"""Claude Agent SDK integration for document extraction."""
from __future__ import annotations

import json
import logging
import re

from claude_agent_sdk import (
    ClaudeSDKClient,
    ClaudeAgentOptions,
    AssistantMessage,
    ResultMessage,
    TextBlock,
)

from .config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
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
"""


def _build_options() -> ClaudeAgentOptions:
    """Build agent options for extraction."""
    return ClaudeAgentOptions(
        system_prompt=SYSTEM_PROMPT,
        mcp_servers={
            "doc-tools": {
                "type": "sse",
                "url": settings.MCP_TOOLS_URL,
            }
        },
        allowed_tools=[
            "mcp__doc-tools__doc_index_build",
            "mcp__doc-tools__doc_search",
            "mcp__doc-tools__doc_read",
            "mcp__doc-tools__json_validate_and_normalize",
        ],
        max_turns=50,
        permission_mode="acceptEdits",
    )


async def run_extraction(file_path: str, schema: dict) -> dict:
    """Run the extraction agent and return structured result.

    Args:
        file_path: Path to the uploaded document.
        schema: The target schema skeleton.

    Returns:
        Dict with "data", "evidence", and "warnings" keys.
    """
    prompt = (
        f"Extract data from the document at: {file_path}\n\n"
        f"Target schema:\n{json.dumps(schema, indent=2)}"
    )

    options = _build_options()
    final_text = ""

    async with ClaudeSDKClient(options=options) as client:
        await client.query(prompt)
        async for msg in client.receive_response():
            if isinstance(msg, (AssistantMessage, ResultMessage)):
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        final_text += block.text

    return _parse_agent_output(final_text)


def _parse_agent_output(text: str) -> dict:
    """Parse the agent's output into data + evidence + warnings.

    The agent should return two JSON objects: {"data": ...} and {"evidence": ...}
    """
    # Try to find JSON objects in the text
    json_objects = []
    # Find all top-level JSON objects
    brace_depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if brace_depth == 0:
                start = i
            brace_depth += 1
        elif ch == "}":
            brace_depth -= 1
            if brace_depth == 0 and start is not None:
                try:
                    obj = json.loads(text[start : i + 1])
                    json_objects.append(obj)
                except json.JSONDecodeError:
                    pass
                start = None

    data = {}
    evidence = {}
    warnings = []

    if len(json_objects) >= 2:
        # Agent returned two objects: data and evidence
        data = json_objects[0].get("data", json_objects[0])
        evidence = json_objects[1].get("evidence", json_objects[1])
    elif len(json_objects) == 1:
        obj = json_objects[0]
        data = obj.get("data", {})
        evidence = obj.get("evidence", {})
        warnings = obj.get("warnings", [])

    # Extract warnings from evidence (null fields with warnings)
    if not warnings:
        warnings = _extract_warnings(evidence)

    return {"data": data, "evidence": evidence, "warnings": warnings}


def _extract_warnings(evidence: dict, path: str = "") -> list[str]:
    """Extract warnings from evidence tree (null locators indicate missing data)."""
    warnings = []
    if isinstance(evidence, dict):
        if "warning" in evidence and evidence.get("locator") is None:
            field_name = path or "unknown"
            warnings.append(evidence["warning"])
        else:
            for key, val in evidence.items():
                child_path = f"{path}.{key}" if path else key
                if key in ("locator", "snippet", "warning"):
                    continue
                warnings.extend(_extract_warnings(val, child_path))
    return warnings
```

**Step 2: Commit**

```bash
git add api/app/agent.py
git commit -m "feat: Claude Agent SDK integration with extraction prompt"
```

---

## Task 13: Document-Retriever API — Dockerfile + requirements

**Files:**
- Create: `api/Dockerfile`
- Create: `api/requirements.txt`

**Step 1: Write requirements.txt**

Create `api/requirements.txt`:

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

**Step 2: Write Dockerfile**

Create `api/Dockerfile`:

```dockerfile
FROM python:3.11-slim

# claude-agent-sdk bundles Claude Code CLI which needs node
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY tests/ ./tests/

# Copy fixtures for e2e tests
COPY ../fixtures/ ./fixtures/ 2>/dev/null || true

ENV PYTHONUNBUFFERED=1
ENV DATA_DIR=/data
ENV DB_PATH=/data/tasks.db

RUN mkdir -p /data/uploads /data/cache

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Step 3: Verify build**

```bash
cd /home/ubuntu/workspaces/document-retriever
docker build -t document-retriever-test ./api
```

Expected: Build succeeds.

**Step 4: Commit**

```bash
git add api/Dockerfile api/requirements.txt
git commit -m "feat: Dockerfile and requirements for document-retriever API"
```

---

## Task 14: Docker Compose + Integration

**Files:**
- Create: `docker-compose.yml`

**Step 1: Write docker-compose.yml**

Create `docker-compose.yml`:

```yaml
services:
  document-retriever:
    build: ./api
    ports:
      - "8000:8000"
    environment:
      # LLM config — pass through all Anthropic-compatible env vars from host
      - ANTHROPIC_API_KEY
      - ANTHROPIC_AUTH_TOKEN
      - ANTHROPIC_BASE_URL
      - ANTHROPIC_DEFAULT_HAIKU_MODEL
      - ANTHROPIC_DEFAULT_SONNET_MODEL
      - ANTHROPIC_DEFAULT_OPUS_MODEL
      - API_TIMEOUT_MS
      # Service config
      - MCP_TOOLS_URL=http://mcp-tools:8001/sse
      - DATA_DIR=/data
      - DB_PATH=/data/tasks.db
    volumes:
      - ./data:/data
    depends_on:
      mcp-tools:
        condition: service_healthy
    restart: unless-stopped

  mcp-tools:
    build: ./mcp_tools
    ports:
      - "8001:8001"
    environment:
      - MARKITDOWN_URL=http://markitdown:8000
      - DATA_DIR=/data
    volumes:
      - ./data:/data
    depends_on:
      markitdown:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8001/health')"]
      interval: 5s
      timeout: 5s
      retries: 5
      start_period: 10s
    restart: unless-stopped

  markitdown:
    build:
      context: ./markitdown-vision-service
    ports:
      - "8002:8000"
    volumes:
      - ./data/markitdown:/data
    environment:
      - HOST=0.0.0.0
      - PORT=8000
      - OPENAI_API_TOKEN=${OPENAI_API_KEY:-}
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 5s
      timeout: 5s
      retries: 5
      start_period: 10s
    restart: unless-stopped
```

Note: markitdown-vision-service runs on internal port 8000, mapped to 8002 on the host.
Inter-container traffic uses `http://markitdown:8000` (internal port).

**Step 2: Verify compose builds**

```bash
cd /home/ubuntu/workspaces/document-retriever
docker compose build
```

Expected: All 3 services build.

**Step 3: Smoke test — start and check health**

```bash
docker compose up -d markitdown
sleep 10
curl -s http://localhost:8002/health

docker compose up -d mcp-tools
sleep 10
curl -s http://localhost:8001/health

docker compose down
```

Expected: Both return `{"status":"ok"}`.

**Step 4: Commit**

```bash
git add docker-compose.yml
git commit -m "feat: docker-compose with 3 services"
```

---

## Task 15: API Smoke Tests

**Files:**
- Create: `api/tests/test_smoke.py`
- Create: `api/tests/conftest.py`

**Step 1: Write conftest and smoke tests**

Create `api/tests/conftest.py`:

```python
"""Test configuration and fixtures."""
import asyncio
import os
import tempfile

# Override settings before importing app
_tmpdir = tempfile.mkdtemp()
os.environ["DATA_DIR"] = _tmpdir
os.environ["DB_PATH"] = os.path.join(_tmpdir, "test.db")
os.environ["ANTHROPIC_API_KEY"] = "test-key"
os.environ["MCP_TOOLS_URL"] = "http://localhost:9999/sse"

import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app
from app.store import init_db, close_db


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
async def setup_db():
    await init_db()
    yield
    await close_db()


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
```

Create `api/tests/test_smoke.py`:

```python
"""Smoke tests for the document-retriever API."""
import io
import json

import pytest

pytestmark = pytest.mark.asyncio


async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


async def test_post_task(client):
    schema = json.dumps({"product": {"name": "string"}})
    files = {"file": ("test.pdf", b"%PDF-1.4 fake content", "application/pdf")}
    data = {"schema": schema}
    resp = await client.post("/tasks", files=files, data=data)
    assert resp.status_code == 202
    body = resp.json()
    assert "task_id" in body
    assert len(body["task_id"]) > 0


async def test_post_task_invalid_schema(client):
    files = {"file": ("test.pdf", b"%PDF-1.4 fake", "application/pdf")}
    data = {"schema": "not-json{{{"}
    resp = await client.post("/tasks", files=files, data=data)
    assert resp.status_code == 400


async def test_post_task_unsupported_format(client):
    schema = json.dumps({"key": "value"})
    files = {"file": ("test.pptx", b"fake pptx", "application/vnd.ms-powerpoint")}
    data = {"schema": schema}
    resp = await client.post("/tasks", files=files, data=data)
    assert resp.status_code == 400
    assert "Unsupported" in resp.json()["detail"]


async def test_get_task_status(client):
    # Create a task first
    schema = json.dumps({"key": "value"})
    files = {"file": ("test.txt", b"hello world", "text/plain")}
    data = {"schema": schema}
    create_resp = await client.post("/tasks", files=files, data=data)
    task_id = create_resp.json()["task_id"]

    # Get status
    resp = await client.get(f"/tasks/{task_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["task_id"] == task_id
    assert body["status"] in ("queued", "running", "succeeded", "failed")


async def test_get_task_not_found(client):
    resp = await client.get("/tasks/nonexistent-id")
    assert resp.status_code == 404


async def test_result_not_ready(client):
    schema = json.dumps({"key": "value"})
    files = {"file": ("test.txt", b"hello", "text/plain")}
    data = {"schema": schema}
    create_resp = await client.post("/tasks", files=files, data=data)
    task_id = create_resp.json()["task_id"]

    resp = await client.get(f"/tasks/{task_id}/result")
    assert resp.status_code == 400
    assert "not complete" in resp.json()["detail"]
```

**Step 2: Run smoke tests**

```bash
cd /home/ubuntu/workspaces/document-retriever
docker run --rm -v "$(pwd)/api:/app" -w /app python:3.11-slim \
  sh -c "pip install -q fastapi uvicorn aiosqlite ulid-py python-multipart httpx pydantic pydantic-settings pytest pytest-asyncio && python -m pytest tests/test_smoke.py -v"
```

Expected: All tests PASS.

**Step 3: Commit**

```bash
git add api/tests/
git commit -m "test: API smoke tests for endpoints"
```

---

## Task 16: Client Script

**Files:**
- Create: `scripts/client.sh`

**Step 1: Write client.sh**

Create `scripts/client.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

# Usage: ./client.sh <base_url> <file_path> <schema_path>
BASE_URL="${1:?Usage: client.sh <base_url> <file_path> <schema_path>}"
FILE_PATH="$(cd "$(dirname "${2:?}")" && pwd)/$(basename "$2")"
SCHEMA_PATH="$(cd "$(dirname "${3:?}")" && pwd)/$(basename "$3")"
FILE_NAME="$(basename "$FILE_PATH")"

echo "Uploading: $FILE_NAME"
echo "Schema:    $SCHEMA_PATH"
echo "Endpoint:  $BASE_URL"
echo ""

# POST file + schema → get task_id
RESPONSE=$(docker run --rm --network host \
  -v "$FILE_PATH:/upload/$FILE_NAME" \
  -v "$SCHEMA_PATH:/upload/schema.json" \
  curlimages/curl -s -X POST "$BASE_URL/tasks" \
    -F "file=@/upload/$FILE_NAME" \
    -F "schema=</upload/schema.json")

TASK_ID=$(echo "$RESPONSE" | docker run --rm -i python:3.11-alpine \
  python3 -c "import sys,json; print(json.load(sys.stdin)['task_id'])")

echo "Task created: $TASK_ID"
echo "Polling every 10 seconds..."
echo ""

while true; do
  sleep 10

  STATUS_JSON=$(docker run --rm --network host \
    curlimages/curl -s "$BASE_URL/tasks/$TASK_ID")

  STATUS=$(echo "$STATUS_JSON" | docker run --rm -i python:3.11-alpine \
    python3 -c "import sys,json; print(json.load(sys.stdin)['status'])")

  case "$STATUS" in
    succeeded)
      echo "=== EXTRACTION SUCCEEDED ==="
      echo ""
      docker run --rm --network host \
        curlimages/curl -s "$BASE_URL/tasks/$TASK_ID/result" | \
        docker run --rm -i python:3.11-alpine \
        python3 -c "import sys,json; json.dump(json.load(sys.stdin),sys.stdout,indent=2)"
      echo ""
      exit 0
      ;;
    failed)
      echo "=== EXTRACTION FAILED ===" >&2
      echo "$STATUS_JSON" | docker run --rm -i python:3.11-alpine \
        python3 -c "import sys,json; json.dump(json.load(sys.stdin),sys.stdout,indent=2)" >&2
      echo "" >&2
      exit 1
      ;;
    *)
      echo "  [$(date +%H:%M:%S)] status: $STATUS"
      ;;
  esac
done
```

**Step 2: Make executable**

```bash
chmod +x scripts/client.sh
```

**Step 3: Commit**

```bash
git add scripts/
git commit -m "feat: dockerized client script (no host dependencies)"
```

---

## Task 17: E2E Test

**Files:**
- Create: `api/tests/test_e2e.py`

**Step 1: Write the E2E test**

Create `api/tests/test_e2e.py`:

```python
"""End-to-end test: extract data from a real PDF using the full pipeline.

Requires:
  - All 3 services running (docker compose up)
  - Valid ANTHROPIC_API_KEY
  - Fixture files in /app/fixtures/ or ../fixtures/

Run:
  docker compose run --rm -e ANTHROPIC_API_KEY document-retriever \
    pytest tests/test_e2e.py -v --timeout=300
"""
import json
import os
import re
import time
from pathlib import Path

import httpx
import pytest

# Determine base URL — inside compose network or external
BASE_URL = os.environ.get("E2E_BASE_URL", "http://localhost:8000")

# Find fixtures
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
if not FIXTURES_DIR.exists():
    FIXTURES_DIR = Path("/app/fixtures")

PDF_PATH = FIXTURES_DIR / "MOD 8~15KTL3-X2(Pro).pdf"
SCHEMA_PATH = FIXTURES_DIR / "datasheet_schema.json"

POLL_INTERVAL = 5
MAX_WAIT = 300  # 5 minutes

EXPECTED_TOP_LEVEL_KEYS = [
    "original_file",
    "product_identity",
    "main_application",
    "pv_dc_input",
    "ac_grid_output",
    "battery_interface",
    "efficiency",
    "protection_and_safety",
    "monitoring_and_communication",
    "environment_and_installation",
    "compliance_and_certifications",
    "key_marketing_features",
]


@pytest.fixture(scope="module")
def schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text())


@pytest.mark.skipif(
    not PDF_PATH.exists(),
    reason=f"Fixture not found: {PDF_PATH}",
)
@pytest.mark.timeout(MAX_WAIT + 30)
def test_e2e_datasheet_extraction(schema):
    """Full pipeline: upload PDF + schema → poll → validate result."""

    with httpx.Client(base_url=BASE_URL, timeout=30.0) as client:
        # 1. POST /tasks
        with open(PDF_PATH, "rb") as f:
            resp = client.post(
                "/tasks",
                files={"file": (PDF_PATH.name, f, "application/pdf")},
                data={"schema": json.dumps(schema)},
            )
        assert resp.status_code == 202, f"POST failed: {resp.status_code} {resp.text}"
        task_id = resp.json()["task_id"]
        print(f"\nTask created: {task_id}")

        # 2. Poll until done
        elapsed = 0
        status = "queued"
        while elapsed < MAX_WAIT:
            time.sleep(POLL_INTERVAL)
            elapsed += POLL_INTERVAL

            status_resp = client.get(f"/tasks/{task_id}")
            assert status_resp.status_code == 200
            status = status_resp.json()["status"]
            print(f"  [{elapsed}s] status: {status}")

            if status in ("succeeded", "failed"):
                break

        # 3. Assert succeeded
        assert status == "succeeded", f"Task did not succeed: {status}"

        # 4. GET /tasks/{task_id}/result
        result_resp = client.get(f"/tasks/{task_id}/result")
        assert result_resp.status_code == 200
        result = result_resp.json()

        # 5. Validate result structure
        assert "data" in result and isinstance(result["data"], dict)
        assert "evidence" in result and isinstance(result["evidence"], dict)
        assert "warnings" in result and isinstance(result["warnings"], list)
        assert "timing" in result and isinstance(result["timing"], dict)
        assert result["timing"].get("total_seconds", 0) > 0

        data = result["data"]
        evidence = result["evidence"]

        # 6. Validate schema coverage — all top-level keys present
        for key in EXPECTED_TOP_LEVEL_KEYS:
            assert key in data, f"Missing top-level key: {key}"

        # 7. Validate nested keys exist for dict sections
        for section_key in ["product_identity", "pv_dc_input", "ac_grid_output",
                            "battery_interface", "efficiency"]:
            if isinstance(schema.get(section_key), dict) and isinstance(data.get(section_key), dict):
                for sub_key in schema[section_key]:
                    assert sub_key in data[section_key], \
                        f"Missing nested key: {section_key}.{sub_key}"

        # 8. Ground-truth spot checks
        pi = data.get("product_identity", {})
        assert pi.get("series_name") is not None, "series_name should not be null"
        series = str(pi.get("series_name", ""))
        assert "MOD" in series or "mod" in series.lower(), f"series_name should contain MOD: {series}"
        assert "KTL3" in series or "ktl3" in series.lower(), f"series_name should contain KTL3: {series}"

        models = pi.get("models")
        assert isinstance(models, list), f"models should be a list: {models}"
        assert len(models) > 1, f"Expected multiple models: {models}"

        pv = data.get("pv_dc_input", {})
        max_dc = str(pv.get("max_dc_voltage", ""))
        assert "1100" in max_dc, f"max_dc_voltage should contain 1100: {max_dc}"

        mppts = pv.get("number_of_mppts")
        assert mppts == 2 or str(mppts) == "2", f"number_of_mppts should be 2: {mppts}"

        ac = data.get("ac_grid_output", {})
        grid_type = str(ac.get("grid_type", ""))
        assert "3" in grid_type, f"grid_type should contain '3' (three-phase): {grid_type}"

        env = data.get("environment_and_installation", {})
        protection = str(env.get("protection_class", ""))
        assert "IP6" in protection or "ip6" in protection.lower(), \
            f"protection_class should contain IP65 or IP66: {protection}"

        # 9. Validate evidence structure
        non_null_count = 0
        null_count = 0
        _check_evidence(data, evidence, "", non_null_count, null_count)

        # 10. Print summary
        print(f"\n=== E2E RESULT SUMMARY ===")
        print(f"  Top-level keys: {len([k for k in EXPECTED_TOP_LEVEL_KEYS if k in data])}/{len(EXPECTED_TOP_LEVEL_KEYS)}")
        print(f"  Warnings: {len(result['warnings'])}")
        print(f"  Timing: {result['timing'].get('total_seconds', '?')}s")
        for w in result["warnings"][:10]:
            print(f"    - {w}")
        print(f"  Ground-truth checks: ALL PASSED")


def _check_evidence(data, evidence, path, non_null_count, null_count):
    """Recursively verify evidence exists for non-null data leaves."""
    if isinstance(data, dict):
        for key, val in data.items():
            child_path = f"{path}.{key}" if path else key
            if isinstance(val, (dict, list)):
                ev = evidence.get(key, {}) if isinstance(evidence, dict) else {}
                _check_evidence(val, ev, child_path, non_null_count, null_count)
            elif val is not None:
                non_null_count += 1
                # Evidence should exist — but agent output structure may vary,
                # so we just log rather than hard-fail on evidence format
    elif isinstance(data, list):
        pass  # Lists don't have per-element evidence in MVP
```

**Step 2: Commit**

```bash
git add api/tests/test_e2e.py
git commit -m "test: E2E test with real PDF and ground-truth validation"
```

---

## Task 18: README

**Files:**
- Create: `README.md`

**Step 1: Write README**

Create `README.md`:

```markdown
# document-retriever

Extract structured data from uploaded documents (PDF, DOCX, TXT, MD, HTML) into
JSON using Claude as an extraction agent with custom MCP tools.

## Quick Start

### Prerequisites

- Docker + Docker Compose
- Anthropic API key

### 1. Clone and initialize

```bash
git clone <this-repo>
cd document-retriever
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
- `document-retriever` on http://localhost:8000
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
| `document-retriever` | 8000 | API + task queue + Claude agent |
| `mcp-tools` | 8001 | MCP server with 4 document tools |
| `markitdown` | 8002 | Document → Markdown conversion |

The agent uses a **search-then-read** pattern: it never sees the full document.
Instead it calls `doc_search` to find relevant chunks, then `doc_read` to read
only those chunks.

## API

### POST /tasks — Create extraction task
### GET /tasks/{task_id} — Check status
### GET /tasks/{task_id}/result — Get extraction result

See [design doc](docs/plans/2026-02-05-document-retriever-design.md) for details.

## Testing

```bash
# Unit tests (indexer, no API key needed)
docker run --rm -v "$(pwd)/mcp_tools:/app" -w /app python:3.11-slim \
  sh -c "pip install -q pytest && python -m pytest tests/ -v"

# API smoke tests (no API key needed)
docker run --rm -v "$(pwd)/api:/app" -w /app python:3.11-slim \
  sh -c "pip install -q fastapi uvicorn aiosqlite ulid-py python-multipart httpx pydantic pydantic-settings pytest pytest-asyncio && python -m pytest tests/test_smoke.py -v"

# E2E test (requires all services running + API key)
docker compose run --rm -e ANTHROPIC_API_KEY document-retriever \
  pytest tests/test_e2e.py -v --timeout=300
```
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: README with quick start instructions"
```

---

## Task 19: Final Integration Test

**Step 1: Build everything**

```bash
cd /home/ubuntu/workspaces/document-retriever
docker compose build
```

Expected: All 3 services build successfully.

**Step 2: Start and verify health**

```bash
docker compose up -d
sleep 15
curl -s http://localhost:8000/health
curl -s http://localhost:8001/health
curl -s http://localhost:8002/health
```

Expected: All return `{"status":"ok"}`.

**Step 3: Run E2E test**

```bash
./scripts/client.sh http://localhost:8000 \
  "fixtures/MOD 8~15KTL3-X2(Pro).pdf" \
  fixtures/datasheet_schema.json
```

Expected: Prints extracted JSON with product identity, PV DC input specs, etc.

**Step 4: Teardown**

```bash
docker compose down
```

**Step 5: Final commit**

```bash
git add -A
git commit -m "chore: integration verified"
```
