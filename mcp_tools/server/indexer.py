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
