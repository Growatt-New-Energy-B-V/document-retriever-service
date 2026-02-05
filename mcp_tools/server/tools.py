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
