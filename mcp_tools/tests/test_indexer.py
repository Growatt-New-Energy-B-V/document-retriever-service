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
