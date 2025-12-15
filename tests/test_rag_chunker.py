"""Lightweight tests for the HTML chunker (no network)."""

import pytest

# Skip if optional tokenizer/soup deps are unavailable
pytest.importorskip("tiktoken")
pytest.importorskip("bs4")

from llm_tools_server.rag.chunker import semantic_chunk_html


@pytest.mark.unit
def test_semantic_chunk_html_creates_parent_and_children():
    """Chunker should return parent/child records with parent links."""
    html = "<html><body><h1>Intro</h1><p>" + ("content " * 30) + "</p></body></html>"

    chunks = semantic_chunk_html(
        html=html,
        url="https://example.com/docs",
        child_min_tokens=5,
        child_max_tokens=50,
        parent_min_tokens=5,
        parent_max_tokens=80,
        absolute_max_tokens=100,
    )

    assert chunks["parents"]
    assert chunks["children"]

    parent_ids = {parent["chunk_id"] for parent in chunks["parents"]}
    assert all(child["parent_id"] in parent_ids for child in chunks["children"])
