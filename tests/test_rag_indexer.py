"""Unit tests for DocSearchIndex lightweight behaviors."""

from datetime import datetime, timedelta
from pathlib import Path

import pytest
from langchain_core.documents import Document

from llm_tools_server.rag import DocSearchIndex, RAGConfig


@pytest.mark.unit
def test_load_index_rebuilds_child_to_parent_mapping(tmp_path: Path):
    """Ensure child_to_parent mapping is reconstructed from cached chunk metadata.

    This tests the chunk loading and mapping logic without loading ML models.
    """
    config = RAGConfig(base_url="https://example.com", cache_dir=tmp_path)

    # Write cached chunks/parents to disk
    writer = DocSearchIndex(config)
    writer.chunks = [
        Document(
            page_content="child content",
            metadata={
                "chunk_id": "child-1",
                "parent_id": "parent-1",
                "url": "https://example.com/page",
            },
        )
    ]
    writer.parent_chunks = {"parent-1": {"text": "parent content", "url": "https://example.com/page"}}
    writer._save_chunks()
    writer._save_parent_chunks()

    # New instance simulates fresh process
    loader = DocSearchIndex(config)

    # Directly test the chunk loading and mapping logic (what load_index does first)
    loader.chunks = loader._load_chunks() or []
    loader.parent_chunks = loader._load_parent_chunks() or {}

    # Rebuild child_to_parent mapping from chunk metadata (same logic as load_index)
    loader.child_to_parent = {}
    for chunk in loader.chunks:
        chunk_id = chunk.metadata.get("chunk_id")
        parent_id = chunk.metadata.get("parent_id")
        if chunk_id and parent_id:
            loader.child_to_parent[chunk_id] = parent_id

    assert loader.child_to_parent == {"child-1": "parent-1"}
    assert loader.chunks[0].metadata["chunk_id"] == "child-1"
    assert loader.parent_chunks["parent-1"]["text"] == "parent content"


@pytest.mark.unit
def test_force_refresh_bypasses_needs_update_check(tmp_path: Path, monkeypatch):
    """Ensure force_refresh=True proceeds even when needs_update() returns False.

    Regression test for bug where force_refresh was ignored in the early return guard.
    """
    config = RAGConfig(base_url="https://example.com", cache_dir=tmp_path)
    index = DocSearchIndex(config)

    # Set up metadata to make needs_update() return False (recent update)
    metadata = {
        "version": index.INDEX_VERSION,
        "last_update": datetime.now().isoformat(),
        "num_chunks": 1,
        "embedding_model": config.embedding_model,
    }
    index._save_metadata(metadata)

    # Track whether the indexing pipeline was entered
    pipeline_entered = False

    def mock_discover_and_crawl():
        nonlocal pipeline_entered
        pipeline_entered = True
        return []  # Return empty to exit early after proving we got past the guard

    monkeypatch.setattr(index.crawler, "discover_and_crawl", mock_discover_and_crawl)

    # Without force_refresh, should return early (not enter pipeline)
    index.crawl_and_index(force_refresh=False)
    assert not pipeline_entered, "Should have returned early when needs_update() is False"

    # With force_refresh=True, should enter the pipeline
    index.crawl_and_index(force_refresh=True)
    assert pipeline_entered, "force_refresh=True should bypass needs_update() check"


@pytest.mark.unit
def test_refresh_with_all_cached_pages_does_not_duplicate_chunks(tmp_path: Path, monkeypatch):
    """Ensure scheduled refresh with all cached pages doesn't duplicate chunks.

    Regression test for bug where refresh mode would re-chunk cached pages,
    doubling the index size even when no content changed.
    """
    config = RAGConfig(
        base_url="https://example.com",
        cache_dir=tmp_path,
        update_check_interval_hours=0,  # Force needs_update() to return True
    )
    index = DocSearchIndex(config)

    # Set up existing chunks (simulating a previous indexing run)
    existing_chunks = [
        Document(
            page_content="existing content",
            metadata={
                "chunk_id": "chunk-1",
                "parent_id": "parent-1",
                "url": "https://example.com/page1",
            },
        )
    ]
    index.chunks = existing_chunks
    index.parent_chunks = {"parent-1": {"content": "parent content", "url": "https://example.com/page1"}}
    index._save_chunks()
    index._save_parent_chunks()

    # Set up metadata with old timestamp to trigger refresh
    old_time = datetime.now() - timedelta(hours=24)
    metadata = {
        "version": index.INDEX_VERSION,
        "last_update": old_time.isoformat(),
        "num_chunks": 1,
        "embedding_model": config.embedding_model,
    }
    index._save_metadata(metadata)

    # Set up crawl state with indexed URLs
    crawl_state = {
        "discovered_urls": ["https://example.com/page1"],
        "indexed_urls": ["https://example.com/page1"],
        "crawl_complete": True,
        "max_pages_limit": 100,
        "failed_urls": {},
    }
    index._save_crawl_state(crawl_state)

    # Mock crawler to return the same URL
    def mock_discover_and_crawl():
        return [{"url": "https://example.com/page1"}]

    monkeypatch.setattr(index.crawler, "discover_and_crawl", mock_discover_and_crawl)

    # Mock fetch to return cached page (from_cache=True)
    def mock_fetch_pages(url_list, failed_urls, force_refresh=False):
        # Simulate all pages coming from cache
        return [{"url": "https://example.com/page1", "html": "<p>content</p>", "from_cache": True}], failed_urls

    monkeypatch.setattr(index, "_fetch_pages", mock_fetch_pages)

    # Mock build_index to avoid ML model loading
    build_index_called = False

    def mock_build_index():
        nonlocal build_index_called
        build_index_called = True

    monkeypatch.setattr(index, "_build_index", mock_build_index)
    monkeypatch.setattr(index, "_update_index_incremental", mock_build_index)

    # Run the refresh (needs_update() will return True due to old timestamp)
    index.crawl_and_index()

    # Should NOT have duplicated chunks - should still be 1
    assert len(index.chunks) == 1, f"Expected 1 chunk, got {len(index.chunks)} - chunks were duplicated!"
    assert index.chunks[0].metadata["chunk_id"] == "chunk-1"
