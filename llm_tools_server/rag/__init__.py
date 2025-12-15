"""RAG (Retrieval-Augmented Generation) module for document search and indexing.

This module provides local RAG capabilities with:
- Sitemap-based and recursive web crawling
- Semantic HTML-aware chunking with parent-child relationships
- Hybrid search (BM25 + semantic vector search)
- Cross-encoder re-ranking
- Contextual retrieval (Anthropic's approach for ~40-50% fewer retrieval failures)
- Incremental index updates
- FAISS vector storage (CPU-optimized, local)

Example usage:
    from llm_tools_server.rag import DocSearchIndex, RAGConfig

    # Configure RAG with contextual retrieval
    config = RAGConfig(
        base_url="https://docs.example.com",
        cache_dir="./doc_index",
        # Enable contextual retrieval (requires local Ollama)
        contextual_retrieval_enabled=True,
        contextual_model="llama3.2",
    )

    # Build index (includes context generation)
    index = DocSearchIndex(config)
    index.crawl_and_index()

    # Search
    results = index.search("How do I configure authentication?", top_k=5)
"""

from .config import RAGConfig
from .contextualizer import ChunkContextualizer
from .crawler import SitemapChanges
from .indexer import DocSearchIndex
from .updater import PeriodicIndexUpdater, UpdateResult, UpdaterStatus

__all__ = [
    "ChunkContextualizer",
    "DocSearchIndex",
    "PeriodicIndexUpdater",
    "RAGConfig",
    "SitemapChanges",
    "UpdateResult",
    "UpdaterStatus",
]
