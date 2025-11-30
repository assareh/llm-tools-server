"""Document indexer with semantic search and hybrid retrieval.

Main class for building and searching document indexes using:
- HTML semantic chunking
- HuggingFace embeddings
- FAISS vector store
- Hybrid BM25 + semantic search
- Cross-encoder re-ranking
"""

import hashlib
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import torch
from bs4 import BeautifulSoup
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from readability import Document as ReadabilityDocument
from sentence_transformers import CrossEncoder
from tqdm import tqdm

from .chunker import semantic_chunk_html
from .config import RAGConfig
from .contextualizer import ChunkContextualizer
from .crawler import DocumentCrawler

logger = logging.getLogger(__name__)


class DocSearchIndex:
    """Main document search index with crawling, chunking, embedding, and hybrid search."""

    # Index version for cache invalidation
    INDEX_VERSION = "1.1.0-chunker-v2"

    # Class-level flag to ensure global configuration is only set once
    _global_config_applied = False

    def __init__(self, config: RAGConfig, server_config=None):
        """Initialize the document search index.

        Args:
            config: RAG configuration
            server_config: Optional ServerConfig for backend settings (used by contextual retrieval).
                          If using contextual retrieval, this provides the LLM backend config.
        """
        self.server_config = server_config
        # Apply global configuration once per process
        if not DocSearchIndex._global_config_applied:
            # Suppress noisy "ruthless removal did not work" messages from readability library
            logging.getLogger("readability.readability").setLevel(logging.WARNING)
            # Disable tokenizers parallelism to prevent fork-related warnings when using WebUI
            # This is safe because we use ThreadPoolExecutor for parallel operations instead
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            DocSearchIndex._global_config_applied = True

        self.config = config

        # Create cache directories
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)

        self.content_dir = self.cache_dir / "pages"
        self.content_dir.mkdir(exist_ok=True)

        self.index_dir = self.cache_dir / "index"
        self.index_dir.mkdir(exist_ok=True)

        # Cache files
        self.metadata_file = self.cache_dir / "metadata.json"
        self.chunks_file = self.cache_dir / "chunks.json"
        self.parent_chunks_file = self.cache_dir / "parent_chunks.json"
        self.crawl_state_file = self.cache_dir / "crawl_state.json"

        # Components (lazy-loaded)
        self.embeddings: HuggingFaceEmbeddings | None = None
        self.vectorstore: FAISS | None = None
        self.bm25_retriever: BM25Retriever | None = None
        self.ensemble_retriever: EnsembleRetriever | None = None
        self.cross_encoder: CrossEncoder | None = None

        # Storage
        self.chunks: list[Document] = []
        self.parent_chunks: dict[str, dict[str, Any]] = {}  # chunk_id -> parent content/metadata
        self.child_to_parent: dict[str, str] = {}  # child_chunk_id -> parent_chunk_id

        # Crawler
        self.crawler = DocumentCrawler(
            base_url=config.base_url,
            cache_dir=self.cache_dir,
            manual_urls=config.manual_urls,
            manual_urls_only=config.manual_urls_only,
            max_crawl_depth=config.max_crawl_depth,
            rate_limit_delay=config.rate_limit_delay,
            max_workers=config.max_workers,
            max_pages=config.max_pages,
            request_timeout=config.request_timeout,
            url_include_patterns=config.url_include_patterns,
            url_exclude_patterns=config.url_exclude_patterns,
            show_progress=config.show_progress,
        )

        # Contextualizer for Anthropic's contextual retrieval approach
        self.contextualizer = ChunkContextualizer(config, self.cache_dir, server_config)

    def needs_update(self) -> bool:
        """Check if index needs rebuilding.

        Returns:
            True if index needs update
        """
        metadata = self._load_metadata()

        if "last_update" not in metadata:
            logger.info("[RAG] No previous index found, needs initial build")
            return True

        # Check version
        if metadata.get("version") != self.INDEX_VERSION:
            logger.info(
                f"[RAG] Index version changed (old: {metadata.get('version')}, new: {self.INDEX_VERSION}), needs rebuild"
            )
            return True

        # Check embedding model compatibility
        saved_model = metadata.get("embedding_model")
        if saved_model and saved_model != self.config.embedding_model:
            logger.info(
                f"[RAG] Embedding model changed (index: {saved_model}, config: {self.config.embedding_model}), "
                "needs rebuild to avoid index corruption"
            )
            return True

        # Check if max_pages has increased (expansion scenario)
        crawl_state = self._load_crawl_state()
        previous_max_pages = crawl_state.get("max_pages_limit")
        if (
            previous_max_pages is not None
            and self.config.max_pages is not None
            and self.config.max_pages > previous_max_pages
        ):
            logger.info(f"[RAG] max_pages increased from {previous_max_pages} to {self.config.max_pages}, needs update")
            return True

        # Check time since last update
        last_update = datetime.fromisoformat(metadata["last_update"])
        time_since_update = datetime.now() - last_update
        update_interval = timedelta(hours=self.config.update_check_interval_hours)

        needs_update = time_since_update >= update_interval
        if needs_update:
            logger.info(f"[RAG] Update interval exceeded ({time_since_update})")
        else:
            logger.info(f"[RAG] Recent update found ({time_since_update} ago)")

        return needs_update

    def crawl_and_index(self, force_rebuild: bool = False, force_refresh: bool = False):
        """Discover URLs, crawl pages, chunk content, and build search index.

        Supports:
        - Incremental updates (resume interrupted crawls)
        - Expanding max_pages limit without full rebuild
        - Stateful crawling with progress tracking
        - Staleness refresh for existing URLs (TTL-based or forced)
        - Automatic embedding-only rebuild when model changes (skips crawling)

        Args:
            force_rebuild: Force full rebuild of index and crawl state (clears all state)
            force_refresh: Force refetch of all cached pages (bypasses page cache, but keeps crawl state)
        """
        if not force_rebuild and not force_refresh and not self.needs_update():
            logger.info("[RAG] Index is up-to-date, loading from cache")
            self.load_index()
            return

        # Check if only the embedding model changed (can skip crawling)
        metadata = self._load_metadata()
        saved_model = metadata.get("embedding_model")
        chunks_file = self.cache_dir / "chunks.json"
        embedding_model_changed = (
            saved_model
            and saved_model != self.config.embedding_model
            and chunks_file.exists()
            and not force_rebuild
            and not force_refresh
        )

        if embedding_model_changed:
            logger.info(
                f"[RAG] Embedding model changed ({saved_model} -> {self.config.embedding_model}), "
                "rebuilding embeddings from saved chunks (skipping crawl)..."
            )
            self.rebuild_embeddings()
            return

        logger.info("[RAG] " + "=" * 70)
        logger.info("[RAG] Starting document indexing pipeline")
        logger.info("[RAG] " + "=" * 70)

        # Load existing crawl state
        crawl_state = self._load_crawl_state()
        indexed_urls_set = set(crawl_state.get("indexed_urls", []))
        discovered_urls = crawl_state.get("discovered_urls", [])
        previous_max_pages = crawl_state.get("max_pages_limit")
        crawl_complete = crawl_state.get("crawl_complete", False)
        failed_urls = crawl_state.get("failed_urls", {})

        # Determine if we're resuming, expanding, or starting fresh
        is_resuming = bool(indexed_urls_set) and not force_rebuild
        is_expanding = (
            previous_max_pages is not None
            and self.config.max_pages is not None
            and self.config.max_pages > previous_max_pages
        )

        # Determine if we're doing a refresh of existing content
        is_refreshing = force_refresh or (is_resuming and self.needs_update())

        if force_rebuild:
            logger.info("[RAG] Force rebuild requested, starting fresh")
            indexed_urls_set = set()
            discovered_urls = []
            crawl_complete = False
            failed_urls = {}
        elif force_refresh:
            logger.info("[RAG] Force refresh requested, will refetch all pages")
        elif is_resuming:
            logger.info(f"[RAG] Resuming crawl: {len(indexed_urls_set)} URLs already indexed")
        elif is_expanding:
            logger.info(
                f"[RAG] Expanding max_pages from {previous_max_pages} to {self.config.max_pages}, "
                "will index additional pages"
            )

        # Phase 1: Discover URLs (if needed)
        if not crawl_complete or is_expanding or force_rebuild:
            logger.info("[RAG] Phase 1/4: Discovering URLs")
            start_time = time.time()
            url_list = self.crawler.discover_and_crawl()

            # Update state
            discovered_urls = [url_info["url"] for url_info in url_list]

            # Only mark crawl complete if we actually found URLs
            # This prevents caching an empty result from a failed crawl
            if discovered_urls:
                crawl_complete = True
                crawl_state.update(
                    {
                        "discovered_urls": discovered_urls,
                        "crawl_complete": crawl_complete,
                        "max_pages_limit": self.config.max_pages,
                    }
                )
                self._save_crawl_state(crawl_state)
                logger.info(f"[RAG] Discovered {len(discovered_urls)} URLs in {time.time() - start_time:.1f}s")
            else:
                logger.warning("[RAG] Crawl found 0 URLs, not caching result")
        else:
            logger.info(f"[RAG] Phase 1/4: Using cached URL list ({len(discovered_urls)} URLs)")
            url_list = [{"url": url} for url in discovered_urls]

        if not url_list:
            logger.error("[RAG] No URLs discovered!")
            return

        # Determine which URLs to fetch
        # - force_refresh: fetch all URLs (existing ones will have cache bypassed)
        # - normal mode: only fetch new URLs not already indexed
        if is_refreshing:
            urls_to_fetch = url_list  # Fetch all URLs, cache bypass handled by force_refresh
            logger.info(f"[RAG] Refresh mode: will check {len(urls_to_fetch)} URLs for staleness")
        else:
            urls_to_fetch = [url_info for url_info in url_list if url_info["url"] not in indexed_urls_set]

        # Filter out URLs that have failed too many times
        skipped_urls = []
        filtered_urls_to_fetch = []
        for url_info in urls_to_fetch:
            url = url_info["url"]
            if url in failed_urls and failed_urls[url].get("failure_count", 0) >= self.config.max_url_retries:
                skipped_urls.append(url)
            else:
                filtered_urls_to_fetch.append(url_info)

        if skipped_urls:
            logger.info(f"[RAG] Skipping {len(skipped_urls)} URLs that exceeded {self.config.max_url_retries} retries")

        urls_to_fetch = filtered_urls_to_fetch

        if not urls_to_fetch:
            logger.info("[RAG] All URLs already indexed or skipped, loading existing index")
            self.load_index()
            return

        if is_refreshing:
            logger.info(f"[RAG] {len(urls_to_fetch)} URLs to check/refresh (out of {len(url_list)} total)")
        else:
            logger.info(f"[RAG] {len(urls_to_fetch)} new URLs to index (out of {len(url_list)} total)")

        # Phase 2: Fetch pages
        logger.info(f"[RAG] Phase 2/4: Fetching {len(urls_to_fetch)} pages...")
        start_time = time.time()
        new_pages, failed_urls = self._fetch_pages(urls_to_fetch, failed_urls, force_refresh=force_refresh)
        logger.info(f"[RAG] Fetched {len(new_pages)} pages in {time.time() - start_time:.1f}s")

        # Save updated failure tracking
        crawl_state["failed_urls"] = failed_urls
        self._save_crawl_state(crawl_state)

        if not new_pages:
            logger.warning("[RAG] No new pages fetched!")
            if indexed_urls_set:
                logger.info("[RAG] Loading existing index")
                self.load_index()
            return

        # Phase 3: Chunk content
        logger.info(f"[RAG] Phase 3/4: Chunking {len(new_pages)} pages into semantic segments...")
        start_time = time.time()

        # Identify refreshed pages (pages that were refetched, not from cache)
        refreshed_urls = {page["url"] for page in new_pages if not page.get("from_cache")}

        # Filter pages to chunk: only chunk pages with fresh content, not cached pages
        # Cached pages already have their chunks in the index, re-chunking would create duplicates
        pages_to_chunk = [page for page in new_pages if not page.get("from_cache")]
        cached_page_count = len(new_pages) - len(pages_to_chunk)
        if cached_page_count > 0:
            logger.info(
                f"[RAG] Skipping {cached_page_count} cached pages (already have chunks), chunking {len(pages_to_chunk)} fresh pages"
            )

        # If resuming/expanding/refreshing, load existing chunks first
        if is_resuming or is_expanding or is_refreshing:
            logger.info("[RAG] Loading existing chunks for incremental update...")
            existing_chunks = self._load_chunks() or []
            existing_parent_chunks = self._load_parent_chunks() or {}
            logger.info(
                f"[RAG] Loaded {len(existing_chunks)} existing child chunks, {len(existing_parent_chunks)} parent chunks"
            )

            # When refreshing, remove old chunks for URLs that were refetched
            if refreshed_urls:
                logger.info(f"[RAG] Removing old chunks for {len(refreshed_urls)} refreshed URLs...")
                # Filter out chunks belonging to refreshed URLs
                existing_chunks = [c for c in existing_chunks if c.metadata.get("url") not in refreshed_urls]
                # Filter out parent chunks for refreshed URLs
                existing_parent_chunks = {
                    k: v for k, v in existing_parent_chunks.items() if v.get("url") not in refreshed_urls
                }
                logger.info(
                    f"[RAG] After removing stale chunks: {len(existing_chunks)} child, "
                    f"{len(existing_parent_chunks)} parent"
                )

            # Set up for incremental update
            self.chunks = existing_chunks
            self.parent_chunks = existing_parent_chunks
            # Rebuild child_to_parent mapping from existing chunks
            self.child_to_parent = {}
            for chunk in self.chunks:
                chunk_id = chunk.metadata.get("chunk_id")
                parent_id = chunk.metadata.get("parent_id")
                if chunk_id and parent_id:
                    self.child_to_parent[chunk_id] = parent_id
        else:
            # Fresh build - initialize empty
            logger.info("[RAG] Starting fresh chunking (no existing chunks)...")
            self.chunks = []
            self.parent_chunks = {}
            self.child_to_parent = {}

        # Create chunks from pages with fresh content only (not cached pages)
        new_chunk_count_before = len(self.chunks)
        page_contents = self._create_chunks(pages_to_chunk)
        new_chunk_count = len(self.chunks) - new_chunk_count_before

        logger.info(
            f"[RAG] Created {new_chunk_count} new child chunks from {len(pages_to_chunk)} pages in {time.time() - start_time:.1f}s"
        )
        logger.info(f"[RAG] Total chunks: {len(self.chunks)} child chunks, {len(self.parent_chunks)} parent chunks")

        # Phase 3.5: Contextual retrieval (if enabled)
        if self.config.contextual_retrieval_enabled and new_chunk_count > 0:
            logger.info("[RAG] Phase 3.5/4: Applying contextual retrieval...")
            start_time = time.time()

            # Convert chunks to dict format for contextualizer
            chunk_dicts = [
                {
                    "chunk_id": chunk.metadata.get("chunk_id"),
                    "content": chunk.page_content,
                    "url": chunk.metadata.get("url"),
                    "metadata": chunk.metadata,
                }
                for chunk in self.chunks[new_chunk_count_before:]  # Only new chunks
            ]

            # Generate and apply contextual prefixes
            contextualized = self.contextualizer.contextualize_chunks(chunk_dicts, page_contents)

            # Update chunks with contextualized content
            for i, ctx_chunk in enumerate(contextualized):
                idx = new_chunk_count_before + i
                self.chunks[idx] = Document(
                    page_content=ctx_chunk["content"],
                    metadata={
                        **self.chunks[idx].metadata,
                        "original_content": ctx_chunk.get("original_content", ctx_chunk["content"]),
                    },
                )

            logger.info(f"[RAG] Contextual retrieval complete in {time.time() - start_time:.1f}s")

        # Save chunks
        self._save_chunks()
        self._save_parent_chunks()

        # Update indexed URLs
        newly_indexed = {page["url"] for page in new_pages}
        indexed_urls_set.update(newly_indexed)
        crawl_state["indexed_urls"] = list(indexed_urls_set)
        self._save_crawl_state(crawl_state)

        # Phase 4: Build/update index
        logger.info(f"[RAG] Phase 4/4: Building search index from {len(self.chunks)} chunks...")
        logger.info("[RAG] This may take a few minutes - generating embeddings and building indexes...")
        start_time = time.time()

        # Determine update strategy:
        # - Refreshing with replaced content: must rebuild (can't incrementally remove from FAISS)
        # - Resuming/expanding without refresh: can use incremental
        # - Fresh build: full rebuild
        if refreshed_urls:
            # Full rebuild required when content was replaced
            logger.info(f"[RAG] Full rebuild required ({len(refreshed_urls)} URLs refreshed)...")
            self._build_index()
        elif is_resuming or is_expanding:
            # Incremental update (only adding new content)
            logger.info("[RAG] Using incremental index update (adding new chunks to existing index)...")
            self._update_index_incremental()
        else:
            # Full rebuild
            logger.info("[RAG] Building fresh index (full rebuild)...")
            self._build_index()

        logger.info(f"[RAG] ✓ Index built successfully in {time.time() - start_time:.1f}s")

        # Save metadata (including embedding model for validation on load)
        self._save_metadata(
            {
                "version": self.INDEX_VERSION,
                "last_update": datetime.now().isoformat(),
                "num_chunks": len(self.chunks),
                "embedding_model": self.config.embedding_model,
            }
        )

        logger.info("[RAG] " + "=" * 70)
        logger.info("[RAG] Indexing complete!")
        logger.info(f"[RAG] Total indexed: {len(indexed_urls_set)} URLs, {len(self.chunks)} chunks")
        logger.info("[RAG] " + "=" * 70)

    def load_index(self):
        """Load index from cache.

        Attempts to load the persisted FAISS index first for fast startup.
        Falls back to rebuilding from chunks if the saved index is missing or corrupted.
        BM25 retriever is always rebuilt (not persisted) but this is fast.
        """
        logger.info("[RAG] Loading index from cache...")

        # Load chunks
        self.chunks = self._load_chunks() or []
        self.parent_chunks = self._load_parent_chunks() or {}

        # Rebuild child_to_parent mapping from chunk metadata
        self.child_to_parent = {}
        for chunk in self.chunks:
            chunk_id = chunk.metadata.get("chunk_id")
            parent_id = chunk.metadata.get("parent_id")
            if chunk_id and parent_id:
                self.child_to_parent[chunk_id] = parent_id

        if not self.chunks:
            logger.warning("[RAG] No cached chunks found")
            return

        # Initialize components
        logger.info("[RAG] Initializing ML models (embeddings, re-rankers)...")
        self._initialize_components()

        # Try to load persisted FAISS index first (fast path)
        faiss_path = str(self.index_dir / "faiss_index")
        faiss_loaded = False

        if Path(faiss_path).exists():
            try:
                # Verify checksum before loading (raises ValueError if tampered)
                self._verify_faiss_checksum(faiss_path)

                logger.info(f"[RAG] Loading persisted FAISS index from {faiss_path}...")
                start = time.time()
                self.vectorstore = FAISS.load_local(
                    faiss_path, self.embeddings, allow_dangerous_deserialization=True  # Checksum verified above
                )
                logger.info(f"[RAG] ✓ Loaded FAISS index in {time.time() - start:.1f}s")
                faiss_loaded = True
            except Exception as e:
                logger.warning(f"[RAG] Failed to load persisted FAISS index: {e}")
                logger.info("[RAG] Will rebuild FAISS index from chunks...")

        if not faiss_loaded:
            # Fall back to rebuilding FAISS from chunks (slow path)
            logger.info(f"[RAG] Building FAISS vector index from {len(self.chunks)} chunks...")
            start = time.time()
            self.vectorstore = self._build_faiss_with_progress(self.chunks)
            logger.info(f"[RAG] ✓ FAISS index built in {time.time() - start:.1f}s")

            # Save the rebuilt index for next time
            logger.info(f"[RAG] Saving FAISS index to {faiss_path}...")
            self.vectorstore.save_local(faiss_path)
            self._save_faiss_checksum(faiss_path)
            logger.info("[RAG] ✓ FAISS index saved")

        # Build BM25 retriever (always rebuilt, fast)
        logger.info(f"[RAG] Building BM25 keyword retriever from {len(self.chunks)} chunks...")
        start = time.time()
        self.bm25_retriever = BM25Retriever.from_documents(self.chunks)
        self.bm25_retriever.k = (
            self.config.search_top_k * self.config.retriever_candidate_multiplier
        )  # Get more candidates for ensemble
        logger.info(f"[RAG] ✓ BM25 retriever built in {time.time() - start:.1f}s")

        # Build ensemble retriever (hybrid search)
        logger.info(
            f"[RAG] Building hybrid ensemble retriever "
            f"(BM25 weight: {self.config.hybrid_bm25_weight}, Semantic weight: {self.config.hybrid_semantic_weight})..."
        )
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[
                self.bm25_retriever,
                self.vectorstore.as_retriever(
                    search_kwargs={"k": self.config.search_top_k * self.config.retriever_candidate_multiplier}
                ),
            ],
            weights=[self.config.hybrid_bm25_weight, self.config.hybrid_semantic_weight],
        )
        logger.info("[RAG] ✓ Ensemble retriever ready")

        logger.info("[RAG] ✓ Index loaded successfully")

        # Start background contextualization if enabled
        if self.config.contextual_retrieval_enabled and self.config.contextual_retrieval_background:
            metadata = self._load_metadata()
            if not metadata.get("contextual_retrieval"):
                logger.info("[RAG] Starting background contextual retrieval...")
                self.start_background_contextualization()

    def rebuild_embeddings(self):
        """Rebuild FAISS index from saved chunks without re-crawling.

        Use this when changing embedding models - it skips the expensive crawling
        and chunking phases and just regenerates embeddings from saved chunks.

        Raises:
            ValueError: If no saved chunks found (need to crawl first)
        """
        logger.info("[RAG] " + "=" * 70)
        logger.info("[RAG] Rebuilding embeddings from saved chunks")
        logger.info("[RAG] " + "=" * 70)

        # Load saved chunks
        chunks_file = self.cache_dir / "chunks.json"

        if not chunks_file.exists():
            raise ValueError("No saved chunks found. Run crawl_and_index() first.")

        logger.info("[RAG] Loading saved chunks...")
        self.chunks = self._load_chunks()
        self.parent_chunks = self._load_parent_chunks()
        logger.info(f"[RAG] Loaded {len(self.chunks)} chunks, {len(self.parent_chunks)} parent chunks")

        if not self.chunks:
            raise ValueError("Chunks file exists but is empty. Run crawl_and_index() first.")

        # Initialize ML models
        logger.info("[RAG] Initializing embedding model...")
        self._initialize_components()

        # Rebuild FAISS index
        logger.info(f"[RAG] Rebuilding FAISS index with model: {self.config.embedding_model}")
        start_time = time.time()
        self._build_retrievers()
        logger.info(f"[RAG] ✓ Index rebuilt in {time.time() - start_time:.1f}s")

        # Update metadata with new embedding model
        self._save_metadata(
            {
                "version": self.INDEX_VERSION,
                "last_update": datetime.now().isoformat(),
                "num_chunks": len(self.chunks),
                "embedding_model": self.config.embedding_model,
            }
        )

        logger.info("[RAG] " + "=" * 70)
        logger.info("[RAG] Embedding rebuild complete!")
        logger.info("[RAG] " + "=" * 70)

    def add_contextual_retrieval(self, batch_size: int = 100, save_every: int = 50):
        """Add contextual retrieval to an existing index.

        This generates LLM context for chunks and rebuilds embeddings.
        Can be run separately after crawl_and_index() completes.
        Progress is saved incrementally and can be resumed if interrupted.

        Args:
            batch_size: Number of chunks to process in parallel
            save_every: Save context cache every N chunks

        Raises:
            ValueError: If no chunks found or index not built
        """
        logger.info("[RAG] " + "=" * 70)
        logger.info("[RAG] Adding contextual retrieval to existing index")
        logger.info("[RAG] " + "=" * 70)

        # Load chunks if not already loaded
        if not self.chunks:
            self.chunks = self._load_chunks()
            self.parent_chunks = self._load_parent_chunks()

        if not self.chunks:
            raise ValueError("No chunks found. Run crawl_and_index() first.")

        # Load page contents from cache for context generation
        logger.info("[RAG] Loading cached page contents...")
        page_contents = self._load_all_page_contents()

        if not page_contents:
            raise ValueError("No cached pages found. Run crawl_and_index() first.")

        logger.info(f"[RAG] Loaded {len(page_contents)} pages, {len(self.chunks)} chunks")

        # Convert Document objects to dicts for contextualizer
        chunk_dicts = []
        for doc in self.chunks:
            chunk_dict = {
                "content": doc.page_content,
                "chunk_id": doc.metadata.get("chunk_id", ""),
                "url": doc.metadata.get("url", ""),
                "metadata": doc.metadata,
            }
            chunk_dicts.append(chunk_dict)

        # Generate contexts (uses cache, saves incrementally)
        logger.info(f"[RAG] Generating contexts for {len(chunk_dicts)} chunks...")
        contextualized = self.contextualizer.contextualize_chunks(chunk_dicts, page_contents)

        # Update chunks with contextualized content
        for i, ctx_chunk in enumerate(contextualized):
            if ctx_chunk.get("original_content"):  # Was contextualized
                self.chunks[i] = Document(
                    page_content=ctx_chunk["content"],
                    metadata={
                        **self.chunks[i].metadata,
                        "original_content": ctx_chunk.get("original_content", ctx_chunk["content"]),
                    },
                )

        # Save updated chunks
        self._save_chunks()

        # Rebuild embeddings with contextualized content
        logger.info("[RAG] Rebuilding embeddings with contextualized chunks...")
        self._initialize_components()
        self._build_retrievers()

        # Update metadata
        self._save_metadata(
            {
                "version": self.INDEX_VERSION,
                "last_update": datetime.now().isoformat(),
                "num_chunks": len(self.chunks),
                "embedding_model": self.config.embedding_model,
                "contextual_retrieval": True,
            }
        )

        logger.info("[RAG] " + "=" * 70)
        logger.info("[RAG] Contextual retrieval added successfully!")
        logger.info("[RAG] " + "=" * 70)

    def start_background_contextualization(self, callback=None):
        """Start contextual retrieval in a background thread.

        The index remains usable while contexts are being generated.
        Once complete, embeddings are rebuilt automatically.

        Args:
            callback: Optional function to call when complete (receives self)

        Returns:
            threading.Thread: The background thread (already started)
        """
        import threading

        def _background_task():
            try:
                print("[RAG] Background contextualization started...", file=sys.stderr)
                logger.info("[RAG] Background contextualization started...")
                self.add_contextual_retrieval()
                print("[RAG] Background contextualization complete!", file=sys.stderr)
                logger.info("[RAG] Background contextualization complete!")
                if callback:
                    callback(self)
            except Exception as e:
                print(f"[RAG] Background contextualization failed: {e}", file=sys.stderr)
                logger.error(f"[RAG] Background contextualization failed: {e}")

        thread = threading.Thread(target=_background_task, daemon=True)
        thread.start()
        print("[RAG] Background contextualization thread started", file=sys.stderr)
        logger.info("[RAG] Background contextualization thread started")
        return thread

    def _load_all_page_contents(self) -> dict[str, str]:
        """Load all cached page contents for contextual retrieval.

        Returns:
            Dict mapping URL -> plain text content
        """
        page_contents = {}

        if not self.content_dir.exists():
            return page_contents

        for cache_file in self.content_dir.glob("*.json"):
            try:
                import json

                data = json.loads(cache_file.read_text())
                url = data.get("url", "")
                html = data.get("html", "")
                if url and html:
                    page_contents[url] = self._extract_page_text(html)
            except Exception as e:
                logger.debug(f"[RAG] Failed to load cached page {cache_file}: {e}")

        return page_contents

    def search(self, query: str, top_k: int | None = None, return_parent: bool = True) -> list[dict[str, Any]]:
        """Search the document index with hybrid retrieval and re-ranking.

        Args:
            query: Search query
            top_k: Number of results to return (default from config)
            return_parent: If True, return parent chunks for context

        Returns:
            List of search results with content, metadata, and scores
        """
        if top_k is None:
            top_k = self.config.search_top_k

        if not self.ensemble_retriever:
            logger.error("[RAG] Index not loaded, call load_index() or crawl_and_index() first")
            return []

        logger.debug(f"[RAG] Searching for: {query}")

        # Get initial candidates from hybrid search
        # Each retriever returns search_top_k * retriever_candidate_multiplier candidates
        # which are then fused using Reciprocal Rank Fusion (RRF)
        expected_candidates = self.config.search_top_k * self.config.retriever_candidate_multiplier
        candidates = self.ensemble_retriever.invoke(query)

        logger.debug(
            f"[RAG] Retrieved {len(candidates)} candidates from hybrid search "
            f"(expected ~{expected_candidates} per retriever before RRF deduplication)"
        )

        # Convert to result format
        results = []
        for doc in candidates:
            result = {
                "text": doc.page_content,
                "url": doc.metadata.get("url", ""),
                "heading_path": doc.metadata.get("heading_path_joined", ""),
                "metadata": doc.metadata,
                "score": 0.0,  # Will be set by re-ranker
            }

            # Get parent chunk if requested
            if return_parent:
                child_id = doc.metadata.get("chunk_id")
                if child_id and child_id in self.child_to_parent:
                    parent_id = self.child_to_parent[child_id]
                    if parent_id in self.parent_chunks:
                        result["parent_text"] = self.parent_chunks[parent_id]["content"]
                        result["parent_metadata"] = self.parent_chunks[parent_id].get("metadata")

            results.append(result)

        # Re-rank if enabled
        if self.config.rerank_enabled and results:
            results = self._rerank_results(query, results)

        # Return top-k
        return results[:top_k]

    def _fetch_pages(
        self, url_list: list[dict[str, Any]], failed_urls: dict[str, Any], force_refresh: bool = False
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Fetch pages with caching and parallel fetching, tracking failures.

        Args:
            url_list: List of URL info dicts
            failed_urls: Dict of failed URL tracking info
            force_refresh: If True, skip cache and refetch all pages

        Returns:
            Tuple of (list of page data dicts, updated failed_urls dict)
        """
        pages = []
        total = len(url_list)
        cache_hits = 0
        failed_count = 0

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all fetch tasks (with cache check)
            future_to_url = {
                executor.submit(self._fetch_page_with_cache, url_info, force_refresh): url_info for url_info in url_list
            }

            # Create progress bar for page fetching
            pbar = tqdm(
                as_completed(future_to_url),
                total=total,
                desc="Fetching pages",
                unit="page",
                disable=not self.config.show_progress,
                file=sys.stderr,
            )

            # Process completed tasks
            for future in pbar:
                url_info = future_to_url[future]
                url = url_info["url"]
                try:
                    result = future.result()
                    if result:
                        pages.append(result)
                        if result.get("from_cache"):
                            cache_hits += 1

                        # Success - clear any previous failures
                        failed_urls.pop(url, None)
                    else:
                        # Fetch returned None (failure)
                        failed_count += 1
                        logger.debug(f"[RAG] Failed to fetch page: {url}")
                        self._track_url_failure(url, failed_urls, "Failed to fetch page")

                except Exception as e:
                    failed_count += 1
                    logger.debug(f"[RAG] Failed to fetch {url}: {e}")
                    self._track_url_failure(url, failed_urls, str(e))

                # Update progress bar postfix with stats
                pbar.set_postfix_str(
                    f"cached={cache_hits}, failed={failed_count}",
                    refresh=True,
                )

        # Log summary
        if cache_hits > 0 and len(pages) > 0:
            logger.info(f"[RAG] Fetch complete: {len(pages)} pages ({cache_hits} from cache, {failed_count} failed)")
        elif failed_count > 0:
            logger.info(f"[RAG] Fetch complete: {len(pages)} pages ({failed_count} failed)")

        return pages, failed_urls

    def _fetch_page_with_cache(self, url_info: dict[str, Any], force_refresh: bool = False) -> dict[str, Any] | None:
        """Fetch a page with caching support.

        Args:
            url_info: URL info dict with url and optional lastmod
            force_refresh: If True, skip cache and refetch page

        Returns:
            Page data dict or None if failed
        """
        url = url_info["url"]
        lastmod = url_info.get("lastmod")

        # Try to load from cache (respects force_refresh and TTL)
        cached = self._load_cached_page(url, lastmod, force_refresh=force_refresh)
        if cached:
            return cached

        # Fetch fresh content
        result = self.crawler.fetch_page(url)
        if result:
            url, html = result

            # Extract main content using readability
            clean_html = self._extract_main_content(html, url)

            page_data = {"url": url, "html": clean_html, "lastmod": lastmod, "from_cache": False}

            # Save to cache
            self._save_cached_page(page_data)

            return page_data

        return None

    def _extract_main_content(self, html: str, url: str) -> str:
        """Extract main content from HTML using readability.

        Uses readability to extract the main content, but falls back to the <main> tag
        if readability removes too many code blocks (>50% loss). This protects technical
        documentation from losing code examples while still removing boilerplate.

        Fallback strategy:
        1. Try readability extraction
        2. If >50% code blocks lost, try extracting <main> tag
        3. If no <main> tag, use original HTML as last resort

        Args:
            html: Raw HTML content
            url: Page URL (used by readability for context)

        Returns:
            Cleaned HTML with just the main content
        """
        try:
            # Count code blocks in original HTML
            original_code_blocks = html.lower().count("<pre") + html.lower().count("<code")

            doc = ReadabilityDocument(html, url=url)
            clean_html = doc.summary()

            # Check if readability failed (returned essentially empty content)
            # or if code blocks were stripped (>50% loss)
            clean_code_blocks = clean_html.lower().count("<pre") + clean_html.lower().count("<code")
            readability_failed = len(clean_html) < 100  # Empty or near-empty output
            code_blocks_stripped = original_code_blocks > 0 and clean_code_blocks < original_code_blocks * 0.5

            if readability_failed or code_blocks_stripped:
                # Try extracting semantic HTML as fallback (cleaner than full HTML)
                fallback_html = self._extract_main_tag(html)
                if fallback_html:
                    reason = "failed" if readability_failed else "stripped code blocks"
                    logger.debug(
                        f"[RAG] Readability {reason} from {url}, using semantic HTML fallback "
                        f"(readability: {len(clean_html)} bytes, fallback: {len(fallback_html)} bytes)"
                    )
                    return fallback_html
                else:
                    # No semantic tags found, fall back to original HTML
                    reason = "failed" if readability_failed else "stripped code blocks"
                    logger.warning(
                        f"[RAG] Readability {reason} from {url}, no semantic tags, using original HTML "
                        f"(original: {len(html)} bytes, clean: {len(clean_html)} bytes)"
                    )
                    return html

            logger.debug(f"[RAG] Extracted main content from {url}")
            return clean_html
        except Exception as e:
            logger.warning(f"[RAG] Failed to extract main content from {url}: {e}, using original HTML")
            return html

    def _extract_main_tag(self, html: str) -> str | None:
        """Extract content from semantic HTML elements as fallback.

        Tries to find the most specific content container:
        1. div with 'mdxContent' class (HashiCorp/Next.js MDX content)
        2. <article> tag
        3. <main> tag

        Args:
            html: Raw HTML content

        Returns:
            HTML string of content, or None if not found
        """
        try:
            soup = BeautifulSoup(html, "html.parser")

            # Try mdxContent div first (HashiCorp docs specific, very clean)
            mdx_content = soup.find("div", class_=lambda x: x and "mdxContent" in x)
            if mdx_content:
                return str(mdx_content)

            # Try article tag
            article = soup.find("article")
            if article:
                return str(article)

            # Fall back to main tag
            main = soup.find("main")
            if main:
                return str(main)

            return None
        except Exception:
            return None

    def _get_page_cache_path(self, url: str) -> Path:
        """Get cache file path for a URL."""
        url_hash = hashlib.sha256(url.encode()).hexdigest()[:32]
        return self.content_dir / f"{url_hash}.json"

    def _load_cached_page(self, url: str, lastmod: str | None, force_refresh: bool = False) -> dict[str, Any] | None:
        """Load cached page content if still valid.

        Args:
            url: Page URL
            lastmod: Last modification date from sitemap
            force_refresh: If True, skip cache entirely and refetch

        Returns:
            Cached page data or None if cache invalid
        """
        # Skip cache entirely if force_refresh requested
        if force_refresh:
            logger.debug(f"[RAG] Force refresh requested, skipping cache for {url}")
            return None

        cache_path = self._get_page_cache_path(url)
        if not cache_path.exists():
            return None

        try:
            cached = json.loads(cache_path.read_text())

            # Check if lastmod matches (if we have one from sitemap)
            if lastmod and cached.get("lastmod") != lastmod:
                logger.debug(f"[RAG] Cache invalidated for {url} (lastmod changed)")
                return None

            # If no lastmod, apply TTL-based invalidation
            if not lastmod and self.config.page_cache_ttl_hours > 0:
                cached_at = cached.get("cached_at")
                if cached_at:
                    cached_time = datetime.fromisoformat(cached_at)
                    age = datetime.now() - cached_time
                    ttl = timedelta(hours=self.config.page_cache_ttl_hours)
                    if age >= ttl:
                        logger.debug(f"[RAG] Cache expired for {url} (age: {age}, TTL: {ttl})")
                        return None

            # Mark as from cache
            cached["from_cache"] = True
            logger.debug(f"[RAG] Loaded from cache: {url}")
            return cached

        except Exception as e:
            logger.debug(f"[RAG] Failed to load cache for {url}: {e}")
            return None

    def _save_cached_page(self, page_data: dict[str, Any]):
        """Save page content to cache with timestamp for TTL-based invalidation."""
        try:
            cache_path = self._get_page_cache_path(page_data["url"])
            # Don't save the from_cache flag, add cached_at timestamp
            save_data = {k: v for k, v in page_data.items() if k != "from_cache"}
            save_data["cached_at"] = datetime.now().isoformat()
            cache_path.write_text(json.dumps(save_data))
            logger.debug(f"[RAG] Cached: {page_data['url']}")
        except Exception as e:
            logger.warning(f"[RAG] Failed to cache page {page_data['url']}: {e}")

    def _extract_page_text(self, html: str) -> str:
        """Extract plain text from HTML for contextual retrieval.

        Args:
            html: HTML content

        Returns:
            Plain text content
        """
        try:
            soup = BeautifulSoup(html, "html.parser")

            # Remove script and style elements
            for element in soup(["script", "style", "nav", "footer", "header"]):
                element.decompose()

            # Get text with reasonable spacing
            text = soup.get_text(separator=" ", strip=True)

            # Clean up excessive whitespace
            text = " ".join(text.split())

            return text
        except Exception as e:
            logger.warning(f"[RAG] Failed to extract text from HTML: {e}")
            return ""

    def _create_chunks(self, pages: list[dict[str, Any]]) -> dict[str, str]:
        """Create parent-child chunks from pages using semantic HTML chunking.

        Appends new chunks to existing ones (for incremental updates).
        Deduplicates pages with identical content (keeping the first URL encountered).

        Args:
            pages: List of page data dicts with HTML

        Returns:
            Dict mapping URL -> plain text content (for contextual retrieval)
        """
        # Don't reset - append to existing chunks for incremental updates
        # (caller sets self.chunks to [] for full rebuild or existing chunks for incremental)

        # Deduplicate pages by content hash to avoid indexing identical content
        # (e.g., same content at different URLs, versioned pages with identical text)
        seen_content_hashes: set[str] = set()
        deduplicated_pages: list[dict[str, Any]] = []
        duplicates_skipped = 0

        for page in pages:
            content_hash = hashlib.sha256(page["html"].encode()).hexdigest()
            if content_hash not in seen_content_hashes:
                seen_content_hashes.add(content_hash)
                deduplicated_pages.append(page)
            else:
                duplicates_skipped += 1
                logger.debug(f"[RAG] Skipping duplicate content: {page['url']}")

        if duplicates_skipped > 0:
            logger.info(f"[RAG] Deduplicated {duplicates_skipped} pages with identical content")

        # Extract page text content for contextual retrieval
        page_contents: dict[str, str] = {}

        # Track chunks created during this call (for progress reporting)
        chunks_before = len(self.chunks)
        parents_before = len(self.parent_chunks)

        # Create progress bar for chunking
        pbar = tqdm(
            deduplicated_pages,
            desc="Chunking pages",
            unit="page",
            disable=not self.config.show_progress,
            file=sys.stderr,
        )

        for page in pbar:
            # Extract plain text for contextual retrieval
            page_contents[page["url"]] = self._extract_page_text(page["html"])

            try:
                # Use semantic chunking
                result = semantic_chunk_html(
                    html=page["html"],
                    url=page["url"],
                    child_min_tokens=self.config.child_chunk_min_tokens,
                    child_max_tokens=self.config.child_chunk_size,
                    parent_min_tokens=self.config.parent_chunk_min_tokens,
                    parent_max_tokens=self.config.parent_chunk_size,
                    absolute_max_tokens=self.config.absolute_max_chunk_tokens,
                )

                parents = result.get("parents", [])
                children = result.get("children", [])

                # Build a set of parent IDs that have children
                parents_with_children = {child.get("parent_id") for child in children if child.get("parent_id")}

                # Store parent chunks
                for parent in parents:
                    chunk_id = parent["chunk_id"]
                    # Convert metadata dataclass to dict for JSON serialization
                    metadata = parent.get("metadata")
                    if hasattr(metadata, "__dict__"):
                        metadata = vars(metadata)

                    self.parent_chunks[chunk_id] = {
                        "content": parent["content"],
                        "metadata": metadata,
                        "url": page["url"],
                        "lastmod": page.get("lastmod"),
                    }

                    # If this parent has no children, add it directly to searchable chunks
                    # This ensures content is searchable even when individual content blocks
                    # are too small to create child chunks
                    if chunk_id not in parents_with_children:
                        doc = Document(
                            page_content=parent["content"],
                            metadata={
                                **metadata,
                                "chunk_id": chunk_id,
                                "parent_id": None,  # This is a parent chunk, no parent above it
                                "url": page["url"],
                                "lastmod": page.get("lastmod"),
                                "is_parent_as_child": True,  # Flag indicating this parent is indexed directly
                            },
                        )
                        self.chunks.append(doc)

                # Create LangChain Documents from child chunks
                for child in children:
                    chunk_id = child["chunk_id"]
                    parent_id = child.get("parent_id")

                    # Track parent relationship
                    if parent_id:
                        self.child_to_parent[chunk_id] = parent_id

                    # Create document
                    metadata = child.get("metadata", {})
                    # Convert dataclass to dict if needed
                    if hasattr(metadata, "__dict__"):
                        metadata = vars(metadata)

                    doc = Document(
                        page_content=child["content"],
                        metadata={
                            **metadata,
                            "chunk_id": chunk_id,
                            "parent_id": parent_id,
                            "url": page["url"],
                            "lastmod": page.get("lastmod"),
                        },
                    )
                    self.chunks.append(doc)

                # Update progress bar with chunk counts
                new_chunks = len(self.chunks) - chunks_before
                new_parents = len(self.parent_chunks) - parents_before
                pbar.set_postfix_str(f"chunks={new_chunks}, parents={new_parents}", refresh=True)

            except Exception as e:
                logger.error(f"[RAG] Failed to chunk {page['url']}: {e}")
                continue

        return page_contents

    def _build_index(self):
        """Build FAISS vector index and retrievers."""
        if not self.chunks:
            logger.error("[RAG] No chunks to index!")
            return

        # Initialize components
        self._initialize_components()

        # Build retrievers
        self._build_retrievers()

    def _initialize_components(self):
        """Initialize embeddings and cross-encoders."""
        if self.embeddings is None:
            logger.info(f"[RAG] Loading embedding model: {self.config.embedding_model}...")
            logger.info("[RAG] (First-time model download may take a minute)")
            start = time.time()
            # Auto-detect best device (MPS for Apple Silicon, CUDA for NVIDIA, else CPU)
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
            logger.info(f"[RAG] Using device: {device}")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.config.embedding_model,
                model_kwargs={"device": device},
                encode_kwargs={"normalize_embeddings": True},
            )
            logger.info(f"[RAG] ✓ Embedding model loaded in {time.time() - start:.1f}s")

        if self.config.rerank_enabled and self.cross_encoder is None:
            logger.info(f"[RAG] Loading cross-encoder: {self.config.rerank_model}...")
            start = time.time()
            self.cross_encoder = CrossEncoder(self.config.rerank_model)
            logger.info(f"[RAG] ✓ Cross-encoder loaded in {time.time() - start:.1f}s")

    def _build_faiss_with_progress(self, chunks: list[Document], batch_size: int = 100) -> FAISS:
        """Build FAISS index with progress bar for embedding generation.

        Args:
            chunks: List of Document objects to embed
            batch_size: Number of chunks to embed at once

        Returns:
            FAISS vectorstore
        """
        if not chunks:
            raise ValueError("No chunks to embed")

        total_chunks = len(chunks)
        logger.info(f"[RAG] Generating embeddings for {total_chunks} chunks...")

        # Process first batch to initialize FAISS
        first_batch = chunks[:batch_size]
        vectorstore = FAISS.from_documents(first_batch, self.embeddings)

        # Process remaining batches with progress bar
        if total_chunks > batch_size:
            remaining_chunks = chunks[batch_size:]
            with tqdm(total=total_chunks, initial=batch_size, desc="Embedding chunks", unit="chunks") as pbar:
                for i in range(0, len(remaining_chunks), batch_size):
                    batch = remaining_chunks[i : i + batch_size]
                    vectorstore.add_documents(batch)
                    pbar.update(len(batch))

        return vectorstore

    def _compute_faiss_checksum(self, faiss_path: str) -> str:
        """Compute SHA256 checksum of FAISS index files.

        Args:
            faiss_path: Path to FAISS index directory

        Returns:
            Hex-encoded SHA256 checksum of all index files
        """
        faiss_dir = Path(faiss_path)
        hasher = hashlib.sha256()

        # Hash all files in the FAISS directory in sorted order for consistency
        for file_path in sorted(faiss_dir.glob("*")):
            if file_path.is_file():
                hasher.update(file_path.name.encode())
                hasher.update(file_path.read_bytes())

        return hasher.hexdigest()

    def _save_faiss_checksum(self, faiss_path: str):
        """Save checksum file for FAISS index.

        Args:
            faiss_path: Path to FAISS index directory
        """
        checksum = self._compute_faiss_checksum(faiss_path)
        checksum_file = Path(faiss_path).parent / "faiss_index.sha256"
        checksum_file.write_text(checksum)
        logger.debug(f"[RAG] FAISS checksum saved: {checksum[:16]}...")

    def _verify_faiss_checksum(self, faiss_path: str) -> bool:
        """Verify FAISS index checksum before loading.

        Args:
            faiss_path: Path to FAISS index directory

        Returns:
            True if checksum matches or no checksum file exists (legacy), False if mismatch

        Raises:
            ValueError: If checksum verification fails (possible tampering)
        """
        checksum_file = Path(faiss_path).parent / "faiss_index.sha256"

        if not checksum_file.exists():
            # Legacy index without checksum - allow loading but warn
            logger.warning("[RAG] No FAISS checksum file found (legacy index). Consider rebuilding.")
            return True

        expected = checksum_file.read_text().strip()
        actual = self._compute_faiss_checksum(faiss_path)

        if expected != actual:
            raise ValueError(
                f"FAISS index checksum mismatch - possible tampering detected. "
                f"Expected: {expected[:16]}..., Got: {actual[:16]}... "
                f"Delete the cache directory and rebuild the index."
            )

        logger.debug(f"[RAG] FAISS checksum verified: {actual[:16]}...")
        return True

    def _build_retrievers(self):
        """Build FAISS vector store and hybrid retriever."""
        # Build FAISS index
        logger.info(f"[RAG] Building FAISS vector index from {len(self.chunks)} chunks...")
        start = time.time()
        self.vectorstore = self._build_faiss_with_progress(self.chunks)
        logger.info(f"[RAG] ✓ FAISS index built in {time.time() - start:.1f}s")

        # Save FAISS index with checksum
        faiss_path = str(self.index_dir / "faiss_index")
        logger.info(f"[RAG] Saving FAISS index to {faiss_path}...")
        self.vectorstore.save_local(faiss_path)
        self._save_faiss_checksum(faiss_path)
        logger.info("[RAG] ✓ FAISS index saved")

        # Build BM25 retriever
        logger.info(f"[RAG] Building BM25 keyword retriever from {len(self.chunks)} chunks...")
        start = time.time()
        self.bm25_retriever = BM25Retriever.from_documents(self.chunks)
        self.bm25_retriever.k = (
            self.config.search_top_k * self.config.retriever_candidate_multiplier
        )  # Get more candidates for ensemble
        logger.info(f"[RAG] ✓ BM25 retriever built in {time.time() - start:.1f}s")

        # Build ensemble retriever (hybrid search)
        logger.info(
            f"[RAG] Building hybrid ensemble retriever "
            f"(BM25 weight: {self.config.hybrid_bm25_weight}, Semantic weight: {self.config.hybrid_semantic_weight})..."
        )
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[
                self.bm25_retriever,
                self.vectorstore.as_retriever(
                    search_kwargs={"k": self.config.search_top_k * self.config.retriever_candidate_multiplier}
                ),
            ],
            weights=[self.config.hybrid_bm25_weight, self.config.hybrid_semantic_weight],
        )
        logger.info("[RAG] ✓ Ensemble retriever ready")

    def _update_index_incremental(self):
        """Update existing index with new documents (incremental).

        This is more efficient than full rebuild for large indexes.
        Loads existing FAISS index and adds new documents to it.
        """
        # Load existing chunks count from metadata
        metadata = self._load_metadata()
        existing_chunk_count = metadata.get("num_chunks", 0)

        if existing_chunk_count == 0:
            # No existing index, do full build
            logger.info("[RAG] No existing index found, performing full build")
            self._build_index()
            return

        # Calculate new chunks to add
        new_chunk_count = len(self.chunks) - existing_chunk_count
        if new_chunk_count <= 0:
            logger.warning("[RAG] No new chunks to add, rebuilding retrievers only")
            self._initialize_components()
            self._build_retrievers()
            return

        logger.info(
            f"[RAG] Incremental update: adding {new_chunk_count} new chunks to existing {existing_chunk_count} chunks..."
        )

        # Initialize components if needed
        logger.info("[RAG] Initializing ML models (embeddings, re-rankers)...")
        self._initialize_components()

        # Load existing FAISS index with checksum verification
        faiss_path = str(self.index_dir / "faiss_index")
        try:
            # Verify checksum before loading (raises ValueError if tampered)
            self._verify_faiss_checksum(faiss_path)

            logger.info(f"[RAG] Loading existing FAISS index from {faiss_path}...")
            start = time.time()
            self.vectorstore = FAISS.load_local(
                faiss_path, self.embeddings, allow_dangerous_deserialization=True  # Checksum verified above
            )
            logger.info(
                f"[RAG] ✓ Loaded existing index with {existing_chunk_count} chunks in {time.time() - start:.1f}s"
            )

            # Get only the new chunks
            new_chunks = self.chunks[existing_chunk_count:]
            logger.info(f"[RAG] Generating embeddings for {len(new_chunks)} new chunks...")
            start = time.time()

            # Add new documents to existing index
            self.vectorstore.add_documents(new_chunks)
            logger.info(f"[RAG] ✓ Added {len(new_chunks)} new chunks in {time.time() - start:.1f}s")

            # Save updated index with new checksum
            logger.info(f"[RAG] Saving updated FAISS index to {faiss_path}...")
            self.vectorstore.save_local(faiss_path)
            self._save_faiss_checksum(faiss_path)
            logger.info("[RAG] ✓ Updated FAISS index saved")

        except Exception as e:
            logger.warning(f"[RAG] Failed to load existing FAISS index: {e}")
            logger.info("[RAG] Performing full rebuild instead...")
            self._build_index()
            return

        # Rebuild BM25 retriever (fast, must use all chunks)
        logger.info(f"[RAG] Rebuilding BM25 retriever with all {len(self.chunks)} chunks...")
        start = time.time()
        self.bm25_retriever = BM25Retriever.from_documents(self.chunks)
        self.bm25_retriever.k = self.config.search_top_k * self.config.retriever_candidate_multiplier
        logger.info(f"[RAG] ✓ BM25 retriever rebuilt in {time.time() - start:.1f}s")

        # Rebuild ensemble retriever
        logger.info("[RAG] Rebuilding ensemble retriever...")
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[
                self.bm25_retriever,
                self.vectorstore.as_retriever(
                    search_kwargs={"k": self.config.search_top_k * self.config.retriever_candidate_multiplier}
                ),
            ],
            weights=[self.config.hybrid_bm25_weight, self.config.hybrid_semantic_weight],
        )
        logger.info("[RAG] ✓ Ensemble retriever ready")

    def _rerank_results(self, query: str, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Re-rank results using cross-encoder.

        This is the second stage of the retrieval pipeline:
        1. Hybrid retrieval (BM25 + semantic via RRF) produces candidates
        2. Cross-encoder re-ranking (this method) produces final ranking

        Args:
            query: Search query
            results: List of results to re-rank

        Returns:
            Re-ranked results sorted by cross-encoder score
        """
        if not results:
            return results

        logger.debug(f"[RAG] Re-ranking {len(results)} results")

        # Cross-encoder re-ranking (accurate, final ranking)
        if self.cross_encoder:
            pairs = [[query, result["text"]] for result in results]
            scores = self.cross_encoder.predict(pairs)

            # Normalize scores to 0-1 range using min-max scaling
            # This ensures consistent scoring regardless of cross-encoder model characteristics
            min_score = float(min(scores))
            max_score = float(max(scores))
            score_range = max_score - min_score

            for result, score in zip(results, scores):
                if score_range > 0:
                    # Normalize to 0-1 range
                    result["score"] = (float(score) - min_score) / score_range
                else:
                    # All scores identical, assign uniform score
                    result["score"] = 1.0

            # Sort by final score
            results = sorted(results, key=lambda x: x["score"], reverse=True)

        return results

    def _load_metadata(self) -> dict[str, Any]:
        """Load metadata from cache."""
        if self.metadata_file.exists():
            try:
                return json.loads(self.metadata_file.read_text())
            except Exception as e:
                logger.warning(f"[RAG] Failed to load metadata: {e}")
        return {}

    def _save_metadata(self, metadata: dict[str, Any]):
        """Save metadata to cache."""
        try:
            self.metadata_file.write_text(json.dumps(metadata, indent=2))
        except Exception as e:
            logger.error(f"[RAG] Failed to save metadata: {e}")

    def _save_chunks(self):
        """Save chunks to disk."""
        try:
            chunk_dicts = [{"page_content": chunk.page_content, "metadata": chunk.metadata} for chunk in self.chunks]
            self.chunks_file.write_text(json.dumps(chunk_dicts))
            logger.info(f"[RAG] Saved {len(self.chunks)} chunks")
        except Exception as e:
            logger.error(f"[RAG] Failed to save chunks: {e}")

    def _load_chunks(self) -> list[Document] | None:
        """Load chunks from disk."""
        if not self.chunks_file.exists():
            return None
        try:
            chunk_dicts = json.loads(self.chunks_file.read_text())
            chunks = [Document(page_content=cd["page_content"], metadata=cd["metadata"]) for cd in chunk_dicts]
            logger.info(f"[RAG] Loaded {len(chunks)} chunks from cache")
            return chunks
        except Exception as e:
            logger.warning(f"[RAG] Failed to load chunks: {e}")
            return None

    def _save_parent_chunks(self):
        """Save parent chunks to disk."""
        try:
            self.parent_chunks_file.write_text(json.dumps(self.parent_chunks, indent=2))
            logger.info(f"[RAG] Saved {len(self.parent_chunks)} parent chunks")
        except Exception as e:
            logger.error(f"[RAG] Failed to save parent chunks: {e}")

    def _load_parent_chunks(self) -> dict[str, dict[str, Any]] | None:
        """Load parent chunks from disk."""
        if not self.parent_chunks_file.exists():
            return None
        try:
            parent_chunks = json.loads(self.parent_chunks_file.read_text())
            logger.info(f"[RAG] Loaded {len(parent_chunks)} parent chunks from cache")
            return parent_chunks
        except Exception as e:
            logger.warning(f"[RAG] Failed to load parent chunks: {e}")
            return None

    def _load_crawl_state(self) -> dict[str, Any]:
        """Load crawl state from disk.

        Returns:
            Crawl state dict with discovered_urls, indexed_urls, failed_urls, etc.
        """
        if not self.crawl_state_file.exists():
            return {}
        try:
            state = json.loads(self.crawl_state_file.read_text())
            failed_urls = state.get("failed_urls", {})
            logger.info(
                f"[RAG] Loaded crawl state: {len(state.get('discovered_urls', []))} discovered, "
                f"{len(state.get('indexed_urls', []))} indexed, {len(failed_urls)} failed"
            )
            return state
        except Exception as e:
            logger.warning(f"[RAG] Failed to load crawl state: {e}")
            return {}

    def _save_crawl_state(self, state: dict[str, Any]):
        """Save crawl state to disk.

        Args:
            state: Crawl state dict with discovered_urls, indexed_urls, etc.
        """
        try:
            self.crawl_state_file.write_text(json.dumps(state, indent=2))
            logger.debug(f"[RAG] Saved crawl state: {len(state.get('indexed_urls', []))} indexed URLs")
        except Exception as e:
            logger.error(f"[RAG] Failed to save crawl state: {e}")

    def _track_url_failure(self, url: str, failed_urls: dict[str, Any], error_msg: str):
        """Track failed URL attempts with error details.

        Args:
            url: The URL that failed
            failed_urls: Dict tracking failed URLs
            error_msg: Error message from the failure
        """
        if url not in failed_urls:
            failed_urls[url] = {"failure_count": 0, "first_error": error_msg, "last_error": error_msg}

        failed_urls[url]["failure_count"] += 1
        failed_urls[url]["last_error"] = error_msg
        failed_urls[url]["last_attempt"] = datetime.now().isoformat()

        count = failed_urls[url]["failure_count"]
        if count >= self.config.max_url_retries:
            logger.warning(
                f"[RAG] URL {url} failed {count} times (limit: {self.config.max_url_retries}), "
                "will skip on future crawls"
            )
        else:
            logger.debug(f"[RAG] URL {url} failure count: {count}/{self.config.max_url_retries}")
