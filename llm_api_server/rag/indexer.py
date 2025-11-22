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
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

from .chunker import semantic_chunk_html
from .config import RAGConfig
from .crawler import DocumentCrawler

logger = logging.getLogger(__name__)

# Disable tokenizers parallelism to prevent fork-related warnings when using WebUI
# This is safe because we use ThreadPoolExecutor for parallel operations instead
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class DocSearchIndex:
    """Main document search index with crawling, chunking, embedding, and hybrid search."""

    # Index version for cache invalidation
    INDEX_VERSION = "1.0.0-parent-child"

    def __init__(self, config: RAGConfig):
        """Initialize the document search index.

        Args:
            config: RAG configuration
        """
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

        # Components (lazy-loaded)
        self.embeddings: HuggingFaceEmbeddings | None = None
        self.vectorstore: FAISS | None = None
        self.bm25_retriever: BM25Retriever | None = None
        self.ensemble_retriever: EnsembleRetriever | None = None
        self.cross_encoder: CrossEncoder | None = None
        self.light_cross_encoder: CrossEncoder | None = None

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
            url_include_patterns=config.url_include_patterns,
            url_exclude_patterns=config.url_exclude_patterns,
        )

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

    def crawl_and_index(self, force_rebuild: bool = False):
        """Discover URLs, crawl pages, chunk content, and build search index.

        Args:
            force_rebuild: Force full rebuild even if index is up-to-date
        """
        if not force_rebuild and not self.needs_update():
            logger.info("[RAG] Index is up-to-date, loading from cache")
            self.load_index()
            return

        logger.info("[RAG] " + "=" * 70)
        logger.info("[RAG] Starting document indexing pipeline")
        logger.info("[RAG] " + "=" * 70)

        # Phase 1: Discover URLs
        logger.info("[RAG] Phase 1/4: Discovering URLs")
        start_time = time.time()
        url_list = self.crawler.discover_and_crawl()
        logger.info(f"[RAG] Discovered {len(url_list)} URLs in {time.time() - start_time:.1f}s")

        if not url_list:
            logger.error("[RAG] No URLs discovered!")
            return

        # Phase 2: Fetch pages
        logger.info("[RAG] Phase 2/4: Fetching pages")
        start_time = time.time()
        pages = self._fetch_pages(url_list)
        logger.info(f"[RAG] Fetched {len(pages)} pages in {time.time() - start_time:.1f}s")

        if not pages:
            logger.error("[RAG] No pages fetched!")
            return

        # Phase 3: Chunk content
        logger.info("[RAG] Phase 3/4: Chunking content")
        start_time = time.time()
        self._create_chunks(pages)
        logger.info(
            f"[RAG] Created {len(self.chunks)} child chunks from {len(self.parent_chunks)} parents in {time.time() - start_time:.1f}s"
        )

        # Save chunks
        self._save_chunks()
        self._save_parent_chunks()

        # Phase 4: Build index
        logger.info("[RAG] Phase 4/4: Building search index")
        start_time = time.time()
        self._build_index()
        logger.info(f"[RAG] Built index in {time.time() - start_time:.1f}s")

        # Save metadata
        self._save_metadata(
            {"version": self.INDEX_VERSION, "last_update": datetime.now().isoformat(), "num_chunks": len(self.chunks)}
        )

        logger.info("[RAG] " + "=" * 70)
        logger.info("[RAG] Indexing complete!")
        logger.info("[RAG] " + "=" * 70)

    def load_index(self):
        """Load index from cache."""
        logger.info("[RAG] Loading index from cache")

        # Load chunks
        self.chunks = self._load_chunks() or []
        self.parent_chunks = self._load_parent_chunks() or {}

        if not self.chunks:
            logger.warning("[RAG] No cached chunks found")
            return

        # Initialize components
        self._initialize_components()

        # Build retrievers
        logger.info(f"[RAG] Building retrievers from {len(self.chunks)} chunks")
        self._build_retrievers()

        logger.info("[RAG] Index loaded successfully")

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

        # Get initial candidates (more than needed for re-ranking)
        candidates_k = self.config.rerank_top_k if self.config.rerank_enabled else top_k
        candidates = self.ensemble_retriever.get_relevant_documents(query, k=candidates_k)

        logger.debug(f"[RAG] Retrieved {len(candidates)} candidates from hybrid search")

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

    def _fetch_pages(self, url_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Fetch pages with caching and parallel fetching.

        Args:
            url_list: List of URL info dicts

        Returns:
            List of page data dicts
        """
        pages = []
        total = len(url_list)
        cache_hits = 0

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all fetch tasks (with cache check)
            future_to_url = {executor.submit(self._fetch_page_with_cache, url_info): url_info for url_info in url_list}

            # Process completed tasks
            for idx, future in enumerate(as_completed(future_to_url), 1):
                url_info = future_to_url[future]
                try:
                    result = future.result()
                    if result:
                        pages.append(result)
                        if result.get("from_cache"):
                            cache_hits += 1

                        if idx % 10 == 0:
                            logger.info(
                                f"[RAG] Fetched {idx}/{total} pages ({100*idx/total:.1f}%) - {cache_hits} from cache"
                            )

                except Exception as e:
                    logger.error(f"[RAG] Failed to fetch {url_info['url']}: {e}")

        if cache_hits > 0:
            logger.info(f"[RAG] Cache hits: {cache_hits}/{len(pages)} ({100*cache_hits/len(pages):.1f}%)")

        return pages

    def _fetch_page_with_cache(self, url_info: dict[str, Any]) -> dict[str, Any] | None:
        """Fetch a page with caching support.

        Args:
            url_info: URL info dict with url and optional lastmod

        Returns:
            Page data dict or None if failed
        """
        url = url_info["url"]
        lastmod = url_info.get("lastmod")

        # Try to load from cache
        cached = self._load_cached_page(url, lastmod)
        if cached:
            return cached

        # Fetch fresh content
        result = self.crawler.fetch_page(url)
        if result:
            url, html = result
            page_data = {"url": url, "html": html, "lastmod": lastmod, "from_cache": False}

            # Save to cache
            self._save_cached_page(page_data)

            return page_data

        return None

    def _get_page_cache_path(self, url: str) -> Path:
        """Get cache file path for a URL."""
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return self.content_dir / f"{url_hash}.json"

    def _load_cached_page(self, url: str, lastmod: str | None) -> dict[str, Any] | None:
        """Load cached page content if still valid.

        Args:
            url: Page URL
            lastmod: Last modification date from sitemap

        Returns:
            Cached page data or None if cache invalid
        """
        cache_path = self._get_page_cache_path(url)
        if not cache_path.exists():
            return None

        try:
            cached = json.loads(cache_path.read_text())

            # Check if lastmod matches (if we have one)
            if lastmod and cached.get("lastmod") != lastmod:
                logger.debug(f"[RAG] Cache invalidated for {url} (lastmod changed)")
                return None

            # Mark as from cache
            cached["from_cache"] = True
            logger.debug(f"[RAG] Loaded from cache: {url}")
            return cached

        except Exception as e:
            logger.debug(f"[RAG] Failed to load cache for {url}: {e}")
            return None

    def _save_cached_page(self, page_data: dict[str, Any]):
        """Save page content to cache."""
        try:
            cache_path = self._get_page_cache_path(page_data["url"])
            # Don't save the from_cache flag
            save_data = {k: v for k, v in page_data.items() if k != "from_cache"}
            cache_path.write_text(json.dumps(save_data))
            logger.debug(f"[RAG] Cached: {page_data['url']}")
        except Exception as e:
            logger.warning(f"[RAG] Failed to cache page {page_data['url']}: {e}")

    def _create_chunks(self, pages: list[dict[str, Any]]):
        """Create parent-child chunks from pages using semantic HTML chunking.

        Args:
            pages: List of page data dicts with HTML
        """
        self.chunks = []
        self.parent_chunks = {}
        self.child_to_parent = {}

        for idx, page in enumerate(pages, 1):
            try:
                # Use semantic chunking
                result = semantic_chunk_html(
                    html=page["html"],
                    url=page["url"],
                    child_min_tokens=150,
                    child_max_tokens=self.config.child_chunk_size,
                    parent_min_tokens=self.config.parent_chunk_size // 3,
                    parent_max_tokens=self.config.parent_chunk_size,
                )

                parents = result.get("parents", [])
                children = result.get("children", [])

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

                if (idx % 50 == 0) or (idx == len(pages)):
                    logger.info(
                        f"[RAG] Chunked {idx}/{len(pages)} pages ({100*idx/len(pages):.1f}%) - {len(self.chunks)} chunks so far"
                    )

            except Exception as e:
                logger.error(f"[RAG] Failed to chunk {page['url']}: {e}")
                continue

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
            logger.info(f"[RAG] Loading embedding model: {self.config.embedding_model}")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.config.embedding_model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )

        if self.config.rerank_enabled:
            if self.cross_encoder is None:
                logger.info(f"[RAG] Loading cross-encoder: {self.config.rerank_model}")
                self.cross_encoder = CrossEncoder(self.config.rerank_model)

            if self.light_cross_encoder is None:
                logger.info(f"[RAG] Loading light cross-encoder: {self.config.light_rerank_model}")
                self.light_cross_encoder = CrossEncoder(self.config.light_rerank_model)

    def _build_retrievers(self):
        """Build FAISS vector store and hybrid retriever."""
        # Build FAISS index
        logger.info(f"[RAG] Building FAISS index from {len(self.chunks)} chunks")
        self.vectorstore = FAISS.from_documents(self.chunks, self.embeddings)

        # Save FAISS index
        faiss_path = str(self.index_dir / "faiss_index")
        self.vectorstore.save_local(faiss_path)
        logger.info(f"[RAG] Saved FAISS index to {faiss_path}")

        # Build BM25 retriever
        logger.info("[RAG] Building BM25 retriever")
        self.bm25_retriever = BM25Retriever.from_documents(self.chunks)
        self.bm25_retriever.k = self.config.search_top_k * 3  # Get more candidates for ensemble

        # Build ensemble retriever (hybrid search)
        logger.info(
            f"[RAG] Building ensemble retriever (BM25: {self.config.hybrid_bm25_weight}, Semantic: {self.config.hybrid_semantic_weight})"
        )
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[
                self.bm25_retriever,
                self.vectorstore.as_retriever(search_kwargs={"k": self.config.search_top_k * 3}),
            ],
            weights=[self.config.hybrid_bm25_weight, self.config.hybrid_semantic_weight],
        )

    def _rerank_results(self, query: str, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Re-rank results using cross-encoder.

        Uses two-stage re-ranking:
        1. Light cross-encoder for first pass
        2. Heavy cross-encoder for final ranking

        Args:
            query: Search query
            results: List of results to re-rank

        Returns:
            Re-ranked results
        """
        if not results:
            return results

        logger.debug(f"[RAG] Re-ranking {len(results)} results")

        # Stage 1: Light cross-encoder (fast, reduces candidates)
        if self.light_cross_encoder and len(results) > self.config.rerank_top_k:
            pairs = [[query, result["text"]] for result in results]
            scores = self.light_cross_encoder.predict(pairs)

            for result, score in zip(results, scores):
                result["light_score"] = float(score)

            # Sort and keep top candidates
            results = sorted(results, key=lambda x: x["light_score"], reverse=True)[: self.config.rerank_top_k]
            logger.debug(f"[RAG] Light re-ranker reduced to {len(results)} candidates")

        # Stage 2: Heavy cross-encoder (accurate, final ranking)
        if self.cross_encoder:
            pairs = [[query, result["text"]] for result in results]
            scores = self.cross_encoder.predict(pairs)

            for result, score in zip(results, scores):
                result["score"] = float(score)

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
