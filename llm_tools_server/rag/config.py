"""RAG configuration dataclass."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RAGConfig:
    """Configuration for RAG document indexing and search.

    Attributes:
        base_url: Starting URL for crawling (e.g., "https://docs.example.com")
        cache_dir: Directory to store index, embeddings, and cached content
        manual_urls: Optional list of specific URLs to index
        manual_urls_only: If True, only index manual_urls (no crawling). If False, manual_urls are additive.

        # Crawling settings
        max_crawl_depth: Maximum depth for recursive crawler (default: 3)
        rate_limit_delay: Seconds between HTTP requests (default: 0.1)
        max_workers: Number of parallel fetching threads (default: 5)
        max_pages: Maximum total pages to crawl (None = unlimited, useful for testing)
        url_include_patterns: List of regex patterns - only crawl matching URLs
        url_exclude_patterns: List of regex patterns - skip matching URLs

        # Chunking settings
        child_chunk_min_tokens: Minimum tokens for child chunks (default: 150)
        child_chunk_size: Maximum tokens per child chunk (default: 350)
        parent_chunk_min_tokens: Minimum tokens for parent chunks (default: 300)
        parent_chunk_size: Maximum tokens per parent chunk (default: 900)
        absolute_max_chunk_tokens: Hard limit for any chunk (default: 1200). Content exceeding this is split.

        # Search settings
        hybrid_bm25_weight: Weight for BM25 in Reciprocal Rank Fusion (default: 0.3).
            NOTE: Uses RRF, not weighted average. Higher weight = ranks from that
            retriever contribute more to final ordering. Formula: score = Σ(weight/(rank+60))
        hybrid_semantic_weight: Weight for semantic search in RRF (default: 0.7).
            With defaults (0.3/0.7), semantic ranks are weighted ~2.3x more than BM25.
        search_top_k: Default number of results to return (default: 5)
        rerank_enabled: Enable cross-encoder re-ranking (default: True)
        parent_context_max_chars: Max characters for parent context in tool results (default: 500, 0=no limit)

        # Model settings
        embedding_model: HuggingFace embedding model name. Options (speed vs quality):
            - "sentence-transformers/all-MiniLM-L6-v2": Fast (22M params), good quality (default)
            - "BAAI/bge-base-en-v1.5": Medium (110M params), better quality
            - "BAAI/bge-large-en-v1.5": Slow (335M params), best quality
            Note: Changing embedding model requires full index rebuild.
        rerank_model: Cross-encoder model for re-ranking

        # Contextual retrieval settings (Anthropic's approach)
        contextual_retrieval_enabled: Enable LLM-generated context prepended to chunks
        contextual_retrieval_background: Run contextual retrieval in background thread (index usable immediately)
        contextual_backend_type: Backend type ("lmstudio" or "ollama"), uses server default if None
        contextual_backend_endpoint: Backend endpoint URL, uses server default if None
        contextual_model: Model name for context generation, uses server default if None
        contextual_max_workers: Number of parallel context generation workers
        contextual_timeout: Timeout per context generation request in seconds
        contextual_prompt: Prompt template for context generation ({document} and {chunk} placeholders)

        # Index settings
        update_check_interval_hours: Hours between index update checks (default: 168 = 7 days)
        page_cache_ttl_hours: TTL for cached pages without lastmod (default: 168 = 7 days, 0 = never expire).
            Pages with lastmod from sitemap are invalidated when lastmod changes.
            Pages without lastmod are invalidated after this TTL expires.

        # Periodic update settings (for long-running applications)
        periodic_update_enabled: Enable background sitemap polling for automatic updates (default: False)
        periodic_update_interval_hours: Hours between sitemap checks (default: 6.0)
        periodic_update_min_interval_minutes: Minimum minutes between checks to prevent rapid polling (default: 30.0)
        update_batch_size: Maximum pages to process per update cycle (default: 50)
        update_on_startup: Run initial crawl_and_index on load_index if index missing (default: True)

        # Tombstone settings (for incremental updates without full rebuild)
        tombstone_rebuild_threshold: Fraction of tombstoned chunks that triggers full rebuild (default: 0.1 = 10%)
        auto_rebuild_enabled: Automatically rebuild index when tombstone threshold exceeded (default: True)

        # Progress display settings
        show_progress: Show progress bars during crawling/indexing (default: True).
            Disable for non-interactive environments or quiet operation.
    """

    # Core settings
    base_url: str
    cache_dir: str | Path = "./rag_cache"
    manual_urls: list[str] | None = None
    manual_urls_only: bool = False

    # Crawling settings
    max_crawl_depth: int = 3
    rate_limit_delay: float = 0.1
    max_workers: int = 5
    max_pages: int | None = None
    request_timeout: float = 10.0  # HTTP request timeout in seconds
    max_url_retries: int = 3  # Skip URLs after this many consecutive failures
    url_include_patterns: list[str] = field(default_factory=list)
    url_exclude_patterns: list[str] = field(default_factory=list)

    # Chunking settings
    child_chunk_min_tokens: int = 150  # Minimum tokens for child chunks
    child_chunk_size: int = 350  # Maximum tokens for child chunks
    parent_chunk_min_tokens: int = 300  # Minimum tokens for parent chunks
    parent_chunk_size: int = 900  # Maximum tokens for parent chunks
    absolute_max_chunk_tokens: int = 1200  # Hard limit - split any content exceeding this

    # Search settings (uses Reciprocal Rank Fusion, not weighted average)
    hybrid_bm25_weight: float = 0.3  # RRF weight for BM25 keyword search
    hybrid_semantic_weight: float = 0.7  # RRF weight for semantic vector search
    search_top_k: int = 5
    retriever_candidate_multiplier: int = 3  # Multiplier for initial retrieval candidates (search_top_k * this)
    rerank_enabled: bool = True
    parent_context_max_chars: int = 500  # Max chars for parent context in tool results (0 = no limit)

    # Model settings
    # Embedding model options (speed vs quality tradeoff):
    #   - "sentence-transformers/all-MiniLM-L6-v2": Fast (22M params), good quality
    #   - "BAAI/bge-base-en-v1.5": Medium (110M params), better quality
    #   - "BAAI/bge-large-en-v1.5": Slow (335M params), best quality
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"  # Fast default, configurable
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"

    # Contextual retrieval settings (Anthropic's approach for ~40-50% fewer retrieval failures)
    # See: https://www.anthropic.com/news/contextual-retrieval
    # Uses the same backend (LM Studio/Ollama) as the main server by default.
    # Set contextual_backend_type/endpoint/model to override with different backend.
    contextual_retrieval_enabled: bool = False  # Enable LLM-generated context for chunks
    contextual_retrieval_background: bool = False  # Run contextual retrieval in background thread
    contextual_backend_type: str | None = None  # "lmstudio" or "ollama", None = use server default
    contextual_backend_endpoint: str | None = None  # Backend URL, None = use server default
    contextual_model: str | None = None  # Model name, None = use server default (BACKEND_MODEL)
    contextual_max_workers: int = 4  # Parallel context generation workers
    contextual_timeout: float = 60.0  # Timeout per context generation request
    contextual_prompt: str = """<document>
{document}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk}
</chunk>

Please give a short succinct context (2-3 sentences) to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Respond only with the context, nothing else."""

    # Index settings
    update_check_interval_hours: int = 168  # 7 days
    page_cache_ttl_hours: int = 168  # 7 days - TTL for cached pages without lastmod (0 = never expire)

    # Periodic update settings (for long-running applications)
    periodic_update_enabled: bool = False  # Disabled by default - opt-in feature
    periodic_update_interval_hours: float = 6.0  # Check sitemap every 6 hours
    periodic_update_min_interval_minutes: float = 30.0  # Minimum 30 minutes between checks
    update_batch_size: int = 50  # Max pages to process per update cycle
    update_on_startup: bool = True  # Run initial crawl if index missing

    # Tombstone settings (for incremental updates without full rebuild)
    tombstone_rebuild_threshold: float = 0.1  # Rebuild when 10% are tombstoned
    auto_rebuild_enabled: bool = True  # Automatically rebuild when threshold exceeded

    # Progress display settings
    show_progress: bool = True  # Show progress bars during crawling/indexing

    def __post_init__(self):
        """Convert cache_dir to Path and validate weights."""
        self.cache_dir = Path(self.cache_dir)

        # Validate hybrid search weights
        total_weight = self.hybrid_bm25_weight + self.hybrid_semantic_weight
        if abs(total_weight - 1.0) > 0.01:  # Allow small floating point error
            raise ValueError(
                f"Hybrid search weights must sum to 1.0, got {total_weight} "
                f"(bm25={self.hybrid_bm25_weight}, semantic={self.hybrid_semantic_weight})"
            )

        # Ensure manual_urls is a list if provided
        if self.manual_urls is None:
            self.manual_urls = []
