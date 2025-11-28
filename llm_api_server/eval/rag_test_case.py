"""RAG-specific test case definitions for retrieval evaluation."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RAGTestCase:
    """Represents a single test case for RAG retrieval evaluation.

    Attributes:
        query: The search query to test
        description: Human-readable description of what this test validates
        relevant_urls: List of URLs that should be retrieved (ground truth)
        relevant_keywords: Keywords that should appear in retrieved content (alternative to URLs)
        top_k: Number of results to retrieve for this test (default: 5)
        metadata: Optional metadata dict for custom tracking/reporting
    """

    query: str
    description: str
    relevant_urls: list[str] = field(default_factory=list)
    relevant_keywords: list[str] = field(default_factory=list)
    top_k: int = 5
    metadata: dict[str, Any] | None = None

    def __post_init__(self):
        """Validate test case has at least one relevance criterion."""
        if not self.relevant_urls and not self.relevant_keywords:
            raise ValueError("RAGTestCase must have at least one of: relevant_urls or relevant_keywords")


@dataclass
class RAGTestResult:
    """Result of running a single RAG test case.

    Attributes:
        test_case: The original test case
        retrieved_results: List of retrieved document results
        metrics: Dict of computed metrics (recall, mrr, ndcg, etc.)
        search_time: Time taken for search in seconds
        config_snapshot: Snapshot of RAG config used (for A/B comparison)
    """

    test_case: RAGTestCase
    retrieved_results: list[dict[str, Any]]
    metrics: dict[str, float]
    search_time: float = 0.0
    config_snapshot: dict[str, Any] | None = None

    @property
    def passed(self) -> bool:
        """Consider test passed if recall > 0 (at least one relevant doc found)."""
        return self.metrics.get("recall", 0) > 0

    @property
    def recall(self) -> float:
        """Recall@k - fraction of relevant docs found in top-k."""
        return self.metrics.get("recall", 0.0)

    @property
    def mrr(self) -> float:
        """Mean Reciprocal Rank - 1/rank of first relevant doc."""
        return self.metrics.get("mrr", 0.0)

    @property
    def ndcg(self) -> float:
        """Normalized Discounted Cumulative Gain."""
        return self.metrics.get("ndcg", 0.0)
