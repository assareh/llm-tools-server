"""RAG Evaluator for testing retrieval quality and reranking effectiveness."""

import logging
import math
import time
from typing import Any

from .rag_test_case import RAGTestCase, RAGTestResult

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """Evaluator for testing RAG retrieval quality.

    Runs test cases against a DocSearchIndex and computes retrieval metrics
    like Recall@k, MRR, and nDCG.

    Args:
        index: A DocSearchIndex instance (must be loaded/indexed)

    Example:
        from llm_tools_server.rag import DocSearchIndex, RAGConfig
        from llm_tools_server.eval import RAGEvaluator, RAGTestCase

        # Setup index
        config = RAGConfig(base_url="https://docs.example.com", cache_dir="./cache")
        index = DocSearchIndex(config)
        index.load_index()

        # Define test cases
        tests = [
            RAGTestCase(
                query="how to configure authentication",
                description="Auth config retrieval",
                relevant_urls=["https://docs.example.com/auth/config"],
            ),
        ]

        # Run evaluation
        evaluator = RAGEvaluator(index)
        results = evaluator.run_tests(tests)
        summary = evaluator.get_summary(results)
        print(f"Mean Recall@5: {summary['mean_recall']:.2%}")
    """

    def __init__(self, index: Any):
        """Initialize evaluator with a DocSearchIndex.

        Args:
            index: DocSearchIndex instance (imported dynamically to avoid circular imports)
        """
        self.index = index

    def run_test(self, test_case: RAGTestCase) -> RAGTestResult:
        """Run a single RAG test case.

        Args:
            test_case: The test case to run

        Returns:
            RAGTestResult with metrics and retrieved documents
        """
        # Run search
        start_time = time.time()
        results = self.index.search(test_case.query, top_k=test_case.top_k)
        search_time = time.time() - start_time

        # Extract retrieved URLs and content
        retrieved_urls = [r.get("url", "") for r in results]
        retrieved_texts = [r.get("text", "") for r in results]

        # Compute metrics
        metrics = self._compute_metrics(test_case, retrieved_urls, retrieved_texts)

        # Capture config snapshot for A/B comparison
        config_snapshot = {
            "rerank_enabled": self.index.config.rerank_enabled,
            "rerank_model": self.index.config.rerank_model,
            "hybrid_bm25_weight": self.index.config.hybrid_bm25_weight,
            "hybrid_semantic_weight": self.index.config.hybrid_semantic_weight,
        }

        return RAGTestResult(
            test_case=test_case,
            retrieved_results=results,
            metrics=metrics,
            search_time=search_time,
            config_snapshot=config_snapshot,
        )

    def run_tests(self, test_cases: list[RAGTestCase]) -> list[RAGTestResult]:
        """Run multiple RAG test cases.

        Args:
            test_cases: List of test cases to run

        Returns:
            List of RAGTestResult objects
        """
        results = []
        for i, test_case in enumerate(test_cases, 1):
            logger.info(f"[RAG Eval] Running test {i}/{len(test_cases)}: {test_case.description}")
            result = self.run_test(test_case)
            results.append(result)
            logger.info(
                f"[RAG Eval] Recall: {result.recall:.2%}, MRR: {result.mrr:.2f}, "
                f"nDCG: {result.ndcg:.2f}, Time: {result.search_time:.2f}s"
            )
        return results

    def run_ab_comparison(
        self, test_cases: list[RAGTestCase], config_a: dict[str, Any], config_b: dict[str, Any]
    ) -> dict[str, Any]:
        """Run A/B comparison with different configurations.

        Temporarily modifies the index config to run tests under different settings,
        then restores the original config.

        Args:
            test_cases: List of test cases to run
            config_a: First config overrides (e.g., {"rerank_enabled": True})
            config_b: Second config overrides (e.g., {"rerank_enabled": False})

        Returns:
            Dict with results for both configs and comparison metrics
        """
        # Save original config
        original_config = {}
        for key in set(config_a.keys()) | set(config_b.keys()):
            if hasattr(self.index.config, key):
                original_config[key] = getattr(self.index.config, key)

        try:
            # Run with config A
            logger.info(f"[RAG Eval] Running A/B test - Config A: {config_a}")
            self._apply_config(config_a)
            # Reinitialize cross-encoder if rerank setting changed
            if "rerank_enabled" in config_a:
                self._reinit_reranker()
            results_a = self.run_tests(test_cases)

            # Run with config B
            logger.info(f"[RAG Eval] Running A/B test - Config B: {config_b}")
            self._apply_config(config_b)
            if "rerank_enabled" in config_b:
                self._reinit_reranker()
            results_b = self.run_tests(test_cases)

        finally:
            # Restore original config
            self._apply_config(original_config)
            if "rerank_enabled" in original_config:
                self._reinit_reranker()

        # Compute summaries
        summary_a = self.get_summary(results_a)
        summary_b = self.get_summary(results_b)

        # Compute deltas
        comparison = {
            "config_a": config_a,
            "config_b": config_b,
            "results_a": results_a,
            "results_b": results_b,
            "summary_a": summary_a,
            "summary_b": summary_b,
            "deltas": {
                "recall": summary_b["mean_recall"] - summary_a["mean_recall"],
                "mrr": summary_b["mean_mrr"] - summary_a["mean_mrr"],
                "ndcg": summary_b["mean_ndcg"] - summary_a["mean_ndcg"],
                "search_time": summary_b["mean_search_time"] - summary_a["mean_search_time"],
            },
        }

        return comparison

    def _apply_config(self, config_overrides: dict[str, Any]):
        """Apply config overrides to the index."""
        for key, value in config_overrides.items():
            if hasattr(self.index.config, key):
                setattr(self.index.config, key, value)

    def _reinit_reranker(self):
        """Reinitialize cross-encoder based on current rerank_enabled setting."""
        if self.index.config.rerank_enabled and self.index.cross_encoder is None:
            from sentence_transformers import CrossEncoder

            logger.info(f"[RAG Eval] Loading cross-encoder: {self.index.config.rerank_model}")
            self.index.cross_encoder = CrossEncoder(self.index.config.rerank_model)
        elif not self.index.config.rerank_enabled:
            self.index.cross_encoder = None

    def _compute_metrics(
        self, test_case: RAGTestCase, retrieved_urls: list[str], retrieved_texts: list[str]
    ) -> dict[str, float]:
        """Compute retrieval metrics for a test case.

        Args:
            test_case: The test case with ground truth
            retrieved_urls: List of retrieved document URLs
            retrieved_texts: List of retrieved document texts

        Returns:
            Dict of metric name -> value
        """
        # Build relevance judgments (binary: 1 if relevant, 0 if not)
        relevance = []
        for i, url in enumerate(retrieved_urls):
            is_relevant = False

            # Check URL match
            if test_case.relevant_urls:
                # Flexible URL matching - check if any relevant URL is contained in retrieved URL
                for relevant_url in test_case.relevant_urls:
                    if relevant_url in url or url in relevant_url:
                        is_relevant = True
                        break

            # Check keyword match in content
            if not is_relevant and test_case.relevant_keywords:
                text_lower = retrieved_texts[i].lower() if i < len(retrieved_texts) else ""
                for keyword in test_case.relevant_keywords:
                    if keyword.lower() in text_lower:
                        is_relevant = True
                        break

            relevance.append(1 if is_relevant else 0)

        # Compute metrics
        metrics = {
            "recall": self._recall_at_k(relevance, test_case),
            "mrr": self._mrr(relevance),
            "ndcg": self._ndcg(relevance),
            "precision": self._precision_at_k(relevance),
            "num_relevant_found": sum(relevance),
            "num_relevant_total": len(test_case.relevant_urls) + (1 if test_case.relevant_keywords else 0),
        }

        return metrics

    def _recall_at_k(self, relevance: list[int], test_case: RAGTestCase) -> float:
        """Compute Recall@k - fraction of relevant docs found.

        Args:
            relevance: Binary relevance judgments for retrieved docs
            test_case: Test case with ground truth count

        Returns:
            Recall score between 0 and 1
        """
        num_relevant_total = len(test_case.relevant_urls) or 1  # Avoid div by zero
        num_relevant_found = sum(relevance)
        return num_relevant_found / num_relevant_total

    def _precision_at_k(self, relevance: list[int]) -> float:
        """Compute Precision@k - fraction of retrieved docs that are relevant.

        Args:
            relevance: Binary relevance judgments for retrieved docs

        Returns:
            Precision score between 0 and 1
        """
        if not relevance:
            return 0.0
        return sum(relevance) / len(relevance)

    def _mrr(self, relevance: list[int]) -> float:
        """Compute Mean Reciprocal Rank - 1/rank of first relevant doc.

        Args:
            relevance: Binary relevance judgments for retrieved docs

        Returns:
            MRR score between 0 and 1
        """
        for i, rel in enumerate(relevance):
            if rel == 1:
                return 1.0 / (i + 1)
        return 0.0

    def _ndcg(self, relevance: list[int]) -> float:
        """Compute Normalized Discounted Cumulative Gain.

        Uses binary relevance (0 or 1) and computes nDCG against ideal ranking.

        Args:
            relevance: Binary relevance judgments for retrieved docs

        Returns:
            nDCG score between 0 and 1
        """
        if not relevance or sum(relevance) == 0:
            return 0.0

        # DCG: sum of rel_i / log2(i + 2) for i in 0..k-1
        dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevance))

        # Ideal DCG: all relevant docs at top
        ideal_relevance = sorted(relevance, reverse=True)
        idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_relevance))

        if idcg == 0:
            return 0.0

        return dcg / idcg

    def get_summary(self, results: list[RAGTestResult]) -> dict[str, Any]:
        """Generate summary statistics from test results.

        Args:
            results: List of RAG test results

        Returns:
            Dictionary with summary statistics
        """
        if not results:
            return {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "pass_rate": 0.0,
                "mean_recall": 0.0,
                "mean_mrr": 0.0,
                "mean_ndcg": 0.0,
                "mean_precision": 0.0,
                "mean_search_time": 0.0,
            }

        total = len(results)
        passed = sum(1 for r in results if r.passed)

        return {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": passed / total * 100,
            "mean_recall": sum(r.recall for r in results) / total,
            "mean_mrr": sum(r.mrr for r in results) / total,
            "mean_ndcg": sum(r.ndcg for r in results) / total,
            "mean_precision": sum(r.metrics.get("precision", 0) for r in results) / total,
            "mean_search_time": sum(r.search_time for r in results) / total,
            "total_search_time": sum(r.search_time for r in results),
        }

    def print_summary(self, results: list[RAGTestResult], title: str = "RAG Evaluation Results"):
        """Print a formatted summary to console.

        Args:
            results: List of RAG test results
            title: Title for the summary
        """
        summary = self.get_summary(results)

        print(f"\n{'=' * 60}")
        print(f" {title}")
        print(f"{'=' * 60}")
        print(f" Total Tests:     {summary['total']}")
        print(f" Passed:          {summary['passed']} ({summary['pass_rate']:.1f}%)")
        print(f" Failed:          {summary['failed']}")
        print(f"{'─' * 60}")
        print(f" Mean Recall@k:   {summary['mean_recall']:.2%}")
        print(f" Mean MRR:        {summary['mean_mrr']:.3f}")
        print(f" Mean nDCG:       {summary['mean_ndcg']:.3f}")
        print(f" Mean Precision:  {summary['mean_precision']:.2%}")
        print(f"{'─' * 60}")
        print(f" Mean Search Time: {summary['mean_search_time']:.3f}s")
        print(f" Total Time:       {summary['total_search_time']:.2f}s")
        print(f"{'=' * 60}\n")

    def print_ab_comparison(self, comparison: dict[str, Any]):
        """Print a formatted A/B comparison to console.

        Args:
            comparison: Result from run_ab_comparison()
        """
        print(f"\n{'=' * 70}")
        print(" A/B Comparison Results")
        print(f"{'=' * 70}")
        print(f" Config A: {comparison['config_a']}")
        print(f" Config B: {comparison['config_b']}")
        print(f"{'─' * 70}")

        sa = comparison["summary_a"]
        sb = comparison["summary_b"]
        deltas = comparison["deltas"]

        def fmt_delta(val: float, pct: bool = False) -> str:
            """Format delta with + or - sign."""
            sign = "+" if val >= 0 else ""
            if pct:
                return f"{sign}{val:.2%}"
            return f"{sign}{val:.3f}"

        print(f" {'Metric':<20} {'Config A':>12} {'Config B':>12} {'Delta':>12}")
        print(f" {'─' * 56}")
        print(
            f" {'Recall@k':<20} {sa['mean_recall']:>11.2%} {sb['mean_recall']:>11.2%} {fmt_delta(deltas['recall'], True):>12}"
        )
        print(f" {'MRR':<20} {sa['mean_mrr']:>12.3f} {sb['mean_mrr']:>12.3f} {fmt_delta(deltas['mrr']):>12}")
        print(f" {'nDCG':<20} {sa['mean_ndcg']:>12.3f} {sb['mean_ndcg']:>12.3f} {fmt_delta(deltas['ndcg']):>12}")
        print(
            f" {'Search Time (s)':<20} {sa['mean_search_time']:>12.3f} {sb['mean_search_time']:>12.3f} {fmt_delta(deltas['search_time']):>12}"
        )
        print(f"{'=' * 70}\n")
