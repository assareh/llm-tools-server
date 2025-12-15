"""LLM API Server Evaluation Framework.

This module provides a comprehensive evaluation framework for testing LLM API endpoints
and RAG retrieval systems.

## LLM API Evaluation

Test LLM API endpoints with validation criteria and generate HTML/JSON reports:

    from llm_tools_server.eval import Evaluator, TestCase, HTMLReporter

    tests = [
        TestCase(
            question="What is 2+2?",
            description="Basic arithmetic test",
            expected_keywords=["4", "four"],
        ),
    ]

    evaluator = Evaluator(api_url="http://localhost:8000")
    results = evaluator.run_tests(tests)

    reporter = HTMLReporter()
    reporter.generate(results, "evaluation_report.html")

## RAG Retrieval Evaluation

Test RAG retrieval quality with metrics like Recall@k, MRR, and nDCG:

    from llm_tools_server.eval import RAGEvaluator, RAGTestCase
    from llm_tools_server.rag import DocSearchIndex, RAGConfig

    # Load index
    config = RAGConfig(base_url="https://docs.example.com", cache_dir="./cache")
    index = DocSearchIndex(config)
    index.load_index()

    # Define test cases with ground truth
    tests = [
        RAGTestCase(
            query="how to configure auth",
            description="Auth config retrieval",
            relevant_urls=["https://docs.example.com/auth/config"],
        ),
    ]

    # Run evaluation
    evaluator = RAGEvaluator(index)
    results = evaluator.run_tests(tests)
    evaluator.print_summary(results)

    # A/B test reranking
    comparison = evaluator.run_ab_comparison(
        tests,
        config_a={"rerank_enabled": True},
        config_b={"rerank_enabled": False},
    )
    evaluator.print_ab_comparison(comparison)
"""

from .evaluator import Evaluator
from .rag_evaluator import RAGEvaluator
from .rag_test_builder import (
    create_test_case_interactive,
    inspect_search_results,
    load_test_cases,
    print_example_usage,
    save_test_cases,
)
from .rag_test_case import RAGTestCase, RAGTestResult
from .reporters import ConsoleReporter, HTMLReporter, JSONReporter
from .test_case import TestCase, TestResult
from .validators import validate_response

__all__ = [
    "ConsoleReporter",
    "Evaluator",
    "HTMLReporter",
    "JSONReporter",
    "RAGEvaluator",
    "RAGTestCase",
    "RAGTestResult",
    "TestCase",
    "TestResult",
    "create_test_case_interactive",
    "inspect_search_results",
    "load_test_cases",
    "print_example_usage",
    "save_test_cases",
    "validate_response",
]
