"""Helper utilities for building RAG test cases.

This module provides tools to help create ground truth test cases for RAG evaluation,
including interactive test case builders and utilities to inspect search results.
"""

import json
from pathlib import Path
from typing import Any

from .rag_test_case import RAGTestCase


def inspect_search_results(index: Any, query: str, top_k: int = 10) -> list[dict[str, Any]]:
    """Run a search and display results for manual relevance assessment.

    Use this to explore what the index returns for a query before creating test cases.

    Args:
        index: DocSearchIndex instance
        query: Search query to test
        top_k: Number of results to retrieve

    Returns:
        List of result dicts with url, score, and text preview
    """
    results = index.search(query, top_k=top_k)

    print(f"\n{'=' * 70}")
    print(f" Search Results for: {query!r}")
    print(f"{'=' * 70}\n")

    simplified = []
    for i, r in enumerate(results, 1):
        url = r.get("url", "N/A")
        score = r.get("score", 0)
        text = r.get("text", "")[:200] + "..." if len(r.get("text", "")) > 200 else r.get("text", "")
        heading = r.get("heading_path", "")

        print(f"[{i}] Score: {score:.3f}")
        print(f"    URL: {url}")
        if heading:
            print(f"    Heading: {heading}")
        print(f"    Text: {text}")
        print()

        simplified.append({"rank": i, "url": url, "score": score, "heading": heading, "text_preview": text})

    return simplified


def create_test_case_interactive(index: Any, query: str, description: str, top_k: int = 10) -> RAGTestCase:
    """Interactively create a test case by reviewing search results.

    Displays search results and prompts user to select which are relevant.

    Args:
        index: DocSearchIndex instance
        query: Search query to test
        description: Description for the test case
        top_k: Number of results to show

    Returns:
        RAGTestCase with user-selected relevant URLs
    """
    results = inspect_search_results(index, query, top_k)

    print(f"{'─' * 70}")
    print("Enter the numbers of RELEVANT results (comma-separated), or 'none':")
    print("Example: 1,3,5")

    user_input = input("> ").strip()

    relevant_urls = []
    if user_input.lower() != "none" and user_input:
        try:
            indices = [int(x.strip()) for x in user_input.split(",")]
            for idx in indices:
                if 1 <= idx <= len(results):
                    relevant_urls.append(results[idx - 1]["url"])
        except ValueError:
            print("Invalid input, creating test case with no relevant URLs")

    test_case = RAGTestCase(
        query=query,
        description=description,
        relevant_urls=relevant_urls,
        top_k=top_k,
    )

    print(f"\nCreated test case with {len(relevant_urls)} relevant URLs")
    return test_case


def save_test_cases(test_cases: list[RAGTestCase], filepath: str | Path):
    """Save test cases to a JSON file.

    Args:
        test_cases: List of RAGTestCase objects
        filepath: Path to save JSON file
    """
    data = []
    for tc in test_cases:
        data.append(
            {
                "query": tc.query,
                "description": tc.description,
                "relevant_urls": tc.relevant_urls,
                "relevant_keywords": tc.relevant_keywords,
                "top_k": tc.top_k,
                "metadata": tc.metadata,
            }
        )

    Path(filepath).write_text(json.dumps(data, indent=2))
    print(f"Saved {len(test_cases)} test cases to {filepath}")


def load_test_cases(filepath: str | Path) -> list[RAGTestCase]:
    """Load test cases from a JSON file.

    Args:
        filepath: Path to JSON file

    Returns:
        List of RAGTestCase objects
    """
    data = json.loads(Path(filepath).read_text())

    test_cases = []
    for item in data:
        test_cases.append(
            RAGTestCase(
                query=item["query"],
                description=item["description"],
                relevant_urls=item.get("relevant_urls", []),
                relevant_keywords=item.get("relevant_keywords", []),
                top_k=item.get("top_k", 5),
                metadata=item.get("metadata"),
            )
        )

    print(f"Loaded {len(test_cases)} test cases from {filepath}")
    return test_cases


# Example test cases for common RAG evaluation scenarios
EXAMPLE_TEST_CASES = """
# Example RAG Test Cases

Here's how to create test cases for your RAG system:

```python
from llm_tools_server.eval import RAGTestCase, RAGEvaluator, load_test_cases

# Option 1: Define test cases in code
tests = [
    RAGTestCase(
        query="how to configure vault namespaces",
        description="Vault namespace configuration docs",
        relevant_urls=[
            "https://developer.hashicorp.com/vault/docs/enterprise/namespaces",
            "https://developer.hashicorp.com/vault/tutorials/enterprise/namespaces",
        ],
        top_k=5,
    ),
    RAGTestCase(
        query="terraform state locking",
        description="Terraform state locking documentation",
        relevant_urls=[
            "https://developer.hashicorp.com/terraform/language/state/locking",
        ],
        top_k=5,
    ),
    # Use keywords when you don't know exact URLs
    RAGTestCase(
        query="consul service mesh",
        description="Consul service mesh overview",
        relevant_keywords=["connect", "sidecar", "proxy"],
        top_k=5,
    ),
]

# Option 2: Load from JSON file
tests = load_test_cases("my_test_cases.json")

# Option 3: Build interactively
from llm_tools_server.eval import create_test_case_interactive
test = create_test_case_interactive(index, "my query", "Test description")
```

## JSON Test Case Format

```json
[
    {
        "query": "how to configure vault namespaces",
        "description": "Vault namespace configuration docs",
        "relevant_urls": [
            "https://developer.hashicorp.com/vault/docs/enterprise/namespaces"
        ],
        "relevant_keywords": [],
        "top_k": 5,
        "metadata": {"category": "vault", "priority": "high"}
    }
]
```

## Running Evaluations

```python
from llm_tools_server.rag import DocSearchIndex, RAGConfig
from llm_tools_server.eval import RAGEvaluator

# Load your index
config = RAGConfig(base_url="https://docs.example.com", cache_dir="./cache")
index = DocSearchIndex(config)
index.load_index()

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
```
"""


def print_example_usage():
    """Print example usage for RAG test cases."""
    print(EXAMPLE_TEST_CASES)
