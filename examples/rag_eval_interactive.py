#!/usr/bin/env python3
"""Interactive tool for building RAG test cases.

This script helps you explore your RAG index and create ground truth
test cases by reviewing search results and marking relevant ones.

Usage:
    uv run python examples/rag_eval_interactive.py --index-dir ./my_index

    # Or use Ivan's index:
    uv run python examples/rag_eval_interactive.py --index-dir ~/Developer/Ivan/hashicorp_docs_index
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_tools_server.eval import (
    RAGTestCase,
    inspect_search_results,
    load_test_cases,
    save_test_cases,
)
from llm_tools_server.rag import DocSearchIndex, RAGConfig


def interactive_session(index: DocSearchIndex, output_file: str = "test_cases.json"):
    """Run interactive test case building session."""
    test_cases = []

    # Try to load existing test cases
    output_path = Path(output_file)
    if output_path.exists():
        try:
            test_cases = load_test_cases(output_file)
            print(f"Loaded {len(test_cases)} existing test cases from {output_file}")
        except Exception as e:
            print(f"Could not load existing file: {e}")

    print("\n" + "=" * 70)
    print(" Interactive RAG Test Case Builder")
    print("=" * 70)
    print("\nCommands:")
    print("  <query>     - Search and review results")
    print("  save        - Save test cases to file")
    print("  list        - List current test cases")
    print("  quit/exit   - Save and quit")
    print("  help        - Show this help")
    print("=" * 70)

    while True:
        try:
            user_input = input("\nEnter query (or command): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            break
        elif user_input.lower() == "save":
            save_test_cases(test_cases, output_file)
            continue
        elif user_input.lower() == "list":
            if not test_cases:
                print("No test cases yet.")
            else:
                print(f"\nCurrent test cases ({len(test_cases)}):")
                for i, tc in enumerate(test_cases, 1):
                    print(f"  {i}. {tc.query[:50]}... ({len(tc.relevant_urls)} URLs)")
            continue
        elif user_input.lower() == "help":
            print("\nCommands: save, list, quit, help")
            print("Or enter a search query to explore results.")
            continue

        # It's a search query
        query = user_input
        results = inspect_search_results(index, query, top_k=10)

        if not results:
            print("No results found.")
            continue

        # Ask for relevance judgments
        print("-" * 70)
        print("Enter numbers of RELEVANT results (comma-separated), or:")
        print("  'skip' - skip this query")
        print("  'keywords' - mark by keywords instead of URLs")

        selection = input("> ").strip()

        if selection.lower() == "skip":
            continue

        relevant_urls = []
        relevant_keywords = []

        if selection.lower() == "keywords":
            kw_input = input("Enter relevant keywords (comma-separated): ").strip()
            relevant_keywords = [k.strip() for k in kw_input.split(",") if k.strip()]
        elif selection:
            try:
                indices = [int(x.strip()) for x in selection.split(",")]
                for idx in indices:
                    if 1 <= idx <= len(results):
                        relevant_urls.append(results[idx - 1]["url"])
            except ValueError:
                print("Invalid input, skipping.")
                continue

        if not relevant_urls and not relevant_keywords:
            print("No relevance marked, skipping.")
            continue

        # Get description
        description = input("Brief description of this test: ").strip()
        if not description:
            description = f"Test for: {query[:50]}"

        # Create test case
        test_case = RAGTestCase(
            query=query,
            description=description,
            relevant_urls=relevant_urls,
            relevant_keywords=relevant_keywords,
        )
        test_cases.append(test_case)
        print(f"Added test case. Total: {len(test_cases)}")

    # Save on exit
    if test_cases:
        save_test_cases(test_cases, output_file)
        print(f"\nSaved {len(test_cases)} test cases to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Interactive RAG test case builder")
    parser.add_argument(
        "--index-dir",
        type=str,
        default="./hashicorp_docs_index",
        help="Path to RAG index directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="test_cases.json",
        help="Output file for test cases",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="https://developer.hashicorp.com",
        help="Base URL (only used if building new index)",
    )
    args = parser.parse_args()

    index_dir = Path(args.index_dir).expanduser()

    if not index_dir.exists():
        print(f"Index directory not found: {index_dir}")
        print("Please provide a valid --index-dir or build an index first.")
        sys.exit(1)

    # Load index
    print(f"Loading index from {index_dir}...")
    config = RAGConfig(base_url=args.base_url, cache_dir=str(index_dir))
    index = DocSearchIndex(config)
    index.load_index()
    print(f"Loaded {len(index.chunks)} chunks")

    # Run interactive session
    interactive_session(index, args.output)


if __name__ == "__main__":
    main()
