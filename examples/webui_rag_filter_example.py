"""Example: Filtering Open Web UI system tasks to skip RAG retrieval.

This example demonstrates how to use the WebUI query filter in a consuming
project to avoid expensive RAG retrieval for system-generated tasks.

Usage in your consuming project (e.g., milesoss, Ivan):
    1. Import the filter
    2. Check queries before doing RAG retrieval
    3. Save API calls, latency, and costs
"""

from llm_api_server import is_webui_system_task


def rag_tool_basic(query: str) -> str:
    """Basic example: Use convenience function to filter system tasks."""
    # Skip RAG for Open Web UI system tasks (title, tags, follow-ups)
    if is_webui_system_task(query):
        return ""  # No RAG context needed

    # Do expensive RAG retrieval for real user queries
    # (Replace this with your actual RAG implementation)
    context = f"RAG results for: {query}"
    return context


def rag_tool_with_logging(query: str, logger) -> str:
    """Advanced example: Filter with logging."""
    if is_webui_system_task(query):
        logger.debug(
            {
                "event": "rag_skipped",
                "reason": "webui_system_task",
                "query": query[:100],  # Truncate for logging
            }
        )
        return ""

    # Log that we're doing RAG retrieval
    logger.debug(
        {
            "event": "rag_retrieval",
            "query": query,
        }
    )

    # Do RAG retrieval
    context = f"RAG results for: {query}"

    logger.debug(
        {
            "event": "rag_completed",
            "context_length": len(context),
        }
    )

    return context


def demo():
    """Demonstrate the filter in action."""

    # Real user queries - these SHOULD trigger RAG
    real_queries = [
        "do you get anything at checkin for being Marriott titanium",
        "what are Hyatt's elite tiers",
        "How many points do I need for Titanium status?",
    ]

    # System tasks - these SHOULD NOT trigger RAG
    system_tasks = [
        """### Task:
Suggest 3-5 relevant follow-up questions or prompts that the user might naturally ask next""",
        """### Task:
Generate a concise, 3-5 word title with an emoji summarizing the chat history.""",
        """### Task:
Generate 1-3 broad tags categorizing the main themes of the chat history""",
    ]

    print("=" * 80)
    print("REAL USER QUERIES (RAG should run)")
    print("=" * 80)

    for query in real_queries:
        is_system = is_webui_system_task(query)
        result = rag_tool_basic(query)

        print(f"\nQuery: {query[:60]}...")
        print(f"Is system task: {is_system}")
        print(f"RAG result: {result or '(skipped)'}")

    print("\n" + "=" * 80)
    print("SYSTEM TASKS (RAG should be skipped)")
    print("=" * 80)

    for query in system_tasks:
        is_system = is_webui_system_task(query)
        result = rag_tool_basic(query)

        print(f"\nQuery: {query[:60]}...")
        print(f"Is system task: {is_system}")
        print(f"RAG result: {result or '(skipped)'}")

    # Performance impact estimation
    print("\n" + "=" * 80)
    print("PERFORMANCE IMPACT")
    print("=" * 80)

    total_queries = len(real_queries) + len(system_tasks)
    system_count = len(system_tasks)
    savings_percent = (system_count / total_queries) * 100

    print(f"\nTotal queries: {total_queries}")
    print(f"System tasks filtered: {system_count}")
    print(f"RAG calls saved: {savings_percent:.1f}%")
    print("\nAssuming 500ms RAG latency and $0.01 per call:")
    print(f"  Time saved: {system_count * 0.5:.1f}s per {total_queries} queries")
    print(f"  Cost saved: ${system_count * 0.01:.2f} per {total_queries} queries")


if __name__ == "__main__":
    demo()
