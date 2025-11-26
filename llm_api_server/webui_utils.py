"""Utilities for Open Web UI integration."""

import re
from re import Pattern
from typing import ClassVar


class WebUIQueryFilter:
    """Filter to detect Open Web UI system-generated tasks.

    Open Web UI generates various system tasks (title generation, tag generation,
    follow-up suggestions) that don't need RAG context retrieval. This filter
    helps identify and skip expensive operations for these tasks.

    Example usage:
        filter = WebUIQueryFilter()

        # Check if query is a system task
        if filter.is_system_task(user_query):
            # Skip RAG retrieval
            context = ""
        else:
            # Do RAG retrieval
            context = rag_index.search(user_query)
    """

    # Default patterns for Open Web UI system tasks
    DEFAULT_PATTERNS: ClassVar[list[str]] = [
        r"^### Task:\s*\nSuggest \d+-\d+ relevant follow-up questions",  # Follow-up generation
        r"^### Task:\s*\nGenerate a concise, \d+-\d+ word title",  # Title generation
        r"^### Task:\s*\nGenerate \d+-\d+ broad tags categorizing",  # Tag generation
    ]

    def __init__(
        self,
        patterns: list[str] | None = None,
        use_defaults: bool = True,
        case_sensitive: bool = False,
    ):
        """Initialize the query filter.

        Args:
            patterns: Custom regex patterns to match system tasks (in addition to defaults)
            use_defaults: Whether to include default Open Web UI patterns
            case_sensitive: Whether pattern matching should be case-sensitive
        """
        self.case_sensitive = case_sensitive
        self._patterns: list[Pattern] = []

        # Compile patterns
        flags = 0 if case_sensitive else re.IGNORECASE

        if use_defaults:
            for pattern in self.DEFAULT_PATTERNS:
                self._patterns.append(re.compile(pattern, flags))

        if patterns:
            for pattern in patterns:
                self._patterns.append(re.compile(pattern, flags))

    def is_system_task(self, query: str) -> bool:
        """Check if a query is a system-generated task.

        Args:
            query: The user query to check

        Returns:
            True if the query matches a system task pattern, False otherwise
        """
        if not query:
            return False

        return any(pattern.search(query) for pattern in self._patterns)

    def add_pattern(self, pattern: str) -> None:
        """Add a custom pattern to the filter.

        Args:
            pattern: Regex pattern to match system tasks
        """
        flags = 0 if self.case_sensitive else re.IGNORECASE
        self._patterns.append(re.compile(pattern, flags))


# Singleton instance for convenience
default_filter = WebUIQueryFilter()


def is_webui_system_task(query: str) -> bool:
    """Convenience function to check if a query is a Web UI system task.

    Uses the default filter with standard Open Web UI patterns.

    Args:
        query: The user query to check

    Returns:
        True if the query is a system task, False otherwise

    Example:
        from llm_api_server.webui_utils import is_webui_system_task

        if is_webui_system_task(user_query):
            # Skip expensive RAG retrieval
            context = ""
        else:
            context = rag_index.search(user_query)
    """
    return default_filter.is_system_task(query)
