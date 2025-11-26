"""Tests for Open Web UI query filtering."""

import pytest

from llm_api_server.webui_utils import WebUIQueryFilter, is_webui_system_task


def test_convenience_function_detects_follow_up_task():
    """Test the convenience function with a follow-up generation task."""
    query = """### Task:
Suggest 3-5 relevant follow-up questions or prompts that the user might naturally ask next in this conversation as a **user**, based on the chat history, to help continue or deepen the discussion.
### Guidelines:
- Write all follow-up questions from the user's point of view, directed to the assistant."""

    assert is_webui_system_task(query) is True


def test_convenience_function_detects_title_task():
    """Test the convenience function with a title generation task."""
    query = """### Task:
Generate a concise, 3-5 word title with an emoji summarizing the chat history.
### Guidelines:
- The title should clearly represent the main theme or subject of the conversation."""

    assert is_webui_system_task(query) is True


def test_convenience_function_detects_tags_task():
    """Test the convenience function with a tag generation task."""
    query = """### Task:
Generate 1-3 broad tags categorizing the main themes of the chat history, along with 1-3 more specific subtopic tags.

### Guidelines:
- Start with high-level domains (e.g. Science, Technology, Philosophy, Arts)"""

    assert is_webui_system_task(query) is True


def test_convenience_function_allows_real_query():
    """Test the convenience function with a real user query."""
    query = "do you get anything at checkin for being Marriott titanium"

    assert is_webui_system_task(query) is False


def test_convenience_function_allows_another_real_query():
    """Test the convenience function with another real user query."""
    query = "who gets guaranteed room availability with Marriott Bonvoy"

    assert is_webui_system_task(query) is False


def test_filter_detects_follow_up_task():
    """Test filter with a follow-up generation task."""
    filter = WebUIQueryFilter()
    query = """### Task:
Suggest 3-5 relevant follow-up questions or prompts that the user might naturally ask next in this conversation as a **user**, based on the chat history, to help continue or deepen the discussion."""

    assert filter.is_system_task(query) is True


def test_filter_detects_title_task():
    """Test filter with a title generation task."""
    filter = WebUIQueryFilter()
    query = """### Task:
Generate a concise, 3-5 word title with an emoji summarizing the chat history."""

    assert filter.is_system_task(query) is True


def test_filter_detects_tags_task():
    """Test filter with a tag generation task."""
    filter = WebUIQueryFilter()
    query = """### Task:
Generate 1-3 broad tags categorizing the main themes of the chat history, along with 1-3 more specific subtopic tags."""

    assert filter.is_system_task(query) is True


def test_filter_allows_real_query():
    """Test filter allows real user queries through."""
    filter = WebUIQueryFilter()
    query = "what are Hyatt's elite tiers"

    assert filter.is_system_task(query) is False


def test_filter_case_insensitive_by_default():
    """Test filter is case-insensitive by default."""
    filter = WebUIQueryFilter()
    query = """### task:
suggest 3-5 relevant follow-up questions"""

    assert filter.is_system_task(query) is True


def test_filter_case_sensitive_when_configured():
    """Test filter can be configured for case-sensitive matching."""
    filter = WebUIQueryFilter(case_sensitive=True)
    query = """### task:
Suggest 3-5 relevant follow-up questions"""

    # Should NOT match because "task" is lowercase
    assert filter.is_system_task(query) is False


def test_filter_custom_patterns():
    """Test filter with custom patterns."""
    filter = WebUIQueryFilter(
        patterns=[r"^CUSTOM: Generate"],
        use_defaults=False,
    )

    # Should match custom pattern
    assert filter.is_system_task("CUSTOM: Generate something") is True

    # Should NOT match default patterns (disabled)
    assert filter.is_system_task("### Task:\nGenerate a title") is False


def test_filter_add_pattern():
    """Test adding patterns dynamically."""
    filter = WebUIQueryFilter(use_defaults=False)

    # Initially doesn't match anything
    assert filter.is_system_task("### Task:\nGenerate") is False

    # Add a pattern
    filter.add_pattern(r"^### Task:")

    # Now it should match
    assert filter.is_system_task("### Task:\nGenerate") is True


def test_filter_empty_query():
    """Test filter handles empty queries gracefully."""
    filter = WebUIQueryFilter()

    assert filter.is_system_task("") is False
    assert filter.is_system_task(None) is False


def test_filter_combined_default_and_custom():
    """Test filter with both default and custom patterns."""
    filter = WebUIQueryFilter(
        patterns=[r"^CUSTOM:"],
        use_defaults=True,
    )

    # Should match default patterns (use full expected phrase)
    assert filter.is_system_task("### Task:\nSuggest 3-5 relevant follow-up questions") is True

    # Should also match custom patterns
    assert filter.is_system_task("CUSTOM: Do something") is True

    # Should NOT match unrelated queries
    assert filter.is_system_task("Regular user question") is False


def test_actual_log_queries():
    """Test with actual queries from the user's log."""
    # Real user queries - should NOT be filtered
    real_queries = [
        "do you get anything at checkin for being Marriott titanium",
        "who gets guaranteed room availability with Marriott Bonvoy",
        "How many points do I need to reach Titanium Elite status with Marriott Bonvoy?",
        "what are Hyatt's elite tiers",
    ]

    for query in real_queries:
        assert is_webui_system_task(query) is False, f"Real query incorrectly filtered: {query}"

    # System tasks - SHOULD be filtered
    system_tasks = [
        """### Task:
Suggest 3-5 relevant follow-up questions or prompts that the user might naturally ask next in this conversation as a **user**, based on the chat history, to help continue or deepen the discussion.
### Guidelines:
- Write all follow-up questions from the user's point of view, directed to the assistant.
- Make questions concise, clear, and directly related to the discussed topic(s).
- Only suggest follow-ups that make sense given the chat content and do not repeat what was already covered.
- If the conversation is very short or not specific, suggest more general (but relevant) follow-ups the user might ask.
- Use the conversation's primary language; default to English if multilingual.
- Response must be a JSON array of strings, no extra text or formatting.
### Output:
JSON format: { "follow_ups": ["Question 1?", "Question 2?", "Question 3?"] }""",
        """### Task:
Generate a concise, 3-5 word title with an emoji summarizing the chat history.
### Guidelines:
- The title should clearly represent the main theme or subject of the conversation.
- Use emojis that enhance understanding of the topic, but avoid quotation marks or special formatting.
- Write the title in the chat's primary language; default to English if multilingual.
- Prioritize accuracy over excessive creativity; keep it clear and simple.""",
        """### Task:
Generate 1-3 broad tags categorizing the main themes of the chat history, along with 1-3 more specific subtopic tags.

### Guidelines:
- Start with high-level domains (e.g. Science, Technology, Philosophy, Arts, Politics, Business, Health, Sports, Entertainment, Education)
- Consider including relevant subfields/subdomains if they are strongly represented throughout the conversation
- If content is too short (less than 3 messages) or too diverse, use only ["General"]
- Use the chat's primary language; default to English if multilingual
- Prioritize accuracy over specificity

### Output:
JSON format: { "tags": ["tag1", "tag2", "tag3"] }""",
    ]

    for task in system_tasks:
        assert is_webui_system_task(task) is True, f"System task not filtered: {task[:100]}..."


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
