"""Automated tests for web search tool (no network)."""

from unittest.mock import patch

import pytest

from llm_tools_server import ServerConfig, create_web_search_tool


@pytest.mark.unit
def test_web_search_requires_api_key():
    """Should short-circuit without calling Ollama when no API key is set."""
    config = ServerConfig()
    config.OLLAMA_API_KEY = ""

    with patch("llm_tools_server.web_search_tool.ollama_web_search") as mock_search:
        tool = create_web_search_tool(config)
        result = tool.func(query="Python programming language", max_results=3)

    assert "requires OLLAMA_API_KEY" in result
    mock_search.assert_not_called()


@pytest.mark.unit
def test_web_search_calls_ollama_and_formats_results():
    """Should invoke Ollama search and format results when API key is provided."""
    config = ServerConfig()
    config.OLLAMA_API_KEY = "test-key"

    fake_results = [
        {"title": "Result One", "url": "https://example.com/1", "description": "First result"},
        {"title": "Result Two", "url": "https://example.com/2", "description": "Second result"},
    ]

    with patch("llm_tools_server.web_search_tool.ollama_web_search", return_value=fake_results) as mock_search:
        tool = create_web_search_tool(config)
        output = tool.func(query="vault", max_results=1, site="hashicorp.com")

    mock_search.assert_called_once()
    called_query = mock_search.call_args[0][0]
    assert called_query.startswith("site:hashicorp.com")
    assert "Result One" in output
    assert "https://example.com/1" in output


@pytest.mark.unit
def test_web_search_handles_errors_gracefully():
    """Should surface a clear error message when Ollama search fails."""
    config = ServerConfig()
    config.OLLAMA_API_KEY = "test-key"

    with patch(
        "llm_tools_server.web_search_tool.ollama_web_search",
        side_effect=RuntimeError("boom"),
    ):
        tool = create_web_search_tool(config)
        output = tool.func(query="anything", max_results=3)

    assert "Web search failed" in output
    assert "boom" in output
