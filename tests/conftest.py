"""Shared pytest fixtures for LLM API Server tests."""

import pytest

from llm_tools_server.config import ServerConfig


@pytest.fixture
def default_config():
    """Provide a default ServerConfig instance for testing."""
    return ServerConfig()


@pytest.fixture
def custom_config():
    """Provide a custom ServerConfig instance for testing."""
    config = ServerConfig()
    config.BACKEND_TYPE = "ollama"
    config.BACKEND_MODEL = "llama2"
    config.OLLAMA_ENDPOINT = "http://localhost:11434"
    config.DEFAULT_PORT = 9000
    config.DEBUG_TOOLS = True
    return config


@pytest.fixture
def sample_messages():
    """Provide sample chat messages for testing."""
    return [
        {"role": "user", "content": "Hello, how are you?"},
    ]


@pytest.fixture
def sample_tools():
    """Provide sample LangChain tools for testing."""
    from langchain_core.tools import tool

    @tool
    def test_tool(query: str) -> str:
        """A test tool that echoes the input."""
        return f"Echo: {query}"

    return [test_tool]
