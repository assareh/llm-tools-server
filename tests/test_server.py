"""Tests for LLMServer class."""

import pytest

from llm_api_server.server import LLMServer


@pytest.mark.unit
class TestLLMServer:
    """Test LLMServer class initialization and basic functionality."""

    def test_server_initialization(self, default_config, sample_tools):
        """Test that server initializes correctly."""
        server = LLMServer(
            name="TestServer",
            model_name="test/model",
            tools=sample_tools,
            config=default_config,
        )

        assert server.name == "TestServer"
        assert server.model_name == "test/model"
        assert len(server.tools) == 1
        assert server.config == default_config
        assert server.app is not None

    def test_tool_execution_success(self, default_config, sample_tools):
        """Test successful tool execution."""
        server = LLMServer(
            name="TestServer",
            model_name="test/model",
            tools=sample_tools,
            config=default_config,
        )

        result = server.execute_tool("test_tool", {"query": "hello"})

        assert "Echo: hello" in result

    def test_tool_execution_not_found(self, default_config, sample_tools):
        """Test tool execution with non-existent tool."""
        server = LLMServer(
            name="TestServer",
            model_name="test/model",
            tools=sample_tools,
            config=default_config,
        )

        result = server.execute_tool("nonexistent_tool", {})

        assert "not found" in result
        assert "test_tool" in result  # Should list available tools

    def test_tool_execution_error(self, default_config):
        """Test tool execution with error in tool function."""
        from langchain_core.tools import tool

        @tool
        def broken_tool(query: str) -> str:
            """A tool that always raises an error."""
            raise ValueError("Tool is broken")

        server = LLMServer(
            name="TestServer",
            model_name="test/model",
            tools=[broken_tool],
            config=default_config,
        )

        result = server.execute_tool("broken_tool", {"query": "test"})

        assert "Error executing tool" in result
        assert "ValueError" in result
        assert "broken" in result

    def test_system_prompt_loading_default(self, default_config, sample_tools, tmp_path):
        """Test loading default system prompt when file doesn't exist."""
        # Set non-existent path
        default_config.SYSTEM_PROMPT_PATH = str(tmp_path / "nonexistent.md")

        server = LLMServer(
            name="TestServer",
            model_name="test/model",
            tools=sample_tools,
            config=default_config,
            default_system_prompt="Custom default prompt",
        )

        prompt = server.get_system_prompt()

        assert prompt == "Custom default prompt"

    def test_system_prompt_loading_from_file(self, default_config, sample_tools, tmp_path):
        """Test loading system prompt from file."""
        prompt_file = tmp_path / "test_prompt.md"
        prompt_file.write_text("This is a test system prompt")

        default_config.SYSTEM_PROMPT_PATH = str(prompt_file)

        server = LLMServer(
            name="TestServer",
            model_name="test/model",
            tools=sample_tools,
            config=default_config,
        )

        prompt = server.get_system_prompt()

        assert prompt == "This is a test system prompt"

    def test_system_prompt_caching(self, default_config, sample_tools, tmp_path):
        """Test that system prompt is cached and reloaded on file change."""
        prompt_file = tmp_path / "test_prompt.md"
        prompt_file.write_text("Original prompt")

        default_config.SYSTEM_PROMPT_PATH = str(prompt_file)

        server = LLMServer(
            name="TestServer",
            model_name="test/model",
            tools=sample_tools,
            config=default_config,
        )

        # First load
        prompt1 = server.get_system_prompt()
        assert prompt1 == "Original prompt"

        # Second load (should use cache)
        prompt2 = server.get_system_prompt()
        assert prompt2 == "Original prompt"

        # Modify file
        import time

        time.sleep(0.01)  # Ensure mtime changes
        prompt_file.write_text("Updated prompt")

        # Third load (should reload from file)
        prompt3 = server.get_system_prompt()
        assert prompt3 == "Updated prompt"


@pytest.mark.unit
class TestServerRoutes:
    """Test Flask routes."""

    def test_health_endpoint(self, default_config, sample_tools):
        """Test /health endpoint."""
        server = LLMServer(
            name="TestServer",
            model_name="test/model",
            tools=sample_tools,
            config=default_config,
        )

        with server.app.test_client() as client:
            response = client.get("/health")

            assert response.status_code == 200
            data = response.get_json()
            assert data["status"] == "healthy"
            assert data["backend"] == default_config.BACKEND_TYPE
            assert data["model"] == "test/model"

    def test_list_models_endpoint(self, default_config, sample_tools):
        """Test /v1/models endpoint."""
        server = LLMServer(
            name="TestServer",
            model_name="test/model",
            tools=sample_tools,
            config=default_config,
        )

        with server.app.test_client() as client:
            response = client.get("/v1/models")

            assert response.status_code == 200
            data = response.get_json()
            assert data["object"] == "list"
            assert len(data["data"]) == 1
            assert data["data"][0]["id"] == "test/model"

    def test_chat_completions_invalid_json(self, default_config, sample_tools):
        """Test /v1/chat/completions with invalid JSON."""
        server = LLMServer(
            name="TestServer",
            model_name="test/model",
            tools=sample_tools,
            config=default_config,
        )

        with server.app.test_client() as client:
            response = client.post(
                "/v1/chat/completions",
                data="not json",
                content_type="application/json",
            )

            assert response.status_code == 400
            data = response.get_json()
            assert "Invalid JSON" in data["error"]

    def test_chat_completions_missing_messages(self, default_config, sample_tools):
        """Test /v1/chat/completions with missing messages field."""
        server = LLMServer(
            name="TestServer",
            model_name="test/model",
            tools=sample_tools,
            config=default_config,
        )

        with server.app.test_client() as client:
            response = client.post(
                "/v1/chat/completions",
                json={"temperature": 0.5},
            )

            assert response.status_code == 400
            data = response.get_json()
            assert "messages" in data["error"]

    def test_chat_completions_empty_messages(self, default_config, sample_tools):
        """Test /v1/chat/completions with empty messages array."""
        server = LLMServer(
            name="TestServer",
            model_name="test/model",
            tools=sample_tools,
            config=default_config,
        )

        with server.app.test_client() as client:
            response = client.post(
                "/v1/chat/completions",
                json={"messages": []},
            )

            assert response.status_code == 400
            data = response.get_json()
            assert "cannot be empty" in data["error"]
