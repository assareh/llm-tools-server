"""Tests for backend communication functions."""

from unittest.mock import Mock, patch

import pytest
import requests

from llm_tools_server.backends import check_lmstudio_health, check_ollama_health


@pytest.fixture(autouse=True)
def reset_backend_session():
    """Reset the module-level session between tests to ensure mocking works."""
    import llm_tools_server.backends as backends_module

    original_session = backends_module._session
    backends_module._session = None
    yield
    backends_module._session = original_session


@pytest.mark.unit
class TestBackendHealthChecks:
    """Test backend health check functions."""

    def test_ollama_health_model_not_found(self, default_config):
        """Test Ollama health check when configured model is not available."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "models": [
                {"name": "llama2"},
                {"name": "mistral"},
            ]
        }
        mock_response.raise_for_status = Mock()

        mock_session = Mock()
        mock_session.get.return_value = mock_response

        with patch("llm_tools_server.backends._get_session", return_value=mock_session):
            is_healthy, message = check_ollama_health(default_config)

        assert is_healthy is False  # Model not in list
        assert "not found" in message

    def test_ollama_health_model_available(self, custom_config):
        """Test Ollama health check with correct model."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "models": [
                {"name": "llama2"},
                {"name": "mistral"},
            ]
        }
        mock_response.raise_for_status = Mock()

        mock_session = Mock()
        mock_session.get.return_value = mock_response

        with patch("llm_tools_server.backends._get_session", return_value=mock_session):
            is_healthy, message = check_ollama_health(custom_config)

        assert is_healthy is True
        assert "llama2" in message
        assert "available" in message

    def test_ollama_health_connection_error(self, default_config):
        """Test Ollama health check with connection error."""
        mock_session = Mock()
        mock_session.get.side_effect = requests.ConnectionError("Connection refused")

        with patch("llm_tools_server.backends._get_session", return_value=mock_session):
            is_healthy, message = check_ollama_health(default_config)

        assert is_healthy is False
        assert "Cannot connect" in message
        assert "Is it running" in message

    def test_ollama_health_timeout(self, default_config):
        """Test Ollama health check with timeout."""
        mock_session = Mock()
        mock_session.get.side_effect = requests.Timeout

        with patch("llm_tools_server.backends._get_session", return_value=mock_session):
            is_healthy, message = check_ollama_health(default_config)

        assert is_healthy is False
        assert "timed out" in message

    def test_lmstudio_health_success(self, default_config):
        """Test successful LM Studio health check."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [
                {"id": "model1"},
                {"id": "model2"},
            ]
        }
        mock_response.raise_for_status = Mock()

        mock_session = Mock()
        mock_session.get.return_value = mock_response

        with patch("llm_tools_server.backends._get_session", return_value=mock_session):
            is_healthy, message = check_lmstudio_health(default_config)

        assert is_healthy is True
        assert "2 model(s) loaded" in message

    def test_lmstudio_health_no_models(self, default_config):
        """Test LM Studio health check with no models loaded."""
        mock_response = Mock()
        mock_response.json.return_value = {"data": []}
        mock_response.raise_for_status = Mock()

        mock_session = Mock()
        mock_session.get.return_value = mock_response

        with patch("llm_tools_server.backends._get_session", return_value=mock_session):
            is_healthy, message = check_lmstudio_health(default_config)

        assert is_healthy is False
        assert "no models are loaded" in message

    def test_lmstudio_health_connection_error(self, default_config):
        """Test LM Studio health check with connection error."""
        mock_session = Mock()
        mock_session.get.side_effect = requests.ConnectionError

        with patch("llm_tools_server.backends._get_session", return_value=mock_session):
            is_healthy, message = check_lmstudio_health(default_config)

        assert is_healthy is False
        assert "Cannot connect" in message


@pytest.mark.unit
class TestRetryLogic:
    """Test retry logic for backend calls."""

    def test_retry_on_connection_error(self, default_config):
        """Test that connection errors trigger retries."""
        from llm_tools_server.backends import _retry_on_connection_error

        mock_func = Mock(
            side_effect=[requests.ConnectionError("Failed"), requests.ConnectionError("Failed"), "Success"]
        )

        with patch("time.sleep"):  # Skip actual sleep delays
            result = _retry_on_connection_error(mock_func, default_config)

        assert result == "Success"
        assert mock_func.call_count == 3

    def test_no_retry_on_http_error(self, default_config):
        """Test that HTTP errors don't trigger retries."""
        from llm_tools_server.backends import _retry_on_connection_error

        http_error = requests.HTTPError("404 Not Found")
        mock_func = Mock(side_effect=http_error)

        with pytest.raises(requests.HTTPError):
            _retry_on_connection_error(mock_func, default_config)

        # Should only call once, no retries
        assert mock_func.call_count == 1

    def test_no_retry_on_timeout(self, default_config):
        """Test that timeouts don't trigger retries."""
        from llm_tools_server.backends import _retry_on_connection_error

        timeout_error = requests.Timeout("Request timed out")
        mock_func = Mock(side_effect=timeout_error)

        with pytest.raises(requests.Timeout):
            _retry_on_connection_error(mock_func, default_config)

        # Should only call once, no retries
        assert mock_func.call_count == 1

    def test_max_retries_exhausted(self, default_config):
        """Test behavior when max retries are exhausted."""
        from llm_tools_server.backends import _retry_on_connection_error

        mock_func = Mock(side_effect=requests.ConnectionError("Always fails"))

        with patch("time.sleep"), pytest.raises(requests.ConnectionError):  # Skip actual sleep delays
            _retry_on_connection_error(mock_func, default_config)

        # Should try 3 times (default BACKEND_RETRY_ATTEMPTS)
        assert mock_func.call_count == 3
