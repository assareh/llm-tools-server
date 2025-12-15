"""Tests for ServerConfig configuration loading."""

import pytest

from llm_tools_server.config import ServerConfig


@pytest.mark.unit
class TestServerConfig:
    """Test ServerConfig class."""

    def test_default_config(self, default_config):
        """Test that default config has expected values."""
        assert default_config.BACKEND_TYPE == "lmstudio"
        assert default_config.DEFAULT_PORT == 8000
        assert default_config.DEFAULT_HOST == "127.0.0.1"
        assert default_config.BACKEND_CONNECT_TIMEOUT == 10
        assert default_config.BACKEND_READ_TIMEOUT == 300
        assert default_config.HEALTH_CHECK_ON_STARTUP is True
        assert default_config.BACKEND_RETRY_ATTEMPTS == 3
        assert default_config.DEBUG_LOG_MAX_BYTES == 10 * 1024 * 1024
        assert default_config.THREADED is True

    def test_custom_config(self, custom_config):
        """Test custom config overrides."""
        assert custom_config.BACKEND_TYPE == "ollama"
        assert custom_config.BACKEND_MODEL == "llama2"
        assert custom_config.DEFAULT_PORT == 9000
        assert custom_config.DEBUG_TOOLS is True

    def test_from_env_with_prefix(self, monkeypatch):
        """Test loading config from environment variables with prefix."""
        monkeypatch.setenv("MYAPP_BACKEND", "ollama")
        monkeypatch.setenv("MYAPP_PORT", "9999")
        monkeypatch.setenv("MYAPP_HOST", "0.0.0.0")
        monkeypatch.setenv("MYAPP_DEBUG_TOOLS", "true")

        config = ServerConfig.from_env("MYAPP_")

        assert config.BACKEND_TYPE == "ollama"
        assert config.DEFAULT_PORT == 9999
        assert config.DEFAULT_HOST == "0.0.0.0"
        assert config.DEBUG_TOOLS is True

    def test_from_env_without_prefix(self, monkeypatch):
        """Test loading config from environment variables without prefix."""
        monkeypatch.setenv("BACKEND", "ollama")
        monkeypatch.setenv("PORT", "8888")
        monkeypatch.setenv("THREADED", "false")

        config = ServerConfig.from_env()

        assert config.BACKEND_TYPE == "ollama"
        assert config.DEFAULT_PORT == 8888
        assert config.THREADED is False

    def test_boolean_parsing(self, monkeypatch):
        """Test boolean environment variable parsing."""
        # Test DEBUG_TOOLS (true if value in list)
        monkeypatch.setenv("DEBUG_TOOLS", "true")
        config1 = ServerConfig.from_env()
        assert config1.DEBUG_TOOLS is True

        monkeypatch.setenv("DEBUG_TOOLS", "1")
        config2 = ServerConfig.from_env()
        assert config2.DEBUG_TOOLS is True

        monkeypatch.setenv("DEBUG_TOOLS", "false")
        config3 = ServerConfig.from_env()
        assert config3.DEBUG_TOOLS is False

        # Test HEALTH_CHECK_ON_STARTUP (true unless value in list)
        monkeypatch.setenv("HEALTH_CHECK_ON_STARTUP", "false")
        config4 = ServerConfig.from_env()
        assert config4.HEALTH_CHECK_ON_STARTUP is False

        monkeypatch.setenv("HEALTH_CHECK_ON_STARTUP", "true")
        config5 = ServerConfig.from_env()
        assert config5.HEALTH_CHECK_ON_STARTUP is True

    def test_numeric_parsing(self, monkeypatch):
        """Test numeric environment variable parsing."""
        monkeypatch.setenv("PORT", "7777")
        monkeypatch.setenv("BACKEND_CONNECT_TIMEOUT", "20")
        monkeypatch.setenv("BACKEND_READ_TIMEOUT", "600")
        monkeypatch.setenv("BACKEND_RETRY_ATTEMPTS", "5")
        monkeypatch.setenv("DEBUG_LOG_MAX_BYTES", "5242880")

        config = ServerConfig.from_env()

        assert config.DEFAULT_PORT == 7777
        assert config.BACKEND_CONNECT_TIMEOUT == 20
        assert config.BACKEND_READ_TIMEOUT == 600
        assert config.BACKEND_RETRY_ATTEMPTS == 5
        assert config.DEBUG_LOG_MAX_BYTES == 5242880
