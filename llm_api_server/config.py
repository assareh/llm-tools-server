"""Base configuration for LLM API Server."""

from typing import Literal, Optional


class ServerConfig:
    """Base configuration class for LLM API Server.

    Projects should subclass this and override as needed.
    """

    # Backend configuration
    BACKEND_TYPE: Literal["lmstudio", "ollama"] = "lmstudio"
    BACKEND_MODEL: str = "openai/gpt-oss-20b"

    # Backend endpoints
    LMSTUDIO_ENDPOINT: str = "http://localhost:1234/v1"
    OLLAMA_ENDPOINT: str = "http://localhost:11434"

    # Server configuration
    DEFAULT_HOST: str = "127.0.0.1"  # Default to localhost for security (use 0.0.0.0 for all interfaces)
    DEFAULT_PORT: int = 8000
    DEFAULT_TEMPERATURE: float = 0.0
    SYSTEM_PROMPT_PATH: str = "system_prompt.md"

    # Model name advertised via API
    MODEL_NAME: str = "llm-server/default"

    # WebUI configuration
    WEBUI_PORT: int = 8001
    ENABLE_WEBUI: bool = True

    # Debug settings
    DEBUG_TOOLS: bool = False
    DEBUG_TOOLS_LOG_FILE: str = "tools_debug.log"
    DEBUG_LOG_MAX_BYTES: int = 10 * 1024 * 1024  # 10MB default
    DEBUG_LOG_BACKUP_COUNT: int = 5  # Keep 5 backup files

    # Backend timeout settings (in seconds)
    BACKEND_CONNECT_TIMEOUT: int = 10  # Connection timeout
    BACKEND_READ_TIMEOUT: int = 300  # Read timeout (5 minutes for long completions)

    # Health check settings
    HEALTH_CHECK_ON_STARTUP: bool = True  # Check backend availability before starting server
    HEALTH_CHECK_TIMEOUT: int = 5  # Timeout for health check requests (in seconds)

    # Retry settings for backend calls
    BACKEND_RETRY_ATTEMPTS: int = 3  # Number of retry attempts for connection errors
    BACKEND_RETRY_INITIAL_DELAY: float = 1.0  # Initial delay in seconds (doubles each retry)

    # Custom prompt suggestions for WebUI (list of dicts with title and content)
    DEFAULT_PROMPT_SUGGESTIONS: Optional[list] = None

    @classmethod
    def from_env(cls, env_prefix: str = ""):
        """Create config from environment variables with optional prefix.

        Args:
            env_prefix: Prefix for environment variables (e.g., "IVAN_", "MILES_")

        Returns:
            ServerConfig instance populated from environment
        """
        import os

        from dotenv import load_dotenv

        load_dotenv()

        config = cls()

        # Helper to get env var with prefix
        def get_env(name: str, default):
            # Try with prefix first, then without
            prefixed = os.getenv(f"{env_prefix}{name}", None)
            if prefixed is not None:
                return prefixed
            return os.getenv(name, default)

        # Load configuration from environment
        config.BACKEND_TYPE = get_env("BACKEND", cls.BACKEND_TYPE)
        config.BACKEND_MODEL = get_env("BACKEND_MODEL", cls.BACKEND_MODEL)
        config.LMSTUDIO_ENDPOINT = get_env("LMSTUDIO_ENDPOINT", cls.LMSTUDIO_ENDPOINT)
        config.OLLAMA_ENDPOINT = get_env("OLLAMA_ENDPOINT", cls.OLLAMA_ENDPOINT)
        config.DEFAULT_HOST = get_env("HOST", cls.DEFAULT_HOST)
        config.DEFAULT_PORT = int(get_env("PORT", str(cls.DEFAULT_PORT)))
        config.DEFAULT_TEMPERATURE = float(get_env("TEMPERATURE", str(cls.DEFAULT_TEMPERATURE)))
        config.SYSTEM_PROMPT_PATH = get_env("SYSTEM_PROMPT_PATH", cls.SYSTEM_PROMPT_PATH)
        config.WEBUI_PORT = int(get_env("WEBUI_PORT", str(cls.WEBUI_PORT)))
        config.DEBUG_TOOLS = get_env("DEBUG_TOOLS", "").lower() in ("true", "1", "yes")
        config.DEBUG_TOOLS_LOG_FILE = get_env("DEBUG_TOOLS_LOG_FILE", cls.DEBUG_TOOLS_LOG_FILE)
        config.DEBUG_LOG_MAX_BYTES = int(get_env("DEBUG_LOG_MAX_BYTES", str(cls.DEBUG_LOG_MAX_BYTES)))
        config.DEBUG_LOG_BACKUP_COUNT = int(get_env("DEBUG_LOG_BACKUP_COUNT", str(cls.DEBUG_LOG_BACKUP_COUNT)))
        config.BACKEND_CONNECT_TIMEOUT = int(get_env("BACKEND_CONNECT_TIMEOUT", str(cls.BACKEND_CONNECT_TIMEOUT)))
        config.BACKEND_READ_TIMEOUT = int(get_env("BACKEND_READ_TIMEOUT", str(cls.BACKEND_READ_TIMEOUT)))
        config.HEALTH_CHECK_ON_STARTUP = get_env("HEALTH_CHECK_ON_STARTUP", "").lower() not in ("false", "0", "no")
        config.HEALTH_CHECK_TIMEOUT = int(get_env("HEALTH_CHECK_TIMEOUT", str(cls.HEALTH_CHECK_TIMEOUT)))
        config.BACKEND_RETRY_ATTEMPTS = int(get_env("BACKEND_RETRY_ATTEMPTS", str(cls.BACKEND_RETRY_ATTEMPTS)))
        config.BACKEND_RETRY_INITIAL_DELAY = float(
            get_env("BACKEND_RETRY_INITIAL_DELAY", str(cls.BACKEND_RETRY_INITIAL_DELAY))
        )

        return config
