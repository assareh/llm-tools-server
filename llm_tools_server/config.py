"""Base configuration for LLM API Server."""

from collections.abc import Callable
from typing import Any, Literal


def _parse_bool_env(value: str, default: bool) -> bool:
    """Parse boolean from environment variable string.

    Args:
        value: Environment variable string value
        default: Default value if string is empty

    Returns:
        Parsed boolean value
    """
    if not value:
        return default
    return value.lower() in ("true", "1", "yes")


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

    # Optional API keys
    OLLAMA_API_KEY: str = ""  # For web search (optional)

    # Server configuration
    DEFAULT_HOST: str = "127.0.0.1"  # Default to localhost for security (use 0.0.0.0 for all interfaces)
    DEFAULT_PORT: int = 8000
    DEFAULT_TEMPERATURE: float = 0.0
    SYSTEM_PROMPT_PATH: str = "system_prompt.md"
    THREADED: bool = True  # Enable threaded mode for concurrent requests
    MAX_TOOL_ITERATIONS: int = 5  # Maximum tool calling loop iterations per request
    TOOL_LOOP_TIMEOUT: int = 120  # Maximum seconds for entire tool loop (0 = no timeout)
    FIRST_ITERATION_TOOL_CHOICE: Literal["auto", "required"] = "auto"  # Tool choice for first iteration

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
    DEBUG_LOG_FORMAT: Literal["text", "json", "yaml"] = "text"  # Log format: text, json, or yaml
    DEBUG_LOG_MAX_RESPONSE_LENGTH: int = 1000  # Max response chars in logs (0 = no truncation)

    # Backend timeout settings (in seconds)
    BACKEND_CONNECT_TIMEOUT: int = 10  # Connection timeout
    BACKEND_READ_TIMEOUT: int = 300  # Read timeout (5 minutes for long completions)

    # Health check settings
    HEALTH_CHECK_ON_STARTUP: bool = True  # Check backend availability before starting server
    HEALTH_CHECK_TIMEOUT: int = 5  # Timeout for health check requests (in seconds)

    # Retry settings for backend calls
    BACKEND_RETRY_ATTEMPTS: int = 3  # Number of retry attempts for connection errors
    BACKEND_RETRY_INITIAL_DELAY: float = 1.0  # Initial delay in seconds (doubles each retry)

    # Rate limiting (requires flask-limiter: pip install flask-limiter)
    RATE_LIMIT_ENABLED: bool = False  # Enable rate limiting on API endpoints
    RATE_LIMIT_DEFAULT: str = "100 per minute"  # Default rate limit (flask-limiter format)
    RATE_LIMIT_STORAGE_URI: str = "memory://"  # Storage backend (memory://, redis://localhost:6379, etc.)

    # Custom prompt suggestions for WebUI (list of dicts with title and content)
    DEFAULT_PROMPT_SUGGESTIONS: list | None = None

    # Request hook for debugging/logging LLM requests
    # Called with (backend_name: str, payload: dict) before each request
    REQUEST_HOOK: Callable[[str, dict[str, Any]], None] | None = None

    @classmethod
    def from_env(cls, env_prefix: str = ""):
        """Create config from environment variables with optional prefix.

        Args:
            env_prefix: Prefix for environment variables (e.g., "MYAPP_", "BOT_")

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
        # Support both BACKEND and BACKEND_TYPE env vars (BACKEND_TYPE takes precedence)
        config.BACKEND_TYPE = get_env("BACKEND_TYPE", None) or get_env("BACKEND", cls.BACKEND_TYPE)
        config.BACKEND_MODEL = get_env("BACKEND_MODEL", cls.BACKEND_MODEL)
        config.LMSTUDIO_ENDPOINT = get_env("LMSTUDIO_ENDPOINT", cls.LMSTUDIO_ENDPOINT)
        config.OLLAMA_ENDPOINT = get_env("OLLAMA_ENDPOINT", cls.OLLAMA_ENDPOINT)
        config.OLLAMA_API_KEY = get_env("OLLAMA_API_KEY", cls.OLLAMA_API_KEY)
        config.DEFAULT_HOST = get_env("HOST", cls.DEFAULT_HOST)
        config.DEFAULT_PORT = int(get_env("PORT", str(cls.DEFAULT_PORT)))
        config.DEFAULT_TEMPERATURE = float(get_env("TEMPERATURE", str(cls.DEFAULT_TEMPERATURE)))
        config.SYSTEM_PROMPT_PATH = get_env("SYSTEM_PROMPT_PATH", cls.SYSTEM_PROMPT_PATH)
        config.THREADED = _parse_bool_env(get_env("THREADED", ""), cls.THREADED)
        config.WEBUI_PORT = int(get_env("WEBUI_PORT", str(cls.WEBUI_PORT)))
        config.DEBUG_TOOLS = _parse_bool_env(get_env("DEBUG_TOOLS", ""), cls.DEBUG_TOOLS)
        config.DEBUG_TOOLS_LOG_FILE = get_env("DEBUG_TOOLS_LOG_FILE", cls.DEBUG_TOOLS_LOG_FILE)
        config.DEBUG_LOG_MAX_BYTES = int(get_env("DEBUG_LOG_MAX_BYTES", str(cls.DEBUG_LOG_MAX_BYTES)))
        config.DEBUG_LOG_BACKUP_COUNT = int(get_env("DEBUG_LOG_BACKUP_COUNT", str(cls.DEBUG_LOG_BACKUP_COUNT)))
        debug_format = get_env("DEBUG_LOG_FORMAT", cls.DEBUG_LOG_FORMAT)
        if debug_format not in ("text", "json", "yaml"):
            debug_format = "text"
        config.DEBUG_LOG_FORMAT = debug_format
        config.DEBUG_LOG_MAX_RESPONSE_LENGTH = int(
            get_env("DEBUG_LOG_MAX_RESPONSE_LENGTH", str(cls.DEBUG_LOG_MAX_RESPONSE_LENGTH))
        )
        config.BACKEND_CONNECT_TIMEOUT = int(get_env("BACKEND_CONNECT_TIMEOUT", str(cls.BACKEND_CONNECT_TIMEOUT)))
        config.BACKEND_READ_TIMEOUT = int(get_env("BACKEND_READ_TIMEOUT", str(cls.BACKEND_READ_TIMEOUT)))
        config.HEALTH_CHECK_ON_STARTUP = _parse_bool_env(
            get_env("HEALTH_CHECK_ON_STARTUP", ""), cls.HEALTH_CHECK_ON_STARTUP
        )
        config.HEALTH_CHECK_TIMEOUT = int(get_env("HEALTH_CHECK_TIMEOUT", str(cls.HEALTH_CHECK_TIMEOUT)))
        config.BACKEND_RETRY_ATTEMPTS = int(get_env("BACKEND_RETRY_ATTEMPTS", str(cls.BACKEND_RETRY_ATTEMPTS)))
        config.BACKEND_RETRY_INITIAL_DELAY = float(
            get_env("BACKEND_RETRY_INITIAL_DELAY", str(cls.BACKEND_RETRY_INITIAL_DELAY))
        )
        config.RATE_LIMIT_ENABLED = _parse_bool_env(get_env("RATE_LIMIT_ENABLED", ""), cls.RATE_LIMIT_ENABLED)
        config.RATE_LIMIT_DEFAULT = get_env("RATE_LIMIT_DEFAULT", cls.RATE_LIMIT_DEFAULT)
        config.RATE_LIMIT_STORAGE_URI = get_env("RATE_LIMIT_STORAGE_URI", cls.RATE_LIMIT_STORAGE_URI)
        config.MAX_TOOL_ITERATIONS = int(get_env("MAX_TOOL_ITERATIONS", str(cls.MAX_TOOL_ITERATIONS)))
        config.TOOL_LOOP_TIMEOUT = int(get_env("TOOL_LOOP_TIMEOUT", str(cls.TOOL_LOOP_TIMEOUT)))
        first_tool_choice = get_env("FIRST_ITERATION_TOOL_CHOICE", cls.FIRST_ITERATION_TOOL_CHOICE)
        if first_tool_choice not in ("auto", "required"):
            first_tool_choice = "auto"
        config.FIRST_ITERATION_TOOL_CHOICE = first_tool_choice

        return config
