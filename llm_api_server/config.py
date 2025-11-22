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

    # Backend timeout settings (in seconds)
    BACKEND_CONNECT_TIMEOUT: int = 10  # Connection timeout
    BACKEND_READ_TIMEOUT: int = 300  # Read timeout (5 minutes for long completions)

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
        config.DEFAULT_PORT = int(get_env("PORT", str(cls.DEFAULT_PORT)))
        config.DEFAULT_TEMPERATURE = float(get_env("TEMPERATURE", str(cls.DEFAULT_TEMPERATURE)))
        config.SYSTEM_PROMPT_PATH = get_env("SYSTEM_PROMPT_PATH", cls.SYSTEM_PROMPT_PATH)
        config.WEBUI_PORT = int(get_env("WEBUI_PORT", str(cls.WEBUI_PORT)))
        config.DEBUG_TOOLS = get_env("DEBUG_TOOLS", "").lower() in ("true", "1", "yes")
        config.DEBUG_TOOLS_LOG_FILE = get_env("DEBUG_TOOLS_LOG_FILE", cls.DEBUG_TOOLS_LOG_FILE)
        config.BACKEND_CONNECT_TIMEOUT = int(get_env("BACKEND_CONNECT_TIMEOUT", str(cls.BACKEND_CONNECT_TIMEOUT)))
        config.BACKEND_READ_TIMEOUT = int(get_env("BACKEND_READ_TIMEOUT", str(cls.BACKEND_READ_TIMEOUT)))

        return config
