"""LLM API Server - A reusable Flask server for LLM backends with tool calling."""

from .builtin_tools import (
    BUILTIN_TOOLS,
    calculate,
    create_web_search_tool,
    get_current_date,
)
from .config import ServerConfig
from .server import LLMServer
from .webui_utils import WebUIQueryFilter, is_webui_system_task

# Optional modules available but not imported by default to avoid dependency bloat:
# - Eval module: from llm_api_server.eval import Evaluator, TestCase, etc.
# - RAG module: from llm_api_server.rag import DocSearchIndex, RAGConfig

__version__ = "0.3.0"
__all__ = [
    "BUILTIN_TOOLS",
    "LLMServer",
    "ServerConfig",
    "WebUIQueryFilter",
    "calculate",
    "create_web_search_tool",
    "get_current_date",
    "is_webui_system_task",
]
