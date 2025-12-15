"""LLM API Server - A reusable Flask server for LLM backends with tool calling."""

from importlib.metadata import version

from .builtin_tools import (
    BUILTIN_TOOLS,
    calculate,
    create_doc_search_tool,
    create_web_search_tool,
    get_current_datetime,
)
from .config import ServerConfig
from .server import LLMServer

# Optional modules available but not imported by default to avoid dependency bloat:
# - Eval module: from llm_api_server.eval import Evaluator, TestCase, etc.
# - RAG module: from llm_api_server.rag import DocSearchIndex, RAGConfig

__version__ = version("llm-api-server")
__all__ = [
    "BUILTIN_TOOLS",
    "LLMServer",
    "ServerConfig",
    "calculate",
    "create_doc_search_tool",
    "create_web_search_tool",
    "get_current_datetime",
]
