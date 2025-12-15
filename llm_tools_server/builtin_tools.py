"""Built-in tools for LLM API Server.

These tools provide common functionality that can be used across different applications.
Users can import individual tools or use the BUILTIN_TOOLS collection.
"""

import ast
import operator
from datetime import datetime
from typing import TYPE_CHECKING

from langchain_core.tools import Tool, tool
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from .config import ServerConfig
    from .rag import DocSearchIndex


@tool
def get_current_datetime() -> str:
    """Get the current date and time in the local timezone.

    Returns the current date and time formatted for human readability,
    including the timezone name. This is useful for answering questions
    about "what time is it" or "what's today's date".

    Returns:
        Current date and time string (e.g., "Wednesday, November 26, 2025 at 2:30 PM PST")
    """
    # Get current time in local timezone
    local_tz = datetime.now().astimezone().tzinfo
    now = datetime.now(local_tz)

    # Format: "Wednesday, November 26, 2025 at 2:30 PM PST"
    date_str = now.strftime("%A, %B %d, %Y")
    time_str = now.strftime("%I:%M %p").lstrip("0")  # Remove leading zero from hour
    tz_name = now.strftime("%Z")

    return f"{date_str} at {time_str} {tz_name}"


@tool
def calculate(expression: str) -> str:
    """Safely evaluate a mathematical expression.

    Supports basic arithmetic operations: +, -, *, /, //, %, ** (power)
    Also supports parentheses for grouping.

    Args:
        expression: Mathematical expression to evaluate (e.g., "2 + 3 * 4", "(10 + 5) / 3")

    Returns:
        Result of the calculation as a string

    Examples:
        - calculate("2 + 3") -> "5"
        - calculate("10 * (5 + 3)") -> "80"
        - calculate("2 ** 8") -> "256"
    """
    # Mapping of allowed operators
    ALLOWED_OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,  # Unary minus
    }

    def eval_node(node):
        """Recursively evaluate AST nodes."""
        if isinstance(node, ast.Constant):  # Numbers
            # Only allow numeric constants; bool subclasses int so block explicitly
            if not isinstance(node.value, (int, float, complex)) or isinstance(node.value, bool):
                raise ValueError(f"Only numeric constants allowed, got {type(node.value).__name__}")
            return node.value
        elif isinstance(node, ast.BinOp):  # Binary operations
            op = ALLOWED_OPERATORS.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
            left = eval_node(node.left)
            right = eval_node(node.right)
            return op(left, right)
        elif isinstance(node, ast.UnaryOp):  # Unary operations
            op = ALLOWED_OPERATORS.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
            operand = eval_node(node.operand)
            return op(operand)
        else:
            raise ValueError(f"Unsupported expression type: {type(node).__name__}")

    try:
        # Parse the expression into an AST
        tree = ast.parse(expression, mode="eval")
        # Evaluate the AST
        result = eval_node(tree.body)
        # Format the result nicely
        if isinstance(result, float) and result.is_integer():
            return str(int(result))
        return str(result)
    except SyntaxError as e:
        return f"Syntax error in expression: {e}"
    except ValueError as e:
        return f"Error: {e}"
    except ZeroDivisionError:
        return "Error: Division by zero"
    except Exception as e:
        return f"Error evaluating expression: {e}"


class WebSearchInput(BaseModel):
    """Input schema for web search tool."""

    query: str = Field(
        description="The search query (e.g., 'Python async programming best practices', 'Docker container networking')"
    )
    max_results: int = Field(default=5, description="Maximum number of results to return. Default is 5.")
    site: str = Field(default="", description="Optional site restriction (e.g., 'hashicorp.com')")


class DocSearchInput(BaseModel):
    """Input schema for document search tool."""

    query: str = Field(
        description="The search query to find relevant documentation (e.g., 'how to configure authentication', 'API rate limits')"
    )
    top_k: int = Field(default=5, description="Number of results to return. Default is 5.")


def create_web_search_tool(config: "ServerConfig") -> Tool:
    """Create a web search tool configured with the given ServerConfig.

    This tool requires the optional 'websearch' dependency.
    Install with: uv sync --extra websearch

    Uses Ollama web search API. Requires OLLAMA_API_KEY to be configured.

    Args:
        config: ServerConfig instance with OLLAMA_API_KEY

    Returns:
        LangChain Tool for web search

    Example:
        >>> from llm_tools_server import ServerConfig, create_web_search_tool
        >>> config = ServerConfig.from_env()
        >>> web_search = create_web_search_tool(config)
        >>> tools = [get_current_datetime, calculate, web_search]
    """
    from .web_search_tool import web_search

    def _web_search_wrapper(query: str, max_results: int = 5, site: str = "") -> str:
        """Wrapper that provides API key from config."""
        return web_search(query, max_results, site, ollama_api_key=config.OLLAMA_API_KEY)

    return Tool(
        name="web_search",
        description="Search the web for general information using Ollama API. Use this for finding current information, documentation, tutorials, Stack Overflow answers, or any online resources. Returns titles, URLs, and descriptions of relevant pages. Requires OLLAMA_API_KEY to be configured.",
        func=_web_search_wrapper,
        args_schema=WebSearchInput,
    )


def create_doc_search_tool(
    index: "DocSearchIndex",
    name: str = "doc_search",
    description: str | None = None,
) -> Tool:
    """Create a document search tool that queries a local RAG index.

    This tool requires the optional 'rag' dependency.
    Install with: uv sync --extra rag

    The tool wraps a DocSearchIndex instance to provide semantic search
    over locally indexed documentation.

    Args:
        index: DocSearchIndex instance (must be initialized and loaded)
        name: Tool name (default: "doc_search")
        description: Custom description (default: generic doc search description)

    Returns:
        LangChain Tool for document search

    Example:
        >>> from llm_tools_server.rag import DocSearchIndex, RAGConfig
        >>> from llm_tools_server import create_doc_search_tool
        >>>
        >>> # Set up RAG index
        >>> config = RAGConfig(base_url="https://docs.example.com", cache_dir="./doc_index")
        >>> index = DocSearchIndex(config)
        >>> index.crawl_and_index()
        >>>
        >>> # Create tool
        >>> doc_search = create_doc_search_tool(index, description="Search Example.com documentation")
        >>> tools = [get_current_datetime, calculate, doc_search]
    """
    from .rag import DocSearchIndex

    if not isinstance(index, DocSearchIndex):
        raise TypeError(f"index must be a DocSearchIndex instance, got {type(index).__name__}")

    default_description = (
        "Search local documentation for answers to technical questions. "
        "Use this tool to find information from indexed documentation before searching the web. "
        "Returns relevant text excerpts with source URLs."
    )

    # Get parent context max chars from config (default 500)
    parent_max_chars = index.config.parent_context_max_chars

    def _doc_search_wrapper(query: str, top_k: int = 5) -> str:
        """Search the document index and format results."""
        results = index.search(query, top_k=top_k, return_parent=True)

        if not results:
            return "No relevant documentation found for your query."

        # Format results for the LLM
        formatted = []
        for i, result in enumerate(results, 1):
            entry = f"**Result {i}**\n"
            entry += f"Source: {result['url']}\n"
            if result.get("heading_path"):
                entry += f"Section: {result['heading_path']}\n"
            entry += f"\n{result['text']}\n"

            # Include parent context if available and different from child
            if result.get("parent_text") and result["parent_text"] != result["text"]:
                parent_text = result["parent_text"]
                if parent_max_chars > 0 and len(parent_text) > parent_max_chars:
                    parent_text = parent_text[:parent_max_chars] + "..."
                entry += f"\n---\nBroader context:\n{parent_text}\n"

            formatted.append(entry)

        return "\n\n".join(formatted)

    return Tool(
        name=name,
        description=description or default_description,
        func=_doc_search_wrapper,
        args_schema=DocSearchInput,
    )


# Collection of all built-in tools
BUILTIN_TOOLS = [
    get_current_datetime,
    calculate,
]


__all__ = [
    "BUILTIN_TOOLS",
    "DocSearchInput",
    "WebSearchInput",
    "calculate",
    "create_doc_search_tool",
    "create_web_search_tool",
    "get_current_datetime",
]
