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


@tool
def get_current_date() -> str:
    """Get the current date in YYYY-MM-DD format using the local timezone.

    Returns:
        Current date as a string in local timezone (e.g., "2025-11-21")
    """
    return datetime.now().astimezone().strftime("%Y-%m-%d")


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
    max_results: int = Field(default=10, description="Maximum number of results to return. Default is 10.")
    site: str = Field(default="", description="Optional site restriction (e.g., 'hashicorp.com')")


def create_web_search_tool(config: "ServerConfig") -> Tool:
    """Create a web search tool configured with the given ServerConfig.

    This tool requires the optional 'websearch' dependency.
    Install with: uv sync --extra websearch

    The tool will try Ollama web search API first (if OLLAMA_API_KEY is configured),
    then fall back to DuckDuckGo search.

    Args:
        config: ServerConfig instance with OLLAMA_API_KEY (optional)

    Returns:
        LangChain Tool for web search

    Example:
        >>> from llm_api_server import ServerConfig, create_web_search_tool
        >>> config = ServerConfig.from_env()
        >>> web_search = create_web_search_tool(config)
        >>> tools = [get_current_date, calculate, web_search]
    """
    from .web_search_tool import web_search

    def _web_search_wrapper(query: str, max_results: int = 10, site: str = "") -> str:
        """Wrapper that provides API key from config."""
        return web_search(query, max_results, site, ollama_api_key=config.OLLAMA_API_KEY)

    return Tool(
        name="web_search",
        description="Search the web for general information using Ollama API (if configured) or DuckDuckGo. Use this for finding current information, documentation, tutorials, Stack Overflow answers, or any online resources. Returns titles, URLs, and descriptions of relevant pages.",
        func=_web_search_wrapper,
        args_schema=WebSearchInput,
    )


# Collection of all built-in tools
BUILTIN_TOOLS = [
    get_current_date,
    calculate,
]


__all__ = [
    "BUILTIN_TOOLS",
    "WebSearchInput",
    "calculate",
    "create_web_search_tool",
    "get_current_date",
]
