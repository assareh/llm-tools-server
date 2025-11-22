"""Built-in tools for LLM API Server.

These tools provide common functionality that can be used across different applications.
Users can import individual tools or use the BUILTIN_TOOLS collection.
"""

import ast
import operator
from datetime import datetime

from langchain_core.tools import tool


@tool
def get_current_date() -> str:
    """Get the current date in YYYY-MM-DD format.

    Returns:
        Current date as a string (e.g., "2025-11-21")
    """
    return datetime.now().strftime("%Y-%m-%d")


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


# Collection of all built-in tools
BUILTIN_TOOLS = [
    get_current_date,
    calculate,
]


__all__ = [
    "BUILTIN_TOOLS",
    "calculate",
    "get_current_date",
]
