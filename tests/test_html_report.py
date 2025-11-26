#!/usr/bin/env python3
"""Test script for HTML report generation with formatted responses."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from llm_api_server.eval import HTMLReporter, TestCase, TestResult


def create_test_results():
    """Create sample test results with various response types."""
    results = []

    # Test 1: Short response (no expand button)
    test1 = TestCase(
        question="What is 2 + 2?",
        description="Simple calculation",
    )
    result1 = TestResult(
        test_case=test1,
        passed=True,
        response="The answer is 4.",
        response_time=0.5,
    )
    results.append(result1)

    # Test 2: Long markdown response with code
    test2 = TestCase(
        question="How do I create a function in Python?",
        description="Explain Python functions",
    )
    result2 = TestResult(
        test_case=test2,
        passed=True,
        response="""# Python Functions

To create a function in Python, you use the `def` keyword followed by the function name and parentheses.

## Basic Syntax

```python
def greet(name):
    \"\"\"Greet a person by name.\"\"\"
    return f"Hello, {name}!"

# Call the function
message = greet("Alice")
print(message)  # Output: Hello, Alice!
```

## Key Points

1. **Function Definition**: Use `def` keyword
2. **Parameters**: Define inside parentheses
3. **Docstring**: Optional documentation string
4. **Return Value**: Use `return` to send back a value

## Advanced Example

You can also have default parameters:

```python
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"

print(greet("Bob"))              # Output: Hello, Bob!
print(greet("Bob", "Hi"))        # Output: Hi, Bob!
```

> **Note**: Functions are first-class objects in Python, meaning they can be passed as arguments to other functions.

## Multiple Return Values

Python functions can return multiple values using tuples:

```python
def get_coordinates():
    x = 10
    y = 20
    return x, y

x_pos, y_pos = get_coordinates()
```

This makes Python functions very flexible and powerful!
""",
        response_time=1.2,
    )
    results.append(result2)

    # Test 3: Failed test with issues
    test3 = TestCase(
        question="Does the API return JSON?",
        description="Check API response format",
    )
    result3 = TestResult(
        test_case=test3,
        passed=False,
        response="The API returns XML instead of JSON.",
        response_time=0.8,
        issues=["Response format is XML, expected JSON", "Missing Content-Type header"],
    )
    results.append(result3)

    # Test 4: Error test
    test4 = TestCase(
        question="What happens on timeout?",
        description="Test timeout handling",
    )
    result4 = TestResult(
        test_case=test4,
        passed=False,
        response=None,
        response_time=30.0,
        error="Request timeout after 30 seconds",
    )
    results.append(result4)

    # Test 5: Response with tables and lists
    test5 = TestCase(
        question="What are the differences between lists and tuples?",
        description="Compare data structures",
    )
    result5 = TestResult(
        test_case=test5,
        passed=True,
        response="""# Lists vs Tuples in Python

## Overview

Both lists and tuples are sequence types in Python, but they have important differences:

| Feature | List | Tuple |
|---------|------|-------|
| Mutability | Mutable (can be changed) | Immutable (cannot be changed) |
| Syntax | `[1, 2, 3]` | `(1, 2, 3)` |
| Performance | Slower | Faster |
| Use Case | Dynamic data | Fixed data |

## When to Use Lists

Use lists when:
- You need to add, remove, or modify elements
- The data is subject to change
- You need methods like `append()`, `remove()`, etc.

Example:
```python
fruits = ['apple', 'banana']
fruits.append('orange')  # OK
fruits[0] = 'grape'      # OK
```

## When to Use Tuples

Use tuples when:
- The data should not change
- You want to use it as a dictionary key
- You need slightly better performance

Example:
```python
coordinates = (10, 20)
# coordinates[0] = 15  # ERROR: tuples are immutable
```

## Memory Efficiency

Tuples are more memory-efficient than lists because:
1. They don't need extra space for growth
2. They can be optimized by the interpreter
3. They don't have methods for modification

This makes tuples a good choice for large datasets that don't need to change!
""",
        response_time=1.5,
    )
    results.append(result5)

    # Test 6: Response with blockquotes
    test6 = TestCase(
        question="How should I handle errors in Python?",
        description="Best practices for error handling",
    )
    result6 = TestResult(
        test_case=test6,
        passed=True,
        response=(
            "# Error Handling Best Practices\n\n"
            "## Use Specific Exceptions\n\n"
            "Always catch specific exceptions rather than broad `Exception`:\n\n"
            "```python\n"
            "try:\n"
            "    result = risky_operation()\n"
            "except ValueError as e:\n"
            "    print(f'Invalid value: {e}')\n"
            "except FileNotFoundError as e:\n"
            "    print(f'File not found: {e}')\n"
            "```\n\n"
            "> **Important**: Catching broad exceptions can hide bugs and make debugging difficult.\n\n"
            "## Clean Up with Finally\n\n"
            "Use `finally` for cleanup operations:\n\n"
            "```python\n"
            "file = open('data.txt')\n"
            "try:\n"
            "    content = file.read()\n"
            "    process(content)\n"
            "finally:\n"
            "    file.close()  # Always executes\n"
            "```\n\n"
            "Or better yet, use context managers:\n\n"
            "```python\n"
            "with open('data.txt') as file:\n"
            "    content = file.read()\n"
            "    process(content)\n"
            "# File automatically closed\n"
            "```\n\n"
            "> **Pro Tip**: Context managers are the Pythonic way to handle resources!\n\n"
            "## Custom Exceptions\n\n"
            "Create custom exceptions for your application:\n\n"
            "```python\n"
            "class ValidationError(Exception):\n"
            "    '''Raised when data validation fails.'''\n"
            "    pass\n\n"
            "def validate_age(age):\n"
            "    if age < 0:\n"
            '        raise ValidationError("Age cannot be negative")\n'
            "    return True\n"
            "```\n\n"
            "This makes your code more maintainable and easier to understand!"
        ),
        response_time=1.8,
    )
    results.append(result6)

    return results


def main():
    """Generate test HTML report."""
    print("\n" + "=" * 60)
    print("Generating HTML Report with Formatted Responses")
    print("=" * 60 + "\n")

    # Create test results
    results = create_test_results()
    print(f"Created {len(results)} test results")

    # Generate HTML report
    reporter = HTMLReporter()
    output_path = Path("test_report.html")

    reporter.generate(
        results=results,
        output_path=output_path,
        title="LLM Evaluation - Formatted Response Test",
    )

    print(f"\nHTML report generated: {output_path.absolute()}")
    print("\nFeatures demonstrated:")
    print("  ✓ Full responses (no truncation)")
    print("  ✓ Markdown formatting (headings, code, tables, lists)")
    print("  ✓ Collapsible long responses (> 300 chars)")
    print("  ✓ Syntax highlighting for code blocks")
    print("  ✓ Proper table rendering")
    print("  ✓ Blockquote styling")
    print("\nOpen the file in your browser to see the results!")
    print(f"\nCommand: open {output_path.absolute()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
