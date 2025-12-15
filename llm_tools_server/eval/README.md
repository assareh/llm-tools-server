# LLM Evaluation Framework

A comprehensive evaluation framework for testing LLM API endpoints with flexible validation and reporting capabilities.

## Features

- **Flexible Test Cases**: Define tests with expected keywords, response length constraints, and custom validators
- **Multiple Reporters**: Generate HTML, JSON, or console reports
- **Performance Metrics**: Track response times and success rates
- **Custom Validation**: Support for custom validation functions
- **Metadata Support**: Attach custom metadata to test cases for categorization

## Quick Start

```python
from llm_tools_server.eval import Evaluator, TestCase, HTMLReporter

# Define test cases
tests = [
    TestCase(
        question="What is 2+2?",
        description="Basic arithmetic test",
        expected_keywords=["4", "four"],
        min_response_length=10
    ),
    TestCase(
        question="Explain photosynthesis",
        description="Biology knowledge test",
        expected_keywords=["plants", "light", "oxygen"],
        min_response_length=100
    )
]

# Run evaluation
evaluator = Evaluator(api_url="http://localhost:8000")

# Check API health first
if not evaluator.check_health():
    print("API is not running!")
    exit(1)

# Run tests
results = evaluator.run_tests(tests)

# Generate HTML report
reporter = HTMLReporter()
reporter.generate(results, "evaluation_report.html")

# Get summary statistics
summary = evaluator.get_summary(results)
print(f"Success Rate: {summary['success_rate']:.1f}%")
```

## API Reference

### TestCase

Define a test case with validation criteria.

```python
TestCase(
    question="Your question here",
    description="What this tests",
    expected_keywords=["keyword1", "keyword2"],  # Optional
    unexpected_keywords=["bad_keyword"],          # Optional
    min_response_length=10,                       # Optional, default: 10
    max_response_length=500,                      # Optional, default: None
    timeout=120,                                  # Optional, default: 120s
    custom_validator=my_validator_function,       # Optional
    metadata={"category": "math"}                 # Optional
)
```

**Parameters:**

- `question` (str): The question/prompt to send to the LLM
- `description` (str): Human-readable description of the test
- `expected_keywords` (list[str]): Keywords that should appear (case-insensitive)
- `unexpected_keywords` (list[str]): Keywords that should NOT appear
- `min_response_length` (int): Minimum response length in characters
- `max_response_length` (int | None): Maximum response length (None for unlimited)
- `timeout` (int): Request timeout in seconds
- `custom_validator` (callable): Custom validation function
- `metadata` (dict): Custom metadata for reporting

### Evaluator

Run test cases against an LLM API endpoint.

```python
evaluator = Evaluator(
    api_url="http://localhost:8000",
    model="default",
    stream=False,
    extra_params={"temperature": 0.7}
)
```

**Parameters:**

- `api_url` (str): Base URL of the LLM API
- `model` (str): Model name to use in requests
- `stream` (bool): Whether to use streaming responses
- `extra_params` (dict): Additional parameters for API requests

**Methods:**

- `check_health() -> bool`: Check if API is healthy
- `send_question(question, timeout) -> (response, time, error)`: Send single question
- `run_test(test_case) -> TestResult`: Run single test
- `run_tests(test_cases, stop_on_failure=False) -> list[TestResult]`: Run multiple tests
- `get_summary(results) -> dict`: Get summary statistics

### HTMLReporter

Generate HTML reports with visual test results.

```python
reporter = HTMLReporter()
reporter.generate(
    results,
    output_path="report.html",
    title="My Evaluation Report"
)
```

Features:
- Color-coded pass/fail results
- Response times
- Failed test details
- Summary statistics
- Responsive design

### JSONReporter

Generate structured JSON reports.

```python
reporter = JSONReporter()
reporter.generate(results, "report.json")
```

Output format:
```json
{
  "generated_at": "2025-11-22T12:00:00",
  "summary": {
    "total": 10,
    "passed": 8,
    "failed": 2,
    "success_rate": 80.0,
    "total_time": 45.2,
    "avg_time": 4.52
  },
  "results": [...]
}
```

### ConsoleReporter

Print results to console with colors.

```python
reporter = ConsoleReporter()
reporter.generate(results, verbose=True)
```

## Custom Validators

Create custom validation logic:

```python
def my_validator(response: str) -> tuple[bool, list[str]]:
    """Custom validator function.

    Args:
        response: The LLM response to validate

    Returns:
        Tuple of (passed, list_of_issues)
    """
    issues = []

    # Check for code blocks
    if "```" not in response:
        issues.append("Response should contain code blocks")

    # Check response length
    if len(response.split()) < 50:
        issues.append("Response should contain at least 50 words")

    return len(issues) == 0, issues

# Use in test case
test = TestCase(
    question="Write a Python function to sort a list",
    description="Code generation with custom validation",
    custom_validator=my_validator
)
```

## Example Use Cases

### Testing a Q&A System

```python
from llm_tools_server.eval import Evaluator, TestCase, HTMLReporter

# Define domain-specific test cases
tests = [
    TestCase(
        question="What are the top travel credit cards?",
        description="Travel card recommendation",
        expected_keywords=["chase", "sapphire", "amex"],
        min_response_length=100
    ),
    TestCase(
        question="How do I transfer points to airlines?",
        description="Transfer partner knowledge",
        expected_keywords=["transfer", "partner", "ratio"],
        min_response_length=150
    )
]

evaluator = Evaluator(api_url="http://localhost:8000", model="mymodel")
results = evaluator.run_tests(tests)

reporter = HTMLReporter()
reporter.generate(results, "qa_evaluation.html", title="Q&A System Evaluation")
```

### Regression Testing

```python
# Define regression test suite
regression_tests = [
    TestCase(
        question="Calculate 15% of 200",
        description="Math calculation regression",
        expected_keywords=["30"],
        min_response_length=5
    ),
    # ... more tests
]

# Run with stop_on_failure for CI/CD
evaluator = Evaluator(api_url="http://localhost:8000")
results = evaluator.run_tests(regression_tests, stop_on_failure=True)

# Generate JSON for CI integration
from llm_tools_server.eval import JSONReporter
reporter = JSONReporter()
reporter.generate(results, "regression_results.json")

# Exit with appropriate code
import sys
summary = evaluator.get_summary(results)
sys.exit(0 if summary['failed'] == 0 else 1)
```

### Performance Benchmarking

```python
import time
from llm_tools_server.eval import Evaluator, TestCase

# Create test cases with different complexities
tests = [
    TestCase(
        question="What is 2+2?",
        description="Simple query",
        metadata={"complexity": "simple"}
    ),
    TestCase(
        question="Explain quantum entanglement in detail",
        description="Complex query",
        metadata={"complexity": "complex"}
    )
]

evaluator = Evaluator(api_url="http://localhost:8000")
results = evaluator.run_tests(tests)

# Analyze performance by complexity
simple_times = [r.response_time for r in results if r.test_case.metadata.get("complexity") == "simple"]
complex_times = [r.response_time for r in results if r.test_case.metadata.get("complexity") == "complex"]

print(f"Simple queries avg: {sum(simple_times)/len(simple_times):.2f}s")
print(f"Complex queries avg: {sum(complex_times)/len(complex_times):.2f}s")
```

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: LLM Evaluation

on: [push, pull_request]

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -e .

      - name: Start LLM server
        run: |
          python server.py &
          sleep 10

      - name: Run evaluation
        run: |
          python evaluation_tests.py

      - name: Upload reports
        uses: actions/upload-artifact@v2
        with:
          name: evaluation-reports
          path: |
            evaluation_report.html
            evaluation_report.json
```

## Best Practices

1. **Start Simple**: Begin with basic keyword and length validation
2. **Add Custom Validators**: Create domain-specific validation logic
3. **Use Metadata**: Tag tests for categorization and filtering
4. **Check Health First**: Always verify API is running before tests
5. **Generate Multiple Reports**: HTML for humans, JSON for automation
6. **Track Over Time**: Store JSON reports to track performance trends
7. **Use Appropriate Timeouts**: Set realistic timeouts based on query complexity

## Troubleshooting

### API Connection Issues

```python
evaluator = Evaluator(api_url="http://localhost:8000")
if not evaluator.check_health():
    print("API is not running. Please start it first.")
    exit(1)
```

### Flaky Tests

If tests pass inconsistently:
- Check if model temperature is set to 0 for deterministic results
- Adjust validation criteria to be more flexible
- Use custom validators for nuanced validation

### Slow Tests

- Reduce timeout values for faster failure
- Use `stop_on_failure=True` to halt on first error
- Run tests in parallel (future feature)

## Future Enhancements

Planned features:
- Parallel test execution
- Streaming response support
- Tool calling validation
- Comparative evaluations (A/B testing)
- Test case templates
- Advanced metrics (BLEU, ROUGE scores)

## License

Same as llm-tools-server parent project.
