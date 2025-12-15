#!/usr/bin/env python
"""Example script demonstrating the LLM evaluation framework.

This script shows how to:
1. Define test cases with validation criteria
2. Run evaluations against an LLM API endpoint
3. Generate HTML and JSON reports

Usage:
    # Start your LLM API server first
    python myapp.py  # or any server using llm-tools-server

    # Then run this evaluation
    python example_evaluation.py
"""

from llm_tools_server.eval import ConsoleReporter, Evaluator, HTMLReporter, JSONReporter, TestCase


def create_test_cases() -> list[TestCase]:
    """Create a set of example test cases.

    Returns:
        List of TestCase objects
    """
    return [
        TestCase(
            question="What is 2+2?",
            description="Basic arithmetic test",
            expected_keywords=["4", "four"],
            min_response_length=5,
        ),
        TestCase(
            question="Explain photosynthesis in one sentence",
            description="Biology knowledge test",
            expected_keywords=["plant", "light"],
            min_response_length=20,
            max_response_length=200,
        ),
        TestCase(
            question="What is the capital of France?",
            description="Geography test",
            expected_keywords=["paris"],
            unexpected_keywords=["london", "berlin"],
            min_response_length=5,
        ),
        TestCase(
            question="Write a Python function to check if a number is even",
            description="Code generation test",
            expected_keywords=["def", "%", "2", "return"],
            min_response_length=50,
        ),
        TestCase(
            question="What are the benefits of exercise?",
            description="Health knowledge test",
            expected_keywords=["health"],
            min_response_length=50,
        ),
    ]


def custom_validator_example(response: str) -> tuple[bool, list[str]]:
    """Example custom validator function.

    Args:
        response: The LLM response to validate

    Returns:
        Tuple of (passed, issues)
    """
    issues = []

    # Check if response is polite
    if not any(word in response.lower() for word in ["please", "thank", "welcome"]):
        issues.append("Response lacks polite language")

    # Check for complete sentences
    if not response.strip().endswith((".", "!", "?")):
        issues.append("Response does not end with proper punctuation")

    return len(issues) == 0, issues


def create_advanced_test_cases() -> list[TestCase]:
    """Create advanced test cases with custom validators and metadata.

    Returns:
        List of advanced TestCase objects
    """
    return [
        TestCase(
            question="How are you today?",
            description="Greeting with custom validation",
            expected_keywords=["good", "well", "fine", "great"],
            custom_validator=custom_validator_example,
            metadata={"category": "conversational", "difficulty": "easy"},
        ),
        TestCase(
            question="Explain quantum computing",
            description="Complex technical topic",
            expected_keywords=["qubit", "quantum"],
            min_response_length=100,
            metadata={"category": "technical", "difficulty": "hard"},
        ),
    ]


def main():
    """Main evaluation script."""
    print("=" * 80)
    print("LLM API Server - Evaluation Framework Example")
    print("=" * 80)
    print()

    # Configuration
    API_URL = "http://localhost:8000"
    MODEL_NAME = "default"

    # Create evaluator
    print(f"Initializing evaluator for {API_URL}...")
    evaluator = Evaluator(api_url=API_URL, model=MODEL_NAME)

    # Check API health
    print("Checking API health...")
    if not evaluator.check_health():
        print("❌ ERROR: LLM API is not running or not healthy")
        print(f"\nPlease start your LLM API server at {API_URL}")
        print("Example: python myapp.py")
        return 1

    print("✅ API is healthy\n")

    # Create test cases
    print("Creating test cases...")
    test_cases = create_test_cases()
    print(f"Created {len(test_cases)} basic test cases\n")

    # Run evaluation
    print("Running evaluation...")
    print("-" * 80)
    results = evaluator.run_tests(test_cases)
    print("-" * 80)
    print()

    # Generate console report
    print("Generating console report...")
    console_reporter = ConsoleReporter()
    console_reporter.generate(results, verbose=False)

    # Generate HTML report
    html_output = "evaluation_report.html"
    print(f"Generating HTML report: {html_output}")
    html_reporter = HTMLReporter()
    html_reporter.generate(results, html_output, title="LLM Evaluation - Basic Tests")
    print(f"✅ HTML report saved to: {html_output}")

    # Generate JSON report
    json_output = "evaluation_report.json"
    print(f"Generating JSON report: {json_output}")
    json_reporter = JSONReporter()
    json_reporter.generate(results, json_output)
    print(f"✅ JSON report saved to: {json_output}")

    # Summary
    summary = evaluator.get_summary(results)
    print()
    print("=" * 80)
    print("Final Summary")
    print("=" * 80)
    print(f"Success Rate: {summary['success_rate']:.1f}%")
    print(f"Total Tests:  {summary['total']}")
    print(f"Passed:       {summary['passed']}")
    print(f"Failed:       {summary['failed']}")
    print(f"Total Time:   {summary['total_time']:.2f}s")
    print(f"Avg Time:     {summary['avg_time']:.2f}s")
    print()

    # Advanced example
    print("=" * 80)
    print("Running Advanced Tests with Custom Validators")
    print("=" * 80)
    print()

    advanced_cases = create_advanced_test_cases()
    advanced_results = evaluator.run_tests(advanced_cases)

    # Generate advanced HTML report
    advanced_html = "advanced_evaluation_report.html"
    html_reporter.generate(advanced_results, advanced_html, title="LLM Evaluation - Advanced Tests")
    print(f"✅ Advanced HTML report saved to: {advanced_html}")
    print()

    # Return exit code
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
