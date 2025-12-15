"""Evaluator for running LLM test cases."""

import json
import time
from typing import Any

import requests

from .test_case import TestCase, TestResult
from .validators import validate_response


class Evaluator:
    """Evaluator for running test cases against an LLM API endpoint.

    Args:
        api_url: Base URL of the LLM API (e.g., "http://localhost:8000")
        model: Model name to use in requests (default: "default")
        stream: Whether to use streaming responses (default: False)
        extra_params: Additional parameters to include in API requests
    """

    def __init__(
        self, api_url: str, model: str = "default", stream: bool = False, extra_params: dict[str, Any] | None = None
    ):
        self.api_url = api_url.rstrip("/")
        self.model = model
        self.stream = stream
        self.extra_params = extra_params or {}

    def check_health(self) -> bool:
        """Check if the LLM API is running and healthy.

        Returns:
            True if API is healthy, False otherwise
        """
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def send_question(self, question: str, timeout: int = 120) -> tuple[str | None, float, str | None, list[str]]:
        """Send a question to the LLM API.

        Args:
            question: The question to ask
            timeout: Request timeout in seconds

        Returns:
            Tuple of (response_text, response_time_seconds, error_message, tools_used)
        """
        try:
            start_time = time.time()

            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": question}],
                "stream": self.stream,
                **self.extra_params,
            }

            response = requests.post(
                f"{self.api_url}/v1/chat/completions",
                json=payload,
                timeout=timeout,
                headers={"Content-Type": "application/json"},
            )

            elapsed = time.time() - start_time

            if response.status_code != 200:
                error_text = response.text[:500] if len(response.text) > 500 else response.text
                return None, elapsed, f"HTTP {response.status_code}: {error_text}", []

            data = response.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            tools_used = data.get("tools_used", [])

            if not content:
                return None, elapsed, "Empty response from API", tools_used

            return content, elapsed, None, tools_used

        except requests.Timeout:
            return None, timeout, "Request timeout", []
        except requests.RequestException as e:
            return None, 0, f"Request error: {e!s}", []
        except (json.JSONDecodeError, KeyError) as e:
            return None, 0, f"Response parsing error: {e!s}", []

    def run_test(self, test_case: TestCase) -> TestResult:
        """Run a single test case.

        Args:
            test_case: The test case to run

        Returns:
            TestResult with pass/fail status and details
        """
        # Send question
        response, response_time, error, tools_used = self.send_question(test_case.question, test_case.timeout)

        # Handle errors
        if error:
            return TestResult(
                test_case=test_case,
                passed=False,
                response=response,
                response_time=response_time,
                error=error,
                tools_used=tools_used,
            )

        # Validate response
        passed, issues = validate_response(test_case, response)

        return TestResult(
            test_case=test_case,
            passed=passed,
            response=response,
            response_time=response_time,
            issues=issues,
            tools_used=tools_used,
        )

    def run_tests(self, test_cases: list[TestCase], stop_on_failure: bool = False) -> list[TestResult]:
        """Run multiple test cases.

        Args:
            test_cases: List of test cases to run
            stop_on_failure: If True, stop running tests after first failure

        Returns:
            List of TestResult objects
        """
        results = []

        for test_case in test_cases:
            result = self.run_test(test_case)
            results.append(result)

            if stop_on_failure and not result.passed:
                break

        return results

    def get_summary(self, results: list[TestResult]) -> dict[str, Any]:
        """Generate summary statistics from test results.

        Args:
            results: List of test results

        Returns:
            Dictionary with summary statistics
        """
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = total - passed

        total_time = sum(r.response_time for r in results)
        avg_time = total_time / total if total > 0 else 0

        success_rate = (passed / total * 100) if total > 0 else 0

        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "success_rate": success_rate,
            "total_time": total_time,
            "avg_time": avg_time,
        }
