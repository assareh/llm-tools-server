"""Test case definitions for LLM evaluation framework."""

from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class TestCase:
    """Represents a single test case for LLM evaluation.

    Attributes:
        question: The question/prompt to send to the LLM
        description: Human-readable description of what this test validates
        expected_keywords: List of keywords that should appear in response (case-insensitive)
        unexpected_keywords: List of keywords that should NOT appear in response
        min_response_length: Minimum expected response length in characters
        max_response_length: Maximum expected response length in characters (None for unlimited)
        timeout: Maximum time in seconds to wait for response
        custom_validator: Optional custom validation function that takes response string and returns (passed, issues)
        metadata: Optional metadata dict for custom tracking/reporting
    """

    question: str
    description: str
    expected_keywords: list[str] = field(default_factory=list)
    unexpected_keywords: list[str] = field(default_factory=list)
    min_response_length: int = 10
    max_response_length: int | None = None
    timeout: int = 120
    custom_validator: Callable[[str], tuple[bool, list[str]]] | None = None
    metadata: dict | None = None

    def __post_init__(self):
        """Normalize keywords to lowercase for case-insensitive matching."""
        self.expected_keywords = [k.lower() for k in self.expected_keywords]
        self.unexpected_keywords = [k.lower() for k in self.unexpected_keywords]


@dataclass
class TestResult:
    """Result of running a single test case.

    Attributes:
        test_case: The original test case
        passed: Whether the test passed all validation
        response: The LLM's response text
        response_time: Time taken to get response in seconds
        issues: List of validation issues found
        error: Error message if request failed
        tools_used: List of tool names that were called during the response
    """

    test_case: TestCase
    passed: bool
    response: str | None = None
    response_time: float = 0.0
    issues: list[str] = field(default_factory=list)
    error: str | None = None
    tools_used: list[str] = field(default_factory=list)
