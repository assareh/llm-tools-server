"""Validation functions for test case responses."""

from .test_case import TestCase


def validate_response(test_case: TestCase, response: str) -> tuple[bool, list[str]]:
    """Validate a response against test case criteria.

    Args:
        test_case: The TestCase to validate against
        response: The response text to validate

    Returns:
        Tuple of (passed, list_of_issues)
    """
    issues = []
    response_lower = response.lower()

    # Check minimum length
    if len(response) < test_case.min_response_length:
        issues.append(f"Response too short ({len(response)} chars, expected >={test_case.min_response_length})")

    # Check maximum length
    if test_case.max_response_length and len(response) > test_case.max_response_length:
        issues.append(f"Response too long ({len(response)} chars, expected <={test_case.max_response_length})")

    # Check for expected keywords
    missing_keywords = []
    for keyword in test_case.expected_keywords:
        if keyword not in response_lower:
            missing_keywords.append(keyword)

    if missing_keywords:
        issues.append(f"Missing expected keywords: {', '.join(missing_keywords)}")

    # Check for unexpected keywords
    found_unexpected = []
    for keyword in test_case.unexpected_keywords:
        if keyword in response_lower:
            found_unexpected.append(keyword)

    if found_unexpected:
        issues.append(f"Found unexpected keywords: {', '.join(found_unexpected)}")

    # Run custom validator if provided
    if test_case.custom_validator:
        try:
            custom_passed, custom_issues = test_case.custom_validator(response)
            if not custom_passed:
                issues.extend(custom_issues)
        except Exception as e:
            issues.append(f"Custom validator error: {e!s}")

    return len(issues) == 0, issues
