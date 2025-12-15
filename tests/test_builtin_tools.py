"""Unit tests for built-in tools."""

import pytest

from llm_tools_server.builtin_tools import calculate


@pytest.mark.unit
def test_calculate_rejects_non_numeric_constants():
    """Calculator should block strings and other non-numeric literals."""
    result = calculate("'hello'")

    assert "only numeric constants" in result.lower()


@pytest.mark.unit
def test_calculate_rejects_none_literal():
    """Calculator should not evaluate None."""
    result = calculate("None")

    assert "only numeric constants" in result.lower()


@pytest.mark.unit
def test_calculate_rejects_boolean_literal():
    """Calculator should not treat booleans as numeric."""
    result = calculate("True")

    assert "only numeric constants" in result.lower()


@pytest.mark.unit
def test_calculate_handles_division_by_zero():
    """Gracefully handle division by zero."""
    result = calculate("1/0")

    assert "division by zero" in result.lower()
