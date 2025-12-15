"""Automated tests for HTML report generation."""

from pathlib import Path

import pytest

from llm_tools_server.eval import HTMLReporter, TestCase, TestResult, reporters


@pytest.fixture
def sample_results():
    """Minimal set of results to exercise rendering paths."""
    short = TestResult(
        test_case=TestCase(question="Short?", description="Short response"),
        passed=True,
        response="Short answer",
        response_time=0.5,
        tools_used=["search_tool"],
    )

    long_text = "Long content " * 40  # > 300 chars triggers collapsible block
    long = TestResult(
        test_case=TestCase(question="Long?", description="Long response"),
        passed=False,
        response=long_text,
        response_time=1.2,
        issues=["Response too long"],
        tools_used=[],
    )

    markdown_result = TestResult(
        test_case=TestCase(question="Markdown?", description="Markdown rendering"),
        passed=True,
        response="## Heading\n\n`code` block",
        response_time=0.3,
        tools_used=["doc_search"],
    )

    return [short, long, markdown_result]


@pytest.mark.unit
def test_html_report_generates_collapsible_sections(tmp_path: Path, sample_results):
    """Ensure the report writes to disk and long responses are collapsible."""
    output_path = tmp_path / "report.html"
    reporter = HTMLReporter()

    reporter.generate(results=sample_results, output_path=output_path, title="Demo Report")

    html = output_path.read_text(encoding="utf-8")
    assert "Demo Report" in html
    assert "Long content" in html
    assert "collapsed" in html  # long response div should be collapsed
    assert "toggleResponse" in html  # toggle JS hook present
    assert "Response too long" in html  # issues list rendered
    assert "search_tool" in html  # tool badge rendered


@pytest.mark.unit
def test_html_report_renders_markdown_when_available(tmp_path: Path):
    """Verify markdown is converted when the markdown dependency is present."""
    markdown_result = TestResult(
        test_case=TestCase(question="Markdown?", description="Markdown rendering"),
        passed=True,
        response="## Heading\n\n`code` block",
        response_time=0.3,
    )
    output_path = tmp_path / "report_markdown.html"
    reporter = HTMLReporter()

    reporter.generate(results=[markdown_result], output_path=output_path, title="Markdown Report")

    html = output_path.read_text(encoding="utf-8")
    if reporters.HAS_MARKDOWN:
        assert "<h2>Heading</h2>" in html
        assert "<code>code</code>" in html
    else:
        # Fallback preserves raw markdown text
        assert "## Heading" in html
        assert "`code` block" in html
