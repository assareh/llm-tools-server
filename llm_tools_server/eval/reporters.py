"""Report generators for evaluation results."""

import html
import json
from datetime import datetime
from pathlib import Path

from .test_case import TestResult

# Optional markdown support for HTML reports
try:
    import markdown

    HAS_MARKDOWN = True
except ImportError:
    HAS_MARKDOWN = False


class HTMLReporter:
    """Generate HTML reports from evaluation results."""

    def generate(self, results: list[TestResult], output_path: str | Path, title: str = "LLM Evaluation Report"):
        """Generate an HTML report from test results.

        Args:
            results: List of test results
            output_path: Path to write HTML report
            title: Report title
        """
        output_path = Path(output_path)

        # Calculate summary stats
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = total - passed
        success_rate = (passed / total * 100) if total > 0 else 0
        total_time = sum(r.response_time for r in results)
        avg_time = total_time / total if total > 0 else 0

        # Generate HTML
        html_content = self._generate_html(
            results=results,
            title=title,
            total=total,
            passed=passed,
            failed=failed,
            success_rate=success_rate,
            total_time=total_time,
            avg_time=avg_time,
        )

        # Write to file
        output_path.write_text(html_content, encoding="utf-8")

    def _generate_html(
        self,
        results: list[TestResult],
        title: str,
        total: int,
        passed: int,
        failed: int,
        success_rate: float,
        total_time: float,
        avg_time: float,
    ) -> str:
        """Generate HTML content for report."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Generate test result rows
        test_rows = []
        for i, result in enumerate(results, 1):
            status_class = "passed" if result.passed else "failed"
            status_icon = "✓" if result.passed else "✗"

            # Build issues/error display
            issues_html = ""
            if result.error:
                issues_html = f'<div class="error">Error: {html.escape(result.error)}</div>'
            elif result.issues:
                issues_list = "".join(f"<li>{html.escape(issue)}</li>" for issue in result.issues)
                issues_html = f'<div class="issues"><ul>{issues_list}</ul></div>'

            # Build tools used display
            if result.tools_used:
                tools_html = "".join(
                    f'<span class="tool-badge">{html.escape(tool)}</span>' for tool in result.tools_used
                )
            else:
                tools_html = '<span class="no-tools">None</span>'

            # Format response (convert markdown to HTML if available)
            if result.response:
                if HAS_MARKDOWN:
                    # Convert markdown to HTML
                    response_html = markdown.markdown(
                        result.response,
                        extensions=["fenced_code", "tables", "nl2br"],
                    )
                else:
                    # Fallback: escape HTML and preserve line breaks
                    response_html = html.escape(result.response).replace("\n", "<br>")

                # Create collapsible response (collapsed by default if > 300 chars)
                is_long = len(result.response) > 300
                collapsed_class = "collapsed" if is_long else ""
                toggle_btn = (
                    f'<button class="toggle-btn" onclick="toggleResponse(this)">{"Expand" if is_long else "Collapse"}</button>'
                    if is_long
                    else ""
                )

                response_display = f"""
                    {toggle_btn}
                    <div class="response-content {collapsed_class}">
                        {response_html}
                    </div>
                """
            else:
                response_display = '<div class="response-content">N/A</div>'

            row = f"""
            <tr class="{status_class}">
                <td>{i}</td>
                <td><span class="status-icon">{status_icon}</span></td>
                <td><strong>{html.escape(result.test_case.description)}</strong><br>
                    <span class="question">{html.escape(result.test_case.question)}</span>
                </td>
                <td>{result.response_time:.2f}s</td>
                <td><div class="tools-container">{tools_html}</div></td>
                <td>
                    <div class="response-container">{response_display}</div>
                    {issues_html}
                </td>
            </tr>
            """
            test_rows.append(row)

        test_rows_html = "\n".join(test_rows)

        # Complete HTML document
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(title)}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: #f5f5f5;
            padding: 20px;
            color: #333;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
        }}
        h1 {{
            font-size: 28px;
            margin-bottom: 10px;
        }}
        .timestamp {{
            opacity: 0.9;
            font-size: 14px;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            padding: 30px;
            background: #fafafa;
            border-bottom: 1px solid #eee;
        }}
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 6px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .stat-label {{
            font-size: 12px;
            text-transform: uppercase;
            color: #666;
            font-weight: 600;
            margin-bottom: 8px;
        }}
        .stat-value {{
            font-size: 32px;
            font-weight: bold;
            color: #333;
        }}
        .stat-value.success {{
            color: #10b981;
        }}
        .stat-value.danger {{
            color: #ef4444;
        }}
        .stat-value.info {{
            color: #3b82f6;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        thead {{
            background: #f9fafb;
            border-bottom: 2px solid #e5e7eb;
        }}
        th {{
            padding: 15px;
            text-align: left;
            font-weight: 600;
            font-size: 13px;
            text-transform: uppercase;
            color: #6b7280;
        }}
        td {{
            padding: 15px;
            border-bottom: 1px solid #f3f4f6;
            vertical-align: top;
        }}
        tr.passed {{
            background: #f0fdf4;
        }}
        tr.failed {{
            background: #fef2f2;
        }}
        tr:hover {{
            background: #f9fafb !important;
        }}
        .status-icon {{
            font-size: 18px;
            font-weight: bold;
        }}
        tr.passed .status-icon {{
            color: #10b981;
        }}
        tr.failed .status-icon {{
            color: #ef4444;
        }}
        .question {{
            font-size: 13px;
            color: #6b7280;
            font-style: italic;
        }}
        .response-container {{
            position: relative;
        }}
        .toggle-btn {{
            background: #3b82f6;
            color: white;
            border: none;
            padding: 6px 12px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
            font-weight: 600;
            margin-bottom: 8px;
            transition: background 0.2s;
        }}
        .toggle-btn:hover {{
            background: #2563eb;
        }}
        .response-content {{
            font-size: 13px;
            line-height: 1.6;
            color: #1f2937;
            background: #f9fafb;
            padding: 15px;
            border-radius: 4px;
            border: 1px solid #e5e7eb;
            overflow-x: auto;
            transition: max-height 0.3s ease;
        }}
        .response-content.collapsed {{
            max-height: 150px;
            overflow: hidden;
            position: relative;
        }}
        .response-content.collapsed::after {{
            content: "";
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            height: 60px;
            background: linear-gradient(to bottom, transparent, #f9fafb);
            pointer-events: none;
        }}
        /* Markdown styling */
        .response-content h1, .response-content h2, .response-content h3 {{
            margin-top: 16px;
            margin-bottom: 8px;
            font-weight: 600;
            color: #111827;
        }}
        .response-content h1 {{ font-size: 18px; }}
        .response-content h2 {{ font-size: 16px; }}
        .response-content h3 {{ font-size: 14px; }}
        .response-content p {{
            margin: 8px 0;
        }}
        .response-content code {{
            background: #1f2937;
            color: #10b981;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 12px;
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
        }}
        .response-content pre {{
            background: #1f2937;
            color: #e5e7eb;
            padding: 12px;
            border-radius: 4px;
            overflow-x: auto;
            margin: 12px 0;
        }}
        .response-content pre code {{
            background: none;
            color: inherit;
            padding: 0;
        }}
        .response-content ul, .response-content ol {{
            margin: 8px 0;
            padding-left: 24px;
        }}
        .response-content li {{
            margin: 4px 0;
        }}
        .response-content blockquote {{
            border-left: 3px solid #3b82f6;
            padding-left: 12px;
            margin: 12px 0;
            color: #6b7280;
            font-style: italic;
        }}
        .response-content table {{
            border-collapse: collapse;
            width: 100%;
            margin: 12px 0;
        }}
        .response-content table th,
        .response-content table td {{
            border: 1px solid #e5e7eb;
            padding: 8px;
            text-align: left;
        }}
        .response-content table th {{
            background: #f3f4f6;
            font-weight: 600;
        }}
        .issues, .error {{
            margin-top: 10px;
            padding: 10px;
            border-radius: 4px;
            font-size: 13px;
        }}
        .issues {{
            background: #fef3c7;
            border-left: 3px solid #f59e0b;
        }}
        .error {{
            background: #fee2e2;
            border-left: 3px solid #ef4444;
            color: #991b1b;
        }}
        .issues ul {{
            margin-left: 20px;
        }}
        .issues li {{
            margin: 5px 0;
            color: #92400e;
        }}
        .tools-container {{
            display: flex;
            flex-wrap: wrap;
            gap: 4px;
        }}
        .tool-badge {{
            display: inline-block;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
            white-space: nowrap;
        }}
        .no-tools {{
            color: #9ca3af;
            font-size: 12px;
            font-style: italic;
        }}
        footer {{
            padding: 20px 30px;
            background: #f9fafb;
            text-align: center;
            font-size: 13px;
            color: #6b7280;
            border-top: 1px solid #e5e7eb;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{html.escape(title)}</h1>
            <div class="timestamp">Generated on {timestamp}</div>
        </header>

        <div class="summary">
            <div class="stat-card">
                <div class="stat-label">Total Tests</div>
                <div class="stat-value info">{total}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Passed</div>
                <div class="stat-value success">{passed}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Failed</div>
                <div class="stat-value danger">{failed}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Success Rate</div>
                <div class="stat-value">{success_rate:.1f}%</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Total Time</div>
                <div class="stat-value">{total_time:.1f}s</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Avg Time</div>
                <div class="stat-value">{avg_time:.2f}s</div>
            </div>
        </div>

        <table>
            <thead>
                <tr>
                    <th style="width: 50px;">#</th>
                    <th style="width: 50px;">Status</th>
                    <th style="width: 25%;">Test</th>
                    <th style="width: 80px;">Time</th>
                    <th style="width: 150px;">Tools Used</th>
                    <th>Response & Issues</th>
                </tr>
            </thead>
            <tbody>
                {test_rows_html}
            </tbody>
        </table>

        <footer>
            Generated by LLM API Server Evaluation Framework
        </footer>
    </div>

    <script>
        function toggleResponse(button) {{
            const responseContent = button.nextElementSibling;
            const isCollapsed = responseContent.classList.contains('collapsed');

            if (isCollapsed) {{
                responseContent.classList.remove('collapsed');
                button.textContent = 'Collapse';
            }} else {{
                responseContent.classList.add('collapsed');
                button.textContent = 'Expand';
            }}
        }}
    </script>
</body>
</html>"""


class JSONReporter:
    """Generate JSON reports from evaluation results."""

    def generate(self, results: list[TestResult], output_path: str | Path):
        """Generate a JSON report from test results.

        Args:
            results: List of test results
            output_path: Path to write JSON report
        """
        output_path = Path(output_path)

        # Calculate summary stats
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = total - passed
        success_rate = (passed / total * 100) if total > 0 else 0
        total_time = sum(r.response_time for r in results)
        avg_time = total_time / total if total > 0 else 0

        # Build JSON structure
        report = {
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "total": total,
                "passed": passed,
                "failed": failed,
                "success_rate": success_rate,
                "total_time": total_time,
                "avg_time": avg_time,
            },
            "results": [
                {
                    "test_number": i,
                    "description": r.test_case.description,
                    "question": r.test_case.question,
                    "passed": r.passed,
                    "response_time": r.response_time,
                    "response": r.response,
                    "issues": r.issues,
                    "error": r.error,
                    "tools_used": r.tools_used,
                    "metadata": r.test_case.metadata,
                }
                for i, r in enumerate(results, 1)
            ],
        }

        # Write to file
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


class ConsoleReporter:
    """Generate console output from evaluation results."""

    def generate(self, results: list[TestResult], verbose: bool = False):
        """Print test results to console.

        Args:
            results: List of test results
            verbose: If True, print full responses for all tests
        """
        # Calculate summary stats
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = total - passed
        success_rate = (passed / total * 100) if total > 0 else 0
        total_time = sum(r.response_time for r in results)

        print("\n" + "=" * 80)
        print("LLM Evaluation Results")
        print("=" * 80 + "\n")

        # Print individual results
        for i, result in enumerate(results, 1):
            status = "✓ PASSED" if result.passed else "✗ FAILED"
            status_color = "\033[92m" if result.passed else "\033[91m"
            reset = "\033[0m"

            print(f"Test {i}/{total}: {result.test_case.description}")
            print(f'Question: "{result.test_case.question}"')
            print(f"{status_color}{status}{reset} ({result.response_time:.2f}s)")

            if result.tools_used:
                tools_str = ", ".join(result.tools_used)
                print(f"  Tools: {tools_str}")

            if result.error:
                print(f"  Error: {result.error}")
            elif result.issues:
                for issue in result.issues:
                    print(f"  - {issue}")

            if verbose or not result.passed:
                print(f"\nResponse:\n{result.response}\n")

            print()

        # Print summary
        print("=" * 80)
        print("Summary")
        print("=" * 80)
        print(f"Total Tests:  {total}")
        print(f"Passed:       {passed}")
        print(f"Failed:       {failed}")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Total Time:   {total_time:.1f}s")
        print()
