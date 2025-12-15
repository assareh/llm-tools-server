"""Unit tests for crawler URL handling (no network)."""

from unittest.mock import Mock

import pytest
import requests

from llm_tools_server.rag.crawler import DocumentCrawler


@pytest.mark.unit
def test_fetch_page_blocks_redirects_to_external_domains(tmp_path, monkeypatch):
    """fetch_page should return empty content when redirected off the base domain."""

    def fake_get(url, headers=None, timeout=None):
        # Simulate robots.txt 404 during init
        if url.endswith("/robots.txt"):
            resp = Mock()
            resp.text = ""
            resp.status_code = 404
            resp.raise_for_status.side_effect = requests.exceptions.HTTPError(response=resp)
            return resp

        # Simulate redirect to external domain for the actual fetch
        resp = Mock()
        resp.url = "https://malicious.example.net/redirected"
        resp.headers = {"content-type": "text/html"}
        resp.text = "<html><body>redirected</body></html>"
        resp.status_code = 200
        resp.ok = True
        resp.raise_for_status = Mock()
        return resp

    monkeypatch.setattr("llm_tools_server.rag.crawler.requests.get", fake_get)

    crawler = DocumentCrawler(base_url="https://docs.example.com", cache_dir=tmp_path)
    result = crawler.fetch_page("https://docs.example.com/start")

    # Returns (url, empty_html, status_code) for tracking, not None
    assert result is not None
    assert result[0] == "https://docs.example.com/start"
    assert result[1] == ""  # No content returned for external redirect
    assert result[2] == 200  # Status code still returned for tracking


@pytest.mark.unit
def test_fetch_page_skips_non_html_content(tmp_path, monkeypatch):
    """Non-HTML content types should return empty content with status code."""

    def fake_get(url, headers=None, timeout=None):
        if url.endswith("/robots.txt"):
            resp = Mock()
            resp.text = ""
            resp.status_code = 404
            resp.raise_for_status.side_effect = requests.exceptions.HTTPError(response=resp)
            return resp

        resp = Mock()
        resp.url = url
        resp.headers = {"content-type": "application/pdf"}
        resp.text = "%PDF-1.4"
        resp.status_code = 200
        resp.ok = True
        resp.raise_for_status = Mock()
        return resp

    monkeypatch.setattr("llm_tools_server.rag.crawler.requests.get", fake_get)

    crawler = DocumentCrawler(base_url="https://docs.example.com", cache_dir=tmp_path)
    result = crawler.fetch_page("https://docs.example.com/guide.pdf")

    # Returns (url, empty_html, status_code) for tracking, not None
    assert result is not None
    assert result[0] == "https://docs.example.com/guide.pdf"
    assert result[1] == ""  # No content for non-HTML
    assert result[2] == 200  # Status code still returned for tracking
