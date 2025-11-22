"""Web crawler for documentation sites.

Supports three crawling modes:
1. Sitemap-based crawling (discovers sitemap.xml automatically)
2. Recursive crawling (follows links from seed URL)
3. Manual URL list (explicit list of URLs to index)
"""

import logging
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
from urllib.parse import urljoin
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Default user agent for crawling
DEFAULT_USER_AGENT = "RAG-DocBot/1.0 (Respectful crawler; +https://github.com/assareh/llm-api-server)"


class DocumentCrawler:
    """Web crawler for documentation sites with sitemap, recursive, and manual modes."""

    def __init__(
        self,
        base_url: str,
        cache_dir: Path,
        manual_urls: list[str] | None = None,
        manual_urls_only: bool = False,
        max_crawl_depth: int = 3,
        rate_limit_delay: float = 0.1,
        max_workers: int = 5,
        max_pages: int | None = None,
        url_include_patterns: list[str] | None = None,
        url_exclude_patterns: list[str] | None = None,
        user_agent: str = DEFAULT_USER_AGENT,
    ):
        """Initialize the document crawler.

        Args:
            base_url: Base URL for crawling
            cache_dir: Directory to cache downloaded content
            manual_urls: Optional list of specific URLs to index
            manual_urls_only: If True, only index manual_urls (no crawling)
            max_crawl_depth: Maximum depth for recursive crawling
            rate_limit_delay: Delay between requests in seconds
            max_workers: Number of parallel workers for fetching
            max_pages: Maximum total pages to crawl (None = unlimited)
            url_include_patterns: Regex patterns - only crawl matching URLs
            url_exclude_patterns: Regex patterns - skip matching URLs
            user_agent: User agent string for requests
        """
        self.base_url = base_url.rstrip("/")
        self.cache_dir = cache_dir
        self.manual_urls = manual_urls or []
        self.manual_urls_only = manual_urls_only
        self.max_crawl_depth = max_crawl_depth
        self.rate_limit_delay = rate_limit_delay
        self.max_workers = max_workers
        self.max_pages = max_pages
        self.user_agent = user_agent

        # Compile URL patterns
        self.url_include_patterns = [re.compile(p) for p in (url_include_patterns or [])]
        self.url_exclude_patterns = [re.compile(p) for p in (url_exclude_patterns or [])]

        # Robots.txt parser and sitemap discovery
        self.robot_parser = RobotFileParser()
        self.robots_loaded = False  # Track if robots.txt loaded successfully
        self.sitemap_urls_from_robots = []  # Sitemap URLs found in robots.txt

        robots_url = urljoin(self.base_url, "/robots.txt")
        self.robot_parser.set_url(robots_url)
        try:
            # Fetch robots.txt to parse both rules and sitemap URLs
            response = requests.get(robots_url, headers={"User-Agent": user_agent}, timeout=30)
            response.raise_for_status()

            # Parse sitemap URLs from robots.txt
            for line in response.text.splitlines():
                line = line.strip()
                if line.lower().startswith("sitemap:"):
                    sitemap_url = line.split(":", 1)[1].strip()
                    self.sitemap_urls_from_robots.append(sitemap_url)

            # Also load into robot parser for can_fetch checks
            # Note: We can't use read() after fetching manually, so we parse the content

            self.robot_parser.parse(response.text.splitlines())
            self.robots_loaded = True

            if self.sitemap_urls_from_robots:
                logger.info(
                    f"[CRAWLER] Loaded robots.txt from {robots_url}, found {len(self.sitemap_urls_from_robots)} sitemap(s)"
                )
            else:
                logger.info(f"[CRAWLER] Loaded robots.txt from {robots_url} (no sitemaps listed)")
        except Exception as e:
            logger.warning(f"[CRAWLER] Failed to load robots.txt from {robots_url}: {e}")
            logger.info("[CRAWLER] Proceeding without robots.txt restrictions")

    def discover_and_crawl(self) -> list[dict[str, Any]]:
        """Discover URLs using sitemap or recursive crawl, plus manual URLs.

        Returns:
            List of URL info dicts with 'url' and optional metadata
        """
        urls = []

        # Add manual URLs if provided
        if self.manual_urls:
            logger.info(f"[CRAWLER] Adding {len(self.manual_urls)} manual URLs")
            urls.extend([{"url": self._normalize_url(url)} for url in self.manual_urls])

        # If manual_urls_only, skip crawling
        if self.manual_urls_only:
            logger.info("[CRAWLER] Manual URLs only mode, skipping automated crawling")
            return urls

        # Try sitemap first
        sitemap_urls = self._discover_sitemap()
        if sitemap_urls:
            logger.info(f"[CRAWLER] Found {len(sitemap_urls)} URLs from sitemap")
            urls.extend(sitemap_urls)
        else:
            # Fallback to recursive crawl
            logger.info("[CRAWLER] No sitemap found, falling back to recursive crawl")
            crawled_urls = self._recursive_crawl()
            logger.info(f"[CRAWLER] Recursive crawl found {len(crawled_urls)} URLs")
            urls.extend(crawled_urls)

        # Deduplicate
        unique_urls = {url_info["url"]: url_info for url_info in urls}
        result = list(unique_urls.values())

        # Apply max_pages limit
        if self.max_pages and len(result) > self.max_pages:
            logger.info(f"[CRAWLER] Limiting to {self.max_pages} pages (found {len(result)})")
            result = result[: self.max_pages]

        logger.info(f"[CRAWLER] Total unique URLs: {len(result)}")
        return result

    def _discover_sitemap(self) -> list[dict[str, Any]]:
        """Try to discover and parse sitemap.xml.

        First tries sitemap URLs from robots.txt, then falls back to common locations.

        Returns:
            List of URL info dicts from sitemap, or empty list if not found
        """
        # Try sitemap URLs from robots.txt first
        sitemap_urls = list(self.sitemap_urls_from_robots)  # Copy list

        # Add common sitemap locations as fallbacks
        sitemap_urls.extend(
            [
                f"{self.base_url}/sitemap.xml",
                f"{self.base_url}/sitemap_index.xml",
                f"{self.base_url}/server-sitemap.xml",
            ]
        )

        for sitemap_url in sitemap_urls:
            try:
                logger.info(f"[CRAWLER] Trying sitemap: {sitemap_url}")
                response = requests.get(sitemap_url, headers={"User-Agent": self.user_agent}, timeout=30)
                response.raise_for_status()

                # Parse the sitemap
                urls = self._parse_sitemap_xml(response.content)
                if urls:
                    logger.info(f"[CRAWLER] Successfully parsed sitemap from {sitemap_url}")
                    return urls

            except Exception as e:
                logger.debug(f"[CRAWLER] Failed to fetch {sitemap_url}: {e}")
                continue

        logger.info("[CRAWLER] No sitemap found at common locations")
        return []

    def _parse_sitemap_xml(self, xml_content: bytes) -> list[dict[str, Any]]:
        """Parse sitemap XML content.

        Args:
            xml_content: Raw XML bytes

        Returns:
            List of URL info dicts
        """
        try:
            tree = ET.fromstring(xml_content)

            # Handle XML namespace
            namespace = {"ns": "http://www.sitemaps.org/schemas/sitemap/0.9"}

            urls = []

            # Check if this is a sitemap index (contains <sitemap> elements)
            sitemap_elements = tree.findall("ns:sitemap", namespace) or tree.findall("sitemap")
            if sitemap_elements:
                logger.info(f"[CRAWLER] Found sitemap index with {len(sitemap_elements)} sub-sitemaps")

                # Collect sub-sitemaps with their lastmod dates
                sub_sitemaps = []
                for sitemap_elem in sitemap_elements:
                    loc = sitemap_elem.find("ns:loc", namespace) or sitemap_elem.find("loc")
                    lastmod = sitemap_elem.find("ns:lastmod", namespace) or sitemap_elem.find("lastmod")
                    if loc is not None and loc.text:
                        sub_sitemaps.append({"url": loc.text, "lastmod": lastmod.text if lastmod is not None else None})

                # Sort sub-sitemaps by lastmod (newest first) to prioritize recent content
                sub_sitemaps.sort(key=lambda x: x.get("lastmod") or "", reverse=True)
                logger.info("[CRAWLER] Processing sub-sitemaps (newest first)")

                # Parse each sub-sitemap in order
                for sitemap_info in sub_sitemaps:
                    try:
                        response = requests.get(
                            sitemap_info["url"], headers={"User-Agent": self.user_agent}, timeout=30
                        )
                        response.raise_for_status()
                        sub_urls = self._parse_sitemap_xml(response.content)
                        urls.extend(sub_urls)
                        time.sleep(self.rate_limit_delay)
                    except Exception as e:
                        logger.warning(f"[CRAWLER] Failed to parse sub-sitemap {sitemap_info['url']}: {e}")
                return urls

            # Regular sitemap with <url> elements
            for url_elem in tree.findall("ns:url", namespace) or tree.findall("url"):
                loc = url_elem.find("ns:loc", namespace) or url_elem.find("loc")
                lastmod = url_elem.find("ns:lastmod", namespace) or url_elem.find("lastmod")

                if loc is not None and loc.text:
                    url = self._normalize_url(loc.text)

                    # Filter by patterns
                    if not self._should_crawl_url(url):
                        continue

                    urls.append({"url": url, "lastmod": lastmod.text if lastmod is not None else None})

            # Sort by lastmod (newest first) - URLs without lastmod go to the end
            urls.sort(key=lambda x: x.get("lastmod") or "", reverse=True)

            return urls

        except Exception as e:
            logger.error(f"[CRAWLER] Failed to parse sitemap XML: {e}")
            return []

    def _recursive_crawl(self) -> list[dict[str, Any]]:
        """Recursively crawl from base_url following links.

        Returns:
            List of URL info dicts
        """
        visited = set()
        to_visit = [(self.base_url, 0)]  # (url, depth)
        urls = []

        while to_visit and (not self.max_pages or len(urls) < self.max_pages):
            current_url, depth = to_visit.pop(0)

            # Skip if already visited
            if current_url in visited:
                continue

            # Skip if max depth exceeded
            if depth > self.max_crawl_depth:
                continue

            # Skip if filtered out
            if not self._should_crawl_url(current_url):
                continue

            visited.add(current_url)
            urls.append({"url": current_url})

            # Fetch page and extract links
            try:
                logger.info(f"[CRAWLER] [{len(urls)}/{self.max_pages or '∞'}] Crawling depth {depth}: {current_url}")
                time.sleep(self.rate_limit_delay)

                response = requests.get(current_url, headers={"User-Agent": self.user_agent}, timeout=30)
                response.raise_for_status()

                # Only parse HTML content, skip XML/RSS/etc
                content_type = response.headers.get("content-type", "")
                if "text/html" not in content_type:
                    logger.debug(
                        f"[CRAWLER] Skipping non-HTML content during recursive crawl: {current_url} ({content_type})"
                    )
                    continue

                soup = BeautifulSoup(response.text, "html.parser")

                # Extract all links
                for link in soup.find_all("a", href=True):
                    href = link["href"]

                    # Skip non-http links
                    if href.startswith(("mailto:", "tel:", "#", "javascript:")):
                        continue

                    # Make absolute URL
                    if href.startswith("/"):
                        href = urljoin(self.base_url, href)
                    elif not href.startswith("http"):
                        href = urljoin(current_url, href)
                    else:
                        # Skip external links
                        if not href.startswith(self.base_url):
                            continue

                    # Normalize and add to queue
                    href = self._normalize_url(href)
                    if href not in visited and (href, depth + 1) not in to_visit:
                        to_visit.append((href, depth + 1))

            except Exception as e:
                logger.warning(f"[CRAWLER] Failed to crawl {current_url}: {e}")
                continue

        return urls

    def fetch_page(self, url: str) -> tuple[str, str] | None:
        """Fetch a single page and return (url, html_content).

        Args:
            url: URL to fetch

        Returns:
            Tuple of (url, html_content) or None if failed
        """
        try:
            # Check robots.txt only if it loaded successfully
            if self.robots_loaded:
                if not self.robot_parser.can_fetch(self.user_agent, url):
                    logger.warning(f"[CRAWLER] robots.txt disallows: {url}")
                    return None
            else:
                logger.debug(f"[CRAWLER] Skipping robots.txt check (not loaded) for: {url}")

            logger.debug(f"[CRAWLER] Fetching: {url}")
            response = requests.get(url, headers={"User-Agent": self.user_agent}, timeout=30)
            response.raise_for_status()

            # Only process HTML content
            content_type = response.headers.get("content-type", "")
            if "text/html" not in content_type:
                logger.warning(f"[CRAWLER] Skipping non-HTML content: {url} ({content_type})")
                return None

            return (url, response.text)

        except Exception as e:
            logger.error(f"[CRAWLER] Failed to fetch {url}: {e}")
            return None

    def _normalize_url(self, url: str) -> str:
        """Normalize URL by removing query params, anchors, trailing slashes.

        Args:
            url: URL to normalize

        Returns:
            Normalized URL
        """
        # Remove query params and anchors
        url = url.split("?")[0].split("#")[0]
        # Remove trailing slash
        url = url.rstrip("/")
        return url

    def _should_crawl_url(self, url: str) -> bool:
        """Check if URL should be crawled based on include/exclude patterns.

        Args:
            url: URL to check

        Returns:
            True if URL should be crawled
        """
        # Check exclude patterns first
        for pattern in self.url_exclude_patterns:
            if pattern.search(url):
                logger.debug(f"[CRAWLER] Excluded by pattern: {url}")
                return False

        # If include patterns specified, URL must match at least one
        if self.url_include_patterns:
            for pattern in self.url_include_patterns:
                if pattern.search(url):
                    return True
            logger.debug(f"[CRAWLER] Not included by any pattern: {url}")
            return False

        return True
