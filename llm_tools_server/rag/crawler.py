"""Web crawler for documentation sites.

Supports three crawling modes:
1. Sitemap-based crawling (discovers sitemap.xml automatically)
2. Recursive crawling (follows links from seed URL)
3. Manual URL list (explicit list of URLs to index)
"""

import json
import logging
import re
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class SitemapChanges:
    """Result of comparing current sitemap with cached version.

    Attributes:
        new_urls: URLs present in sitemap but not previously indexed
        updated_urls: URLs with changed lastmod timestamps (url, new_lastmod, old_lastmod)
        removed_urls: URLs previously indexed but no longer in sitemap
        unchanged_urls: URLs with same lastmod (no update needed)
        sitemap_unchanged: True if sitemap itself hasn't changed (no need to check pages)
        checked_at: Timestamp when check was performed
        error: Error message if sitemap check failed, None otherwise
    """

    new_urls: list[str] = field(default_factory=list)
    updated_urls: list[tuple[str, str | None, str | None]] = field(
        default_factory=list
    )  # (url, new_lastmod, old_lastmod)
    removed_urls: list[str] = field(default_factory=list)
    unchanged_urls: list[str] = field(default_factory=list)
    sitemap_unchanged: bool = False
    checked_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    error: str | None = None

    @property
    def has_changes(self) -> bool:
        """Return True if there are any changes to process."""
        return bool(self.new_urls or self.updated_urls or self.removed_urls)

    @property
    def total_changes(self) -> int:
        """Return total number of URLs that need processing."""
        return len(self.new_urls) + len(self.updated_urls) + len(self.removed_urls)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "new_urls": self.new_urls,
            "updated_urls": [(u, n, o) for u, n, o in self.updated_urls],
            "removed_urls": self.removed_urls,
            "unchanged_count": len(self.unchanged_urls),
            "sitemap_unchanged": self.sitemap_unchanged,
            "checked_at": self.checked_at.isoformat(),
            "has_changes": self.has_changes,
            "total_changes": self.total_changes,
            "error": self.error,
        }


# Default user agent for crawling
DEFAULT_USER_AGENT = "RAG-DocBot/1.0 (Respectful crawler; +https://github.com/assareh/llm-tools-server)"


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
        request_timeout: float = 10.0,
        url_include_patterns: list[str] | None = None,
        url_exclude_patterns: list[str] | None = None,
        user_agent: str = DEFAULT_USER_AGENT,
        show_progress: bool = True,
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
            request_timeout: HTTP request timeout in seconds
            url_include_patterns: Regex patterns - only crawl matching URLs
            url_exclude_patterns: Regex patterns - skip matching URLs
            user_agent: User agent string for requests
            show_progress: Show progress bars during crawling (default: True)
        """
        self.base_url = base_url.rstrip("/")
        self.cache_dir = cache_dir
        self.sitemap_cache_file = cache_dir / "sitemap_cache.json"
        self.manual_urls = manual_urls or []
        self.manual_urls_only = manual_urls_only
        self.max_crawl_depth = max_crawl_depth
        self.rate_limit_delay = rate_limit_delay
        self.max_workers = max_workers
        self.max_pages = max_pages
        self.request_timeout = request_timeout
        self.user_agent = user_agent
        self.show_progress = show_progress

        # Compile URL patterns
        self.url_include_patterns = [re.compile(p) for p in (url_include_patterns or [])]
        self.url_exclude_patterns = [re.compile(p) for p in (url_exclude_patterns or [])]

        # Robots.txt parser and sitemap discovery
        self.robot_parser = RobotFileParser()
        self.robots_loaded = False  # Track if robots.txt loaded successfully
        self.sitemap_urls_from_robots = []  # Sitemap URLs found in robots.txt

        # Try to load robots.txt, first from base URL, then from root domain
        robots_urls_to_try = [urljoin(self.base_url, "/robots.txt")]

        # If base URL is a subdomain, also try the root domain
        parsed = urlparse(self.base_url)
        domain_parts = parsed.netloc.split(".")
        if len(domain_parts) > 2:  # e.g., developer.hashicorp.com -> hashicorp.com
            root_domain = ".".join(domain_parts[-2:])
            root_url = f"{parsed.scheme}://{root_domain}"
            robots_urls_to_try.append(urljoin(root_url, "/robots.txt"))

        robots_loaded_from = None
        for robots_url in robots_urls_to_try:
            try:
                # Fetch robots.txt to parse both rules and sitemap URLs
                headers = {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}
                response = requests.get(robots_url, headers=headers, timeout=request_timeout)
                response.raise_for_status()

                # Parse sitemap URLs from robots.txt
                for line in response.text.splitlines():
                    line = line.strip()
                    if line.lower().startswith("sitemap:"):
                        sitemap_url = line.split(":", 1)[1].strip()
                        self.sitemap_urls_from_robots.append(sitemap_url)

                # Also load into robot parser for can_fetch checks
                # Note: We can't use read() after fetching manually, so we parse the content
                self.robot_parser.set_url(robots_url)
                self.robot_parser.parse(response.text.splitlines())
                self.robots_loaded = True
                robots_loaded_from = robots_url
                break  # Successfully loaded, stop trying

            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code == 404:
                    logger.debug(f"[CRAWLER] No robots.txt found at {robots_url} (404)")
                else:
                    logger.debug(f"[CRAWLER] Failed to load robots.txt from {robots_url}: {e}")
            except Exception as e:
                logger.debug(f"[CRAWLER] Failed to load robots.txt from {robots_url}: {e}")

        # Log final result
        if self.robots_loaded:
            if self.sitemap_urls_from_robots:
                logger.info(
                    f"[CRAWLER] Loaded robots.txt from {robots_loaded_from}, found {len(self.sitemap_urls_from_robots)} sitemap(s)"
                )
            else:
                logger.info(f"[CRAWLER] Loaded robots.txt from {robots_loaded_from} (no sitemaps listed)")
        else:
            logger.info("[CRAWLER] No robots.txt found, proceeding without restrictions")

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

        # Sort globally by lastmod (newest first) before applying max_pages limit
        # This ensures we get the most recent content regardless of sitemap structure
        # URLs without lastmod dates sort to the end
        result.sort(key=lambda x: x.get("lastmod") or "", reverse=True)

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

        logger.info(f"[CRAWLER] Searching for sitemap ({len(sitemap_urls)} locations to try)...")

        for idx, sitemap_url in enumerate(sitemap_urls, 1):
            try:
                logger.info(f"[CRAWLER] [{idx}/{len(sitemap_urls)}] Trying sitemap: {sitemap_url}")
                headers = {"User-Agent": self.user_agent, "Accept-Encoding": "gzip, deflate"}
                response = requests.get(sitemap_url, headers=headers, timeout=self.request_timeout)
                response.raise_for_status()

                # Parse the sitemap
                urls = self._parse_sitemap_xml(response.content)
                if urls:
                    logger.info(f"[CRAWLER] ✓ Successfully parsed sitemap from {sitemap_url}")
                    return urls
                else:
                    logger.info(f"[CRAWLER] Sitemap fetched but 0 URLs after filtering: {sitemap_url}")

            except Exception as e:
                logger.info(f"[CRAWLER] Failed to fetch {sitemap_url}: {e}")
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
            # Note: Use explicit 'is not None' checks because empty XML elements are falsy
            sitemap_elements = tree.findall("ns:sitemap", namespace)
            if not sitemap_elements:
                sitemap_elements = tree.findall("sitemap")
            if sitemap_elements:
                logger.info(f"[CRAWLER] Found sitemap index with {len(sitemap_elements)} sub-sitemaps")

                # Load sitemap cache for lastmod-based invalidation
                sitemap_cache = self._load_sitemap_cache()
                cached_sitemaps = sitemap_cache.get("sub_sitemaps", {})

                # Collect sub-sitemaps with their lastmod dates
                sub_sitemaps = []
                for sitemap_elem in sitemap_elements:
                    loc = sitemap_elem.find("ns:loc", namespace)
                    if loc is None:
                        loc = sitemap_elem.find("loc")
                    lastmod = sitemap_elem.find("ns:lastmod", namespace)
                    if lastmod is None:
                        lastmod = sitemap_elem.find("lastmod")
                    if loc is not None and loc.text:
                        sub_sitemaps.append({"url": loc.text, "lastmod": lastmod.text if lastmod is not None else None})

                # Sort sub-sitemaps by lastmod (newest first) to prioritize recent content
                sub_sitemaps.sort(key=lambda x: x.get("lastmod") or "", reverse=True)

                # Check cache hits vs misses
                cache_hits = 0
                cache_misses = 0
                for sm in sub_sitemaps:
                    cached = cached_sitemaps.get(sm["url"])
                    if cached and cached.get("lastmod") == sm.get("lastmod") and sm.get("lastmod"):
                        cache_hits += 1
                    else:
                        cache_misses += 1

                logger.info(f"[CRAWLER] Processing sub-sitemaps: {cache_hits} cached, {cache_misses} to fetch...")

                # Track updates for saving cache
                updated_cache = {"sub_sitemaps": {}}

                # Parse each sub-sitemap in order with progress bar
                pbar = tqdm(
                    sub_sitemaps,
                    desc="Parsing sitemaps",
                    unit="sitemap",
                    disable=not self.show_progress,
                    file=sys.stderr,
                )
                for sitemap_info in pbar:
                    sitemap_url = sitemap_info["url"]
                    sitemap_lastmod = sitemap_info.get("lastmod")
                    sitemap_name = sitemap_url.split("/")[-1]

                    try:
                        # Check if we have a valid cache hit
                        cached = cached_sitemaps.get(sitemap_url)
                        if cached and cached.get("lastmod") == sitemap_lastmod and sitemap_lastmod:
                            # Cache hit - use cached URLs
                            pbar.set_postfix_str(f"{sitemap_name[:20]} (cached)", refresh=True)
                            sub_urls = cached.get("urls", [])
                            urls.extend(sub_urls)
                            # Preserve in updated cache
                            updated_cache["sub_sitemaps"][sitemap_url] = cached
                        else:
                            # Cache miss - fetch fresh
                            pbar.set_postfix_str(f"{sitemap_name[:20]} (fetching)", refresh=True)

                            headers = {"User-Agent": self.user_agent, "Accept-Encoding": "gzip, deflate"}
                            response = requests.get(sitemap_url, headers=headers, timeout=self.request_timeout)
                            response.raise_for_status()
                            sub_urls = self._parse_sitemap_xml(response.content)
                            urls.extend(sub_urls)

                            # Update cache with new data
                            updated_cache["sub_sitemaps"][sitemap_url] = {
                                "lastmod": sitemap_lastmod,
                                "urls": sub_urls,
                            }

                            time.sleep(self.rate_limit_delay)

                        # Update progress bar with URL count
                        pbar.set_postfix_str(f"{len(urls)} URLs found", refresh=True)

                    except Exception as e:
                        logger.warning(f"[CRAWLER] Failed to parse sub-sitemap {sitemap_url}: {e}")

                # Save updated cache
                self._save_sitemap_cache(updated_cache)

                logger.info(f"[CRAWLER] Finished parsing {len(sub_sitemaps)} sub-sitemaps, total URLs: {len(urls)}")
                return urls

            # Regular sitemap with <url> elements
            url_elements = tree.findall("ns:url", namespace)
            if not url_elements:
                url_elements = tree.findall("url")
            for url_elem in url_elements:
                loc = url_elem.find("ns:loc", namespace)
                if loc is None:
                    loc = url_elem.find("loc")
                lastmod = url_elem.find("ns:lastmod", namespace)
                if lastmod is None:
                    lastmod = url_elem.find("lastmod")

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
        visited: set[str] = set()
        queued: set[str] = {self.base_url}  # Track URLs in queue for O(1) lookup
        to_visit: list[tuple[str, int]] = [(self.base_url, 0)]  # (url, depth)
        urls: list[dict[str, Any]] = []

        logger.info(f"[CRAWLER] Starting recursive crawl from {self.base_url} (max depth: {self.max_crawl_depth})...")

        # Create progress bar for recursive crawl
        pbar = tqdm(
            desc="Discovering URLs",
            unit="page",
            disable=not self.show_progress,
            file=sys.stderr,
            total=self.max_pages,  # If max_pages is set, use it as total
        )

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

            # Update progress bar
            pbar.update(1)
            pbar.set_postfix_str(f"depth={depth}, queue={len(to_visit)}", refresh=True)

            # Fetch page and extract links
            try:
                time.sleep(self.rate_limit_delay)

                headers = {"User-Agent": self.user_agent, "Accept-Encoding": "gzip, deflate"}
                response = requests.get(current_url, headers=headers, timeout=self.request_timeout)
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
                        # Skip external links (use _is_same_site to handle www/non-www)
                        if not self._is_same_site(href):
                            continue

                    # Normalize and add to queue (use queued set for O(1) lookup)
                    href = self._normalize_url(href)
                    if href not in visited and href not in queued:
                        queued.add(href)
                        to_visit.append((href, depth + 1))

            except Exception as e:
                logger.warning(f"[CRAWLER] Failed to crawl {current_url}: {e}")
                continue

        pbar.close()
        return urls

    def fetch_page(self, url: str) -> tuple[str, str, int] | None:
        """Fetch a single page and return (url, html_content, status_code).

        Args:
            url: URL to fetch

        Returns:
            Tuple of (url, html_content, status_code) or None if blocked by robots.txt
            On HTTP errors, returns (url, "", status_code) to allow status tracking
        """
        try:
            # Check robots.txt only if it loaded successfully
            if self.robots_loaded:
                if not self.robot_parser.can_fetch(self.user_agent, url):
                    logger.warning(f"[CRAWLER] robots.txt disallows: {url}")
                    return None  # No status code - blocked before request
            else:
                logger.debug(f"[CRAWLER] Skipping robots.txt check (not loaded) for: {url}")

            logger.debug(f"[CRAWLER] Fetching: {url}")
            # Explicitly set Accept-Encoding to avoid Brotli issues (some servers send br but decompression can fail)
            headers = {
                "User-Agent": self.user_agent,
                "Accept-Encoding": "gzip, deflate",  # Exclude 'br' (Brotli) to avoid decompression errors
            }
            response = requests.get(url, headers=headers, timeout=self.request_timeout)
            status_code = response.status_code

            # Verify final URL is still within base domain (blocks redirects to external sites)
            # Use _is_same_site() to handle www/non-www redirects (e.g., example.com -> www.example.com)
            final_url = response.url
            if not self._is_same_site(final_url):
                logger.warning(f"[CRAWLER] Redirect to external domain blocked: {url} -> {final_url}")
                return (url, "", status_code)  # Return status but no content

            # Check for HTTP errors
            if not response.ok:
                logger.warning(f"[CRAWLER] HTTP {status_code} for {url}")
                return (url, "", status_code)

            # Only process HTML content
            content_type = response.headers.get("content-type", "")
            if "text/html" not in content_type:
                logger.warning(f"[CRAWLER] Skipping non-HTML content: {url} ({content_type})")
                return (url, "", status_code)

            return (url, response.text, status_code)

        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response is not None else 0
            logger.error(f"[CRAWLER] HTTP {status_code} for {url}: {e}")
            return (url, "", status_code)

        except Exception as e:
            logger.error(f"[CRAWLER] Failed to fetch {url}: {e}")
            return (url, "", 0)  # 0 indicates network/connection error

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

    def _is_same_site(self, url: str) -> bool:
        """Check if URL is on the same site as base_url (treating www. as equivalent to apex).

        This handles redirects between www and non-www versions of the same domain.

        Args:
            url: URL to check

        Returns:
            True if URL is on the same site as base_url
        """

        def normalize_domain(domain: str) -> str:
            return domain.lower().removeprefix("www.")

        parsed_base = urlparse(self.base_url)
        parsed_url = urlparse(url)

        return normalize_domain(parsed_base.netloc) == normalize_domain(parsed_url.netloc)

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

    def _load_sitemap_cache(self) -> dict[str, Any]:
        """Load sitemap cache from disk.

        Returns:
            Dict with 'sub_sitemaps' mapping sitemap URL -> {lastmod, urls}
        """
        if not self.sitemap_cache_file.exists():
            return {"sub_sitemaps": {}}

        try:
            with open(self.sitemap_cache_file) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"[CRAWLER] Failed to load sitemap cache: {e}")
            return {"sub_sitemaps": {}}

    def _save_sitemap_cache(self, cache: dict[str, Any]):
        """Save sitemap cache to disk.

        Args:
            cache: Cache dict with sub_sitemaps mapping
        """
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            with open(self.sitemap_cache_file, "w") as f:
                json.dump(cache, f, indent=2)
        except Exception as e:
            logger.warning(f"[CRAWLER] Failed to save sitemap cache: {e}")

    def get_sitemap_changes(self, indexed_urls: dict[str, str | None]) -> SitemapChanges:
        """Compare current sitemap with previously indexed URLs to detect changes.

        This method is designed for periodic update checking - it fetches the sitemap
        and compares lastmod timestamps with the previously indexed URLs to determine
        which pages need to be re-indexed.

        Args:
            indexed_urls: Dict mapping URL -> lastmod timestamp (or None if no lastmod)
                         This should come from the index's crawl_state tracking.

        Returns:
            SitemapChanges with new, updated, removed, and unchanged URLs
        """
        result = SitemapChanges()

        try:
            # Fetch current sitemap URLs
            logger.info("[CRAWLER] Checking sitemap for changes...")
            current_urls = self._discover_sitemap()

            if not current_urls:
                # No sitemap found - try recursive crawl for comparison
                logger.info("[CRAWLER] No sitemap found, falling back to recursive crawl for change detection")
                current_urls = self._recursive_crawl()

            if not current_urls:
                result.error = "No URLs discovered from sitemap or crawl"
                return result

            # Build lookup of current URLs -> lastmod
            current_url_map: dict[str, str | None] = {}
            for url_info in current_urls:
                url = url_info.get("url")
                if url:
                    # Normalize URL for consistent comparison
                    url = self._normalize_url(url)
                    current_url_map[url] = url_info.get("lastmod")

            # Compare with indexed URLs
            indexed_set = set(indexed_urls.keys())
            current_set = set(current_url_map.keys())

            # New URLs: in current but not in indexed
            new_urls = current_set - indexed_set
            result.new_urls = list(new_urls)

            # Removed URLs: in indexed but not in current
            removed_urls = indexed_set - current_set
            result.removed_urls = list(removed_urls)

            # Check for updates in URLs that exist in both
            common_urls = indexed_set & current_set
            for url in common_urls:
                old_lastmod = indexed_urls.get(url)
                new_lastmod = current_url_map.get(url)

                # Determine if update needed based on lastmod comparison
                if self._lastmod_indicates_change(old_lastmod, new_lastmod):
                    result.updated_urls.append((url, new_lastmod, old_lastmod))
                else:
                    result.unchanged_urls.append(url)

            # Log summary
            logger.info(
                f"[CRAWLER] Sitemap changes detected: "
                f"{len(result.new_urls)} new, "
                f"{len(result.updated_urls)} updated, "
                f"{len(result.removed_urls)} removed, "
                f"{len(result.unchanged_urls)} unchanged"
            )

            return result

        except Exception as e:
            logger.error(f"[CRAWLER] Failed to check sitemap changes: {e}")
            result.error = str(e)
            return result

    def _lastmod_indicates_change(self, old_lastmod: str | None, new_lastmod: str | None) -> bool:
        """Determine if lastmod timestamps indicate a page has changed.

        Args:
            old_lastmod: Previously stored lastmod (from index)
            new_lastmod: Current lastmod (from sitemap)

        Returns:
            True if the page should be considered changed and re-indexed
        """
        # If both are None, we can't detect change via lastmod - assume no change
        # (Page will be checked via TTL-based cache invalidation instead)
        if old_lastmod is None and new_lastmod is None:
            return False

        # If old had lastmod but new doesn't, assume no change
        # (Site may have stopped providing lastmod)
        if old_lastmod is not None and new_lastmod is None:
            return False

        # If old had no lastmod but new does, consider it an update
        # (Site started providing lastmod, or we have new info)
        if old_lastmod is None and new_lastmod is not None:
            return True

        # Both have lastmod - compare them
        # Simple string comparison works for ISO 8601 dates
        return old_lastmod != new_lastmod

    def get_current_sitemap_urls(self) -> dict[str, str | None]:
        """Get current URLs from sitemap with their lastmod timestamps.

        This is a lightweight method that just fetches the sitemap without
        fetching page content. Useful for periodic checks.

        Returns:
            Dict mapping URL -> lastmod timestamp (or None)
        """
        result: dict[str, str | None] = {}

        try:
            # Try sitemap first
            sitemap_urls = self._discover_sitemap()
            if sitemap_urls:
                for url_info in sitemap_urls:
                    url = url_info.get("url")
                    if url:
                        url = self._normalize_url(url)
                        result[url] = url_info.get("lastmod")
                return result

            # Fallback to recursive crawl (won't have lastmod)
            crawled_urls = self._recursive_crawl()
            for url_info in crawled_urls:
                url = url_info.get("url")
                if url:
                    url = self._normalize_url(url)
                    result[url] = None  # No lastmod from recursive crawl

            return result

        except Exception as e:
            logger.error(f"[CRAWLER] Failed to get current sitemap URLs: {e}")
            return result
