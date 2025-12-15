# Feature Request: Incremental RAG Index Updates

## Summary

Add support for incremental index updates in the RAG module by tracking ETags and Last-Modified headers for crawled pages, enabling efficient re-indexing of only changed content.

## Background

The current RAG implementation (`llm_tools_server/rag/`) uses an all-or-nothing approach to index freshness:

1. `DocSearchIndex` checks if the index is older than `index_ttl_hours` (default: 24 hours)
2. If stale, it **re-crawls and re-indexes everything**
3. No tracking of individual page changes

For large documentation sites with thousands of pages, this is inefficient because:
- Most pages don't change between crawls
- Re-embedding unchanged content wastes compute time
- Full re-indexes can take minutes to hours for large sites

## Proposed Changes

### 1. Page Metadata Tracking

**File: `llm_tools_server/rag/crawler.py`**

Track HTTP response headers for each crawled URL:

```python
@dataclass
class CrawledPage:
    url: str
    content: str
    etag: str | None = None
    last_modified: str | None = None
    content_hash: str | None = None  # SHA-256 of content for fallback
```

Store metadata in a JSON file alongside the index:

```python
# page_metadata.json
{
    "https://docs.example.com/intro": {
        "etag": "\"abc123\"",
        "last_modified": "Wed, 21 Oct 2024 07:28:00 GMT",
        "content_hash": "sha256:...",
        "last_crawled": "2024-11-27T10:00:00Z"
    },
    ...
}
```

### 2. Conditional HTTP Requests

**File: `llm_tools_server/rag/crawler.py`**

Add conditional request support to `DocumentCrawler`:

```python
def fetch_page(self, url: str, metadata: dict | None = None) -> CrawledPage | None:
    headers = {}

    if metadata:
        if metadata.get("etag"):
            headers["If-None-Match"] = metadata["etag"]
        if metadata.get("last_modified"):
            headers["If-Modified-Since"] = metadata["last_modified"]

    response = requests.get(url, headers=headers, timeout=self.config.request_timeout)

    if response.status_code == 304:
        return None  # Page unchanged, skip re-indexing

    return CrawledPage(
        url=url,
        content=response.text,
        etag=response.headers.get("ETag"),
        last_modified=response.headers.get("Last-Modified"),
        content_hash=hashlib.sha256(response.text.encode()).hexdigest(),
    )
```

### 3. Incremental Index Updates

**File: `llm_tools_server/rag/indexer.py`**

Add methods for partial index updates:

```python
class DocSearchIndex:
    def incremental_update(self) -> UpdateResult:
        """Check for changed pages and update index incrementally."""
        metadata = self._load_page_metadata()

        changed_urls = []
        removed_urls = []

        for url in self._discover_urls():
            page = self.crawler.fetch_page(url, metadata.get(url))
            if page is None:
                continue  # 304 Not Modified
            if page.content_hash != metadata.get(url, {}).get("content_hash"):
                changed_urls.append(url)

        # Detect removed pages
        current_urls = set(self._discover_urls())
        indexed_urls = set(metadata.keys())
        removed_urls = list(indexed_urls - current_urls)

        # Update only changed/removed documents
        self._remove_documents(removed_urls)
        self._add_documents(changed_urls)

        return UpdateResult(
            changed=len(changed_urls),
            removed=len(removed_urls),
            unchanged=len(current_urls) - len(changed_urls),
        )
```

### 4. FAISS Incremental Updates

FAISS supports adding vectors but not efficient removal. Options:

**Option A: Tombstone + Periodic Rebuild**
- Mark removed documents as "deleted" in metadata
- Filter them out at query time
- Full rebuild when tombstones exceed threshold (e.g., 10%)

**Option B: Use FAISS IDMap**
- Wrap index with `faiss.IndexIDMap`
- Assign stable IDs to documents
- Use `remove_ids()` for deletions (requires `IndexFlatL2` or similar)

**Option C: Rebuild BM25, Append FAISS**
- BM25 index is fast to rebuild
- Only append new vectors to FAISS
- Periodic full FAISS rebuild for cleanup

Recommended: **Option A** for simplicity, with configurable rebuild threshold.

### 5. Configuration

**File: `llm_tools_server/rag/config.py`**

```python
@dataclass
class RAGConfig:
    # ... existing fields ...

    # Incremental update settings
    incremental_enabled: bool = True
    tombstone_rebuild_threshold: float = 0.1  # Rebuild when 10% are tombstones
    track_page_metadata: bool = True
```

## Usage After Implementation

```python
from llm_tools_server.rag import DocSearchIndex, RAGConfig

config = RAGConfig(
    base_url="https://docs.example.com",
    cache_dir="./doc_index",
    incremental_enabled=True,
)

index = DocSearchIndex(config)

# First time: full crawl and index
index.crawl_and_index()

# Subsequent runs: only update changed pages
result = index.incremental_update()
print(f"Updated {result.changed} pages, removed {result.removed}, {result.unchanged} unchanged")
```

## Testing Considerations

- Mock HTTP server returning 304 for unchanged pages
- Verify metadata persistence across runs
- Test document removal detection
- Benchmark incremental vs full rebuild performance
- Test tombstone threshold triggering rebuild
- Verify search results remain accurate after incremental updates

## Implementation Phases

### Phase 1: Metadata Tracking
- Add `CrawledPage` dataclass with headers
- Store/load page metadata JSON
- Add content hashing

### Phase 2: Conditional Requests
- Implement `If-None-Match` / `If-Modified-Since` headers
- Handle 304 responses
- Fallback to content hash comparison

### Phase 3: Incremental Index Updates
- Implement `incremental_update()` method
- Add tombstone tracking for deletions
- Configure rebuild threshold

### Phase 4: Optimization
- Parallel conditional requests
- Batch embedding updates
- Performance benchmarking

## References

- [HTTP Conditional Requests (MDN)](https://developer.mozilla.org/en-US/docs/Web/HTTP/Conditional_requests)
- [FAISS IndexIDMap](https://github.com/facebookresearch/faiss/wiki/Pre--and-post-processing#idmap-converting-between-index-ids-and-dataset-ids)
- [Original feedback from Ivan team](https://github.com/assareh/llm-tools-server)

## Origin

This feature request originated from feedback from the Ivan (HashiCorp documentation assistant) team, who identified incremental updates as a key optimization for production use with large documentation sites.
