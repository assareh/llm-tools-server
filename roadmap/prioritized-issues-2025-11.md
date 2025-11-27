# Prioritized Issues - LLM API Server

**Created:** 2025-11-27
**Source:** Comprehensive Code Review
**Version:** 0.5.1

---

## Overview

| Priority | Count | Effort (Total) |
|----------|-------|----------------|
| 🔴 P0 - Critical | 3 | ~2 hours |
| 🟠 P1 - High | 5 | ~6 hours |
| 🟡 P2 - Medium | 8 | ~2-3 days |
| 🔵 P3 - Low | 7 | ~1 week |

---

## 🔴 P0 - Critical (Fix This Sprint)

Issues that affect data integrity, security, or cause silent failures.

### P0-1: Parent Context Missing After Cache Load

| Field | Value |
|-------|-------|
| **File** | `llm_api_server/rag/indexer.py` |
| **Lines** | 357-377 |
| **Effort** | 30 minutes |
| **Impact** | RAG answers lose parent context after any server restart |
| **Assignee** | |

**Problem:**
`load_index()` loads chunks and parent_chunks but never rebuilds `child_to_parent` mapping. Search results return without parent context.

**Fix:**
```python
def load_index(self):
    """Load index from cache."""
    self.chunks = self._load_chunks() or []
    self.parent_chunks = self._load_parent_chunks() or {}

    # ADD: Rebuild child_to_parent mapping from chunk metadata
    self.child_to_parent = {}
    for chunk in self.chunks:
        chunk_id = chunk.metadata.get("chunk_id")
        parent_id = chunk.metadata.get("parent_id")
        if chunk_id and parent_id:
            self.child_to_parent[chunk_id] = parent_id

    if not self.chunks:
        return
    # ... rest of method
```

**Test:**
1. Build index, verify search returns `parent_text`
2. Restart server (load from cache)
3. Verify search still returns `parent_text`

---

### P0-2: Cache Not Invalidated After Chunker Rewrite

| Field | Value |
|-------|-------|
| **File** | `llm_api_server/rag/indexer.py` |
| **Lines** | 47 |
| **Effort** | 5 minutes |
| **Impact** | Existing caches use old chunking logic, missing new improvements |
| **Assignee** | |

**Problem:**
`INDEX_VERSION` is still `"1.0.0-parent-child"` even though chunking logic changed significantly.

**Fix:**
```python
# Line 47
INDEX_VERSION = "1.1.0-chunker-v2"  # Bump version
```

**Note:** After deploying, users will need to rebuild indexes (automatic on next `crawl_and_index()`).

---

### P0-3: Calculator Accepts Non-Numeric Types

| Field | Value |
|-------|-------|
| **File** | `llm_api_server/builtin_tools.py` |
| **Lines** | 73-76 |
| **Effort** | 15 minutes |
| **Impact** | Security - violates principle of least privilege |
| **Assignee** | |

**Problem:**
`ast.Constant` check accepts any constant (strings, None, booleans), not just numbers.

**Fix:**
```python
def eval_node(node):
    """Recursively evaluate AST nodes."""
    if isinstance(node, ast.Constant):
        # ADD: Validate numeric type
        if not isinstance(node.value, (int, float, complex)):
            raise ValueError(f"Only numeric constants allowed, got {type(node.value).__name__}")
        return node.value
    # ... rest of function
```

**Test:**
```python
def test_calculator_rejects_strings():
    result = calculate("'hello'")
    assert "error" in result.lower()

def test_calculator_rejects_none():
    result = calculate("None")
    assert "error" in result.lower()
```

---

## 🟠 P1 - High (Fix Next Sprint)

Issues that affect reliability or developer experience significantly.

### P1-1: FAISS Deserialization Security Risk

| Field | Value |
|-------|-------|
| **File** | `llm_api_server/rag/indexer.py` |
| **Lines** | 793-794 |
| **Effort** | 2 hours |
| **Impact** | Security - pickle deserialization can execute arbitrary code |
| **Assignee** | |

**Problem:**
`allow_dangerous_deserialization=True` without integrity verification. Attacker with cache directory access could inject malicious payloads.

**Fix Options:**

**Option A: Add checksum verification (recommended)**
```python
def _save_faiss_index(self):
    faiss_path = str(self.index_dir / "faiss_index")
    self.vectorstore.save_local(faiss_path)

    # Generate checksum
    checksum = self._compute_directory_checksum(faiss_path)
    (self.index_dir / "faiss_index.sha256").write_text(checksum)

def _load_faiss_index(self):
    faiss_path = str(self.index_dir / "faiss_index")
    checksum_file = self.index_dir / "faiss_index.sha256"

    # Verify checksum
    if checksum_file.exists():
        expected = checksum_file.read_text().strip()
        actual = self._compute_directory_checksum(faiss_path)
        if expected != actual:
            raise ValueError("FAISS index checksum mismatch - possible tampering")

    self.vectorstore = FAISS.load_local(faiss_path, self.embeddings,
                                        allow_dangerous_deserialization=True)
```

**Option B: Document the risk**
Add warning in CLAUDE.md about cache directory permissions.

---

### P1-2: Race Condition in System Prompt Caching

| Field | Value |
|-------|-------|
| **File** | `llm_api_server/server.py` |
| **Lines** | 137-148 |
| **Effort** | 1 hour |
| **Impact** | Thread safety - concurrent requests may read inconsistent prompts |
| **Assignee** | |

**Problem:**
Threaded mode is default, but mtime check and file read aren't atomic.

**Fix:**
```python
import threading

class LLMServer:
    def __init__(self, ...):
        # ... existing code ...
        self._prompt_lock = threading.Lock()

    def get_system_prompt(self) -> str:
        prompt_path = Path(self.config.SYSTEM_PROMPT_PATH)
        if not prompt_path.exists():
            return self.default_system_prompt

        try:
            with self._prompt_lock:
                current_mtime = prompt_path.stat().st_mtime
                if self._system_prompt_cache is not None and self._system_prompt_mtime == current_mtime:
                    return self._system_prompt_cache

                content = prompt_path.read_text(encoding="utf-8")
                # Verify mtime didn't change during read
                if prompt_path.stat().st_mtime != current_mtime:
                    # File changed during read, retry
                    content = prompt_path.read_text(encoding="utf-8")
                    current_mtime = prompt_path.stat().st_mtime

                self._system_prompt_cache = content
                self._system_prompt_mtime = current_mtime
                return self._system_prompt_cache
        except Exception as e:
            print(f"Error reading system prompt: {e}")
            return self.default_system_prompt
```

---

### P1-3: Inconsistent Boolean Environment Variable Parsing

| Field | Value |
|-------|-------|
| **File** | `llm_api_server/config.py` |
| **Lines** | 100-114 |
| **Effort** | 30 minutes |
| **Impact** | DX - confusing behavior for users setting env vars |
| **Assignee** | |

**Problem:**
```python
# THREADED defaults to True (inverted logic)
config.THREADED = get_env("THREADED", "").lower() not in ("false", "0", "no")

# DEBUG_TOOLS defaults to False (normal logic)
config.DEBUG_TOOLS = get_env("DEBUG_TOOLS", "").lower() in ("true", "1", "yes")
```

**Fix:**
```python
def _parse_bool_env(value: str, default: bool) -> bool:
    """Parse boolean from environment variable string."""
    if not value:
        return default
    return value.lower() in ("true", "1", "yes")

# In from_env():
config.THREADED = _parse_bool_env(get_env("THREADED", ""), cls.THREADED)
config.DEBUG_TOOLS = _parse_bool_env(get_env("DEBUG_TOOLS", ""), cls.DEBUG_TOOLS)
config.HEALTH_CHECK_ON_STARTUP = _parse_bool_env(get_env("HEALTH_CHECK_ON_STARTUP", ""), cls.HEALTH_CHECK_ON_STARTUP)
config.RATE_LIMIT_ENABLED = _parse_bool_env(get_env("RATE_LIMIT_ENABLED", ""), cls.RATE_LIMIT_ENABLED)
```

---

### P1-4: Unbounded Message Accumulation in Tool Loop

| Field | Value |
|-------|-------|
| **File** | `llm_api_server/server.py` |
| **Lines** | 292-398 |
| **Effort** | 1 hour |
| **Impact** | Memory - could grow unbounded with large tool results |
| **Assignee** | |

**Problem:**
`full_messages` list grows with each tool iteration. Large tool results accumulate.

**Fix:**
```python
# Add constant at top of file
MAX_TOOL_RESULT_CHARS = 10000  # Or make configurable

def _execute_tool_calls(self, tool_calls: list, tools_used: list[str]) -> list[dict]:
    result_messages = []
    for tool_call in tool_calls:
        # ... existing code to get tool_result ...

        # ADD: Truncate large results
        if len(tool_result) > MAX_TOOL_RESULT_CHARS:
            tool_result = tool_result[:MAX_TOOL_RESULT_CHARS] + f"\n\n[Truncated - {len(tool_result)} total chars]"

        # ... rest of method
```

---

### P1-5: Crawler Follows Redirects to External Domains

| Field | Value |
|-------|-------|
| **File** | `llm_api_server/rag/crawler.py` |
| **Lines** | 391-419 |
| **Effort** | 30 minutes |
| **Impact** | Security - could index content from unauthorized domains |
| **Assignee** | |

**Problem:**
`requests.get()` follows redirects but doesn't verify final URL is still within allowed domain.

**Fix:**
```python
def fetch_page(self, url: str) -> tuple[str, str] | None:
    try:
        # ... existing robots check ...

        response = requests.get(url, headers={"User-Agent": self.user_agent},
                               timeout=self.request_timeout)

        # ADD: Verify final URL is within base domain
        final_url = response.url
        if not final_url.startswith(self.base_url):
            logger.warning(f"[CRAWLER] Redirect to external domain blocked: {url} -> {final_url}")
            return None

        response.raise_for_status()
        # ... rest of method
```

---

## 🟡 P2 - Medium (Backlog - Next Month)

Issues that affect code quality, performance, or maintainability.

### P2-1: No RAG Module Tests

| Field | Value |
|-------|-------|
| **Files** | `tests/test_rag_*.py` (new) |
| **Effort** | 1-2 days |
| **Impact** | Quality - most complex module has zero test coverage |
| **Assignee** | |

**Scope:**
- `test_rag_chunker.py` - Heading hierarchy, code block handling, split/merge logic
- `test_rag_indexer.py` - Cache save/load, parent mapping, incremental updates
- `test_rag_crawler.py` - URL filtering, robots.txt, sitemap parsing

**Priority order:** Chunker > Indexer > Crawler

---

### P2-2: Parent Context Truncated to 500 Chars

| Field | Value |
|-------|-------|
| **File** | `llm_api_server/builtin_tools.py` |
| **Lines** | 227-228 |
| **Effort** | 30 minutes |
| **Impact** | Quality - long overview sections are cut off |
| **Assignee** | |

**Problem:**
```python
if result.get("parent_text") and result["parent_text"] != result["text"]:
    entry += f"\n---\nBroader context:\n{result['parent_text'][:500]}...\n"
```

**Fix:**
Make configurable via `DocSearchInput` or RAGConfig.

---

### P2-3: Hybrid Search Semantics Undocumented

| Field | Value |
|-------|-------|
| **File** | `llm_api_server/rag/config.py`, docs |
| **Lines** | 32-36, 70-72 |
| **Effort** | 2 hours |
| **Impact** | DX - users don't understand RRF vs weighted average |
| **Assignee** | |

**Problem:**
Config exposes `hybrid_bm25_weight` and `hybrid_semantic_weight`, but these are RRF weights (rank-based), not score weights.

**Fix:**
1. Improve docstring to explain RRF formula
2. Add example showing effect of different weight ratios
3. Consider renaming to `hybrid_bm25_rrf_weight`

---

### P2-4: Global State Modified at Import Time

| Field | Value |
|-------|-------|
| **File** | `llm_api_server/rag/indexer.py` |
| **Lines** | 36-40 |
| **Effort** | 30 minutes |
| **Impact** | Quality - side effects affect entire process |
| **Assignee** | |

**Problem:**
```python
logging.getLogger("readability.readability").setLevel(logging.WARNING)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
```

**Fix:**
Move to `DocSearchIndex.__init__()` or make it opt-in via config.

---

### P2-5: No Connection Pooling for Backend Requests

| Field | Value |
|-------|-------|
| **File** | `llm_api_server/backends.py` |
| **Effort** | 1 hour |
| **Impact** | Performance - new connection per request |
| **Assignee** | |

**Fix:**
```python
# Module-level session (or per-server instance)
_session = None

def get_session():
    global _session
    if _session is None:
        _session = requests.Session()
    return _session

def call_ollama(...):
    session = get_session()
    response = session.post(endpoint, json=payload, stream=stream, timeout=timeout)
```

---

### P2-6: Readability May Strip Technical Content

| Field | Value |
|-------|-------|
| **File** | `llm_api_server/rag/indexer.py` |
| **Lines** | 520-537 |
| **Effort** | 2 hours |
| **Impact** | Quality - code blocks/tables may be dropped |
| **Assignee** | |

**Problem:**
`ReadabilityDocument.summary()` is optimized for news articles, not technical docs.

**Fix options:**
1. Make readability optional/configurable
2. Compare pre/post content length, fallback if >30% loss
3. Check for code blocks before/after

---

### P2-7: Division by Zero in Cache Stats

| Field | Value |
|-------|-------|
| **File** | `llm_api_server/rag/indexer.py` |
| **Lines** | 481-482 |
| **Effort** | 5 minutes |
| **Impact** | Bug - crashes if pages list is empty |
| **Assignee** | |

**Problem:**
```python
if cache_hits > 0:
    logger.info(f"... ({100*cache_hits/len(pages):.1f}%)")  # len(pages) could be 0
```

**Fix:**
```python
if cache_hits > 0 and len(pages) > 0:
```

---

### P2-8: Evaluator Error Response Not Truncated

| Field | Value |
|-------|-------|
| **File** | `llm_api_server/eval/evaluator.py` |
| **Lines** | 72-73 |
| **Effort** | 15 minutes |
| **Impact** | DX - error messages could be huge |
| **Assignee** | |

**Problem:**
```python
return None, elapsed, f"HTTP {response.status_code}: {response.text}", []
```

**Fix:**
```python
error_text = response.text[:500] if len(response.text) > 500 else response.text
return None, elapsed, f"HTTP {response.status_code}: {error_text}", []
```

---

## 🔵 P3 - Low (Backlog - Quarterly)

Nice-to-have improvements.

### P3-1: Missing Type Hints

| Files | `backends.py`, `validators.py`, `crawler.py` |
| Effort | 2 hours |

- `backends.py:59,93` - `tools: list` → `tools: list[BaseTool]`
- `validators.py:4` - `test_case` untyped
- Various internal functions

---

### P3-2: Magic Numbers Without Constants

| Files | `chunker.py` |
| Effort | 30 minutes |

- Line 50: `MIN_CONTENT_LENGTH = 20` (already named, needs docstring)
- Line 89: `absolute_max_tokens = 1200` (should be configurable)
- Line 619: `child_min_tokens // 2` (needs named constant)

---

### P3-3: Crawler O(n) Check for Visited URLs

| File | `llm_api_server/rag/crawler.py` |
| Line | 382 |
| Effort | 15 minutes |

**Problem:**
```python
if href not in visited and (href, depth + 1) not in to_visit:  # O(n) list scan
```

**Fix:**
```python
to_visit_set = set()  # Add alongside to_visit list
```

---

### P3-4: Environment Variable Name Mismatch

| File | `llm_api_server/config.py` |
| Line | 91 |
| Effort | 15 minutes |

**Problem:**
```python
config.BACKEND_TYPE = get_env("BACKEND", cls.BACKEND_TYPE)  # "BACKEND" not "BACKEND_TYPE"
```

**Fix:**
Accept both for backwards compatibility:
```python
config.BACKEND_TYPE = get_env("BACKEND_TYPE", get_env("BACKEND", cls.BACKEND_TYPE))
```

---

### P3-5: Flask App Name Could Conflict

| File | `llm_api_server/server.py` |
| Line | 63 |
| Effort | 15 minutes |

**Problem:**
```python
self.app = Flask(name.lower())  # "ivan" and "Ivan" would conflict
```

**Fix:**
Use unique identifier:
```python
self.app = Flask(f"{name.lower()}_{id(self)}")
```

---

### P3-6: Cross-Encoder Scores Unnormalized

| File | `llm_api_server/rag/indexer.py` |
| Lines | 840-870 |
| Effort | 1 hour |

Raw MS MARCO scores are returned. A sigmoid transformation would make scores comparable/thresholdable.

---

### P3-7: No Deduplication of Crawled Content

| File | `llm_api_server/rag/indexer.py` |
| Effort | 2 hours |

No content hashing to detect duplicate pages (e.g., same content at different URLs). Duplicates waste storage and skew search results.

---

## Sprint Planning Template

### Sprint N (2 weeks)

**P0 Issues (Required):**
- [ ] P0-1: Parent context missing after cache load (30 min)
- [ ] P0-2: Bump INDEX_VERSION (5 min)
- [ ] P0-3: Calculator type validation (15 min)

**P1 Issues (Pick 2-3):**
- [ ] P1-1: FAISS checksum verification (2 hr)
- [ ] P1-2: System prompt race condition (1 hr)
- [ ] P1-3: Boolean env parsing (30 min)

**Total estimate:** ~4-5 hours

---

### Sprint N+1

**P1 Issues (Remaining):**
- [ ] P1-4: Tool result truncation (1 hr)
- [ ] P1-5: Crawler redirect validation (30 min)

**P2 Issues (Pick 2-3):**
- [ ] P2-1: RAG module tests - chunker (1 day)
- [ ] P2-7: Division by zero fix (5 min)
- [ ] P2-8: Evaluator error truncation (15 min)

---

## Acceptance Criteria Checklist

For each issue:
- [ ] Fix implemented
- [ ] Unit test added (where applicable)
- [ ] Manual testing completed
- [ ] Documentation updated (if API change)
- [ ] `./lint.sh` passes
- [ ] PR reviewed

---

## Notes

1. **P0 issues should block release** - don't ship 0.6.0 without fixing these
2. **P1-1 (FAISS security)** - evaluate risk vs effort; document mitigation if not implementing
3. **P2-1 (RAG tests)** - consider splitting across multiple sprints
4. **Consuming projects (Ivan, milesoss)** - test after P0 fixes before releasing

---

*Last updated: 2025-11-27*
