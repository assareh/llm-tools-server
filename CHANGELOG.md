# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.8.2] - 2025-11-28

### Fixed
- **Tool Choice Required Retry** - Retry once with nudge when `tool_choice=required` is ignored by model
  - Some models ignore the `tool_choice` parameter and return text instead of tool calls
  - Now detects when `tool_choice=required` was set but no tool calls were returned
  - Appends a nudge message ("Please use one of the available tools...") and retries once
  - Only retries once per request to avoid infinite loops
  - Logs a warning when the retry occurs for debugging

## [0.8.1] - 2025-11-28

### Fixed
- **Explicit tool_choice=none** - Now explicitly sends `tool_choice="none"` in payload for final response generation
  - Previously, `tool_choice` was only set when tools were present, so it was never sent with empty tools list
  - Ensures backend receives explicit signal not to attempt tool calls during final synthesis

## [0.8.0] - 2025-11-28

### Added
- **Tool Choice Support** - Control how models use tools via `tool_choice` parameter
  - Supports `"auto"` (model decides), `"required"` (force tool use), and `"none"` (disable tools)
  - New `FIRST_ITERATION_TOOL_CHOICE` config option (default: `"auto"`)
  - Environment variable: `FIRST_ITERATION_TOOL_CHOICE=required` to force tools on first call
  - Tool choice passed to backend and logged for debugging

### Changed
- **Ollama Backend** - Switched from native `/api/chat` to OpenAI-compatible `/v1/chat/completions` endpoint
  - Enables full `tool_choice` support (native API didn't support this parameter)
  - Response format now matches LM Studio (standard OpenAI format)
  - Both backends now use nearly identical code paths
- Tools are only sent to backend when `tool_choice != "none"` and tools exist
- Final response generation explicitly uses `tool_choice="none"` to prevent tool calls

## [0.7.3] - 2025-11-28

### Added
- **RAG Staleness Refresh** - Three mechanisms to refresh existing cached content (GitHub issue #1)
  - **`force_refresh=True` option** - Bypasses page cache to refetch all pages while keeping crawl state
  - **TTL-based cache invalidation** - Pages without `lastmod` expire after `page_cache_ttl_hours` (default: 7 days)
  - **Periodic recrawl of existing URLs** - When `update_check_interval_hours` triggers, existing URLs are now checked for staleness

### Changed
- When pages are refreshed, old chunks are removed before new ones are added
- If any content was replaced during refresh, a full index rebuild is triggered (FAISS doesn't support incremental removal)
- Page cache now stores `cached_at` timestamp for TTL-based expiration

### Configuration
- New `page_cache_ttl_hours` option in `RAGConfig` (default: 168 = 7 days, 0 = never expire)

## [0.7.2] - 2025-11-28

### Fixed
- **RAG Chunking Config** - Wire up previously unused config parameters
  - Added `child_chunk_min_tokens` (default: 150) - configurable minimum for child chunks
  - Added `parent_chunk_min_tokens` (default: 300) - configurable minimum for parent chunks
  - Removed unused `child_chunk_overlap` and `parent_chunk_overlap` parameters
  - Semantic chunker uses heading-based sectioning, not sliding-window overlap

- **RAG Boilerplate Removal** - Apply boilerplate selectors in chunker
  - Previously defined `BOILERPLATE_SELECTORS` are now applied before chunking
  - Removes nav, footer, sidebar, TOC elements that may remain after readability extraction
  - Especially useful when fallback to `<main>`/`<article>` tags or original HTML is used

- **Thinker Model Streaming** - Fix content loss when model doesn't use markers
  - Previously, if a model didn't output `[BEGIN FINAL RESPONSE]` markers, content was discarded
  - Now tracks full content and outputs it if no markers are found
  - Fixes streaming for models that don't follow thinker marker protocol

### Changed
- Improved hybrid search debug logging with expected candidate counts
- Updated RAGConfig docstring to reflect correct parameter names

### Documentation
- Expanded README RAG section with architecture overview and detailed feature descriptions
- Clarified RRF (Reciprocal Rank Fusion) weight behavior in hybrid search examples

## [0.7.1] - 2025-11-28

### Added
- **Tool Loop Timeout** - New `TOOL_LOOP_TIMEOUT` config option
  - Maximum seconds for the entire tool loop (default: 120, 0 = no timeout)
  - Prevents runaway tool loops from blocking requests indefinitely
  - Environment variable: `<PREFIX>_TOOL_LOOP_TIMEOUT`

### Changed
- **Graceful Tool Loop Completion** - Improved behavior when hitting limits
  - When max iterations or timeout is reached, now makes one final backend call WITHOUT tools
  - Forces LLM to synthesize a proper response from all gathered tool results
  - Replaces the previous canned "I apologize" error message with actual content
  - New `_generate_final_response()` method handles the final synthesis call
  - Proper error handling if the final response call fails

## [0.7.0] - 2025-11-28

### Added
- **RAG Evaluation Module** - Comprehensive retrieval quality metrics
  - Precision, recall, and F1 score at configurable k values
  - Mean Reciprocal Rank (MRR) for ranking quality
  - Normalized Discounted Cumulative Gain (NDCG) for graded relevance
  - Context relevance scoring for RAG pipelines
  - Answer faithfulness and groundedness metrics
  - Implementation: `llm_api_server/eval/rag_evaluator.py`

## [0.6.4] - 2025-11-27

### Added
- **Structured Debug Logging** - JSON and YAML log formats for tool debugging
  - New `DEBUG_LOG_FORMAT` option: "text" (default), "json", or "yaml"
  - JSON format: Machine-parseable, one JSON object per log entry (use with `jq`)
  - YAML format: Human-readable structured format with literal blocks for multiline
  - New `DEBUG_LOG_MAX_RESPONSE_LENGTH` option: Max chars in logs, 0 = no truncation (default: 1000)
  - All log events include: timestamp, event type, tool name, duration, inputs/outputs
  - Event types: request, backend_call, tool_loop_iteration, tool_call, tool_error, etc.

## [0.6.3] - 2025-11-27

### Fixed
- **RAG Readability Empty Output** - Detect and handle readability failures
  - Now detects when readability returns empty/near-empty content (<100 bytes)
  - Falls back to semantic HTML extraction (article/main tags) when readability fails
  - Fixes extraction for sites like frequentmiler.com where readability returns nothing

## [0.6.2] - 2025-11-27

### Fixed
- **RAG Readability Fallback** - Smart content extraction when code blocks stripped
  - When readability strips >50% of code blocks, tries semantic HTML extraction
  - Extraction priority: `mdxContent` div → `<article>` → `<main>` → original HTML
  - For HashiCorp docs: reduces content from ~200KB to ~12KB while preserving all code
  - Works with any site using semantic HTML5 structure

## [0.6.1] - 2025-11-27

### Fixed
- **RAG Readability Extraction** - Removed overly aggressive byte-size fallback threshold
  - Previous 30% retention threshold caused fallback to original HTML on nearly all pages
  - Modern JS-heavy sites (Next.js, React) have 80-95% boilerplate, so low retention is expected
  - Now only falls back when code blocks are stripped (>50% loss)
  - Results in much smaller, cleaner indexed content without navigation/script noise

## [0.6.0] - 2025-11-27

### Added
- **CI Pipeline** - GitHub Actions workflow for automated testing
  - Runs on push/PR to main branch
  - Tests Python 3.11 and 3.12
  - Runs Black formatting check and Ruff linting
  - Runs full pytest suite

- **API Stability Documentation** - Added to README.md
  - Stable API surface definition
  - Semver compatibility guarantees
  - Support matrix (Python versions, backends, extras)

### Documentation
- Updated 1.0-readiness assessment with resolved status
- `think` parameter support explicitly deferred to post-1.0

## [0.5.1] - 2025-11-27

### Added
- **FAISS Security** - SHA256 checksum verification for FAISS index deserialization
  - Prevents malicious pickle injection attacks on cached indexes
  - Checksums saved on build, verified on load
  - Legacy indexes without checksums show warning but still load
  - Implementation: `rag/indexer.py:775-836`

- **RAG Module Tests** - Comprehensive unit tests for RAG components
  - `test_rag_chunker.py` - Semantic chunking parent/child relationships
  - `test_rag_crawler.py` - URL fetching, redirect blocking, content-type filtering
  - `test_rag_indexer.py` - Index loading and child-to-parent mapping
  - All tests run without network dependencies

- **Content Deduplication** - RAG indexer now deduplicates pages with identical content
  - Uses SHA256 content hashing to detect duplicates
  - Keeps first URL encountered, skips subsequent duplicates
  - Implementation: `rag/indexer.py:637-652`

### Changed
- **Cross-encoder Scores** - Now normalized to 0-1 range using min-max scaling
  - Ensures consistent scoring regardless of model characteristics
  - Implementation: `rag/indexer.py:991-1003`

- **Readability Fallback** - Now detects and prevents content loss
  - Falls back to original HTML if >30% content stripped
  - Falls back if code blocks are removed (technical docs protection)
  - Implementation: `rag/indexer.py:533-575`

### Documentation
- Improved RRF (Reciprocal Rank Fusion) documentation in `RAGConfig`
- Clarified that hybrid weights scale rank contributions, not scores

## [0.5.0] - 2025-11-27

### Added
- **Doc Search Tool Factory** - `create_doc_search_tool()` for creating RAG-powered tools
  - Wraps `DocSearchIndex` for LLM tool calling
  - Customizable tool name and description
  - Configurable parent context truncation via `RAGConfig.parent_context_max_chars`
  - Implementation: `builtin_tools.py:168-248`

- **Connection Pooling** - Backend requests now use `requests.Session`
  - Reuses HTTP connections for better performance
  - Module-level session shared across requests
  - Implementation: `backends.py:14-23`

### Changed
- Parent chunks without children are now indexed directly for search
  - Ensures all content is searchable even when blocks are too small for child chunks

### Fixed
- Index parent chunks that have no children (content was being lost)
- Capture intro content before first heading in RAG chunker

## [0.4.1] - 2025-11-26

### Changed
- Renamed `get_current_date()` to `get_current_datetime()` - now returns full date, time, and timezone in human-readable format (e.g., "Wednesday, November 26, 2025 at 2:30 PM PST")

### Fixed
- Fix XML element fallback logic in sitemap parser (empty XML elements are falsy, causing `or` fallback to fail)
- Don't cache empty crawl results to prevent persisting failed crawls
- Suppress noisy "ruthless removal did not work" messages from readability library

## [0.4.0] - 2025-11-25

### Added
- **RAG Module** - Comprehensive document search system
  - `DocSearchIndex` class for crawling, chunking, embedding, and searching documents
  - Three crawling modes: Sitemap (auto-discover) → Recursive (fallback) → Manual URLs
  - Semantic HTML chunking with parent-child relationships
  - Hybrid search combining BM25 keyword + semantic vector search
  - Cross-encoder re-ranking for improved relevance
  - Incremental updates with timestamp-based staleness detection
  - Local-first architecture with FAISS vector store
  - Robots.txt support with sitemap discovery
  - HTML page caching for faster index rebuilds
  - Requires optional `rag` extra: `uv sync --extra rag`
  - Implementation: `llm_api_server/rag/`

- **RAG Features**
  - Stateful crawling with resume capability and incremental indexing
  - Sitemap discovery from robots.txt with date-based prioritization
  - Readability-lxml integration for main content extraction
  - Configurable HTTP request timeout (default reduced from 30s to 10s)
  - 3-strike skip list for persistently failing URLs
  - Automatic max_pages expansion detection

### Changed
- RAG crawler now uses shorter 10s timeout by default for better responsiveness

### Fixed
- Graceful handling of robots.txt load failures in crawler
- ChunkMetadata JSON serialization and XML parsing warnings resolved
- Disabled tokenizers parallelism to prevent fork warnings
- Replaced deprecated `get_relevant_documents` with `invoke` in RAG indexer
- Proper component initialization before rebuilding retrievers in incremental updates

### Reverted
- Removed Open Web UI query filter experiment (was causing issues)

## [0.3.0] - 2025-11-22

### Added
- **Web Search Tool** - Optional built-in tool for web searching
  - Dual search strategy: Ollama web search API (premium) with DuckDuckGo fallback (free)
  - `create_web_search_tool(config)` factory function
  - Requires optional `websearch` extra: `uv sync --extra websearch`
  - Uses `OLLAMA_API_KEY` from config if available
  - Graceful fallback to DuckDuckGo when API unavailable or rate-limited
  - Site filtering support (e.g., `site:hashicorp.com query`)
  - Implementation: `llm_api_server/web_search_tool.py`

- **Enhanced HTML Reports** - Beautiful markdown-formatted evaluation reports
  - Full responses with no truncation (removed 500-character limit)
  - Markdown to HTML conversion with `markdown` library
  - Collapsible long responses (>300 chars) with expand/collapse buttons
  - Syntax highlighting for code blocks
  - Professional formatting for tables, lists, and blockquotes
  - Smooth CSS transitions and modern styling
  - Requires optional `eval` extra: `uv sync --extra eval`
  - Implementation: `llm_api_server/eval/reporters.py:84-460`

- **Configuration**
  - Added `OLLAMA_API_KEY` to `ServerConfig` for web search authentication
  - Environment variable support: `OLLAMA_API_KEY` or `<PREFIX>_OLLAMA_API_KEY`

### Changed
- HTML reports now display full responses instead of truncated previews
- Code blocks in HTML reports use dark theme with syntax highlighting
- Response sections are collapsible for better readability

### Documentation
- Updated README.md with web search tool usage and eval framework features
- Updated CLAUDE.md with developer documentation for new features
- Added installation instructions for `websearch` and `eval` extras

## [0.2.0] - 2025-11-22

### Added
- **Evaluation Framework** - Comprehensive testing framework for LLM applications
  - `TestCase` class for defining test criteria with keyword validation
  - `Evaluator` class for running tests against API endpoints
  - HTML report generator with visual pass/fail results
  - JSON report generator for CI/CD integration
  - Console reporter with color-coded output
  - Custom validator support for domain-specific validation
  - Performance metrics (response time, success rate)
  - Example evaluation script (`example_evaluation.py`)
  - Complete documentation in `llm_api_server/eval/README.md`

- **Backend Configuration** (from roadmap Tier 1 & 2)
  - Backend request timeouts (`BACKEND_CONNECT_TIMEOUT`, `BACKEND_READ_TIMEOUT`)
  - Backend health checks on startup (`HEALTH_CHECK_ON_STARTUP`, `HEALTH_CHECK_TIMEOUT`)
  - Configurable host binding (`DEFAULT_HOST`) - defaults to `127.0.0.1` for security

- **Request Validation** (from roadmap Tier 1)
  - JSON request validation with 400 Bad Request responses
  - Validates `messages` field existence and format
  - Better error messages for malformed requests

### Fixed
- **Critical Fixes** (from roadmap Tier 1)
  - Backend requests now have proper timeouts (prevents infinite hangs)
  - Improved exception handling - replaced bare `except:` with specific exception types
  - Fixed WebUI subprocess pipe handling (prevents buffer-filling hangs)
  - Better error handling for backend connection failures

### Changed
- **Breaking Change**: Default host binding changed from `0.0.0.0` to `127.0.0.1` for security
  - To bind to all interfaces (allow network access), set `HOST=0.0.0.0` or `<PREFIX>_HOST=0.0.0.0`
  - Security warning displayed when binding to `0.0.0.0`

### Security
- Default localhost-only binding (`127.0.0.1`) prevents unintended network exposure
- Health checks now timeout properly instead of hanging indefinitely

## [0.1.0] - 2025-11-21

### Initial Release

Core LLM API Server package extracted from Ivan project.

**Features:**
- OpenAI-compatible API (`/v1/chat/completions`, `/v1/models`, `/health`)
- Support for Ollama and LM Studio backends
- Tool calling with LangChain integration
- System prompt auto-reload
- Open WebUI integration
- Streaming and non-streaming responses
- Debug logging for tool execution

**Supported Backends:**
- Ollama (http://localhost:11434)
- LM Studio (http://localhost:1234/v1)
