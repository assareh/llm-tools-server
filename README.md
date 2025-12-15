# LLM API Server

A reusable Flask server providing an OpenAI-compatible API for local LLM backends (Ollama, LM Studio) with tool calling support.

**Requirements:** Python 3.11+

## API Stability

This project follows [Semantic Versioning](https://semver.org/). Starting with version 1.0:

- **Stable API surface:**
  - `LLMServer` class instantiation and `run()` method
  - `ServerConfig` class and `from_env()` factory
  - Built-in tools: `BUILTIN_TOOLS`, `get_current_datetime`, `calculate`, `create_web_search_tool`, `create_doc_search_tool`
  - RAG module: `DocSearchIndex`, `RAGConfig`, `search()` method
  - Evaluation framework: `Evaluator`, `TestCase`, `HTMLReporter`, `JSONReporter`
  - REST endpoints: `/health`, `/v1/models`, `/v1/chat/completions`

- **Compatibility guarantees:**
  - Minor versions (1.x) maintain backward compatibility
  - Breaking changes only in major versions with migration guides
  - Deprecated features announced at least one minor version before removal

- **Support matrix:**
  - Python: 3.11, 3.12
  - Backends: Ollama, LM Studio
  - Optional extras: `webui`, `websearch`, `rag`, `eval`, `dev`

## Features

- **OpenAI-compatible API** - Drop-in replacement for OpenAI's `/v1/chat/completions` endpoint
- **Multiple backends** - Supports Ollama and LM Studio
- **Tool calling** - Full support for function calling with LangChain tools
- **Streaming responses** - Real-time token streaming
- **RAG module** - Local documentation search with hybrid retrieval, semantic chunking, and re-ranking
- **Evaluation framework** - Test and validate LLM responses with beautiful HTML reports
  - Full markdown-formatted responses (no truncation)
  - Collapsible long responses with syntax highlighting
  - Professional styling for code, tables, and lists
- **Built-in tools** - Date/time, calculator, and web search (via Ollama API)
- **WebUI integration** - Optional Open Web UI frontend
- **Smart caching** - System prompt auto-reload on file changes
- **Debug logging** - Comprehensive tool execution logging
- **Graceful tool loop completion** - When hitting iteration/timeout limits, generates a synthesized response

## Installation

### Using uv (recommended)

```bash
# Clone and install
git clone https://github.com/assareh/llm-tools-server.git
cd llm-tools-server
uv sync

# With optional dependencies
uv sync --extra webui      # For Open Web UI support
uv sync --extra websearch  # For web search tool
uv sync --extra rag        # For RAG document search module
uv sync --extra eval       # For HTML reports with markdown formatting
uv sync --extra dev        # For development tools
uv sync --all-extras       # Install everything
```

### Using pip

```bash
pip install llm-tools-server
```

Or install from source:

```bash
git clone https://github.com/assareh/llm-tools-server.git
cd llm-tools-server
pip install -e .
```

### Optional dependencies

```bash
# For Open Web UI support
pip install llm-tools-server[webui]

# For web search tool
pip install llm-tools-server[websearch]

# For RAG document search module
pip install llm-tools-server[rag]

# For HTML reports with markdown formatting
pip install llm-tools-server[eval]

# For development
pip install llm-tools-server[dev]
```

## Quick Start

```python
from llm_tools_server import LLMServer, ServerConfig
from langchain_core.tools import tool

# Define your tools
@tool
def get_weather(location: str) -> str:
    """Get weather for a location."""
    return f"Weather in {location}: Sunny, 72°F"

ALL_TOOLS = [get_weather]

# Configure server
config = ServerConfig.from_env("MYAPP_")  # Reads MYAPP_BACKEND, MYAPP_PORT, etc.
config.BACKEND_TYPE = "lmstudio"
config.BACKEND_MODEL = "openai/gpt-oss-20b"
config.MODEL_NAME = "myapp/assistant"
config.SYSTEM_PROMPT_PATH = "system_prompt.md"

# Create and run server
server = LLMServer(
    name="MyApp",
    model_name=config.MODEL_NAME,
    tools=ALL_TOOLS,
    config=config,
    default_system_prompt="You are a helpful assistant."
)

if __name__ == "__main__":
    server.run(port=8000)
```

## Built-in Tools

LLM API Server includes common tools that you can use out of the box:

### Using Built-in Tools

```python
from llm_tools_server import LLMServer, BUILTIN_TOOLS, ServerConfig
from llm_tools_server import get_current_datetime, calculate, create_web_search_tool
from langchain_core.tools import tool

config = ServerConfig.from_env("MYAPP_")

# Option 1: Use all built-in tools
server = LLMServer(
    name="MyApp",
    model_name="myapp/assistant",
    tools=BUILTIN_TOOLS,
    config=config
)

# Option 2: Import specific tools
server = LLMServer(
    name="MyApp",
    model_name="myapp/assistant",
    tools=[get_current_datetime, calculate],
    config=config
)

# Option 3: Add web search tool (requires websearch extra)
web_search = create_web_search_tool(config)  # Uses config.OLLAMA_API_KEY
server = LLMServer(
    name="MyApp",
    model_name="myapp/assistant",
    tools=BUILTIN_TOOLS + [web_search],
    config=config
)

# Option 4: Combine built-in tools with custom tools
@tool
def get_weather(location: str) -> str:
    """Get weather for a location."""
    return f"Weather in {location}: Sunny, 72°F"

server = LLMServer(
    name="MyApp",
    model_name="myapp/assistant",
    tools=BUILTIN_TOOLS + [get_weather],  # Combine both
    config=config
)
```

### Available Built-in Tools

- **`get_current_datetime()`** - Returns the current date and time in local timezone (e.g., "Wednesday, November 26, 2025 at 2:30 PM PST")
- **`calculate(expression: str)`** - Safely evaluates mathematical expressions
  - Supports: `+`, `-`, `*`, `/`, `//`, `%`, `**` (power)
  - Example: `calculate("2 + 3 * 4")` → `"14"`
- **`create_web_search_tool(config)`** - Web search using Ollama API
  - Requires optional `websearch` dependency: `uv sync --extra websearch`
  - Requires `OLLAMA_API_KEY` to be configured
  - Parameters: `query`, `max_results` (default 5), `site` (optional filter)
- **`create_doc_search_tool(index, name, description)`** - Search local RAG index
  - Requires optional `rag` dependency: `uv sync --extra rag`
  - Parameters: `query`, `top_k` (default 5)

### Sample System Prompt Content

When using built-in tools, include relevant instructions in your system prompt. Here are templates you can customize:

**Basic Tools (datetime, calculate):**

```markdown
## Available Tools

You have access to the following tools:

- **get_current_datetime**: Get the current date and time. Use this when users ask about the current time, today's date, or time-sensitive questions.

- **calculate**: Evaluate mathematical expressions. Use for any arithmetic calculations. Supports +, -, *, /, // (floor division), % (modulo), ** (power), and parentheses.
  Examples: "2 + 3 * 4" → 14, "2 ** 8" → 256, "(10 + 5) / 3" → 5
```

**With Web Search:**

```markdown
## Available Tools

- **web_search**: Search the web for current information. Use for:
  - Recent events or news
  - Current documentation or tutorials
  - Stack Overflow answers
  - Any information that may have changed since your training

  Parameters:
  - query: Search terms
  - max_results: Number of results (default 5)
  - site: Optional domain filter (e.g., "stackoverflow.com")
```

**With Document Search (RAG):**

```markdown
## Available Tools

- **doc_search**: Search the indexed documentation. ALWAYS use this tool first when users ask questions about [YOUR PRODUCT/DOCS]. This searches locally indexed documentation and returns relevant excerpts with source URLs.

  Parameters:
  - query: What to search for
  - top_k: Number of results (default 5)

When answering questions:
1. First search the documentation using doc_search
2. If the documentation doesn't have the answer, say so clearly
3. Always cite your sources with the URLs provided
```

**Complete Example (All Tools):**

```markdown
## Available Tools

You have access to these tools to help answer questions:

### get_current_datetime
Get the current date and time in local timezone.

### calculate
Safely evaluate mathematical expressions. Supports +, -, *, /, //, %, ** and parentheses.

### doc_search
Search [YOUR PRODUCT] documentation. Use this FIRST for any questions about [YOUR PRODUCT].

### web_search
Search the web for general information. Use for topics not covered in the documentation.

## Guidelines

1. For [YOUR PRODUCT] questions: Always search documentation first with doc_search
2. For calculations: Use the calculate tool instead of computing manually
3. For current events or external topics: Use web_search
4. Always cite sources when using search tools
```

## RAG Module (Document Search)

LLM API Server includes a RAG (Retrieval-Augmented Generation) module for building local documentation search systems with hybrid retrieval and cross-encoder re-ranking.

### Installation

```bash
# Install with RAG dependencies
uv sync --extra rag
```

### Architecture Overview

The RAG module uses a four-component architecture:

1. **DocumentCrawler** (`crawler.py`) - URL discovery with sitemap/recursive/manual modes
2. **Chunker** (`chunker.py`) - Semantic HTML parsing with parent-child relationships
3. **DocSearchIndex** (`indexer.py`) - Main orchestrator for indexing and search
4. **RAGConfig** (`config.py`) - Comprehensive configuration with sensible defaults

### Key Features

**Intelligent Crawling:**
- **Three modes with smart fallback:** Sitemap → Recursive → Manual
- **Robots.txt compliance** with intelligent parsing
- **URL normalization** (removes query params, anchors, trailing slashes)
- **Include/exclude patterns** with regex support
- **Parallel fetching** with configurable rate limiting

**Semantic HTML Chunking:**
- **Token-aware processing** using tiktoken (matches GPT tokenization)
- **Preserves atomic content** - code blocks and tables kept intact
- **Parent-child hierarchy** - focused retrieval with full context recovery
- **Heading-based sectioning** - respects document structure (h1-h6)
- **Boilerplate removal** - CSS selector-based filtering of nav, footer, etc.

**Hybrid Search:**
- **BM25 keyword search** - captures exact term matches
- **Semantic vector search** - captures conceptual similarity
- **Reciprocal Rank Fusion (RRF)** - combines rankings from both retrievers
- **Configurable weights** - tune keyword vs semantic emphasis in RRF formula

**Cross-Encoder Re-ranking:**
- **MS MARCO cross-encoder** for accurate final ranking
- **Min-max score normalization** for consistent ranking
- **Configurable candidate pool** (rerank_top_k)

**Production-Ready Features:**
- **Incremental indexing** - resume interrupted crawls, add new docs efficiently
- **SHA256 checksums** - index integrity verification
- **Content deduplication** - by hash before indexing
- **GPU auto-detection** - MPS (Apple Silicon), CUDA, or CPU fallback
- **Rich metadata** - heading path, section ID, document type, code identifiers

**Local-First Design:**
- **FAISS** for vector storage (no cloud dependencies)
- **HuggingFace embeddings** - configurable from fast (MiniLM) to accurate (BGE-large)
- **Everything runs locally** - no API keys required for core functionality

**Contextual Retrieval (Optional):**
- **LLM-generated context** - Anthropic's approach for ~40-50% better retrieval
- **Background processing** - index usable immediately while contexts generate
- **Resumable** - progress saved every 50 chunks, can interrupt and resume
- **Uses server backend** - same LM Studio/Ollama configured for requests

### Quick Start

```python
from llm_tools_server.rag import DocSearchIndex, RAGConfig

# Configure RAG
config = RAGConfig(
    base_url="https://docs.example.com",
    cache_dir="./doc_index",
    # Optional: manual URLs (additive or exclusive)
    manual_urls=["https://docs.example.com/important-page"],
    manual_urls_only=False,  # False = add to crawled URLs, True = only index these
)

# Build index (first time)
index = DocSearchIndex(config)
index.crawl_and_index()

# Search
results = index.search("How do I configure authentication?", top_k=5)

for result in results:
    print(f"Score: {result['score']:.3f}")
    print(f"URL: {result['url']}")
    print(f"Section: {result['heading_path']}")
    print(f"Text: {result['text'][:200]}...")
    if 'parent_text' in result:
        print(f"Parent context: {result['parent_text'][:200]}...")
    print()
```

### Configuration Options

```python
config = RAGConfig(
    base_url="https://docs.example.com",
    cache_dir="./doc_index",

    # Crawling settings
    manual_urls=["https://..."],           # Optional list of specific URLs
    manual_urls_only=False,                # True = only index manual URLs
    max_crawl_depth=3,                     # Maximum recursion depth
    rate_limit_delay=0.1,                  # Seconds between requests
    max_workers=5,                         # Parallel fetching threads
    max_pages=None,                        # Limit total pages (None = unlimited)
    request_timeout=10.0,                  # HTTP request timeout in seconds
    max_url_retries=3,                     # Skip URLs after N consecutive failures
    url_include_patterns=["docs/.*"],      # Regex patterns to include
    url_exclude_patterns=[".*/api/.*"],    # Regex patterns to exclude

    # Chunking settings
    child_chunk_size=350,                  # Tokens per child chunk
    child_chunk_overlap=50,                # Overlap tokens between child chunks
    parent_chunk_size=900,                 # Tokens per parent chunk
    parent_chunk_overlap=100,              # Overlap tokens between parent chunks

    # Search settings
    hybrid_bm25_weight=0.3,                # BM25 keyword search weight
    hybrid_semantic_weight=0.7,            # Semantic vector search weight
    search_top_k=5,                        # Default results to return
    rerank_enabled=True,                   # Enable cross-encoder re-ranking
    rerank_top_k=80,                       # Candidates for re-ranking

    # Model settings (embedding model: speed vs quality tradeoff)
    # Options:
    #   - "sentence-transformers/all-MiniLM-L6-v2": Fast (22M params), good quality (default)
    #   - "BAAI/bge-base-en-v1.5": Medium (110M params), better quality
    #   - "BAAI/bge-large-en-v1.5": Slow (335M params), best quality
    # Note: Changing embedding model requires full index rebuild
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    rerank_model="cross-encoder/ms-marco-MiniLM-L-12-v2",  # Cross-encoder for re-ranking

    # Contextual retrieval settings (optional, requires server_config)
    contextual_retrieval_enabled=False,    # Enable LLM-generated context for chunks
    contextual_retrieval_background=False, # Run in background thread (index usable immediately)
    contextual_max_workers=4,              # Parallel context generation workers

    # Index settings
    update_check_interval_hours=168,       # Check for updates (7 days)
    page_cache_ttl_hours=168,              # TTL for cached pages without lastmod (7 days, 0 = never)

    # Progress display
    show_progress=True,                    # Show tqdm progress bars
)
```

### Crawling Modes

The RAG module automatically selects the best crawling strategy:

1. **Default behavior:** Tries to discover `sitemap.xml` → uses if found
2. **Fallback:** If no sitemap → recursive crawling from base_url
3. **Manual mode:** `manual_urls_only=True` → only index specified URLs

### Incremental Updates

```python
# Check if update needed
if index.needs_update():
    print("Index is stale, rebuilding...")
    index.crawl_and_index()
else:
    print("Index is up-to-date")
    index.load_index()

# Force rebuild (clears all state, re-crawls everything)
index.crawl_and_index(force_rebuild=True)

# Force refresh (refetches all cached pages, bypasses TTL)
index.crawl_and_index(force_refresh=True)
```

### Cache Invalidation

Pages are automatically refreshed based on:

1. **Sitemap `lastmod`** - Pages with `lastmod` in sitemap are refetched when the date changes
2. **TTL expiration** - Pages without `lastmod` expire after `page_cache_ttl_hours` (default: 7 days)
3. **Manual refresh** - Use `force_refresh=True` to bypass all caching

### Advanced: Hybrid Search Weights

Adjust the balance between keyword and semantic search using Reciprocal Rank Fusion (RRF).
The weights scale each retriever's contribution to the RRF score: `score += weight / (rank + 60)`.

```python
# More keyword-focused (good for technical docs with specific terms)
config = RAGConfig(
    base_url="https://docs.example.com",
    hybrid_bm25_weight=0.5,      # Equal weight to BM25 ranks
    hybrid_semantic_weight=0.5,  # Equal weight to semantic ranks
)

# More semantic-focused (good for concept-based queries)
config = RAGConfig(
    base_url="https://docs.example.com",
    hybrid_bm25_weight=0.2,      # Lower influence from BM25 ranks
    hybrid_semantic_weight=0.8,  # Higher influence from semantic ranks
)
```

### Advanced: Contextual Retrieval

Contextual retrieval prepends LLM-generated context to each chunk before embedding, improving retrieval accuracy by ~40-50% (Anthropic's approach).

```python
from llm_tools_server import ServerConfig
from llm_tools_server.rag import DocSearchIndex, RAGConfig

# Get server config for backend settings
server_config = ServerConfig.from_env()

# Enable contextual retrieval
rag_config = RAGConfig(
    base_url="https://docs.example.com",
    contextual_retrieval_enabled=True,      # Enable context generation
    contextual_retrieval_background=True,   # Run in background (recommended)
)

# Pass server_config to use same backend for context generation
index = DocSearchIndex(rag_config, server_config)
index.crawl_and_index()
index.load_index()  # Starts background contextualization if enabled
```

Or run contextual retrieval explicitly (can be interrupted and resumed):

```python
index.crawl_and_index()              # Build index first (fast)
index.add_contextual_retrieval()     # Run separately, saves progress every 50 chunks
```

Reference: https://www.anthropic.com/news/contextual-retrieval

### Advanced: Embedding Models

Change embedding model without re-crawling:

```python
config = RAGConfig(
    base_url="https://docs.example.com",
    embedding_model="BAAI/bge-large-en-v1.5",  # Upgrade to better model
)
index = DocSearchIndex(config)
index.crawl_and_index()  # Detects model change, rebuilds embeddings only
```

Or explicitly rebuild embeddings:

```python
index.rebuild_embeddings()  # Uses saved chunks, skips crawling
```

## Configuration

### Using ServerConfig

```python
from llm_tools_server import ServerConfig

# Create from environment variables
config = ServerConfig.from_env("MYAPP_")

# Or configure directly
config = ServerConfig()
config.BACKEND_TYPE = "ollama"  # or "lmstudio"
config.BACKEND_MODEL = "openai/gpt-oss-20b"
config.OLLAMA_ENDPOINT = "http://localhost:11434"
config.DEFAULT_PORT = 8000
config.DEFAULT_TEMPERATURE = 0.0
```

### Environment Variables

With prefix `MYAPP_`:

- `MYAPP_BACKEND` - Backend type (ollama, lmstudio)
- `MYAPP_BACKEND_MODEL` - Model identifier
- `MYAPP_PORT` - Server port (default: 8000)
- `MYAPP_TEMPERATURE` - Default temperature (default: 0.0)
- `MYAPP_SYSTEM_PROMPT_PATH` - Path to system prompt file
- `MYAPP_DEBUG_TOOLS` - Enable tool debug logging (true/false)
- `MYAPP_MAX_TOOL_ITERATIONS` - Maximum tool loop iterations (default: 5)
- `MYAPP_TOOL_LOOP_TIMEOUT` - Maximum seconds for tool loop (default: 120, 0 = no timeout)
- `OLLAMA_ENDPOINT` - Ollama API endpoint
- `OLLAMA_API_KEY` - Ollama API key for web search (optional)
- `LMSTUDIO_ENDPOINT` - LM Studio API endpoint

## API Endpoints

### `GET /health`

Health check endpoint.

```bash
curl http://localhost:8000/health
```

### `GET /v1/models`

List available models.

```bash
curl http://localhost:8000/v1/models
```

### `POST /v1/chat/completions`

OpenAI-compatible chat completions endpoint.

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "myapp/assistant",
    "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
    "stream": false
  }'
```

**Per-Request Model Override:** You can specify a different backend model per request by setting the `model` field to a model name different from the server's configured `model_name`. This passes through to LM Studio/Ollama, allowing dynamic model selection:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5:7b",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Evaluation Framework

LLM API Server includes a comprehensive evaluation framework for testing your LLM applications.

### Quick Example

```python
from llm_tools_server.eval import Evaluator, TestCase, HTMLReporter

# Define test cases
tests = [
    TestCase(
        question="What is 2+2?",
        description="Basic arithmetic test",
        expected_keywords=["4", "four"],
        min_response_length=10
    ),
    TestCase(
        question="What are the best travel credit cards?",
        description="Domain knowledge test",
        expected_keywords=["chase", "sapphire", "travel"],
        min_response_length=100
    )
]

# Run evaluation
evaluator = Evaluator(api_url="http://localhost:8000")
results = evaluator.run_tests(tests)

# Generate HTML report
reporter = HTMLReporter()
reporter.generate(results, "evaluation_report.html")

# Print summary
summary = evaluator.get_summary(results)
print(f"Success Rate: {summary['success_rate']:.1f}%")
```

### Features

- **Flexible validation** - Expected keywords, response length, custom validators
- **Beautiful HTML reports** - Markdown-formatted responses with syntax highlighting
  - Full responses (no truncation)
  - Collapsible long responses with expand/collapse buttons
  - Code blocks with syntax highlighting
  - Tables, lists, and blockquote formatting
  - Requires optional `eval` extra: `uv sync --extra eval`
- **Multiple report formats** - HTML, JSON, and console output
- **Performance tracking** - Response times and success rates
- **Custom validators** - Write domain-specific validation logic
- **CI/CD ready** - JSON reports and exit codes for automation

### Documentation

See [`llm_tools_server/eval/README.md`](llm_tools_server/eval/README.md) for complete documentation and examples.

Run the example script:
```bash
python example_evaluation.py
```

## Advanced Usage

### Custom Initialization Hook

```python
def init_database():
    print("Initializing database...")
    # Your initialization code here

server = LLMServer(
    name="MyApp",
    model_name="myapp/assistant",
    tools=ALL_TOOLS,
    config=config,
    init_hook=init_database  # Called before server starts
)
```

### Custom Logger Names

```python
server = LLMServer(
    name="MyApp",
    model_name="myapp/assistant",
    tools=ALL_TOOLS,
    config=config,
    logger_names=["myapp.tools", "myapp.backend", "tools"]
)
```

### System Prompt Auto-Reload

Create a `system_prompt.md` file:

```markdown
You are MyApp, an AI assistant specialized in...
```

The server automatically reloads this file when it changes (based on modification time).

## Development

### Using uv (recommended)

```bash
# Clone and install with dev dependencies
git clone https://github.com/assareh/llm-tools-server.git
cd llm-tools-server
uv sync --extra dev

# Run tests
uv run pytest

# Format and lint
./lint.sh

# Or manually
uv run black llm_tools_server/
uv run ruff check --fix llm_tools_server/
```

### Using pip

```bash
# Clone the repository
git clone https://github.com/assareh/llm-tools-server.git
cd llm-tools-server

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black llm_tools_server/

# Lint
ruff check --fix llm_tools_server/
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

Built with:
- [Flask](https://flask.palletsprojects.com/)
- [LangChain](https://www.langchain.com/)
- [Open Web UI](https://github.com/open-webui/open-webui)

## Support

If you find this project helpful, consider supporting its development:

[![Buy Me A Coffee](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/assareh)

Your support helps maintain and improve this project!
