# Claude Assistant Guide - LLM API Server

Developer guide for maintaining the LLM API Server framework.

## Quick Reference

```bash
# Setup (first time)
uv sync --extra dev             # Install all dependencies + dev tools

# Before every commit
./lint.sh                       # Format and lint

# Manual linting
uv run black llm_api_server/    # Format code
uv run ruff check --fix llm_api_server/  # Lint with auto-fix

# Running commands
uv run <command>                # Run any command in the project environment
```

## Project Overview

LLM API Server is a reusable Flask framework for building OpenAI-compatible API servers with:
- LM Studio and Ollama backend support
- LangChain tool calling integration
- Open Web UI integration
- Extensible configuration system

## Linting Routine

### Standard Workflow

Always run before committing:

```bash
./lint.sh
```

This script:
1. Formats code with Black (120 char lines)
2. Lints with Ruff (auto-fixes most issues)
3. Verifies all checks pass

### Configuration

All linting settings in `pyproject.toml`:
- **Black**: 120 character lines, Python 3.11+ support
- **Ruff**: Fast linter replacing flake8/isort/pylint
- **MyPy**: Optional type checking

## Development Guidelines

### Code Style
- Line length: 120 characters max
- Python version: 3.11+ compatibility
- Type hints: Use modern syntax where possible
- Imports: Auto-sorted by Ruff (stdlib → third-party → first-party)

### Package Structure

```
llm-api-server/
├── llm_api_server/
│   ├── __init__.py         # Package exports
│   ├── server.py           # Core LLMServer class
│   ├── backends.py         # Backend integrations
│   ├── config.py           # ServerConfig base class
│   ├── builtin_tools.py    # Built-in tools (date, calculate, web search)
│   ├── web_search_tool.py  # Web search implementation (Ollama + DuckDuckGo)
│   ├── webui.py            # Open Web UI integration
│   ├── eval/               # Evaluation framework module
│   │   ├── evaluator.py    # Test execution engine
│   │   ├── reporters.py    # HTML/JSON/Console reporters
│   │   └── ...
│   └── rag/                # RAG document search module
│       ├── config.py       # RAGConfig dataclass
│       ├── crawler.py      # Web crawling (sitemap + recursive + manual)
│       ├── chunker.py      # Semantic HTML chunking
│       ├── indexer.py      # Main DocSearchIndex class
│       └── __init__.py     # Module exports
├── setup.py                # Package installation
├── pyproject.toml          # Packaging & linting config
└── README.md               # Package documentation
```

### Making Changes

1. **Core server** (`server.py`): Flask app, routing, tool calling loop
2. **Backends** (`backends.py`): Ollama/LM Studio communication
3. **Config** (`config.py`): Configuration and environment loading
4. **Built-in tools** (`builtin_tools.py`): Common tools (date, calculate, web search factory)
5. **Web search** (`web_search_tool.py`): Ollama API web search implementation
6. **Web UI** (`webui.py`): Open Web UI subprocess management

### Adding Features

When adding new features, consider:
- **Backwards compatibility**: This is used by multiple projects
- **Configuration options**: Make features configurable
- **Documentation**: Update README.md and docstrings
- **Examples**: Update consuming projects (Ivan, milesoss)

## Testing

Since this is a framework library:

1. **Local testing**: Install in consuming project
   ```bash
   cd ../milesoss  # or ../Ivan
   uv sync  # Will pull llm-api-server from GitHub
   uv run python milesoss.py --no-webui
   ```

2. **Integration testing**: Verify in both Ivan and milesoss

3. **API testing**: Test OpenAI-compatible endpoints
   ```bash
   curl http://localhost:8000/v1/models
   curl http://localhost:8000/health
   ```

## Installation Options

```bash
# Using uv (recommended)
uv sync                   # Install core dependencies
uv sync --extra dev       # With development tools
uv sync --extra webui     # With Open Web UI
uv sync --extra websearch # With web search tool
uv sync --extra rag       # With RAG document search module
uv sync --extra eval      # With HTML report markdown formatting
uv sync --all-extras      # Everything

# Using pip (legacy)
pip install -e .
pip install -e '.[dev]'
pip install -e '.[webui]'
```

## Git Workflow

Standard GitHub workflow:

```bash
# Make changes
./lint.sh  # Format and lint

# Commit
git add .
git commit -m "feat: description

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"

# Push to GitHub
git push
```

Changes are now distributed via GitHub. Consuming projects install with:
```
llm-api-server @ git+https://github.com/assareh/llm-api-server.git
```

## Key Components

### LLMServer Class
Main server class that consuming projects instantiate:
```python
server = LLMServer(
    name="MyApp",
    model_name="myapp/model",
    tools=ALL_TOOLS,
    config=config,
    default_system_prompt="You are...",
    init_hook=initialization_function,
    logger_names=["myapp.tools"]
)
```

### ServerConfig Base Class
Extensible configuration loaded from environment:
```python
class MyConfig(ServerConfig):
    CUSTOM_SETTING: str = "default"

config = ServerConfig.from_env("MYAPP_")
```

**Key configuration options:**
- `MAX_TOOL_ITERATIONS`: Maximum tool calling loop iterations (default: 5)
- `RATE_LIMIT_ENABLED`: Enable API rate limiting (default: False)
- `RATE_LIMIT_DEFAULT`: Rate limit string, e.g., "100 per minute"
- `RATE_LIMIT_STORAGE_URI`: Storage backend for rate limiting (default: "memory://")

**Debug logging options:**
- `DEBUG_TOOLS`: Enable debug logging for tool calls (default: False)
- `DEBUG_TOOLS_LOG_FILE`: Log file path (default: "tools_debug.log")
- `DEBUG_LOG_FORMAT`: Log format - "text", "json", or "yaml" (default: "text")
- `DEBUG_LOG_MAX_RESPONSE_LENGTH`: Max response chars in logs, 0 = no truncation (default: 1000)

**Structured log formats:**
- `text`: Human-readable format with separators and labeled fields
- `json`: Machine-parseable, one JSON object per log entry (use with `jq`)
- `yaml`: Human-readable structured format with literal blocks for multiline

Example JSON log parsing:
```bash
# Filter by tool name
cat tools_debug.log | jq 'select(.tool == "doc_search")'

# Find slow tool calls (>5 seconds)
cat tools_debug.log | jq 'select(.duration_ms > 5000)'

# Get full responses without truncation
DEBUG_LOG_FORMAT=json DEBUG_LOG_MAX_RESPONSE_LENGTH=0
```

### Backend Support
- **Ollama**: Native Ollama API format
- **LM Studio**: OpenAI-compatible format
- **Tool calling**: Automatic conversion and execution

### Built-in Tools

The framework provides reusable tools:

**Always available:**
- `get_current_datetime()` - Returns current date and time in local timezone
- `calculate(expression)` - Safe mathematical expression evaluator

**Optional (requires `--extra websearch` and `OLLAMA_API_KEY`):**
- `create_web_search_tool(config)` - Web search using Ollama API:
  - Requires `OLLAMA_API_KEY` to be configured
  - Supports site filtering: `site:hashicorp.com query`
  - Implementation: `llm_api_server/web_search_tool.py`

**Optional (requires `--extra rag`):**
- `create_doc_search_tool(index, name, description)` - RAG document search tool:
  - Wraps a `DocSearchIndex` for LLM tool calling
  - Hybrid search using Reciprocal Rank Fusion (BM25 + semantic)
  - Cross-encoder re-ranking for final result ordering
  - Returns formatted results with source URLs and parent context
  - Customizable tool name and description per use case
  - Implementation: `llm_api_server/builtin_tools.py`

**Usage:**
```python
from llm_api_server import BUILTIN_TOOLS, create_web_search_tool, create_doc_search_tool
from llm_api_server.rag import DocSearchIndex, RAGConfig

# With web search
web_search = create_web_search_tool(config)

# With RAG doc search
rag_config = RAGConfig(base_url="https://docs.example.com", cache_dir="./doc_index")
index = DocSearchIndex(rag_config)
index.crawl_and_index()
doc_search = create_doc_search_tool(index, description="Search Example docs")

tools = BUILTIN_TOOLS + [web_search, doc_search]
```

### Evaluation Framework

The framework includes a comprehensive evaluation system in `llm_api_server/eval/`:

**Components:**
- `Evaluator` - Runs test cases against LLM API
- `TestCase` - Defines questions and validation rules
- `TestResult` - Contains test outcomes and metrics
- `HTMLReporter` - Generates beautiful HTML reports with markdown formatting
- `JSONReporter` - Machine-readable JSON output
- `ConsoleReporter` - Terminal-friendly output

**HTML Report Features (requires `--extra eval`):**
- **Markdown to HTML conversion** using `markdown` library
- **Full responses** - No truncation, all content visible
- **Collapsible sections** - Long responses start collapsed with expand/collapse buttons
- **Syntax highlighting** - Code blocks, tables, lists, blockquotes
- **Professional styling** - Dark code blocks, formatted tables, styled blockquotes
- Implementation: `llm_api_server/eval/reporters.py:84-460`

**Key files:**
- `evaluator.py` - Test execution engine
- `test_case.py` - Data models for tests and results
- `reporters.py` - HTML/JSON/Console report generation
- `validators.py` - Response validation logic

See `llm_api_server/eval/README.md` for complete documentation.

### RAG Module (Document Search)

The framework includes a comprehensive RAG system in `llm_api_server/rag/`:

**Components:**
- `DocSearchIndex` - Main indexer with crawling, chunking, embedding, and search
- `RAGConfig` - Configuration dataclass with all settings
- `DocumentCrawler` - Web crawler (sitemap + recursive + manual URLs)
- `semantic_chunk_html()` - HTML-aware chunking with parent-child relationships

**Features:**
- **Three crawling modes**: Sitemap (auto-discover) → Recursive (fallback) → Manual (explicit)
- **Semantic HTML chunking** - Respects document structure (headings, code, tables)
- **Parent-child chunks** - Hierarchical relationships for context
- **Hybrid search** - BM25 + semantic via Reciprocal Rank Fusion (RRF)
- **Cross-encoder re-ranking** - MS MARCO model for accurate final ordering
- **Incremental updates** - Check timestamps, only rebuild if stale
- **Local-first** - FAISS vector store, HuggingFace embeddings (all-MiniLM-L6-v2)

**Architecture:**
1. **Crawler** discovers URLs (sitemap XML parsing, recursive link following, or manual list)
2. **Chunker** processes HTML into parent-child chunks using heading hierarchy
3. **Indexer** generates embeddings and builds FAISS + BM25 indexes
4. **Searcher** uses ensemble retriever (hybrid) + cross-encoder re-ranking

**Key files:**
- `config.py` - RAGConfig with all crawling/chunking/search settings
- `crawler.py` - DocumentCrawler class, robots.txt support, URL filtering
- `chunker.py` - semantic_chunk_html(), token-aware chunking with tiktoken
- `indexer.py` - DocSearchIndex main class

**Usage:**
```python
from llm_api_server.rag import DocSearchIndex, RAGConfig

config = RAGConfig(
    base_url="https://docs.example.com",
    cache_dir="./doc_index",
    # Hybrid search uses Reciprocal Rank Fusion (RRF), not weighted average.
    # These weights scale rank contributions: 0.7/0.3 means semantic ranks
    # are weighted ~2.3x more than BM25 ranks in the fusion formula.
    hybrid_bm25_weight=0.3,
    hybrid_semantic_weight=0.7,
)

index = DocSearchIndex(config)
index.crawl_and_index()
results = index.search("query", top_k=5)
```

**Ported from Ivan:**
This module was generalized from Ivan's HashiCorp doc search implementation:
- Removed HashiCorp-specific logic (product extraction, domain-specific synonyms)
- Made crawling more general (works with any doc site)
- Kept all advanced features (parent-child chunking, hybrid search, re-ranking)
- Added configurable crawling modes (sitemap + recursive + manual)

**Dependencies (optional `--extra rag`):**
- langchain, langchain-community, langchain-huggingface
- faiss-cpu (vector store)
- sentence-transformers (cross-encoder re-ranking)
- beautifulsoup4 (HTML parsing)
- tiktoken (token counting)
- rank-bm25 (keyword search)

## Consuming Projects

Current projects using this framework:
- **Ivan**: HashiCorp documentation assistant
- **milesoss**: Credit card rewards optimizer

When making changes, test in both projects.

## Resources

- [README.md](README.md) - Package documentation
- [Black Docs](https://black.readthedocs.io/)
- [Ruff Docs](https://docs.astral.sh/ruff/)
- [Flask](https://flask.palletsprojects.com/)
- [LangChain](https://python.langchain.com/)

---

*Last updated: 2025-11-26*
*Version: 0.7.0*
