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

## Installation

### Using uv (recommended)

```bash
# Clone and install
git clone https://github.com/assareh/llm-api-server.git
cd llm-api-server
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
pip install llm-api-server
```

Or install from source:

```bash
git clone https://github.com/assareh/llm-api-server.git
cd llm-api-server
pip install -e .
```

### Optional dependencies

```bash
# For Open Web UI support
pip install llm-api-server[webui]

# For web search tool
pip install llm-api-server[websearch]

# For RAG document search module
pip install llm-api-server[rag]

# For HTML reports with markdown formatting
pip install llm-api-server[eval]

# For development
pip install llm-api-server[dev]
```

## Quick Start

```python
from llm_api_server import LLMServer, ServerConfig
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
from llm_api_server import LLMServer, BUILTIN_TOOLS, ServerConfig
from llm_api_server import get_current_datetime, calculate, create_web_search_tool
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
  - Parameters: `query`, `max_results` (default 10), `site` (optional filter)

## RAG Module (Document Search)

LLM API Server includes a powerful RAG (Retrieval-Augmented Generation) module for building local documentation search systems.

### Installation

```bash
# Install with RAG dependencies
uv sync --extra rag
```

### Features

- **Three crawling modes:**
  - Sitemap-based (discovers sitemap.xml automatically)
  - Recursive web crawling (follows links)
  - Manual URL list (explicit control)
- **Semantic HTML chunking** - Respects document structure (headings, code blocks, tables)
- **Parent-child chunks** - Hierarchical context for better retrieval
- **Hybrid search** - Combines BM25 keyword search + semantic vector search
- **Cross-encoder re-ranking** - Two-stage re-ranking for improved relevance
- **Incremental updates** - Only re-indexes changed content
- **Local-first** - Everything runs locally (FAISS, HuggingFace embeddings)

### Quick Start

```python
from llm_api_server.rag import DocSearchIndex, RAGConfig

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

    # Model settings
    embedding_model="all-MiniLM-L6-v2",    # HuggingFace model (~80MB)
    rerank_model="cross-encoder/ms-marco-MiniLM-L-12-v2",      # Heavy cross-encoder
    light_rerank_model="cross-encoder/ms-marco-MiniLM-L-6-v2", # Light cross-encoder

    # Index settings
    update_check_interval_hours=168,       # Check for updates (7 days)
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

# Force rebuild
index.crawl_and_index(force_rebuild=True)
```

### Advanced: Hybrid Search Weights

Adjust the balance between keyword and semantic search:

```python
# More keyword-focused (good for technical docs with specific terms)
config = RAGConfig(
    base_url="https://docs.example.com",
    hybrid_bm25_weight=0.5,      # 50% keyword
    hybrid_semantic_weight=0.5,  # 50% semantic
)

# More semantic-focused (good for concept-based queries)
config = RAGConfig(
    base_url="https://docs.example.com",
    hybrid_bm25_weight=0.2,      # 20% keyword
    hybrid_semantic_weight=0.8,  # 80% semantic
)
```

## Configuration

### Using ServerConfig

```python
from llm_api_server import ServerConfig

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

## Evaluation Framework

LLM API Server includes a comprehensive evaluation framework for testing your LLM applications.

### Quick Example

```python
from llm_api_server.eval import Evaluator, TestCase, HTMLReporter

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

See [`llm_api_server/eval/README.md`](llm_api_server/eval/README.md) for complete documentation and examples.

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

## Example Projects

See these projects using LLM API Server:

- **[Ivan](https://github.com/assareh/ivan)** - HashiCorp documentation expert
- **[milesoss](https://github.com/assareh/milesoss)** - Credit card rewards optimizer

## Development

### Using uv (recommended)

```bash
# Clone and install with dev dependencies
git clone https://github.com/assareh/llm-api-server.git
cd llm-api-server
uv sync --extra dev

# Run tests
uv run pytest

# Format and lint
./lint.sh

# Or manually
uv run black llm_api_server/
uv run ruff check --fix llm_api_server/
```

### Using pip

```bash
# Clone the repository
git clone https://github.com/assareh/llm-api-server.git
cd llm-api-server

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black llm_api_server/

# Lint
ruff check --fix llm_api_server/
```

## License

MIT License - see LICENSE file for details.

## Projects Using LLM API Server

- **[Ivan](https://github.com/assareh/ivan)** - AI assistant for HashiCorp solutions engineers
- **[Miles OSS](https://github.com/assareh/milesoss)** - Credit card rewards optimization assistant

## Acknowledgments

Built with:
- [Flask](https://flask.palletsprojects.com/)
- [LangChain](https://www.langchain.com/)
- [Open Web UI](https://github.com/open-webui/open-webui)

## Support

If you find this project helpful, consider supporting its development:

[![Buy Me A Coffee](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/assareh)

Your support helps maintain and improve this project!
