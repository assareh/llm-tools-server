# LLM API Server Tests

Test suite for the LLM API Server framework.

## Setup

Install test dependencies:

```bash
uv sync --extra dev
```

## Running Tests

Run all tests:

```bash
uv run pytest
```

Run with coverage:

```bash
uv run pytest --cov=llm_tools_server --cov-report=html
```

Run only unit tests (fast, no external dependencies):

```bash
uv run pytest -m unit
```

Run specific test file:

```bash
uv run pytest tests/test_config.py
```

Run specific test:

```bash
uv run pytest tests/test_config.py::TestServerConfig::test_default_config
```

## Test Organization

- `conftest.py` - Shared fixtures and test configuration
- `test_config.py` - Configuration loading and environment variable tests
- `test_backends.py` - Backend health checks and retry logic tests
- `test_server.py` - Server initialization, tool execution, and route tests
- `test_web_search.py` - Web search tool (mocked Ollama, no network)
- `test_html_report.py` - HTML reporter generation (asserts collapsible sections/markdown)
- `test_builtin_tools.py` - Calculator safety regressions
- `test_rag_indexer.py` - Cache load/child-parent mapping regression
- `test_rag_chunker.py` - Chunker parent/child linking (skips if optional deps missing)
- `test_rag_crawler.py` - Redirect/content-type filtering (mocked requests)

## Test Markers

Tests are marked with the following categories:

- `@pytest.mark.unit` - Unit tests (fast, no external dependencies)
- `@pytest.mark.integration` - Integration tests (may require backends running)

## Writing Tests

### Using Fixtures

Common fixtures are defined in `conftest.py`:

```python
def test_something(default_config, sample_tools):
    server = LLMServer(...)
    # test code
```

### Mocking External Calls

Use `unittest.mock` or `pytest-mock` for mocking:

```python
from unittest.mock import patch

def test_backend_call(default_config):
    with patch("llm_tools_server.backends.requests.get") as mock_get:
        mock_get.return_value.json.return_value = {"data": []}
        # test code
```

### Testing Flask Routes

Use the test client:

```python
def test_endpoint(default_config, sample_tools):
    server = LLMServer(...)
    with server.app.test_client() as client:
        response = client.get("/health")
        assert response.status_code == 200
```

## Test Coverage

Current test coverage includes:

- ✅ Configuration loading and validation
- ✅ Environment variable parsing
- ✅ Backend health checks (Ollama and LM Studio)
- ✅ Retry logic with exponential backoff
- ✅ Server initialization
- ✅ Tool execution and error handling
- ✅ System prompt loading and caching
- ✅ Flask route validation
- ✅ Request validation (missing fields, invalid JSON)

## Future Test Additions

Areas that could benefit from additional tests:

- Full chat completion flow with mocked backend responses
- Streaming response generation
- Tool calling loop with multiple iterations
- WebUI integration testing
- Performance/load testing

## CI/CD Integration

Tests can be run in CI/CD pipelines:

```yaml
# Example GitHub Actions
- name: Run tests
  run: |
    uv sync --extra dev
    uv run pytest -v
```

## Troubleshooting

**Import errors**: Ensure you've installed the package in development mode:
```bash
uv sync --extra dev
```

**Fixture not found**: Check that fixtures are defined in `conftest.py` or imported correctly.

**Test isolation**: Tests should be independent. Use fixtures and avoid shared state.
